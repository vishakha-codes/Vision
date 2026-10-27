"""Deploy the v2 kids chat cloud side (idempotent, boto3 only; no CDK / Docker).

    cd vision && .venv/bin/python -m kids_chat.infra.deploy                # everything
    .venv/bin/python -m kids_chat.infra.deploy --only runtime              # re-zip + update the runtime
    .venv/bin/python -m kids_chat.infra.deploy --only guardrails

Creates, in the caller's account / us-east-1, all tagged
``project=screen-guardian`` + ``component=v2-chat`` (never touches the v1
agent's resources):

- Bedrock Guardrails ``sg-v2-kids`` (aggressive) and ``sg-v2-teens`` (medium),
  each with a published version
- AgentCore Memory ``screen_guardian_kids_chat_mem``: 7-day events,
  summary + userPreference strategies only (no semantic facts: PII risk)
- S3 bucket ``sg-v2-kids-chat-<account>`` for the runtime code zip
- IAM role ``sg-v2-kids-chat-runtime`` (least privilege)
- AgentCore Runtime ``screen_guardian_kids_chat`` (direct code zip, PYTHON_3_12)

The pattern (direct code zip in S3, stdlib HTTP server) follows
``vision/infra/deploy.py`` (read-only reference). Outputs go to
``kids_chat/infra/outputs.json``; every resource is logged by hand in
``kids_chat/infra/resources-log.md``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
REGION = "us-east-1"
ACCOUNT = ""  # the caller's account, set in main()
TAGS = {"project": "screen-guardian", "component": "v2-chat"}
TAGS_LIST = [{"Key": k, "Value": v} for k, v in TAGS.items()]
BUCKET = ""
ROLE_NAME = "sg-v2-kids-chat-runtime"
RUNTIME_NAME = "screen_guardian_kids_chat"
MEMORY_NAME = "screen_guardian_kids_chat_mem"
# Nova only: Anthropic models are NOT_AVAILABLE in this account (use-case form not submitted).
CHAT_MODEL = "us.amazon.nova-2-lite-v1:0"
FALLBACK_MODEL = "us.amazon.nova-lite-v1:0"
EXPLAIN_MODEL = "us.amazon.nova-pro-v1:0"  # best pointing of the Nova models in a side-by-side test
MODELS = [CHAT_MODEL, FALLBACK_MODEL, EXPLAIN_MODEL]
OUT = HERE / "outputs.json"
BUILD = HERE / "build"
AGENT_FILES = ("main.py", "core.py", "prompts.py", "memory.py", "screen_tools.py", "guard.py")


def log(msg: str) -> None:
    print(f"[kids-chat-deploy] {msg}", flush=True)


def load_outputs() -> dict:
    return json.loads(OUT.read_text()) if OUT.is_file() else {}


def save_outputs(o: dict) -> None:
    OUT.write_text(json.dumps(o, indent=2, sort_keys=True) + "\n")


# ---------------------------------------------------------------- Guardrails

BLOCKED_KIDS = "Hmm, that's not something I can help with. Let's ask a grown-up you trust!"
BLOCKED_KIDS_OUT = "Oops, I can't share that one. A grown-up you trust can help!"
BLOCKED_TEENS = "That's outside what I can help with here. A trusted adult is a good person to ask."
BLOCKED_TEENS_OUT = "I can't share that here. A trusted adult can help with this one."


def _topic(name: str, definition: str, examples: list[str]) -> dict:
    assert len(definition) <= 200, (name, len(definition))
    assert len(examples) <= 5 and all(len(e) <= 100 for e in examples), name
    return {"name": name, "definition": definition, "examples": examples, "type": "DENY"}


COMMON_TOPICS = [
    _topic("SelfHarm", "Methods, plans, instructions or encouragement for self-harm, suicide, wanting to die, or starving or hurting oneself.",
           ["How can I hurt myself?", "What is the easiest way to die?", "I want to kill myself", "How do I stop eating so I get thin?"]),
    _topic("DrugsAlcoholVaping", "Getting, using, hiding or making drugs, alcohol, vapes or cigarettes, or how they feel to use.",
           ["How do I buy a vape?", "How can I get drunk without my parents knowing?", "What does weed feel like?"]),
    _topic("GettingWeapons", "Buying, getting, carrying or hiding real guns, ammunition, explosives or knives, especially as a child. Not history, science or stories.",
           ["Where can I buy a gun?", "How can a kid get bullets?", "Can I take a knife to school?", "Where do people get fireworks to blow things up?"]),
    _topic("Gambling", "Betting money, casino games, online gambling, loot-box betting or how to gamble.",
           ["How do I bet on football online?", "Best casino sites?", "How do I win at slots?"]),
    _topic("StrangersPersonalInfo", "Meeting people from the internet in person, or sharing home address, school, phone number, passwords or photos online.",
           ["Should I meet my online friend at the park?", "Can I send my address to a gamer friend?", "What school should I tell him I go to?"]),
    _topic("Extremism", "Promoting extremist groups, terrorism, hateful ideologies, or recruiting people to them.",
           ["How do I join a terrorist group?", "Why is that extremist group right?"]),
    _topic("DangerousChallenges", "Risky online challenges or stunts that could injure someone, such as choking, fire, or eating dangerous items.",
           ["How do I do the blackout challenge?", "Is the fire challenge fun?", "How many detergent pods can I eat?"]),
    _topic("BypassProtection", "Disabling, bypassing, tricking or uninstalling Screen Guardian, parental controls, the PIN, filters or blur.",
           ["How do I turn off the screen guardian without the PIN?", "How can I stop the blur from working?", "What is the parent PIN?"]),
    _topic("DescribeBlockedImagery", "Describing, guessing or giving details about what a covered, blurred or blocked picture showed, including body parts or nudity.",
           ["What was in the picture you blurred?", "Describe the image that got covered.", "Which body parts were showing?"]),
]

KIDS_TOPICS = [
    _topic("RomanceSex", "Sexual activity, sexual body parts, dating advice, flirting, kissing, or romantic or sexual role-play.",
           ["How do people have sex?", "Pretend to be my girlfriend", "How do I kiss someone?", "Tell me a romantic story about us"]),
] + COMMON_TOPICS

TEENS_TOPICS = [
    _topic("ExplicitSexualContent", "Explicit or graphic sexual content, pornography, sexting, nudes or sexual role-play. Not factual, clinical puberty or sex-education questions.",
           ["Write me a sex scene", "How do I send nudes?", "Pretend to be my girlfriend and flirt", "Where can I find porn?"]),
] + COMMON_TOPICS


def _filters(levels: dict[str, str]) -> list[dict]:
    out = []
    for kind, lvl in levels.items():
        f = {"type": kind, "inputStrength": lvl, "outputStrength": "NONE" if kind == "PROMPT_ATTACK" else lvl}
        if kind != "PROMPT_ATTACK":
            f["inputModalities"] = ["TEXT", "IMAGE"]
            f["outputModalities"] = ["TEXT"]
        else:
            f["inputModalities"] = ["TEXT"]
        out.append(f)
    return out


PII_BLOCK = ["ADDRESS", "PHONE", "US_SOCIAL_SECURITY_NUMBER", "CREDIT_DEBIT_CARD_NUMBER", "PASSWORD", "PIN",
             "AWS_ACCESS_KEY", "AWS_SECRET_KEY"]
# NAME is anonymised on the child's input (their own / friends' names never reach memory),
# but not on the model's output, so history answers ("Who was Ada Lovelace?") still read.
PII_ANON = ["EMAIL", "AGE", "IP_ADDRESS", "USERNAME", "NAME"]


def _pii() -> list[dict]:
    out = [{"type": t, "action": "BLOCK", "inputAction": "BLOCK", "outputAction": "BLOCK",
            "inputEnabled": True, "outputEnabled": True} for t in PII_BLOCK]
    for t in PII_ANON:
        out.append({"type": t, "action": "ANONYMIZE", "inputAction": "ANONYMIZE", "outputAction": "ANONYMIZE",
                    "inputEnabled": True, "outputEnabled": t != "NAME"})
    return out


def guardrail_config(band: str) -> dict:
    if band == "kids":
        levels = {"SEXUAL": "HIGH", "VIOLENCE": "HIGH", "HATE": "HIGH", "INSULTS": "HIGH", "MISCONDUCT": "HIGH", "PROMPT_ATTACK": "HIGH"}
        topics, words = KIDS_TOPICS, [{"text": w} for w in ("stupid idiot", "shut up loser", "kill yourself")]
        blocked_in, blocked_out = BLOCKED_KIDS, BLOCKED_KIDS_OUT
    else:
        levels = {"SEXUAL": "HIGH", "VIOLENCE": "MEDIUM", "HATE": "HIGH", "INSULTS": "MEDIUM", "MISCONDUCT": "HIGH", "PROMPT_ATTACK": "HIGH"}
        topics, words = TEENS_TOPICS, [{"text": "kill yourself"}]
        blocked_in, blocked_out = BLOCKED_TEENS, BLOCKED_TEENS_OUT
    return dict(
        name=f"sg-v2-{band}",
        description=f"Screen Guardian v2 kids chat ({'aggressive / ages 6-11' if band == 'kids' else 'medium / ages 12-16'})",
        topicPolicyConfig={"topicsConfig": topics},
        contentPolicyConfig={"filtersConfig": _filters(levels)},
        wordPolicyConfig={"wordsConfig": words, "managedWordListsConfig": [{"type": "PROFANITY"}]},
        sensitiveInformationPolicyConfig={"piiEntitiesConfig": _pii()},
        blockedInputMessaging=blocked_in,
        blockedOutputsMessaging=blocked_out,
    )


def guardrails(o: dict) -> None:
    br = boto3.client("bedrock", region_name=REGION)
    existing = {g["name"]: g for g in br.list_guardrails(maxResults=100)["guardrails"]}
    for band in ("kids", "teens"):
        cfg = guardrail_config(band)
        name = cfg["name"]
        if name in existing:
            gid = existing[name]["id"]
            br.update_guardrail(guardrailIdentifier=gid, **cfg)
            log(f"updated guardrail {name} ({gid}) DRAFT")
        else:
            gid = br.create_guardrail(tags=[{"key": k, "value": v} for k, v in TAGS.items()], **cfg)["guardrailId"]
            log(f"created guardrail {name} ({gid})")
        for _ in range(60):
            if br.get_guardrail(guardrailIdentifier=gid)["status"] == "READY":
                break
            time.sleep(2)
        ver = br.create_guardrail_version(guardrailIdentifier=gid, description="v2 chat")["version"]
        for _ in range(60):
            if br.get_guardrail(guardrailIdentifier=gid, guardrailVersion=ver)["status"] == "READY":
                break
            time.sleep(2)
        log(f"guardrail {name} version {ver} READY")
        o[f"guardrail_{band}_id"] = gid
        o[f"guardrail_{band}_version"] = ver
        o[f"guardrail_{band}_arn"] = f"arn:aws:bedrock:{REGION}:{ACCOUNT}:guardrail/{gid}"


# ---------------------------------------------------------------- Memory

def memory(o: dict) -> None:
    ctl = boto3.client("bedrock-agentcore-control", region_name=REGION)
    found = [m for m in ctl.list_memories()["memories"] if m.get("id", "").startswith(MEMORY_NAME)]
    if found:
        mid = found[0]["id"]
    else:
        mid = ctl.create_memory(
            name=MEMORY_NAME,
            description="Screen Guardian v2 kids chat: per-child-profile history (masked text only), summaries, interests",
            eventExpiryDuration=7,
            memoryStrategies=[
                {"summaryMemoryStrategy": {"name": "session_summary",
                                           "description": "Per-session topic summaries (parent view, cross-session context)",
                                           "namespaces": ["/summaries/{actorId}/{sessionId}/"]}},
                {"userPreferenceMemoryStrategy": {"name": "interests",
                                                  "description": "Child interests and reading-level preferences",
                                                  "namespaces": ["/preferences/{actorId}/"]}},
            ],
            tags=TAGS,
        )["memory"]["id"]
        log(f"created memory {mid}")
    for _ in range(120):
        m = ctl.get_memory(memoryId=mid)["memory"]
        if m["status"] in ("ACTIVE", "FAILED"):
            break
        time.sleep(5)
    o["memory_id"], o["memory_arn"], o["memory_status"] = mid, m["arn"], m["status"]
    o["memory_strategies"] = {s["type"]: s["strategyId"] for s in m.get("strategies", [])}
    log(f"memory {mid} {m['status']}")


# ---------------------------------------------------------------- IAM + S3

def bucket(o: dict) -> None:
    s3 = boto3.client("s3", region_name=REGION)
    try:
        s3.head_bucket(Bucket=BUCKET)
    except ClientError:
        s3.create_bucket(Bucket=BUCKET)
        s3.put_public_access_block(Bucket=BUCKET, PublicAccessBlockConfiguration={
            "BlockPublicAcls": True, "IgnorePublicAcls": True, "BlockPublicPolicy": True, "RestrictPublicBuckets": True})
        s3.put_bucket_tagging(Bucket=BUCKET, Tagging={"TagSet": TAGS_LIST})
        s3.put_bucket_lifecycle_configuration(Bucket=BUCKET, LifecycleConfiguration={"Rules": [
            {"ID": "expire-old-zips", "Status": "Enabled", "Filter": {"Prefix": "runtime/"}, "Expiration": {"Days": 14}}]})
        log(f"created bucket {BUCKET}")
    o["bucket"] = f"arn:aws:s3:::{BUCKET}"


def role(o: dict) -> None:
    iam = boto3.client("iam")
    trust = {"Version": "2012-10-17", "Statement": [{
        "Effect": "Allow", "Principal": {"Service": "bedrock-agentcore.amazonaws.com"}, "Action": "sts:AssumeRole",
        "Condition": {"StringEquals": {"aws:SourceAccount": ACCOUNT}}}]}
    try:
        arn = iam.get_role(RoleName=ROLE_NAME)["Role"]["Arn"]
    except iam.exceptions.NoSuchEntityException:
        arn = iam.create_role(RoleName=ROLE_NAME, AssumeRolePolicyDocument=json.dumps(trust),
                              Description="Screen Guardian v2 kids chat AgentCore runtime", Tags=TAGS_LIST)["Role"]["Arn"]
        log(f"created role {ROLE_NAME}")
    models = sorted(set(MODELS))
    gr = [o[f"guardrail_{b}_arn"] for b in ("kids", "teens") if o.get(f"guardrail_{b}_arn")]
    policy = {"Version": "2012-10-17", "Statement": [
        {"Sid": "Models", "Effect": "Allow", "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
         "Resource": [f"arn:aws:bedrock:{REGION}:{ACCOUNT}:inference-profile/{m}" for m in models]
         + [f"arn:aws:bedrock:*::foundation-model/{m.split('.', 1)[1]}" for m in models]},
        {"Sid": "Guardrails", "Effect": "Allow", "Action": ["bedrock:ApplyGuardrail"], "Resource": gr},
        {"Sid": "Memory", "Effect": "Allow",
         "Action": ["bedrock-agentcore:CreateEvent", "bedrock-agentcore:ListEvents", "bedrock-agentcore:GetEvent",
                    "bedrock-agentcore:RetrieveMemoryRecords", "bedrock-agentcore:ListMemoryRecords",
                    "bedrock-agentcore:ListSessions", "bedrock-agentcore:ListActors"],
         "Resource": o["memory_arn"]},
        {"Sid": "Code", "Effect": "Allow", "Action": ["s3:GetObject"], "Resource": f"arn:aws:s3:::{BUCKET}/runtime/*"},
        {"Sid": "Logs", "Effect": "Allow", "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents",
                                                     "logs:DescribeLogStreams"],
         "Resource": [f"arn:aws:logs:{REGION}:{ACCOUNT}:log-group:/aws/bedrock-agentcore/runtimes/*"]},
        {"Effect": "Allow", "Action": ["logs:DescribeLogGroups"], "Resource": f"arn:aws:logs:{REGION}:{ACCOUNT}:log-group:*"},
        {"Sid": "Traces", "Effect": "Allow", "Action": ["xray:PutTraceSegments", "xray:PutTelemetryRecords",
                                                       "xray:GetSamplingRules", "xray:GetSamplingTargets"], "Resource": "*"},
        {"Sid": "Metrics", "Effect": "Allow", "Action": "cloudwatch:PutMetricData", "Resource": "*",
         "Condition": {"StringLike": {"cloudwatch:namespace": ["bedrock-agentcore"]}}},
        {"Sid": "WorkloadIdentity", "Effect": "Allow",
         "Action": ["bedrock-agentcore:GetWorkloadAccessToken", "bedrock-agentcore:GetWorkloadAccessTokenForJWT",
                    "bedrock-agentcore:GetWorkloadAccessTokenForUserId"],
         "Resource": [f"arn:aws:bedrock-agentcore:{REGION}:{ACCOUNT}:workload-identity-directory/default",
                      f"arn:aws:bedrock-agentcore:{REGION}:{ACCOUNT}:workload-identity-directory/default/workload-identity/*"]},
    ]}
    iam.put_role_policy(RoleName=ROLE_NAME, PolicyName="inline", PolicyDocument=json.dumps(policy))
    o["role_runtime"] = arn
    time.sleep(8)  # IAM propagation


# ---------------------------------------------------------------- Runtime

def build_zip() -> Path:
    if BUILD.exists():
        shutil.rmtree(BUILD)
    pkg = BUILD / "runtime"
    pkg.mkdir(parents=True)
    subprocess.run(["uv", "pip", "install", "--quiet", "--target", str(pkg), "--python-platform", "aarch64-manylinux2014",
                    "--python-version", "3.12", "--only-binary", ":all:", "boto3>=1.43"], check=True)
    for f in AGENT_FILES:
        shutil.copy(PKG / "agent" / f, pkg / f)
    out = BUILD / "runtime.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(pkg.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                z.write(p, p.relative_to(pkg))
    return out


def runtime_env(o: dict) -> dict:
    return {
        "SG_MEMORY_ID": o["memory_id"],
        "SG_GR_KIDS_ID": o["guardrail_kids_id"], "SG_GR_KIDS_VER": str(o["guardrail_kids_version"]),
        "SG_GR_TEENS_ID": o["guardrail_teens_id"], "SG_GR_TEENS_VER": str(o["guardrail_teens_version"]),
        "SG_CHAT_MODEL": CHAT_MODEL, "SG_FALLBACK_MODEL": FALLBACK_MODEL, "SG_EXPLAIN_MODEL": EXPLAIN_MODEL,
        "PYTHONUNBUFFERED": "1",
    }


def runtime(o: dict) -> None:
    ctl = boto3.client("bedrock-agentcore-control", region_name=REGION)
    s3 = boto3.client("s3", region_name=REGION)
    zpath = build_zip()
    key = f"runtime/{int(time.time())}/runtime.zip"
    s3.upload_file(str(zpath), BUCKET, key)
    log(f"uploaded s3://{BUCKET}/{key} ({zpath.stat().st_size // 1024} KiB)")
    artifact = {"codeConfiguration": {"code": {"s3": {"bucket": BUCKET, "prefix": key}},
                                      "runtime": "PYTHON_3_12", "entryPoint": ["main.py"]}}
    common = dict(agentRuntimeArtifact=artifact, roleArn=o["role_runtime"], networkConfiguration={"networkMode": "PUBLIC"},
                  environmentVariables=runtime_env(o), protocolConfiguration={"serverProtocol": "HTTP"},
                  description="Screen Guardian v2 kids chat (guardrailed, memory per child profile)")
    found = [r for r in ctl.list_agent_runtimes(maxResults=100)["agentRuntimes"] if r["agentRuntimeName"] == RUNTIME_NAME]
    if found:
        rid = found[0]["agentRuntimeId"]
        ctl.update_agent_runtime(agentRuntimeId=rid, **common)
        log(f"updated runtime {rid}")
    else:
        rid = None
        for attempt in range(6):
            try:
                rid = ctl.create_agent_runtime(agentRuntimeName=RUNTIME_NAME, tags=TAGS, **common)["agentRuntimeId"]
                break
            except ClientError as exc:  # role not yet assumable
                if attempt == 5 or "role" not in str(exc).lower():
                    raise
                time.sleep(8)
        log(f"created runtime {rid}")
    for _ in range(120):
        r = ctl.get_agent_runtime(agentRuntimeId=rid)
        if r["status"] in ("READY", "CREATE_FAILED", "UPDATE_FAILED"):
            break
        time.sleep(5)
    o["runtime_id"], o["runtime_arn"], o["runtime_status"] = rid, r["agentRuntimeArn"], r["status"]
    o["runtime_version"] = r.get("agentRuntimeVersion")
    o["models"] = {"chat": CHAT_MODEL, "fallback": FALLBACK_MODEL, "explain": EXPLAIN_MODEL}
    if r["status"] != "READY":
        log(f"runtime status {r['status']}: {r.get('failureReason', '')}")


STEPS = [("guardrails", guardrails), ("memory", memory), ("bucket", bucket), ("role", role), ("runtime", runtime)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=[n for n, _ in STEPS], default=None)
    args = ap.parse_args()
    global ACCOUNT, BUCKET
    ACCOUNT = boto3.client("sts").get_caller_identity()["Account"]
    BUCKET = f"sg-v2-kids-chat-{ACCOUNT}"
    o = load_outputs()
    o.update({"region": REGION, "account": ACCOUNT})
    for name, fn in STEPS:
        if args.only and name != args.only:
            continue
        log(f"--- {name}")
        fn(o)
        save_outputs(o)
    print(json.dumps(o, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
