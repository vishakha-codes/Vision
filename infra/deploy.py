"""Deploy the Screen Guardian agent lane (idempotent, boto3; no CDK / Docker).

    cd vision && .venv/bin/python infra/deploy.py            # create or update everything
    .venv/bin/python infra/deploy.py --only runtime          # re-zip + update the runtime

What it creates in the caller's account / us-east-1 (all tagged
``project=screen-guardian``), modelled on the media-agent-harness shape
(orchestrator on AgentCore Runtime, tools behind AgentCore Gateway), minimal:

- IAM roles: ``screen-guardian-tool-store``, ``screen-guardian-tool-cool``,
  ``screen-guardian-gateway``, ``screen-guardian-agent-runtime`` (least privilege)
- Lambda ``sg-tool-store`` (DynamoDB tools), ``sg-tool-cool`` (VPC → the box)
- SG ``screen-guardian-tool-cool`` + box SG ingress 8443 from it (private IP)
- AgentCore Gateway ``screen-guardian-gw`` (MCP, AWS_IAM) + targets
  ``sg-store`` and ``sg-cool``
- AgentCore Runtime ``screen_guardian_steward`` (direct code zip in S3)

Pre-existing (made by hand, see cloud/deploy/resources-log.md): DynamoDB
``screen-guardian-agent``, S3 ``screen-guardian-agent-<account-id>``, Guardrail
``sg-kids``, secret ``screen-guardian/l2-api-token``, the box + its role.
Outputs → ``infra/outputs.json``. Account-specific ids (box, SG, subnet, VPC,
Guardrail) come from ``vision/aws.local.json``; both files stay out of git.
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

VISION = Path(__file__).resolve().parents[1]
REGION = "us-east-1"
ACCOUNT = ""  # set by load_ids()
TAG = {"project": "screen-guardian"}
TAGS_LIST = [{"Key": "project", "Value": "screen-guardian"}]
TABLE = "screen-guardian-agent"
BUCKET = ""
GUARDRAIL_ID = ""
GUARDRAIL_VERSION = "3"
SECRET_ID = "screen-guardian/l2-api-token"
BOX_INSTANCE = BOX_SG = BOX_SUBNET = VPC = ""  # set by load_ids(); the subnet is the box's
LOCAL_IDS = VISION / "aws.local.json"
MODELS = "us.anthropic.claude-sonnet-4-6,us.amazon.nova-2-lite-v1:0,us.amazon.nova-pro-v1:0"
RUNTIME_NAME = "screen_guardian_steward"
GATEWAY_NAME = "screen-guardian-gw"
OUT = VISION / "infra" / "outputs.json"
BUILD = VISION / "infra" / "build"

iam = boto3.client("iam")
lam = boto3.client("lambda", region_name=REGION)
ec2 = boto3.client("ec2", region_name=REGION)
ctl = boto3.client("bedrock-agentcore-control", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)
sm = boto3.client("secretsmanager", region_name=REGION)


def log(msg: str) -> None:
    print(f"[deploy] {msg}", flush=True)


def load_outputs() -> dict:
    return json.loads(OUT.read_text()) if OUT.is_file() else {}


def save_outputs(o: dict) -> None:
    OUT.write_text(json.dumps(o, indent=2, sort_keys=True))


# ---------------------------------------------------------------- IAM

def trust(service: str) -> str:
    stmt = {"Effect": "Allow", "Principal": {"Service": service}, "Action": "sts:AssumeRole"}
    if service == "bedrock-agentcore.amazonaws.com":
        stmt["Condition"] = {"StringEquals": {"aws:SourceAccount": ACCOUNT}}
    return json.dumps({"Version": "2012-10-17", "Statement": [stmt]})


def ensure_role(name: str, service: str, policy: dict, managed: list[str] = ()) -> str:
    try:
        arn = iam.get_role(RoleName=name)["Role"]["Arn"]
    except iam.exceptions.NoSuchEntityException:
        arn = iam.create_role(RoleName=name, AssumeRolePolicyDocument=trust(service),
                              Description="Screen Guardian agent lane", Tags=TAGS_LIST)["Role"]["Arn"]
        log(f"created role {name}")
    iam.put_role_policy(RoleName=name, PolicyName="inline", PolicyDocument=json.dumps(policy))
    for m in managed:
        iam.attach_role_policy(RoleName=name, PolicyArn=m)
    return arn


def logs_stmt(prefix: str) -> dict:
    return {"Effect": "Allow", "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents",
                                          "logs:DescribeLogStreams", "logs:DescribeLogGroups"],
            "Resource": [f"arn:aws:logs:{REGION}:{ACCOUNT}:log-group:{prefix}*"]}


def roles(o: dict) -> None:
    table_arn = f"arn:aws:dynamodb:{REGION}:{ACCOUNT}:table/{TABLE}"
    o["role_tool_store"] = ensure_role("screen-guardian-tool-store", "lambda.amazonaws.com", {
        "Version": "2012-10-17", "Statement": [
            logs_stmt("/aws/lambda/sg-tool-store"),
            {"Effect": "Allow", "Action": ["dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:UpdateItem"], "Resource": table_arn},
        ]})
    o["role_tool_cool"] = ensure_role("screen-guardian-tool-cool", "lambda.amazonaws.com", {
        "Version": "2012-10-17", "Statement": [logs_stmt("/aws/lambda/sg-tool-cool")]},
        managed=["arn:aws:iam::aws:policy/service-role/AWSLambdaVPCAccessExecutionRole"])
    fns = [f"arn:aws:lambda:{REGION}:{ACCOUNT}:function:sg-tool-store", f"arn:aws:lambda:{REGION}:{ACCOUNT}:function:sg-tool-cool"]
    o["role_gateway"] = ensure_role("screen-guardian-gateway", "bedrock-agentcore.amazonaws.com", {
        "Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Action": "lambda:InvokeFunction", "Resource": fns}]})
    models = [m for m in MODELS.split(",")]
    o["role_runtime"] = ensure_role("screen-guardian-agent-runtime", "bedrock-agentcore.amazonaws.com", {
        "Version": "2012-10-17", "Statement": [
            {"Sid": "Models", "Effect": "Allow", "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
             "Resource": [f"arn:aws:bedrock:{REGION}:{ACCOUNT}:inference-profile/{m}" for m in models]
             + [f"arn:aws:bedrock:*::foundation-model/{m.split('.', 1)[1]}" for m in models]},
            {"Sid": "Guardrail", "Effect": "Allow", "Action": ["bedrock:ApplyGuardrail"],
             "Resource": f"arn:aws:bedrock:{REGION}:{ACCOUNT}:guardrail/{GUARDRAIL_ID}"},
            {"Sid": "Gateway", "Effect": "Allow", "Action": ["bedrock-agentcore:InvokeGateway"],
             "Resource": f"arn:aws:bedrock-agentcore:{REGION}:{ACCOUNT}:gateway/*"},
            {"Sid": "Code", "Effect": "Allow", "Action": ["s3:GetObject"], "Resource": f"arn:aws:s3:::{BUCKET}/runtime/*"},
            logs_stmt("/aws/bedrock-agentcore/runtimes/"),
            {"Effect": "Allow", "Action": ["logs:DescribeLogGroups"], "Resource": f"arn:aws:logs:{REGION}:{ACCOUNT}:log-group:*"},
            {"Sid": "Metrics", "Effect": "Allow", "Action": "cloudwatch:PutMetricData", "Resource": "*",
             "Condition": {"StringLike": {"cloudwatch:namespace": ["ScreenGuardian", "bedrock-agentcore"]}}},
            {"Sid": "Traces", "Effect": "Allow", "Action": ["xray:PutTraceSegments", "xray:PutTelemetryRecords",
                                                            "xray:GetSamplingRules", "xray:GetSamplingTargets"], "Resource": "*"},
            {"Sid": "WorkloadIdentity", "Effect": "Allow",
             "Action": ["bedrock-agentcore:GetWorkloadAccessToken", "bedrock-agentcore:GetWorkloadAccessTokenForJWT",
                        "bedrock-agentcore:GetWorkloadAccessTokenForUserId"],
             "Resource": [f"arn:aws:bedrock-agentcore:{REGION}:{ACCOUNT}:workload-identity-directory/default",
                          f"arn:aws:bedrock-agentcore:{REGION}:{ACCOUNT}:workload-identity-directory/default/workload-identity/*"]},
        ]})
    time.sleep(10)  # IAM propagation


# ---------------------------------------------------------------- network

def network(o: dict) -> None:
    name = "screen-guardian-tool-cool"
    found = ec2.describe_security_groups(Filters=[{"Name": "group-name", "Values": [name]}, {"Name": "vpc-id", "Values": [VPC]}])["SecurityGroups"]
    if found:
        sg = found[0]["GroupId"]
    else:
        sg = ec2.create_security_group(GroupName=name, Description="sg-tool-cool Lambda: egress to the L2 box only",
                                       VpcId=VPC, TagSpecifications=[{"ResourceType": "security-group", "Tags": TAGS_LIST}])["GroupId"]
        log(f"created SG {sg}")
        # replace the default allow-all egress with 8443 to the box SG only
        ec2.revoke_security_group_egress(GroupId=sg, IpPermissions=[{"IpProtocol": "-1", "IpRanges": [{"CidrIp": "0.0.0.0/0"}]}])
        ec2.authorize_security_group_egress(GroupId=sg, IpPermissions=[{"IpProtocol": "tcp", "FromPort": 8443, "ToPort": 8443,
                                                                        "UserIdGroupPairs": [{"GroupId": BOX_SG}]}])
    try:
        ec2.authorize_security_group_ingress(GroupId=BOX_SG, IpPermissions=[{
            "IpProtocol": "tcp", "FromPort": 8443, "ToPort": 8443,
            "UserIdGroupPairs": [{"GroupId": sg, "Description": "sg-tool-cool Lambda (private)"}]}])
        log("box SG: 8443 from the tool Lambda SG")
    except ClientError as exc:
        if "Duplicate" not in str(exc):
            raise
    o["sg_tool_cool"] = sg
    inst = ec2.describe_instances(InstanceIds=[BOX_INSTANCE])["Reservations"][0]["Instances"][0]
    o["box_private_ip"] = inst["PrivateIpAddress"]


# ---------------------------------------------------------------- Lambda

def zip_files(files: dict[str, Path]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for arc, path in files.items():
            z.write(path, arc)
    return buf.getvalue()


def ensure_lambda(name: str, role: str, code: bytes, env: dict, vpc: dict | None = None) -> str:
    kw = dict(Runtime="python3.12", Handler="handler.handler", Role=role, Timeout=15, MemorySize=256,
              Architectures=["arm64"], Environment={"Variables": env})
    if vpc:
        kw["VpcConfig"] = vpc
    try:
        lam.get_function(FunctionName=name)
        lam.update_function_code(FunctionName=name, ZipFile=code, Architectures=["arm64"])
        lam.get_waiter("function_updated_v2").wait(FunctionName=name)
        lam.update_function_configuration(FunctionName=name, **{k: v for k, v in kw.items() if k != "Architectures"})
        lam.get_waiter("function_updated_v2").wait(FunctionName=name)
    except lam.exceptions.ResourceNotFoundException:
        for attempt in range(6):
            try:
                lam.create_function(FunctionName=name, Code={"ZipFile": code}, Tags=TAG,
                                    Description="Screen Guardian Gateway tool", **kw)
                break
            except ClientError as exc:  # role not yet assumable
                if "assume" not in str(exc).lower() or attempt == 5:
                    raise
                time.sleep(5)
        lam.get_waiter("function_active_v2").wait(FunctionName=name)
        log(f"created lambda {name}")
    return lam.get_function(FunctionName=name)["Configuration"]["FunctionArn"]


def lambdas(o: dict) -> None:
    src = VISION / "agent" / "tools_lambda"
    o["lambda_store"] = ensure_lambda("sg-tool-store", o["role_tool_store"],
                                      zip_files({"handler.py": src / "store.py"}), {"SG_TABLE": TABLE})
    token = sm.get_secret_value(SecretId=SECRET_ID)["SecretString"].strip()
    cert = subprocess.run(["ssh", "-i", str(VISION / ".secrets" / "screen-guardian-cool.pem"), "-o", "BatchMode=yes",
                           f"ubuntu@{box_public_ip()}", "cat /home/ubuntu/sg-tls/cert.pem"],
                          capture_output=True, text=True, check=True).stdout
    o["lambda_cool"] = ensure_lambda(
        "sg-tool-cool", o["role_tool_cool"], zip_files({"handler.py": src / "cool.py"}),
        {"SG_BOX_URL": f"https://{o['box_private_ip']}:8443", "SG_L2_TOKEN": token, "SG_BOX_CERT_PEM": cert},
        vpc={"SubnetIds": [BOX_SUBNET], "SecurityGroupIds": [o["sg_tool_cool"]]})


def box_public_ip() -> str:
    return ec2.describe_instances(InstanceIds=[BOX_INSTANCE])["Reservations"][0]["Instances"][0]["PublicIpAddress"]


# ---------------------------------------------------------------- Gateway

SCHEMAS = json.loads((VISION / "agent" / "tool_schemas.json").read_text())["targets"]
STORE_TOOLS = SCHEMAS["sg-store"]
COOL_TOOLS = SCHEMAS["sg-cool"]


def gateway(o: dict) -> None:
    gws = [g for g in ctl.list_gateways()["items"] if g["name"] == GATEWAY_NAME]
    if gws:
        gid = gws[0]["gatewayId"]
    else:
        gid = ctl.create_gateway(name=GATEWAY_NAME, roleArn=o["role_gateway"], protocolType="MCP",
                                 authorizerType="AWS_IAM", description="Screen Guardian agent tools (COOL + store)",
                                 tags=TAG)["gatewayId"]
        log(f"created gateway {gid}")
    for _ in range(60):
        g = ctl.get_gateway(gatewayIdentifier=gid)
        if g["status"] in ("READY", "FAILED"):
            break
        time.sleep(3)
    o["gateway_id"], o["gateway_url"], o["gateway_arn"] = gid, g["gatewayUrl"], g["gatewayArn"]
    existing = {t["name"]: t["targetId"] for t in ctl.list_gateway_targets(gatewayIdentifier=gid)["items"]}
    for tname, arn, tools in (("sg-store", o["lambda_store"], STORE_TOOLS), ("sg-cool", o["lambda_cool"], COOL_TOOLS)):
        cfg = {"mcp": {"lambda": {"lambdaArn": arn, "toolSchema": {"inlinePayload": tools}}}}
        cred = [{"credentialProviderType": "GATEWAY_IAM_ROLE"}]
        if tname in existing:
            ctl.update_gateway_target(gatewayIdentifier=gid, targetId=existing[tname], name=tname,
                                      targetConfiguration=cfg, credentialProviderConfigurations=cred)
            o[f"target_{tname}"] = existing[tname]
        else:
            o[f"target_{tname}"] = ctl.create_gateway_target(gatewayIdentifier=gid, name=tname, targetConfiguration=cfg,
                                                             credentialProviderConfigurations=cred)["targetId"]
            log(f"created target {tname}")


# ---------------------------------------------------------------- Runtime

def build_runtime_zip() -> Path:
    if BUILD.exists():
        shutil.rmtree(BUILD)
    pkg = BUILD / "runtime"
    pkg.mkdir(parents=True)
    subprocess.run(["uv", "pip", "install", "--quiet", "--target", str(pkg), "--python-platform", "aarch64-manylinux2014",
                    "--python-version", "3.12", "--only-binary", ":all:", "boto3"], check=True)
    for f in ("main.py", "steward.py", "gateway_client.py"):
        shutil.copy(VISION / "agent" / "runtime" / f, pkg / f)
    out = BUILD / "runtime.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(pkg.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                z.write(p, p.relative_to(pkg))
    return out


def runtime(o: dict) -> None:
    zpath = build_runtime_zip()
    key = f"runtime/{int(time.time())}/runtime.zip"
    s3.upload_file(str(zpath), BUCKET, key)
    log(f"uploaded s3://{BUCKET}/{key} ({zpath.stat().st_size // 1024} KiB)")
    artifact = {"codeConfiguration": {"code": {"s3": {"bucket": BUCKET, "prefix": key}},
                                      "runtime": "PYTHON_3_12", "entryPoint": ["main.py"]}}
    env = {"SG_GATEWAY_URL": o["gateway_url"], "SG_GUARDRAIL_ID": GUARDRAIL_ID,
           "SG_GUARDRAIL_VERSION": GUARDRAIL_VERSION, "SG_BEDROCK_MODELS": MODELS, "PYTHONUNBUFFERED": "1"}
    found = [r for r in ctl.list_agent_runtimes()["agentRuntimes"] if r["agentRuntimeName"] == RUNTIME_NAME]
    common = dict(agentRuntimeArtifact=artifact, roleArn=o["role_runtime"], networkConfiguration={"networkMode": "PUBLIC"},
                  environmentVariables=env, protocolConfiguration={"serverProtocol": "HTTP"},
                  description="Screen Guardian L3 policy steward (events only, never frames)")
    if found:
        rid = found[0]["agentRuntimeId"]
        ctl.update_agent_runtime(agentRuntimeId=rid, **common)
        log(f"updated runtime {rid}")
    else:
        res = ctl.create_agent_runtime(agentRuntimeName=RUNTIME_NAME, tags=TAG, **common)
        rid = res["agentRuntimeId"]
        log(f"created runtime {rid}")
    for _ in range(100):
        r = ctl.get_agent_runtime(agentRuntimeId=rid)
        if r["status"] in ("READY", "CREATE_FAILED", "UPDATE_FAILED"):
            break
        time.sleep(5)
    o["runtime_id"], o["runtime_arn"], o["runtime_status"] = rid, r["agentRuntimeArn"], r["status"]
    o["runtime_version"] = r.get("agentRuntimeVersion")
    if r["status"] != "READY":
        log(f"runtime status {r['status']}: {r.get('failureReason', '')}")


def load_ids() -> None:
    """Fill the account-specific globals from aws.local.json and STS."""
    global ACCOUNT, BUCKET, GUARDRAIL_ID, BOX_INSTANCE, BOX_SG, BOX_SUBNET, VPC
    if not LOCAL_IDS.exists():
        sys.exit(f"missing {LOCAL_IDS} (copy aws.local.example.json and fill it in)")
    ids = json.loads(LOCAL_IDS.read_text())
    ACCOUNT = boto3.client("sts").get_caller_identity()["Account"]
    BUCKET = f"screen-guardian-agent-{ACCOUNT}"
    GUARDRAIL_ID = ids["steward_guardrail_id"]
    BOX_INSTANCE, BOX_SG = ids["box_instance_id"], ids["box_security_group_id"]
    BOX_SUBNET, VPC = ids["box_subnet_id"], ids["vpc_id"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["roles", "network", "lambdas", "gateway", "runtime"], default=None)
    args = ap.parse_args()
    load_ids()
    o = load_outputs()
    steps = [("roles", roles), ("network", network), ("lambdas", lambdas), ("gateway", gateway), ("runtime", runtime)]
    for name, fn in steps:
        if args.only and name != args.only:
            continue
        log(f"--- {name}")
        fn(o)
        save_outputs({k: v for k, v in o.items()})
    print(json.dumps(o, indent=2))


if __name__ == "__main__":
    sys.exit(main())
