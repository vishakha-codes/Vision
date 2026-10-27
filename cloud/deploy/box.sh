#!/usr/bin/env bash
# Graviton L2 box: start | ip | sg | allow-my-ip | deploy | status | stop.  Run from vision/.
#
#   cloud/deploy/box.sh start     # start the box, print the new public IP
#   cloud/deploy/box.sh sg        # SG 22 + 8443 from this Mac's current /32 only
#   cloud/deploy/box.sh allow-my-ip  # just 8443 from this Mac's current /32 (after an IP change)
#   cloud/deploy/box.sh deploy    # rsync code (no .venv/.secrets/island/media), restart sg-l2
#   cloud/deploy/box.sh status    # systemd state + /health over TLS
#   cloud/deploy/box.sh stop      # stop (never terminate)
#
# The SSH key stays in vision/.secrets/ (used with -i only). The instance and SG
# ids come from vision/aws.local.json (not in git; template aws.local.example.json)
# or from SG_INSTANCE_ID / SG_SECURITY_GROUP.
set -euo pipefail

VISION="$(cd "$(dirname "$0")/../.." && pwd)"
local_id() {
  python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get(sys.argv[2], ""))' \
    "$VISION/aws.local.json" "$1" 2>/dev/null || true
}
REGION="${AWS_REGION:-us-east-1}"
INSTANCE="${SG_INSTANCE_ID:-$(local_id box_instance_id)}"
SGID="${SG_SECURITY_GROUP:-$(local_id box_security_group_id)}"
KEY="${SG_SSH_KEY:-$VISION/.secrets/screen-guardian-cool.pem}"
if [[ -z "$INSTANCE" || -z "$SGID" ]]; then
  echo "set box_instance_id and box_security_group_id in $VISION/aws.local.json" >&2
  exit 2
fi

ip() {
  aws ec2 describe-instances --region "$REGION" --instance-ids "$INSTANCE" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text
}

ssh_box() { ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "ubuntu@$(ip)" "$@"; }

case "${1:-}" in
  start)
    aws ec2 start-instances --region "$REGION" --instance-ids "$INSTANCE" >/dev/null
    aws ec2 wait instance-running --region "$REGION" --instance-ids "$INSTANCE"
    echo "public ip: $(ip)  (changes on every start: no Elastic IP)"
    ;;
  ip) ip ;;
  sg)
    me="$(curl -s https://checkip.amazonaws.com)/32"
    for port in 22 8443; do
      aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$SGID" \
        --ip-permissions "IpProtocol=tcp,FromPort=$port,ToPort=$port,IpRanges=[{CidrIp=$me,Description=screen-guardian-mac}]" \
        >/dev/null 2>&1 && echo "allowed $port from $me" || echo "$port from $me already allowed"
    done
    echo "remove stale /32 rules by hand (never 0.0.0.0/0)"
    ;;
  allow-my-ip)
    me="$(curl -s https://checkip.amazonaws.com | tr -d '[:space:]')/32"
    if aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$SGID" \
        --ip-permissions "IpProtocol=tcp,FromPort=8443,ToPort=8443,IpRanges=[{CidrIp=$me,Description=screen-guardian-mac}]" \
        >/dev/null 2>&1; then
      echo "allowed 8443 from $me"
    else
      echo "8443 from $me already allowed (or no permission: check aws sts get-caller-identity)"
    fi
    aws ec2 describe-security-groups --region "$REGION" --group-ids "$SGID" \
      --query 'SecurityGroups[0].IpPermissions[?ToPort==`8443`].IpRanges[].CidrIp' --output text
    echo "stale /32 rules: remove by hand (never 0.0.0.0/0)"
    ;;
  deploy)
    rsync -az --delete -e "ssh -i $KEY" \
      --exclude .venv --exclude .secrets --exclude island --exclude .cache --exclude .prefs \
      --exclude .pytest_cache --exclude __pycache__ --exclude experiments --exclude 'logs*.md' \
      --exclude '*.mp4' --exclude '.models/320n.onnx' --exclude infra/build \
      "$VISION/" "ubuntu@$(ip):/home/ubuntu/vision/"
    # Keep idle TLS connections 75 s (the Mac pings /health every 30 s), once.
    ssh_box "grep -q -- '--timeout-keep-alive' /etc/systemd/system/sg-l2.service || sudo sed -i 's/--no-access-log\$/--no-access-log --timeout-keep-alive 75/' /etc/systemd/system/sg-l2.service; sudo systemctl daemon-reload"
    ssh_box 'sudo systemctl restart sg-l2 && sleep 4 && systemctl is-active sg-l2'
    ;;
  status)
    ssh_box 'systemctl is-active sg-l2; curl -sk https://127.0.0.1:8443/health | head -c 600; echo'
    ;;
  stop)
    aws ec2 stop-instances --region "$REGION" --instance-ids "$INSTANCE" >/dev/null
    echo "stopping $INSTANCE (EBS kept; public IP released)"
    ;;
  *) sed -n '2,10p' "$0"; exit 2 ;;
esac
