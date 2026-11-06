#!/usr/bin/env bash
# Provision (idempotently) and deploy Frontier Lab Intelligence to one EC2 instance.
#
#   AWS credentials: the standard AWS CLI chain (env vars, profile, SSO).
#   Usage: deploy/aws/deploy.sh            # create or update
#          FLI_INSTANCE_TYPE=t4g.medium deploy/aws/deploy.sh
#
# What it creates (all tagged Name=$FLI_NAME): an ed25519 key pair, a security group (22 from your
# current IP only; 80/443 public), a t4g.small Ubuntu 24.04 instance with a 30 GB gp3 disk and
# IMDSv2 required, and an Elastic IP. Re-running ships the current code and restarts the stack.
#
# The site is public, so .env must set FLI_BASIC_AUTH=user:password (password >= 12 characters);
# the script refuses to run otherwise. The AWS overlay sets FLI_AUTH_ALL=1: every page and API
# read needs the login, /healthz stays open for probes, and labels are recorded under the login.
set -euo pipefail

REGION="${AWS_REGION:-eu-central-1}"
NAME="${FLI_NAME:-fli}"
TYPE="${FLI_INSTANCE_TYPE:-t4g.small}"
KEY_NAME="${NAME}-key"
KEY_FILE="${FLI_KEY_FILE:-$HOME/.ssh/${KEY_NAME}.pem}"
REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
export AWS_PAGER="" AWS_DEFAULT_REGION="$REGION"
aws() { command aws --region "$REGION" "$@"; }
log() { printf '\033[1m==> %s\033[0m\n' "$*"; }
die() { printf 'deploy.sh: %s\n' "$*" >&2; exit 1; }

# --- refuse to publish an instance without authentication (checked before anything is created) ---
env_value() {  # last assignment of $1 in .env; quotes and an unquoted value's trailing comment removed
  local line
  line="$(grep -E "^[[:space:]]*(export[[:space:]]+)?$1[[:space:]]*=" "$REPO_DIR/.env" | tail -n 1)" || return 0
  line="${line#*=}"
  line="${line#"${line%%[![:space:]]*}"}"  # leading blanks
  case "$line" in
    \"*) line="${line#\"}"; line="${line%%\"*}" ;;
    \'*) line="${line#\'}"; line="${line%%\'*}" ;;
    *) line="${line%%[[:space:]]#*}"; line="${line%"${line##*[![:space:]]}"}" ;;
  esac
  printf '%s' "$line"
}
[ -f "$REPO_DIR/.env" ] || die "missing $REPO_DIR/.env (copy .env.example and fill it in)"
BASIC_AUTH="$(env_value FLI_BASIC_AUTH)"
AUTH_USER="${BASIC_AUTH%%:*}"
AUTH_PASS="${BASIC_AUTH#*:}"
if [[ "$BASIC_AUTH" != *:* || -z "$AUTH_USER" || -z "$AUTH_PASS" ]]; then
  die "refusing to deploy a public instance without auth: set FLI_BASIC_AUTH=user:password in $REPO_DIR/.env"
fi
if [ "${#AUTH_PASS}" -lt 12 ]; then
  die "FLI_BASIC_AUTH password is too short for a public site (min 12 characters; e.g. openssl rand -base64 18)"
fi
unset AUTH_PASS
# FLI_DEPLOY_CHECK_ONLY=1 validates .env and stops before touching AWS (CI, or to check a new .env)
if [ "${FLI_DEPLOY_CHECK_ONLY:-0}" = 1 ]; then log "auth check passed (FLI_DEPLOY_CHECK_ONLY=1: stopping here)"; exit 0; fi

# --- key pair ------------------------------------------------------------------------------------
if ! aws ec2 describe-key-pairs --key-names "$KEY_NAME" >/dev/null 2>&1; then
  log "creating key pair $KEY_NAME -> $KEY_FILE"
  mkdir -p "$(dirname "$KEY_FILE")"
  aws ec2 create-key-pair --key-name "$KEY_NAME" --key-type ed25519 \
    --query KeyMaterial --output text > "$KEY_FILE"
  chmod 600 "$KEY_FILE"
fi
[ -f "$KEY_FILE" ] || { echo "key pair exists in AWS but $KEY_FILE is missing"; exit 1; }

# --- security group ----------------------------------------------------------------------------
VPC_ID="$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)"
SG_ID="$(aws ec2 describe-security-groups --filters Name=group-name,Values="${NAME}-sg" Name=vpc-id,Values="$VPC_ID" \
  --query 'SecurityGroups[0].GroupId' --output text)"
if [ "$SG_ID" = "None" ]; then
  log "creating security group ${NAME}-sg"
  SG_ID="$(aws ec2 create-security-group --group-name "${NAME}-sg" --vpc-id "$VPC_ID" \
    --description "Frontier Lab Intelligence: ssh from operator, web public" --query GroupId --output text)"
fi
MY_IP="$(curl -fsS https://checkip.amazonaws.com | tr -d '[:space:]')/32"
allow() {  # proto port cidr (duplicates are fine)
  aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol "$1" --port "$2" --cidr "$3" >/dev/null 2>&1 || true
}
allow tcp 22 "$MY_IP"
allow tcp 80 0.0.0.0/0
allow tcp 443 0.0.0.0/0

# --- instance ------------------------------------------------------------------------------------
INSTANCE_ID="$(aws ec2 describe-instances \
  --filters Name=tag:Name,Values="$NAME" Name=instance-state-name,Values=pending,running,stopping,stopped \
  --query 'Reservations[0].Instances[0].InstanceId' --output text)"
if [ "$INSTANCE_ID" = "None" ]; then
  AMI_ID="$(aws ssm get-parameters --names /aws/service/canonical/ubuntu/server/24.04/stable/current/arm64/hvm/ebs-gp3/ami-id \
    --query 'Parameters[0].Value' --output text)"
  log "launching $TYPE from $AMI_ID"
  INSTANCE_ID="$(aws ec2 run-instances --image-id "$AMI_ID" --instance-type "$TYPE" --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" --metadata-options HttpTokens=required,HttpEndpoint=enabled \
    --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=30,VolumeType=gp3,DeleteOnTermination=true,Encrypted=true}' \
    --user-data "file://$REPO_DIR/deploy/aws/user-data.sh" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME}]" "ResourceType=volume,Tags=[{Key=Name,Value=$NAME}]" \
    --query 'Instances[0].InstanceId' --output text)"
fi
aws ec2 start-instances --instance-ids "$INSTANCE_ID" >/dev/null 2>&1 || true
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"

# --- elastic ip ----------------------------------------------------------------------------------
ALLOC_ID="$(aws ec2 describe-addresses --filters Name=tag:Name,Values="$NAME" --query 'Addresses[0].AllocationId' --output text)"
if [ "$ALLOC_ID" = "None" ]; then
  ALLOC_ID="$(aws ec2 allocate-address --domain vpc \
    --tag-specifications "ResourceType=elastic-ip,Tags=[{Key=Name,Value=$NAME}]" --query AllocationId --output text)"
fi
aws ec2 associate-address --instance-id "$INSTANCE_ID" --allocation-id "$ALLOC_ID" --allow-reassociation >/dev/null
IP="$(aws ec2 describe-addresses --allocation-ids "$ALLOC_ID" --query 'Addresses[0].PublicIp' --output text)"
HOST="${IP//./-}.sslip.io"
log "instance $INSTANCE_ID at $IP ($HOST)"

# --- wait for ssh + docker -------------------------------------------------------------------------
SSH=(ssh -i "$KEY_FILE" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 "ubuntu@$IP")
log "waiting for bootstrap (docker install) to finish"
for _ in $(seq 1 60); do
  if "${SSH[@]}" 'test -f /var/lib/cloud/fli-bootstrap-done && docker compose version' >/dev/null 2>&1; then break; fi
  sleep 10
done

# --- ship code + secrets, start the stack ----------------------------------------------------
log "syncing code"
rsync -az --delete -e "ssh -i $KEY_FILE -o StrictHostKeyChecking=accept-new" \
  --exclude .git --exclude .venv --exclude '__pycache__' --exclude .pytest_cache --exclude .ruff_cache \
  --exclude 'data/raw' --exclude 'data/dev' --exclude 'data/fli.sqlite*' --exclude 'data/models/fastembed' \
  --exclude .env "$REPO_DIR/" "ubuntu@$IP:/opt/fli/"
scp -q -i "$KEY_FILE" "$REPO_DIR/.env" "ubuntu@$IP:/opt/fli/.env"
"${SSH[@]}" "chmod 600 /opt/fli/.env"

if [ "${FLI_RESET_DB:-0}" = "1" ]; then  # re-seed Postgres from the shipped snapshot (keeps TLS certs)
  log "FLI_RESET_DB=1: dropping the Postgres volume so init re-imports data/snapshot"
  "${SSH[@]}" "cd /opt/fli && export FLI_HOSTNAME=$HOST FLI_PORT=127.0.0.1:8000 && \
    docker compose -f docker-compose.yml -f deploy/aws/docker-compose.aws.yml rm -sf db init web worker && \
    (docker volume rm fli_pgdata || true)"
fi
WORKERS="${FLI_WORKER:-0}"  # scheduled ingestion + pipeline; off unless FLI_WORKER=1
log "building and starting containers (first build takes a few minutes; worker replicas: $WORKERS)"
# FLI_PORT binds the web container's published port to the instance's loopback only: the public
# path is Caddy (TLS) -> web:8000 on the compose network, never port 8000 directly.
"${SSH[@]}" "cd /opt/fli && export FLI_HOSTNAME=$HOST FLI_PORT=127.0.0.1:8000 && \
  docker compose -f docker-compose.yml -f deploy/aws/docker-compose.aws.yml up -d --build --remove-orphans \
  --scale worker=$WORKERS"

log "waiting for https://$HOST/healthz"
healthy=0
for _ in $(seq 1 40); do
  if curl -fsS "https://$HOST/healthz" >/dev/null 2>&1; then healthy=1; break; fi
  sleep 10
done
[ "$healthy" = 1 ] || die "web did not answer https://$HOST/healthz within 400 s; inspect with: ${SSH[*]} 'cd /opt/fli && docker compose logs --tail 80 web caddy'"

# every page must now ask for the login; /api/health is read with it (credentials on stdin, not argv)
code="$(curl -s -o /dev/null -w '%{http_code}' "https://$HOST/" || true)"
[ "$code" = 401 ] || die "https://$HOST/ answered HTTP $code without credentials (expected 401): auth is not enforced"
quoted="$(printf '%s' "$BASIC_AUTH" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g')"  # curl config string escapes
printf 'user = "%s"\n' "$quoted" | curl -fsS -K - "https://$HOST/api/health" | head -c 400 || true
echo
log "deployed: https://$HOST (log in with the FLI_BASIC_AUTH user from .env)"
