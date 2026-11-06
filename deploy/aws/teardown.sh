#!/usr/bin/env bash
# Delete everything deploy.sh created (instance + disk, Elastic IP, security group, key pair).
set -euo pipefail
REGION="${AWS_REGION:-eu-central-1}"
NAME="${FLI_NAME:-fli}"
export AWS_PAGER=""
aws() { command aws --region "$REGION" "$@"; }

INSTANCE_ID="$(aws ec2 describe-instances --filters Name=tag:Name,Values="$NAME" \
  Name=instance-state-name,Values=pending,running,stopping,stopped --query 'Reservations[0].Instances[0].InstanceId' --output text)"
if [ "$INSTANCE_ID" != "None" ]; then
  echo "terminating $INSTANCE_ID"
  aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" >/dev/null
  aws ec2 wait instance-terminated --instance-ids "$INSTANCE_ID"
fi
ALLOC_ID="$(aws ec2 describe-addresses --filters Name=tag:Name,Values="$NAME" --query 'Addresses[0].AllocationId' --output text)"
[ "$ALLOC_ID" != "None" ] && aws ec2 release-address --allocation-id "$ALLOC_ID" && echo "released $ALLOC_ID"
SG_ID="$(aws ec2 describe-security-groups --filters Name=group-name,Values="${NAME}-sg" --query 'SecurityGroups[0].GroupId' --output text)"
[ "$SG_ID" != "None" ] && aws ec2 delete-security-group --group-id "$SG_ID" && echo "deleted $SG_ID"
aws ec2 delete-key-pair --key-name "${NAME}-key" >/dev/null 2>&1 && echo "deleted key pair ${NAME}-key" || true
echo "done (local key file ~/.ssh/${NAME}-key.pem left in place; delete it if no longer needed)"
