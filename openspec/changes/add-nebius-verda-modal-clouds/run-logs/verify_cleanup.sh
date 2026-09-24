#!/usr/bin/env bash
# Post-run cleanup verification across the three clouds used by task 8.7.
cd /home/michael/yeto
echo "=== verified at $(date -u +%FT%TZ) ==="
echo
echo "=== sky status ==="
timeout 300 .venv/bin/sky status 2>&1 | grep -v FutureWarning

echo
echo "=== modal app list ==="
timeout 180 .venv/bin/modal app list --json 2>/dev/null | .venv/bin/python -c '
import json, sys
for a in json.load(sys.stdin):
    print(a.get("description"), "|", a.get("state"), "| tasks", a.get("tasks"), "| stopped_at", a.get("stopped_at"))
'

echo
echo "=== nebius instances (eu-north1 project) ==="
PID=$(.venv/bin/python -c "import yaml, os; print(yaml.safe_load(open(os.path.expanduser('~/.sky/config.yaml')))['nebius']['region_configs']['eu-north1']['project_id'])")
~/.nebius/bin/nebius compute instance list --parent-id "$PID" --format json 2>/dev/null | .venv/bin/python -c '
import json, sys
items = json.load(sys.stdin).get("items", [])
print("instances:", len(items))
for i in items:
    print(" -", i["metadata"]["name"], i.get("status", {}).get("state"))
'

echo
echo "=== aws us-east-1 running/pending instances ==="
AWS_PAGER= .venv/bin/python -m awscli ec2 describe-instances --region us-east-1 \
  --filters Name=instance-state-name,Values=running,pending \
  --query 'Reservations[].Instances[].[InstanceId,InstanceType,Tags[?Key==`Name`].Value|[0]]' \
  --output text 2>&1 | head -10
