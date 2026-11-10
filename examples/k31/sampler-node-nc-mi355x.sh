#!/usr/bin/env bash
# Kimi-K3.1 sampler node on National Compute (8 x MI355X, SGLang ROCm): the node scripts of
# scripts/sampler_node/ in order, with the K3.1 server contract on (draft; not run on a node).
# Every value is a placeholder; the store credentials stay in a mode-0600 env file.
set -euo pipefail

: "${NC_ACCOUNT:?nc account number}"
: "${IMAGE:?SGLang ROCm image (sglang-miles-kimi-k31 build)}"
: "${MODEL:?HF repo of the Kimi-K3.1 MXFP4 checkpoint}"
: "${MODEL_DIR:?nvme path the checkpoint lands in}"
: "${R2_ENV_FILE:?0600 env file: AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_ENDPOINT_URL AWS_REGION R2_BUCKET}"
: "${RELEASE_STORE:?local release store directory}"
: "${RUN:?run prefix in the bucket}"
: "${SPEC:?engine sampler spec (examples/k31/sampler-spec.nc-mi355x.json filled in)}"
NODES=$(cd "$(dirname "$0")/../../scripts/sampler_node" && pwd)

"$NODES/nc_lease.sh" --account "$NC_ACCOUNT" --out node.json
"$NODES/bootstrap.sh" --image "$IMAGE" --model "$MODEL" --r2-env-file "$R2_ENV_FILE" \
  --release-store "$RELEASE_STORE" --release-prefix "$RUN" \
  ${RELEASE_PEERS:+--release-peers "$RELEASE_PEERS"} --serve-port "${SERVE_PORT:-8431}"
"$NODES/run_sglang.sh" --image "$IMAGE" --model-dir "$MODEL_DIR" --tp 8 --r2-env-file "$R2_ENV_FILE" \
  -- --context-length 32768 --trust-remote-code
"$NODES/run_engine_sampler.sh" --spec "$SPEC" --r2-env-file "$R2_ENV_FILE" --pip-install
