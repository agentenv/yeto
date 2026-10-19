#!/bin/sh
# Engine image entrypoint (S25 N0.2): check the image against its version lock,
# then run the command. A failed check stops the container (exit 1) before any
# engine starts. The report is kept at /tmp/yeto-image-assert.json.
set -e
python3 /opt/yeto/image/assert_image_versions.py /opt/yeto/image/versions.lock.json \
  --report /tmp/yeto-image-assert.json
exec "$@"
