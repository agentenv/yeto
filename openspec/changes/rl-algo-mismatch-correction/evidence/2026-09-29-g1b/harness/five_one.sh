#!/bin/bash
# 5.1: can the runtime image import Miles' example MIS module? (no GPU work)
for dir in /root /root/miles /work/yeto; do
  echo "== cwd=$dir PYTHONPATH=/root/miles:/work/yeto"
  (cd $dir && PYTHONPATH=/root/miles:/work/yeto python -c "import examples.infra_features.train_infer_mismatch_helper.mis as m; print('IMPORT OK', m.__file__)" 2>&1 | tail -1)
done
echo "== vendored"; (cd /work/yeto && PYTHONPATH=/root/miles:/work/yeto python -c "import yeto.rl.algos.vendor.miles_mis as m; print('IMPORT OK', m.__file__)" 2>&1 | tail -1)
echo "== license"; head -3 /root/miles/LICENSE; git -C /root/miles rev-parse HEAD
