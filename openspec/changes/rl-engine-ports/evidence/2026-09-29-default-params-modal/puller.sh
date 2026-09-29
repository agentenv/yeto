E=/home/michael/work/gpu-default-modal/evidence
while true; do
  for c in ta-01M3PH7HTM7RT7SRZJCEXZXQ2R ta-01M3PH7HP0S3F2FVPKJ3GN780R; do
    for i in 0 1; do
      timeout 60 modal container exec $c -- sh -c "cat /root/yeto-output/rl-island-$i.jsonl 2>/dev/null" > $E/pulled/.tmp.$c.$i 2>/dev/null
      [ -s $E/pulled/.tmp.$c.$i ] && mv $E/pulled/.tmp.$c.$i $E/pulled/rl-island-$i.jsonl
    done
  done
  sleep 45
done
