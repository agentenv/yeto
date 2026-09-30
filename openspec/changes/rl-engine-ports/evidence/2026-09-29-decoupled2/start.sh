sudo docker rm -f wk >/dev/null 2>&1
sudo docker run -d --name wk --gpus '"device=0,1"' --network host --ipc host --shm-size 64g --ulimit memlock=-1 -v /work:/work -e HF_HOME=/work/hf radixark/miles@sha256:90940828dcd4d54fd907ff668b43537cbd94778047580e4160d6560af548b74d sleep infinity
sudo docker exec wk bash /work/setup_in.sh 2>&1 | tail -12
