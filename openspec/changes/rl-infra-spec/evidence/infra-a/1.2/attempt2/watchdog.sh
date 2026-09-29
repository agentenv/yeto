#!/bin/bash
sleep 5700
HOME=/home/michael/work/infra-a-gpu/b12/home /tmp/modal-venv/bin/modal app stop -y yeto-infra-a-b12 > /home/michael/work/infra-a-gpu/b12/watchdog.out 2>&1
