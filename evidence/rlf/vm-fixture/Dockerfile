FROM python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7 AS build
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && rm -rf /var/lib/apt/lists/*
WORKDIR /tmp/build
COPY public/challenge.c ./challenge.c
RUN gcc -O0 -fno-stack-protector -no-pie -z noexecstack -Wno-implicit-function-declaration -o challenge challenge.c

RUN nm -n /tmp/build/challenge | grep ' win$'

FROM python:3.12.11-slim-bookworm@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7
RUN apt-get update && apt-get install -y --no-install-recommends socat netcat-openbsd && rm -rf /var/lib/apt/lists/*
COPY --from=build --chmod=0555 /tmp/build/challenge /opt/challenge
COPY --chmod=0555 entrypoint.sh /opt/entrypoint.sh
USER 10001:10001
EXPOSE 31337
HEALTHCHECK --interval=1s --timeout=2s --retries=20 CMD nc -z 127.0.0.1 31337
ENTRYPOINT ["/opt/entrypoint.sh"]
