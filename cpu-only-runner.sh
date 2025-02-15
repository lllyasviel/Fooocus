#!/bin/bash

CPU_ONLY_FOOCUS_HOME="~/.cpu-only-fooocus"
CPU_ONLY_FOOCUS_TMP="$CPU_ONLY_FOOCUS_HOME/tmp"
CPU_ONLY_FOOCUS_MODULES="$CPU_ONLY_FOOCUS_HOME/modules"

mkdir -pv "$CPU_ONLY_FOOCUS_TMP"
mkdir -pv "$CPU_ONLY_FOOCUS_MODULES"

docker build -t cpu-only-fooocus -f cpu-only-Dockerfile .
docker run -it --rm -p 7865:7865 cpu-only-fooocus

#docker run -it --rm -p 7865:7865 -v "$CPU_ONLY_FOOCUS_TMP":/tmp/fooocus -v "$CPU_ONLY_FOOCUS_MODULES":/app/models cpu-only-fooocus

