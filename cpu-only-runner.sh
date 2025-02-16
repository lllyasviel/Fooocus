#!/bin/bash

# create mount volumes
CPU_ONLY_FOOCUS_HOME="~/.cpu-only-fooocus"
CPU_ONLY_FOOCUS_TMP="$CPU_ONLY_FOOCUS_HOME/tmp"
CPU_ONLY_FOOCUS_MODULES="$CPU_ONLY_FOOCUS_HOME/modules"
CPU_ONLY_FOOCUS_OUTPUTS="$CPU_ONLY_FOOCUS_HOME/outputs"

mkdir -pv "$CPU_ONLY_FOOCUS_TMP"
mkdir -pv "$CPU_ONLY_FOOCUS_MODULES"
mkdir -pv "$CPU_ONLY_FOOCUS_OUTPUTS"

docker-compose -f cpu-only-docker-compose.yml up -d

