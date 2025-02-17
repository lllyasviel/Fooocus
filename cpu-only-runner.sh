#!/bin/bash

# Define variables
CPU_ONLY_FOOCUS_HOME="$HOME/.cpu-only-fooocus"
CPU_ONLY_FOOCUS_TMP="$CPU_ONLY_FOOCUS_HOME/tmp"
CPU_ONLY_FOOCUS_MODULES="$CPU_ONLY_FOOCUS_HOME/modules"
CPU_ONLY_FOOCUS_OUTPUTS="$CPU_ONLY_FOOCUS_HOME/outputs"
CPU_ONLY_CONTAINER_NAME="cpu-only-fooocus"

# Create mount volumes
mkdir -pv "$CPU_ONLY_FOOCUS_TMP"
mkdir -pv "$CPU_ONLY_FOOCUS_MODULES"
mkdir -pv "$CPU_ONLY_FOOCUS_OUTPUTS"

# Check if the container is running
if docker ps -a --format '{{.Names}}' | grep -q "^${CPU_ONLY_CONTAINER_NAME}$"; then
    echo "Container '${CPU_ONLY_CONTAINER_NAME}' already exists."
    read -p "Do you want to rebuild the image? (y/N): " rebuild_choice
    if [[ "$rebuild_choice" =~ ^[Yy]$ ]]; then
        echo "Stopping and removing existing container..."
        docker stop "${CPU_ONLY_CONTAINER_NAME}"
        docker rm "${CPU_ONLY_CONTAINER_NAME}"
        echo "Rebuilding the Docker image..."
        docker-compose -f cpu-only-docker-compose.yml up --build -d
    else
        echo "Starting existing container..."
        docker start "${CPU_ONLY_CONTAINER_NAME}"
    fi
else
    echo "Starting a new container..."
    docker-compose -f cpu-only-docker-compose.yml up -d
fi

