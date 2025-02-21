#!/bin/bash

# Define variables
AMDGPU_FOOCUS_HOME="$HOME/.amdgpu-fooocus"
AMDGPU_FOOCUS_TMP="$AMDGPU_FOOCUS_HOME/tmp"
AMDGPU_FOOCUS_MODULES="$AMDGPU_FOOCUS_HOME/modules"
AMDGPU_FOOCUS_OUTPUTS="$AMDGPU_FOOCUS_HOME/outputs"
AMDGPU_CONTAINER_NAME="amdgpu-fooocus"

# Create mount volumes
mkdir -pv "$AMDGPU_FOOCUS_TMP"
mkdir -pv "$AMDGPU_FOOCUS_MODULES"
mkdir -pv "$AMDGPU_FOOCUS_OUTPUTS"

# Check if the container is running
if docker ps -a --format '{{.Names}}' | grep -q "^${AMDGPU_CONTAINER_NAME}$"; then
    echo "Container '${AMDGPU_CONTAINER_NAME}' already exists."
    read -p "Do you want to rebuild the image? (y/N): " rebuild_choice
    if [[ "$rebuild_choice" =~ ^[Yy]$ ]]; then
        echo "Stopping and removing existing container..."
        docker stop "${AMDGPU_CONTAINER_NAME}"
        docker rm "${AMDGPU_CONTAINER_NAME}"
        echo "Rebuilding the Docker image..."
        docker-compose -f amdgpu-docker-compose.yml up --build -d
    else
        echo "Starting existing container..."
        docker start "${AMDGPU_CONTAINER_NAME}"
    fi
else
    echo "Starting a new container..."
    docker-compose -f amdgpu-docker-compose.yml up -d
fi

