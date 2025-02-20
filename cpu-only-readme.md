# CPU-Only Fooocus Docker Setup

## Overview
This guide explains how to run Fooocus inside a Docker container using only the CPU, without utilizing a GPU.

## Prerequisites
Ensure you have the following installed on your system:
- Docker
- Docker Compose
- Bash shell (for executing the script)

## Running the Docker Container
To start the container, run the following command:
```sh
./cpu-only-runner.sh
```
This script will:
- Create necessary directories for mounting volumes.
- Check if a container named `cpu-only-fooocus` already exists.
  - If it exists, it will prompt whether to rebuild the container.
  - If it does not exist, it will create and start a new one using `docker-compose`.

## Accessing the UI
Once the container is running, access the Fooocus UI by opening the following URL in your browser:
```
http://localhost:7865/
```

## Viewing Logs
To follow the logs of the running container, use:
```sh
docker logs cpu-only-fooocus --follow
```

## Stopping the Container
To stop the running container, execute:
```sh
docker stop cpu-only-fooocus
```

## Configuration Details
The container uses `cpu-only-docker-compose.yml`, which:
- Runs the container with CPU-only mode using the `CMDARGS=--listen --always-cpu --preset realistic` environment variable.
- Mounts directories from `~/.cpu-only-fooocus` to store models and outputs.
- Exposes port `7865` for accessing the UI.

For further modifications, update `cpu-only-docker-compose.yml` or `cpu-only-runner.sh` accordingly.

## Troubleshooting
- Ensure Docker and Docker Compose are installed and running.
- If the container does not start properly, check the logs using `docker logs cpu-only-fooocus`.
- If you encounter permission issues, try running the script with `sudo`.

Enjoy using Fooocus with CPU-only mode!


