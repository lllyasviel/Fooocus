# Fooocus on Docker

The docker image is based on NVIDIA CUDA 12.3 and PyTorch 2.0, see [Dockerfile](Dockerfile) and [requirements_docker.txt](requirements_docker.txt) for details.

## Quick start

**This is just an easy way for testing. Please find more information in the [notes](#notes).**

1. Clone this repository
2. Build the image with `docker compose build`
3. Run the docker container with `docker compose up`. Building the image takes some time.

When you see the message  `Use the app with http://0.0.0.0:7865/` in the console, you can access the URL in your browser.

Your models and outputs are stored in the `fooocus-data` volume, which, depending on OS, is stored in `/var/lib/docker/volumes`.

## Details

### Update the container manually

When you are using `docker compose up` continuously, the container is not updated to the latest version of Fooocus automatically.
Run `git pull` before executing `docker compose build --no-cache` to build an image with the latest Fooocus version.
You can then start it with `docker compose up`

### Import models, outputs
If you want to import files from models or the outputs folder, you can uncomment the following settings in the [docker-compose.yml](docker-compose.yml):
```
#- ./models:/import/models   # Once you import files, you don't need to mount again.
#- ./outputs:/import/outputs  # Once you import files, you don't need to mount again.
```
After running `docker compose up`, your files will be copied into `/content/data/models` and `/content/data/outputs`
Since `/content/data` is a persistent volume folder, your files will be persisted even when you re-run `docker compose up --build` without above volume settings.


### Paths inside the container

|Path|Details|
|-|-|
|/content/app|The application stored folder|
|/content/app/models.org|Original 'models' folder.<br> Files are copied to the '/content/app/models' which is symlinked to '/content/data/models' every time the container boots. (Existing files will not be overwritten.) |
|/content/data|Persistent volume mount point|
|/content/data/models|The folder is symlinked to '/content/app/models'|
|/content/data/outputs|The folder is symlinked to '/content/app/outputs'|

### Environments

You can change `config.txt` parameters by using environment variables.
**The priority of using the environments is higher than the values defined in `config.txt`, and they will be saved to the `config_modification_tutorial.txt`**

Docker specified environments are there. They are used by 'entrypoint.sh'
|Environment|Details|
|-|-|
|DATADIR|'/content/data' location.|
|CMDARGS|Arguments for [entry_with_update.py](entry_with_update.py) which is called by [entrypoint.sh](entrypoint.sh)|
|config_path|'config.txt' location|
|config_example_path|'config_modification_tutorial.txt' location|

You can also use the same json key names and values explained in the 'config_modification_tutorial.txt' as the environments.
See examples in the [docker-compose.yml](docker-compose.yml)

## Notes

- Please keep 'path_outputs' under '/content/app'. Otherwise, you may get an error when you open the history log.
- Docker on Mac/Windows still has issues in the form of slow volume access when you use "bind mount" volumes. Please refer to [this article](https://docs.docker.com/storage/volumes/#use-a-volume-with-docker-compose) for not using "bind mount".
- The MPS backend (Metal Performance Shaders, Apple Silicon M1/M2/etc.) is not yet supported in Docker, see https://github.com/pytorch/pytorch/issues/81224
- You can also use `docker compose up -d` to start the container detached and connect to the logs with `docker compose logs -f`. This way you can also close the terminal and keep the container running.