# Fooocus on Docker

It's based on NVIDIA CUDA 12.3. See [Dockerfile](Dockerfile) for details.

PyTorch version is 2.1. See [requirements_docker.txt](requirements_docker.txt) for details.

## Quick start

**This is just an easy way for testing. Please also see [Notes](#notes) for using.**

Clone git, Then run the docker container `docker compose up --build`
It takes a time to build the container image.
When you see a messeage  `Use the app with http://localhost:7865/` in the console message, you can access the URL on your browser.

Your models, outputs are stored into 'fooocus-data' volume which may be stored in '/var/lib/docker/volumes'.

## Use pre-built image

The pre-built image is provided at `whitehara/fooocus`

You can modify 'docker-compose.yml', then `docker compose up`

Note: when you use the pre-built image, you need to make the 'fooocus-data' volume which permission is '777' ( all users can read/write.) before you run the container.
Otherwise, you will see the permission error.

## Details

### Update the container manually

When you are using `docker compose up --build` continuously, the container is not updated to the newest version of Fooocus automatically.
If you want to use the newest one, you need to run `git pull` before you run `docker compose up --build`

### Import models, outputs
If you want to import files from models, outputs folder, you can uncomment following settings in docker-compose.yml
```
#- ./models:/import/models   # Once you import files, you don't need to mount again.
#- ./outputs:/import/outputs  # Once you import files, you don't need to mount again.
```
Then run `docker compose up --build`, your files will be copied into /content/data/models, /content/data/outputs
Since /content/data is a persistent volume folder, your files will be there when you re-run 'docker compose up --build` without above volume settings.


### Paths inside the container

|Path|Details|
|-|-|
|/content/app|The application stored folder|
|/content/app/models.org|Original 'models' folder.<br> Files are copied to the '/content/app/models' which is symlinked to '/content/data/models' every time the container boots. (Existing files will not be overwritten.) |
|/content/data|Persistent volume mount point|
|/content/data/models|The folder is symlinked to '/content/app/models'|
|/content/data/outputs|The folder is symlinked to '/content/app/outputs'|

### Environments

You can change 'config.txt' parameters by using environments.
**The priority of using the environments is higher than the values defined in 'config.txt'. And they will be saved to the 'config_modification_tuorial.txt'**

Docker specified environments are there. They are used by 'entrypoint.sh'
|Environment|Details|
|-|-|
|DATADIR|'/content/data' location.|
|CMDARGS|Arguments for [entry_with_update.py](entry_with_update.py) which is called by [entrypoint.sh](entrypoint.sh)|
|config_path|'config.txt' location|
|config_example_path|'config_modification_tutorial.txt' location|

You can also use the same json key names and values explained in the 'config_modification_tutorial.txt' as the environments.
See samples in the [docker-compose.yml](docker-compose.yml)

## Notes

- Please keep 'path_outputs' under '/content/app'. Otherwise, you will see an error when you open the history log.
- Docker on Mac/Windows has a slow volume access problem when you use "bind mount" volume. When you see this problem, don't use "bind mount". Please refer to [this article](https://docs.docker.com/storage/volumes/#use-a-volume-with-docker-compose) for not using "bind mount".