# Fooocus على Docker

صورة Docker مبنية على NVIDIA CUDA 12.4 وPyTorch 2.1، راجع [Dockerfile](Dockerfile) و[requirements_docker.txt](requirements_docker.txt) للتفاصيل.

## المتطلبات

- جهاز بمواصفات كافية لتشغيل Fooocus، وبرامج تشغيل Nvidia الخاصة
- Docker أو Docker Compose أو Podman

## بداية سريعة

**مزيد من المعلومات في [الملاحظات](#notes).**

### التشغيل مع Docker Compose

1. استنساخ هذا المستودع
2. تشغيل حاوية Docker باستخدام `docker compose up`.

### التشغيل مع Docker

```sh
docker run -p 7865:7865 -v fooocus-data:/content/data -it \
--gpus all \
-e CMDARGS=--listen \
-e DATADIR=/content/data \
-e config_path=/content/data/config.txt \
-e config_example_path=/content/data/config_modification_tutorial.txt \
-e path_checkpoints=/content/data/models/checkpoints/ \
-e path_loras=/content/data/models/loras/ \
-e path_embeddings=/content/data/models/embeddings/ \
-e path_vae_approx=/content/data/models/vae_approx/ \
-e path_upscale_models=/content/data/models/upscale_models/ \
-e path_inpaint=/content/data/models/inpaint/ \
-e path_controlnet=/content/data/models/controlnet/ \
-e path_clip_vision=/content/data/models/clip_vision/ \
-e path_fooocus_expansion=/content/data/models/prompt_expansion/fooocus_expansion/ \
-e path_outputs=/content/app/outputs/ \
ghcr.io/lllyasviel/fooocus
```
### التشغيل مع Podman

```sh
podman run -p 7865:7865 -v fooocus-data:/content/data -it \
--security-opt=no-new-privileges --cap-drop=ALL --security-opt label=type:nvidia_container_t --device=nvidia.com/gpu=all \
-e CMDARGS=--listen \
-e DATADIR=/content/data \
-e config_path=/content/data/config.txt \
-e config_example_path=/content/data/config_modification_tutorial.txt \
-e path_checkpoints=/content/data/models/checkpoints/ \
-e path_loras=/content/data/models/loras/ \
-e path_embeddings=/content/data/models/embeddings/ \
-e path_vae_approx=/content/data/models/vae_approx/ \
-e path_upscale_models=/content/data/models/upscale_models/ \
-e path_inpaint=/content/data/models/inpaint/ \
-e path_controlnet=/content/data/models/controlnet/ \
-e path_clip_vision=/content/data/models/clip_vision/ \
-e path_fooocus_expansion=/content/data/models/prompt_expansion/fooocus_expansion/ \
-e path_outputs=/content/app/outputs/ \
ghcr.io/lllyasviel/fooocus
```

عندما ترى الرسالة `Use the app with http://0.0.0.0:7865/` في وحدة التحكم، يمكنك الوصول إلى الرابط في متصفحك.

نماذجك ومخرجاتك مخزنة في وحدة التخزين `fooocus-data`، والتي، حسب نظام التشغيل، تكون مخزنة في `/var/lib/docker/volumes/` (أو `~/.local/share/containers/storage/volumes/` عند استخدام `podman`).

## بناء الحاوية محلياً

قم باستنساخ المستودع أولاً، وافتح terminal في المجلد.

البناء باستخدام `docker`:
```sh
docker build . -t fooocus
```

البناء باستخدام `podman`:
```sh
podman build . -t fooocus
```

## تفاصيل

### تحديث الحاوية يدوياً (`docker compose`)

عند استخدام `docker compose up` باستمرار، لا يتم تحديث الحاوية إلى أحدث إصدار من Fooocus تلقائياً.
قم بتشغيل `git pull` قبل تنفيذ `docker compose build --no-cache` لبناء صورة بأحدث إصدار من Fooocus.
يمكنك بعد ذلك تشغيلها باستخدام `docker compose up`

### استيراد النماذج والمخرجات

إذا كنت تريد استيراد ملفات من مجلد النماذج أو المخرجات، يمكنك إضافة روابط التثبيت التالية في [docker-compose.yml](docker-compose.yml) أو الطريقة المفضلة لديك لتشغيل الحاوية:
```
#- ./models:/import/models   # بمجرد استيراد الملفات، لا حاجة لإعادة التثبيت.
#- ./outputs:/import/outputs  # بمجرد استيراد الملفات، لا حاجة لإعادة التثبيت.
```
بعد تشغيل الحاوية، سيتم نسخ ملفاتك إلى `/content/data/models` و`/content/data/outputs`.
نظراً لأن `/content/data` هو مجلد تخزين دائم، فسيتم الاحتفاظ بملفاتك حتى عند إعادة تشغيل الحاوية دون روابط التثبيت أعلاه.


### المسارات داخل الحاوية

|المسار|التفاصيل|
|-|-|
|/content/app|مجلد تخزين التطبيق|
|/content/app/models.org|مجلد 'models' الأصلي.<br> يتم نسخ الملفات إلى '/content/app/models' المرتبط symbolically بـ '/content/data/models' في كل مرة يتم فيها تشغيل الحاوية. (لن يتم استبدال الملفات الموجودة.)|
|/content/data|نقطة تثبيت وحدة التخزين الدائمة|
|/content/data/models|المجلد مرتبط symbolically بـ '/content/app/models'|
|/content/data/outputs|المجلد مرتبط symbolically بـ '/content/app/outputs'|

### المتغيرات البيئية

يمكنك تغيير معاملات `config.txt` باستخدام المتغيرات البيئية.
**أولوية استخدام المتغيرات البيئية أعلى من القيم المعرفة في `config.txt`، وسيتم حفظها في `config_modification_tutorial.txt`**

المتغيرات البيئية المخصصة لـ Docker موجودة هنا. يتم استخدامها بواسطة 'entrypoint.sh'
|المتغير البيئي|التفاصيل|
|-|-|
|DATADIR|موقع '/content/data'.|
|CMDARGS|الوسائط لـ [entry_with_update.py](entry_with_update.py) التي يتم استدعاؤها بواسطة [entrypoint.sh](entrypoint.sh)|
|config_path|موقع 'config.txt'|
|config_example_path|موقع 'config_modification_tutorial.txt'|
|HF_MIRROR|نطاق موقع مرآة huggingface|

يمكنك أيضاً استخدام نفس أسماء المفاتيح والقيم json الموضحة في 'config_modification_tutorial.txt' كمتغيرات بيئية.
راجع الأمثلة في [docker-compose.yml](docker-compose.yml)

## ملاحظات

- يُرجى إبقاء 'path_outputs' داخل '/content/app'. وإلا، قد تحصل على خطأ عند فتح سجل التاريخ.
- لا يزال Docker على Mac/Windows يعاني من مشاكل في شكل بطء الوصول إلى وحدات التخزين عند استخدام "bind mount". يُرجى الرجوع إلى [هذه المقالة](https://docs.docker.com/storage/volumes/#use-a-volume-with-docker-compose) لعدم استخدام "bind mount".
- الخلفية MPS (Metal Performance Shaders، Apple Silicon M1/M2/etc.) غير مدعومة بعد في Docker، راجع https://github.com/pytorch/pytorch/issues/81224
- يمكنك أيضاً استخدام `docker compose up -d` لتشغيل الحاوية بشكل منفصل والاتصال بالسجلات باستخدام `docker compose logs -f`. بهذه الطريقة يمكنك إغلاق terminal وإبقاء الحاوية قيد التشغيل.
