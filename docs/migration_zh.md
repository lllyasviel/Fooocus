# 参数对照表

`AdvancedParams` 用 `adp` 代替，名称变动的原则是和 Fooocus 进行统一：

| Fooocus-API                              | FooocusAPI                           | 备注                  |
|------------------------------------------|--------------------------------------|---------------------|
| prompt                                   | prompt                               |                     |
| negative_prompt                          | negative_prompt                      |                     |
| style_selections                         | style_selections                     |                     |
| performance_selection                    | performance_selection                |                     |
| aspect_ratios_selection                  | aspect_ratios_selection              |                     |
| image_number                             | image_number                         |                     |
| image_seed                               | image_seed                           |                     |
| sharpness                                | sharpness                            |                     |
| guidance_scale                           | guidance_scale                       |                     |
| base_model_name                          | base_model_name                      |                     |
| refiner_model_name                       | refiner_model_name                   |                     |
| refiner_switch                           | refiner_switch                       |                     |
| loras                                    | loras                                | 传入格式相同，都是 Lora 对象列表 |
|                                          | input_image_checkbox                 | 可以忽略，它总是为 True      |
|                                          | current_tab                          | 可以忽略，根据参数会自动判断      |
| uov_method                               | uov_method                           |                     |
| **input_image**                          | **uov_input_image**                  | 使用 Fooocus 的变量名称    |
| outpaint_selections                      | outpaint_selections                  |                     |
| **input_image**                          | **inpaint_input_image**              | 使用 Fooocus 的变量名称    |
| inpaint_additional_prompt                | inpaint_additional_prompt            |                     |
| **input_mask**                           | **inpaint_mask_image_upload**        | 使用 Fooocus 的变量名称    |
| adp.disable_preview                      | disable_preview                      |                     |
| adp.disable_intermediate_results         | disable_intermediate_results         |                     |
| adp.disable_seed_increment               | disable_seed_increment               |                     |
| adp.black_out_nsfw                       | black_out_nsfw                       |                     |
| adp.adm_scaler_positive                  | adm_scaler_positive                  |                     |
| adp.adm_scaler_negative                  | adm_scaler_negative                  |                     |
| adp.adm_scaler_end                       | adm_scaler_end                       |                     |
| adp.adaptive_cfg                         | adaptive_cfg                         |                     |
| adp.clip_skip                            | clip_skip                            |                     |
| adp.sampler_name                         | sampler_name                         |                     |
| adp.scheduler_name                       | scheduler_name                       |                     |
| adp.vae_name                             | vae_name                             |                     |
| adp.overwrite_step                       | overwrite_step                       |                     |
| adp.overwrite_switch                     | overwrite_switch                     |                     |
| adp.overwrite_width                      | overwrite_width                      |                     |
| adp.overwrite_height                     | overwrite_height                     |                     |
| adp.overwrite_vary_strength              | overwrite_vary_strength              |                     |
| adp.overwrite_upscale_strength           | overwrite_upscale_strength           |                     |
| adp.mixing_image_prompt_and_vary_upscale | mixing_image_prompt_and_vary_upscale |                     |
| adp.mixing_image_prompt_and_inpaint      | mixing_image_prompt_and_inpaint      |                     |
| adp.debugging_cn_preprocessor            | debugging_cn_preprocessor            |                     |
| adp.skipping_cn_preprocessor             | skipping_cn_preprocessor             |                     |
| adp.canny_low_threshold                  | canny_low_threshold                  |                     |
| adp.canny_high_threshold                 | canny_high_threshold                 |                     |
| adp.refiner_swap_method                  | refiner_swap_method                  |                     |
| adp.controlnet_softness                  | controlnet_softness                  |                     |
| adp.freeu_enabled                        | freeu_enabled                        |                     |
| adp.freeu_b1                             | freeu_b1                             |                     |
| adp.freeu_b2                             | freeu_b2                             |                     |
| adp.freeu_s1                             | freeu_s1                             |                     |
| adp.freeu_s2                             | freeu_s2                             |                     |
| adp.debugging_inpaint_preprocessor       | debugging_inpaint_preprocessor       |                     |
| adp.inpaint_disable_initial_latent       | inpaint_disable_initial_latent       |                     |
| adp.inpaint_engine                       | inpaint_engine                       |                     |
| adp.inpaint_strength                     | inpaint_strength                     |                     |
| adp.inpaint_respective_field             | inpaint_respective_field             |                     |
| adp.inpaint_mask_upload_checkbox         | inpaint_mask_upload_checkbox         |                     |
| adp.invert_mask_checkbox                 | invert_mask_checkbox                 |                     |
| adp.inpaint_erode_or_dilate              | inpaint_erode_or_dilate              |                     |
| **image_prompts**                        | **controlnet_image**                 | 只是属性名称变更            |
|                                          | generate_image_grid                  | 新增，这是个测试选项，建议默认     |
| outpaint_distance_left                   | outpaint_distance                    | 这四个属性合并为了一个属性       |
| outpaint_distance_right                  |                                      | 可以通过一个列表传递这四个值      |
| outpaint_distance_top                    |                                      | 例如：[100, 50, 0, 0]  |
| outpaint_distance_bottom                 |                                      | 方向是：左, 上, 右, 下      |
| **upscale_value**                        | **upscale_multiple**                 | 属性名变更               |
|                                          | preset                               | 新增，可以通过该属性指定使用的预设   |
|                                          | stream_output                        | 新增流式输出，类似 LLM 的流式输出 |
| **save_meta**                            | **save_metadata_to_images**          |                     |
| **meta_scheme**                          | **metadata_scheme**                  |                     |
| **save_extension**                       | **output_format**                    |                     |
| save_name                                |                                      | 移除，不支持自定义文件名        |
| read_wildcards_in_order                  | read_wildcards_in_order              |                     |
| require_base64                           | require_base64                       | 该参数后续可能会被移除         |
| async_process                            | async_process                        |                     |
| webhook_url                              | webhook_url                          |                     |

简单说来就是

- 将所有 `AdvancedParams` 平移到上一级
- 修改部分参数名
    - `input_image` -> `inpaint_input_image`
    - `inpaint_mask` -> `inpaint_mask_image_upload`
    - `input_image` -> `uov_input_image`
    - `image_prompts` -> `controlnet_image`
    - `upscale_value` -> `upscale_value`
    - `save_meta` -> `upscale_multiple`
    - `meta_scheme` -> `save_metadata_to_images`
    - `save_extension` -> `output_format`
- 移除部分参数名
    - `save_name`
- 增加部分参数
    - `input_image_checkbox`
    - `current_tab`
    - `generate_image_grid`
    - `preset`
    - `stream_output`
- 合并部分参数
    - `outpaint_distance_left,right,top,bottom` 四个参数合并为 `outpaint_distance`

## 四种返回示例

### 异步任务

在参数中指定 `async_process` 为 `True`

```python
import requests
import json

endpoint = "http://127.0.0.1:7866/v1/engine/generate/"

params = {
    "prompt": "",
    "negative_prompt": "",
    "performance_selection": "Lightning",
    "async_process": True,
    "webhook_url": ""
}

res = requests.post(
    url=endpoint,
    data=json.dumps(params),
    timeout=60
)

print(res.json())
```

输出如下：

```python
{'id': -1, 'task_id': '85c10c81e9e2482d90a64c3704137d3a', 'req_params': {}, 'in_queue_mills': -1, 'start_mills': -1, 'finish_mills': -1, 'task_status': 'pending', 'progress': -1, 'preview': '', 'webhook_url': '', 'result': []}
```

你可以通过 `task_id` 访问 `http://127.0.0.1:7866/tasks/{task_id}` 获取任务信息，如果该任务正在执行，返回信息中会包含 `preview`

返回数据示例：

```python
# 未开始
{
    "id": -1,
    "in_queue_mills": 1720085748199,
    "finish_mills": null,
    "progress": null,
    "result": null,
    "req_params": {
        # 完整的请求参数
        ...
    },
    "task_id": "85c10c81e9e2482d90a64c3704137d3a",
    "start_mills": null,
    "task_status": null,
    "webhook_url": ""
}

# 执行中
{
    "id": -1,
    "task_id": "85c10c81e9e2482d90a64c3704137d3a",
    "req_params": {
        ...
    },
    "in_queue_mills": 1720086131653,
    "start_mills": 1720086131865,
    "finish_mills": -1,
    "task_status": "running",
    "progress": 18,
    "preview": "a long text",
    "webhook_url": "",
    "result": []
}

# 已完成
{
    "id": 71,
    "in_queue_mills": 1720085748199,
    "finish_mills": 1720085770046,
    "progress": 100,
    "result": [
        "http://127.0.0.1:7866/outputs/2024-07-04/2024-07-04_17-36-09_5201.png"
    ],
    "req_params": {
        ...
    },
    "task_id": "85c10c81e9e2482d90a64c3704137d3a",
    "start_mills": 1720085748425,
    "task_status": "finished",
    "webhook_url": ""
}
```

### 流式输出

这是一个类似 LLM 流式输出的方式，你会持续收到来自服务器的信息，直到结束，参照上面的示例：

```python
import requests
import json

endpoint = "http://127.0.0.1:7866/v1/engine/generate/"

params = {
    "prompt": "",
    "negative_prompt": "",
    "performance_selection": "Lightning",
    "stream_output": True,
    "webhook_url": ""
}

res = requests.post(
    url=endpoint,
    data=json.dumps(params),
    stream=True,
    timeout=60
)

for line in res.iter_lines():
    if line:
        print(line.decode('utf-8'))
```

你会获得类似下面的输出：

```python
data: {"progress": 2, "preview": null, "message": "Loading models ...", "images": []}
data:
data: {"progress": 13, "preview": null, "message": "Preparing task 1/1 ...", "images": []}
data:
data: {"progress": 13, "preview": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAASAAAA...", 'message': 'Sampling step 1/4, image 1/1 ...', 'images': []}
data:
data: {"progress": 34, "preview": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAASAAAA...", 'message': 'Sampling step 2/4, image 1/1 ...', 'images': []}
data:
data: {"progress": 56, "preview": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAASAAAA...", 'message': 'Sampling step 3/4, image 1/1 ...', 'images': []}
data:
data: {"progress": 78, "preview": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAASAAAA...", 'message': 'Sampling step 4/4, image 1/1 ...', 'images': []}
data:
data: {"progress": 100, "preview": null, "message": "Saving image 1/1 to system ...", "images": []}
data:
data: {"progress": 100, "preview": null, "message": "Finished", "images": ["http://10.0.0.245:7866/outputs/2024-07-05/2024-07-05_09-31-10_1752.png"]}
data:
```

我们在稍微修改下：

```python
import requests
import json

endpoint = "http://127.0.0.1:7866/v1/engine/generate/"

params = {
    "prompt": "",
    "negative_prompt": "",
    "performance_selection": "Lightning",
    "stream_output": True,
    "webhook_url": ""
}

res = requests.post(
    url=endpoint,
    data=json.dumps(params),
    stream=True,
    timeout=60
)

for line in res.iter_lines(chunk_size=8192):
    line = line.decode('utf-8').split('\n')[0]

    try:
        json_data = json.loads(line[6:])
        if json_data["preview"] is not None:
            json_data["preview"] = "data:image/png;base64,iVBORw0KGgoAAAANSU..."
    except json.decoder.JSONDecodeError:
        continue
    print(json_data)
```

然后你就得到了一系列类似这样的输出：

```python
{'progress': 13, 'preview': None, 'message': 'Preparing task 1/1 ...', 'images': []}
{'progress': 13, 'preview': 'data:image/png;base64,iVBORw0KGgoAAAANSU...', 'message': 'Sampling step 1/4, image 1/1 ...', 'images': []}
{'progress': 34, 'preview': 'data:image/png;base64,iVBORw0KGgoAAAANSU...', 'message': 'Sampling step 2/4, image 1/1 ...', 'images': []}
{'progress': 56, 'preview': 'data:image/png;base64,iVBORw0KGgoAAAANSU...', 'message': 'Sampling step 3/4, image 1/1 ...', 'images': []}
{'progress': 78, 'preview': 'data:image/png;base64,iVBORw0KGgoAAAANSU...', 'message': 'Sampling step 4/4, image 1/1 ...', 'images': []}
{'progress': 100, 'preview': None, 'message': 'Saving image 1/1 to system ...', 'images': []}
{'progress': 100, 'preview': None, 'message': 'Finished', 'images': ['http://10.0.0.245:7866/outputs/2024-07-05/2024-07-05_10-02-22_2536.png']}
```

这还挺适合前端套壳用的（可惜我完全搞不懂前端，要不高低套一个），比如我用 AI 生成了一个 [example.html](./docs/example.html) ，服务启动后点击 `Generate` 按钮，你就会得到一个有预览、有进度的生成过程。

### 二进制输出

这个就简单了，在 `header` 中指定 `Accept: image/xxx` 即可，此时 `image_number` 强制为 `1`, 其优先级高于参数中的 `stream_output` 和 `async_process`, 这种情况下会返回一个二进制图片，格式是 `Accept` 中指定的。目前支持的格式有 `image/png`, `image/jpeg`, `image/webp` 和 `image/jpg`。

```python
import requests
import json
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

endpoint = "http://127.0.0.1:7866/v1/engine/generate/"

headers = {
    "Accept": "image/png"
}

params = {
    "prompt": "",
    "negative_prompt": "",
    "performance_selection": "Lightning",
    "webhook_url": ""
}

res = requests.post(
    url=endpoint,
    data=json.dumps(params),
    headers=headers,
    timeout=60
)

image_stream = BytesIO(res.content)
image = Image.open(image_stream)

plt.imshow(image)
plt.show()
```

### 同步任务

默认参数的情况下，该接口会是一个同步接口

```python
import requests
import json

endpoint = "http://127.0.0.1:7866/v1/engine/generate/"

params = {
    "prompt": "",
    "negative_prompt": "",
    "performance_selection": "Lightning",
    "webhook_url": ""
}

res = requests.post(
    url=endpoint,
    data=json.dumps(params),
    timeout=60
)

print(res.json())
```

返回结果和通过 ID 查询结果相同，你可以参照下面 tasks 接口返回格式

# 任务查询

和 [Fooocus-API](https://github.com/mrhan1993/Fooocus-API) 不同的是历史记录的保存将是自动进行的，没有保留开关。数据库使用 `SQLite3` 并存放在 `outputs/db.sqlite3` 中。同时吸取了上次的教训，极大简化了表结构，将请求参数作为 JSON 存放在 `req_params` 字段。为了降低读写，仅在任务进入队列时和完成后进行数据库操作。其仅作为生成记录使用，任务状态的追踪会在内存中完成。

此外，该版本会保留输入图像，上传的图像会计算哈希值并保存在 `inputs` 目录，数据库中的 `req_params` 会将图片参数替换为 `url` 信息进行保存，这意味着更完整的历史记录保存，无论是文生图还是图生图又或者是其他

## /tasks

这是个复合接口，但其返回格式是固定的，该接口总是会返回下面格式的 JSON 数据，无论参数如何指定

```python
{
    "history": [],
    "current": [],  # 尽管是个列表，但其中不会超过一个元素。
    "pending": []
}
```

所有的元素其格式都是和数据库中的 scheme 匹配的，除了 `current` 会多一个 `preview` ，比如下图：

![](./assets/tasks.png)

该接口还支持更加精细的用法，参考下面的示例：

> 该接口返回格式总是固定的，不管参数如何调整

```shell
curl http://localhost:7866/tasks?query=current
# 仅返回当前任务，query 参数还可以指定的值为 'all', 'pending', 'history'

curl http://localhost:7866/tasks?query=history&page=3&page_size=5
# history 和 pending 支持分页和页面大小

curl http://localhost:7866/tasks?query=history&start_at=2024-07-03T12:22:30
# 你可以指定一个时间范围进行查询，这会返回该时间段的所有记录。时间格式是 ISO8601，如果你不指定 end_at 则截止当前时间

curl http://localhost:7866/tasks?query=history&start_at=2024-07-03T12:22:30&action=delete
# 删除指定时间范围的任务，数据库记录和生成文件。目前仅支持这一种删除方法(不会删除 input 文件)。

curl http://localhost:7866/tasks/38ba92b188a64233a7336218cd902865
# 这会返回该任务的信息，但它只是一个字典。相当于从上面列表中取出指定 task_id 的任务，如果它刚好是当前任务，那它也会包含 preview
```