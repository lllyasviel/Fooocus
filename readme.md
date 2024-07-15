[中文](./docs/readme.md)

## Introduction

FastAPI is a modern, fast (high-performance) web framework for building APIs. This project builds the Rest interface of [Fooocus](https://github.com/lllyasviel/Fooocus) based on Fastapi.

For a partial description of Fooocus, please refer to the [Fooocus](https://github.com/lllyasviel/Fooocus) documentation, which mainly introduces the interface section.

Compared to the previous API project [Fooocus-API](https://github.com/mrhan1993/Fooocus-API), there are major changes:

You can get a guide [here](./docs/migration.md) if you use the previous API project.

- Remove the task queue and no longer maintain a separate queue
- Full use of Fooocus' generated code
- Can start at the same time as WebUI
- Removed the form submission interface, leaving only the JSON submission interface
- Main functions merged into one interface
- Simplified parameter structure, consistent with Fooocus' WebUI
- Added streaming output function
- preset support
- More complete task history function

advantage：
- Reduce development load
- More complete Fooocus support
- Easier and faster tracking Fooocus version

## Functional features

- Full Fooocus support
- You can start both the API and WebUI at the same time, or choose not to start WebUI
- Use X-API-KEY for authentication
- all-in-one interface
- use URL provide INPUT image
- streaming output, binary image, asynchronous task and synchronous task support
- persistent task history
- enhanced task history management
- task query function
- Custom upscale rate, Limit by Fooocus, max is 2800px
- preset support
- WebHook support

## Install

Based on Fooocus, there are several dependencies, so you can install it in the same way as Fooocus.

## Startup

Same as Fooocus, use `--apikey` to specify the API authentication key.

Default API port is WebUI port plus 1, that is 7866, use `--port` to modify WebUI port

Environment variable API_PORT is used to specify the API port. It takes precedence over the default setting.

example for WebUI and API:

```shell
python launch.py --listen 0.0.0.0 --port 7865
```

Only API example:

```shell
python launch.py --listen 0.0.0.0 --port 7865 --nowebui
```

## Security

- Use API key for authentication, the key is passed in the request header `X-API-KEY`.

Use `--apikey` to specify the API authentication key on startup.

## EndPoints

### Generate

`POST /v1/engine/generate/`
- **Abstract**: Generate API route
- **RequestBody**: Required, JSON format, based on `CommonRequest` model.
- **Response**:
  - 200: Success response, return the generation result.

Explain：

Request parameter model `CommonRequest` contains all parameters of WebUI, but some parameters you need to pay attention to when using them, including the following categories:

Invalid parameters in the API, this part of the parameter contains:

- `input_image_checkbox`, This parameter always set to True
- `inpaint_mask_upload_checkbox`, This parameter always set to True
- `invert_mask_checkbox`, This parameter always set to False
- `current_tab`, This parameter checks the image information in the parameter and is automatically set. The check order is 'ip' -> 'uov' -> 'inpaint'

Parameters that are not recommended:

- `generate_image_grid`, It is not clear what this parameter does. It is recommended to leave it false.

The following parameters need to be set according to the usage scenario:

- `mixing_image_prompt_and_vary_upscale`
- `mixing_image_prompt_and_inpaint`

In addition, some API-specific parameters are also included:

- `preset`, You can use this parameter to specify a preset that takes precedence over the global default and below the passed parameter, but if the passed parameter is equal to the default value, the preset parameter is used
- `stream_output`, true for streaming output, default false
- `require_base64`, not used
- `async_process`, async task, default false, a synchronous task will be returned when `stream_output` is false at the same time
- `webhook_url`, Webhook addr, if set, the task will be sent to the address after the task is completed.

> `stream_output` has a higher priority than `async_process`, that is, when both are `true`, return streaming output. When all are false, task will be synchronously returned. when you set `Accept: image/xxx` in the request header, the response will be a binary image

### Stop or Skip

`POST /v1/engine/control/`
- **Label**: GenerateV1
- **Abstract**: Stop or Skip task
- **Describe**: stop or skip task, only valid for the current task, stop will stop the current task and continue to the next task. Skip will skip the current generation and continue the task.
- **Params**:
  - `action` (string): action type, can be "stop" or "skip".
- **Response**:
  - `{"message": "{message}"}`

### Get tasks
`GET /tasks`
- **Label**: Query
- **Abstract**: get all tasks
- **Describe**: filter tasks and support paging and time filtering.
- **Params**:
  - `query` (string, default: "all"): task type, one of "all", "history", "current".
  - `page` (integer, default: 0): page number, used for history and pending tasks.
  - `page_size` (integer, default: 10): page size of each page
  - `start_at` (string): filter tasks start time, only valid for history. format by ISO8601 example: "2024-06-30T17:57:07.045812"
  - `end_at` (string): default to current time format by ISO8601, example: "2024-06-30T17:57:07.045812", filter tasks end time, only valid for history.
  - `action` (string): used for delete operation, only valid for history, will delete database record and generated images.
- **Response**:
  - 200: {"history": List[[RecordResponse](#recordresponse)], "current": List[[RecordResponse](#recordresponse)], "pending": List[[RecordResponse](#recordresponse)]}

> Although all models are based on `RecordResponse`, the `current` one will have a preview field

### Get task by id
`GET /tasks/{task_id}`
- **Label**: Query
- **Abstract**: get task by id
- **Params**:
  - `task_id` (string): task id
- **Response**:
  - 200: [RecordResponse](#recordresponse)

### Get all models
`GET /v1/engines/all-models`
- **Label**: Query
- **Abstract**: get all models
- **Response**:
  - 200: return all local checkpoint and lora models.

### Get all styles
`GET /v1/engines/styles`
- **Label**: Query
- **Abstract**: get all styles
- **Response**:
  - 200: return a list of styles.

### Get output file
`GET /outputs/{date}/{file_name}`
- **Label**: Query
- **Abstract**: used for get output image
- **params**:
  - `date` (string): date, the generated image is created in the day folder for classification, the part is the generation date. 
  - `file_name` (string): file name
- **Response**:
  - 200: success response, return output content.
  - 422: validation error.

> if you set `Accept: image/xxx` in the request header, server will convert the output to the specified format and return it. `image/png` `image/jpeg` `image/webp` `image/jpg` are supported.

### Describe image
`POST /v1/tools/describe-image`
- **Label**: GenerateV1
- **Abstract**: get tags from image
- **Describe**: get tags from image, Photo or Anime
- **params**:
  - `image_type` (string): default: "Photo", image type
- **RequestBody**: required `multipart/form-data` format, include image file.
- **Response**:
  - 200: success response, return `DescribeImageResponse` model.
  - 422: validation error.

### Root
`GET /`
- **Label**: Query
- **Abstract**: root endpoint
- **Response**: Redirect to `/docs`

## Components

### Model

#### CommonRequest
- Attributes:
  - `prompt` (string): prompt for generation image.
  - `negative_prompt` (string): negative prompt for filtering unwanted content.
  - `style_selections` (array): style selections.
  - `performance_selection` (Performance): performance, default `Speed`, one of `Quality` `Speed` `Extreme Speed` `Lightning` `Hyper-SD`
  - `aspect_ratios_selection` (string): aspect radios selection, default 1152*896
  - `image_number` (int): number of images to generate, default 1, range 1-32
  - `output_format` (string): out image format, default `png`, one of `jpg` `webp` `png`
  - `image_seed` (int): seed, -1 for random
  - `read_wildcards_in_order` (bool): read wildcards in order, default false
  - `sharpness` (float): sharpness, default 2.0, range 0.0-30.0
  - `guidance_scale` (float): guidance scale, default 4, range 1.0-30.0
  - `base_model_name` (string): base model name, default `juggernautXL_v8Rundiffusion.safetensors` for now
  - `refiner_model_name` (string): refiner model name, default None
  - `refiner_switch` (float): refiner switch, default 0.5, range 0.1-1.0
  - `loras` (Lora): lora list to use, default `sd_xl_offset_example-lora_1.0.safetensors`, format: [Lora](#lora)
  - `input_image_checkbox` (bool): this will always to be true
  - `current_tab` (string): current tab, default `uov` one of `uov` `inpaint` `outpaint`, you don't need to pass this parameter.
  - `uov_method` (string): uov method, default `disable`, all choice [UpscaleOrVaryMethod](#upscaleorvarymethod)
  - `uov_input_image` (string): URL or Base64 image for Upscale or vary, default "None"
  - `outpaint_selections` (array): Outpaint selection, example ["Left", "Right", "Top", "Bottom"]
  - `inpaint_input_image` (string): URL or Base64 image for inpaint
  - `inpaint_additional_prompt` (string): Inpaint additional prompt, default "None"
  - `inpaint_mask_image_upload` (string): URL or Base64 image for inpaint mask
  - `inpaint_mask_upload_checkbox` (bool): this will always true
  - `disable_preview` (bool): disable preview, default false
  - `disable_intermediate_results` (bool): disable intermediate, default false
  - `disable_seed_increment` (bool): disable seed increment, default false
  - `black_out_nsfw` (bool): black out nsfw result, default false
  - `adm_scaler_positive` (float): The scaler multiplied to positive ADM (use 1.0 to disable). default 1.5, range 0.0-3.0
  - `adm_scaler_negative` (float): The scaler multiplied to negative ADM (use 1.0 to disable). default 1.5, range 0.0-3.0
  - `adm_scaler_end` (float): ADM Guidance End At Step, default 0.8, range 0.0-1.0
  - `adaptive_cfg` (float): Adaptive cfg, default 7, range 1.0-30.0
  - `clip_skip` (float): clip skip, default 2, range 1-12
  - `sampler_name` (string): sampler name, default dpmpp_2m_sde_gpu
  - `scheduler_name` (string): scheduler name, default karras
  - `vae_name` (string): VAE name, default Default (model)
  - `overwrite_step` (int): overwrite steps in Performance, default -1
  - `overwrite_switch` (float): overwrite refiner_switch, default -1
  - `overwrite_width` (int): overwrite width in aspect_ratios_selection, default -1, range -1-2048
  - `overwrite_height` (int): overwrite height in aspect_ratios_selection, default -1, range -1-2048
  - `overwrite_vary_strength` (float): overwrite vary_strength, default -1, range 0.0-1.0
  - `overwrite_upscale_strength` (float): overwrite upscale_strength, default -1, range 0.0-1.0
  - `mixing_image_prompt_and_vary_upscale` (bool): mixing image prompt and vary_upscale, default false
  - `mixing_image_prompt_and_inpaint` (bool): mixing image prompt and inpaint, default false
  - `debugging_cn_preprocessor` (bool): debugging cn preprocessor, default false
  - `skipping_cn_preprocessor` (bool): skipping cn preprocessor, default false
  - `canny_low_threshold` (int): default 64, range 1-255
  - `canny_high_threshold` (int): default 128, range 1-255
  - `refiner_swap_method` (string): default joint
  - `controlnet_softness` (float): default 0.25, range 0.0-1.0
  - `freeu_enabled` (bool): enable freeu, default false
  - `freeu_b1` (float): default 1.01
  - `freeu_b2` (float): default 1.02
  - `freeu_s1` (float): default 0.99
  - `freeu_s2` (float): default 0.95
  - `debugging_inpaint_preprocessor` (bool): debugging inpaint preprocessor, default false
  - `inpaint_disable_initial_latent` (bool): default false
  - `inpaint_engine` (string): default v2.6
  - `inpaint_strength` (float): default 1.0, range 0.0-1.0
  - `inpaint_respective_field` (float): default0.618, range 0.0-1.0
  - `invert_mask_checkbox` (bool): default false, this always false
  - `inpaint_erode_or_dilate` (int): default 0, range -64-64
  - `save_metadata_to_images` (bool): save metadata to images, default true
  - `metadata_scheme` (string): default foocus, one of fooocus, a111
  - `controlnet_image` (ImagePrompt): ImagePrompt
  - `generate_image_grid` (bool): default false, suggested to false
  - `outpaint_distance` (List[int]): outpaint distance, default [0, 0, 0, 0], left, top, right, bottom, this params must with outpaint_selections at the same time
  - `upscale_multiple` (float): default 1.0, range 1.0-5.0, work only upscale method is `Upscale (Custom)`
  - `preset` (string): preset, default initial
  - `stream_output` (bool): stream output, default false
  - `require_base64` (bool): not used
  - `async_process` (bool): async process, default false
  - `webhook_url` (string): Webhook URL, default ""


#### Lora

- Attributes:
  - `enabled` (bool): enable Lora, default false
  - `model_name` (string): Lora file name, default None
  - `weight` (float): Lora weight, default 0.5, range -2-2

#### UpscaleOrVaryMethod

- Attributes:
  - "Disabled"
  - "Vary (Subtle)"
  - "Vary (Strong)"
  - "Upscale (1.5x)"
  - "Upscale (2x)"
  - "Upscale (Fast 2x)"
  - "Upscale (Custom)"

#### ImagePrompt

- Attributes:
  - `cn_img` (str): ImageUrl or base64 image
  - `cn_stop` (float): default 0.6, range 0-1
  - `cn_weight` (float): default 0.5, range 0-2
  - `cn_type` (string): default ImagePrompt, one of ImagePrompt FaceSwap, PyraCanny, CPDS

#### DescribeImageResponse

- Attributes:
  - `describe` (string): Image prompt.

#### RecordResponse
- Attributes:：
  - `id` (int): id in sqlite, no use for user.
  - `task_id` (str): task ID, generate by `uuid.uuid4().hex`.
  - `req_params` (CommonRequest): required parameters, convert to url for input image
  - `in_queue_mills` (int): in queue time in millis.
  - `start_mills` (int): start task time in millis.
  - `finish_mills` (int): finish task time in millis.
  - `task_status` (str): task status.
  - `progress` (float): task progress.
  - `preview` (str): preview
  - `webhook_url` (str): Webhook URL.
  - `result` (List): result for generate

## Thanks

This project is based on [Fooocus-API](https://github.com/mrhan1993/Fooocus-API), thanks all the developers who participated in this project.

[Contribute](https://github.com/mrhan1993/Fooocus-API/graphs/contributors)
