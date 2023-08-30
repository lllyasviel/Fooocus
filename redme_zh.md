# Fooocus

<img src="https://github.com/lllyasviel/Fooocus/assets/19834515/f79c5981-cf80-4ee3-b06b-3fef3f8bfbc7" width=100%>

Fooocus是用于一个图片生成的软件.

Fooocus是受到Stable Diffusion和Midjourney启发的产物:

* 像Stable Diffusion一样, 离线、开源、免费.

* 类似于Midjourney, 无需手动调节各种参数, 用户只需要专注于提示词和图片.

Fooocus内置了自动化的[优化选项](#tech_list). 用户无需关注那些恼人的技术参数, 可以尽情享受单纯的人机互动 to "explore new mediums of thought and expanding the imaginative powers of the human species" `[1]`.

Fooocus拥有极简的安装流程. 从点击"download"开始, 只需要不超过3次鼠标点击就可以生成第一张图片. 最低显存需求为4GB (Nvidia).

Fooocus还包含许多用于生成最佳图片的"fooocus-only"高级特性. [点击这里去浏览完整高级特性列表.](https://github.com/lllyasviel/Fooocus/discussions/117)

`[1]` David Holz, 2019.

## Download

### Windows

你可以在这里直接下载Fooocus:

**[>>> 点此下载(Github Releases) <<<](https://github.com/lllyasviel/Fooocus/releases/download/1.0.35/Fooocus_win64_1-1-1035.7z)**

当下载完成后, 请解压文件, 运行"run.bat".

![image](https://github.com/lllyasviel/Fooocus/assets/19834515/c49269c4-c274-4893-b368-047c401cc58c)

当你首次运行时，程序会自动下载以下模块:

1. [将会下载sd_xl_base_1.0_0.9vae.safetensors](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0_0.9vae.safetensors) 并保存于"Fooocus\models\checkpoints\sd_xl_base_1.0_0.9vae.safetensors".
2. [将会下载sd_xl_refiner_1.0_0.9vae.safetensors](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0_0.9vae.safetensors) 并保存于"Fooocus\models\checkpoints\sd_xl_refiner_1.0_0.9vae.safetensors".

![image](https://github.com/lllyasviel/Fooocus/assets/19834515/d386f817-4bd7-490c-ad89-c1e228c23447)

如果你已经下载过以上文件的话, 可以将他们复制到上述路径中。以加快安装过程.

下图是一次运行在**16GB 内存** 和 **6GB 显存** (Nvidia 3060 laptop)笔记本电脑上的测试. 在这台机器上每次迭代大约耗时1.35秒.

![image](https://github.com/lllyasviel/Fooocus/assets/19834515/938737a5-b105-4f19-b051-81356cb7c495)

请注意，最低系统要求为 **4GB Nvidia 显卡显存 (4GB VRAM)** 和 **8GB 系统内存(8GB RAM)**. 以及Microsoft’s Virtual Swap技术支持, 此选项应当在安装Windows系统时自动启动, 所以应该无需为此特意执行什么. 但当不确定或它被手动关闭时(?), 又或者 **报错 "RuntimeError: CPUAllocator"**, 可以按照如下方法开启:

<details>
<summary>点击这里来查看图文教程. </summary>

![image](https://github.com/lllyasviel/Fooocus/assets/19834515/2a06b130-fe9b-4504-94f1-2763be4476e9)

**报错"RuntimeError: CPUAllocator"时, 请确保至少有40GB空余硬盘空间!**

</details>

当你使用类似硬件但无法达到预期效果时, 可以在这里新建一个issue.

### Colab

(最后测试 - 2023 Aug 14)

| Colab | Info
| --- | --- |
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lllyasviel/Fooocus/blob/main/colab.ipynb) | Fooocus Colab (Official Version)

你可以无视Colab的"you must restart the runtime in order to use newly installed XX"的提示.

感谢[camenduru](https://github.com/camenduru)提供的代码!

### Linux

命令行代码:

    git clone https://github.com/lllyasviel/Fooocus.git
    cd Fooocus
    conda env create -f environment.yaml
    conda activate fooocus
    pip install -r requirements_versions.txt

然后下载以下模组: [点此下载sd_xl_base_1.0_0.9vae.safetensors](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0_0.9vae.safetensors) 并保存于 "Fooocus\models\checkpoints\sd_xl_base_1.0_0.9vae.safetensors", 以及 [点此下载sd_xl_refiner_1.0_0.9vae.safetensors](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0_0.9vae.safetensors) 并保存于 "Fooocus\models\checkpoints\sd_xl_refiner_1.0_0.9vae.safetensors". **或者使用启动器来让Fooocus自动下载所需的模组**:

    python launch.py

当你想打开一个远程端口时,

    python launch.py --listen

### Mac/Windows(AMD GPUs)

Coming soon ...

## List of "Hidden" Tricks
<a name="tech_list"></a>

Below things are already inside the software, and **users do not need to do anything about these**.

~Note that some of these tricks are currently (2023 Aug 11) impossible to reproduce in Automatic1111's interface or ComfyUI's node system.~ (Update Aug 21: We are working on implementing some of these as webui extensions/features.)

1. Native refiner swap inside one single k-sampler. The advantage is that now the refiner model can reuse the base model's momentum (or ODE's history parameters) collected from k-sampling to achieve more coherent sampling. In Automatic1111's high-res fix and ComfyUI's node system, the base model and refiner use two independent k-samplers, which means the momentum is largely wasted, and the sampling continuity is broken. Fooocus uses its own advanced k-diffusion sampling that ensures seamless, native, and continuous swap in a refiner setup. (Update Aug 13: Actually I discussed this with Automatic1111 several days ago and it seems that the “native refiner swap inside one single k-sampler” is [merged]( https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12371) into the dev branch of webui. Great!)
2. Negative ADM guidance. Because the highest resolution level of XL Base does not have cross attentions, the positive and negative signals for XL's highest resolution level cannot receive enough contrasts during the CFG sampling, causing the results look a bit plastic or overly smooth in certain cases. Fortunately, since the XL's highest resolution level is still conditioned on image aspect ratios (ADM), we can modify the adm on the positive/negative side to compensate for the lack of CFG contrast in the highest resolution level. (Update Aug 16, the IOS App [Drawing Things](https://apps.apple.com/us/app/draw-things-ai-generation/id6444050820) will support Negative ADM Guidance. Great!)
3. We implemented a carefully tuned variation of the Section 5.1 of ["Improving Sample Quality of Diffusion Models Using Self-Attention Guidance"](https://arxiv.org/pdf/2210.00939.pdf). The weight is set to very low, but this is Fooocus's final guarantee to make sure that the XL will never yield overly smooth or plastic appearance (examples [here](https://github.com/lllyasviel/Fooocus/discussions/117)). This can almostly eliminate all cases that XL still occasionally produce overly smooth results even with negative ADM guidance. (Update 2023 Aug 18, the Gaussian kernel of SAG is changed to an anisotropic kernel for better structure preservation and fewer artifacts.)
4. We modified the style templates a bit and added the "cinematic-default".
5. We tested the "sd_xl_offset_example-lora_1.0.safetensors" and it seems that when the lora weight is below 0.5, the results are always better than XL without lora.
6. The parameters of samplers are carefully tuned.
7. Because XL uses positional encoding for generation resolution, images generated by several fixed resolutions look a bit better than that from arbitrary resolutions (because the positional encoding is not very good at handling int numbers that are unseen during training). This suggests that the resolutions in UI may be hard coded for best results.
8. Separated prompts for two different text encoders seem unnecessary. Separated prompts for base model and refiner may work but the effects are random, and we refrain from implement this.
9. DPM family seems well-suited for XL, since XL sometimes generates overly smooth texture but DPM family sometimes generate overly dense detail in texture. Their joint effect looks neutral and appealing to human perception.

## Advanced Features

[点击这里浏览高级特性.](https://github.com/lllyasviel/Fooocus/discussions/117)

## Thanks

The codebase starts from an odd mixture of [Automatic1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui) and [ComfyUI](https://github.com/comfyanonymous/ComfyUI). (And they both use GPL license.)

## Update Log

The log is [here](update_log.md).
