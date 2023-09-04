# Fooocus

## <div align="center"><b><a href="readme.md">English</a> | <a href="readme_CN.md">简体中文</a></b></div>

<img src="https://github.com/lllyasviel/Fooocus/assets/19834515/f79c5981-cf80-4ee3-b06b-3fef3f8bfbc7" width=100%>

Fooocus是一个AI图像生成软件。

Fooocus参考借鉴了Stable Diffusion和Midjourney的设计理念，融合了两者的优点：

* Stable Diffusion的优势和长处, 软件离线运行、开源、免费。

* Midjourney的优势和长处, 几乎无需手动设置参数，用户只需聚焦在提示词创作。

Fooocus包括了一系列自动化的[内部优化和质量提升](#tech_list). 用户可以忘掉复杂的参数设置，只需发挥人类无限想象力，在简单的操作界面进行创作。原文（Original Text）：Users can forget all those difficult technical parameters, and just enjoy the interaction between human and computer to "explore new mediums of thought and expanding the imaginative powers of the human species" `[1]`.

Fooocus安装简单。从下载到生成您的第一张图片只需鼠标操作不超过3步。最低仅需4G GPU（Nvidia）显存。

面向高级用户的完美输出结果诉求，Fooocus也提供了许多"fooocus-only"高级功能。 [点击这里浏览高级功能。](https://github.com/lllyasviel/Fooocus/discussions/117)

`[1]` David Holz, 2019.

## 下载

### Windows

你可以通过下方链接直接下载Fooocus：

**[>>> 点击这里即可下载 <<<](https://github.com/lllyasviel/Fooocus/releases/download/1.0.35/Fooocus_win64_1-1-1035.7z)**

完成下载后，解压，然后运行"run.bat"。

![image](https://github.com/lllyasviel/Fooocus/assets/19834515/c49269c4-c274-4893-b368-047c401cc58c)

首次运行时，会自动下载下述模型:

1. 会自动下载 [sd_xl_base_1.0_0.9vae.safetensors from here](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0_0.9vae.safetensors) 并放置于 "Fooocus\models\checkpoints\sd_xl_base_1.0_0.9vae.safetensors".
2. 会自动下载 [sd_xl_refiner_1.0_0.9vae.safetensors from here](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0_0.9vae.safetensors) 并放置于 "Fooocus\models\checkpoints\sd_xl_refiner_1.0_0.9vae.safetensors".

![image](https://github.com/lllyasviel/Fooocus/assets/19834515/d386f817-4bd7-490c-ad89-c1e228c23447)

如果你预先已经下载过了这些文件，你可以自行复制到上述位置以加快安装速度。

请注意：如果看见诸如 **"MetadataIncompleteBuffer"** 的提示, 则代表你的模型文件损坏，请重新下载模型。

下方为一个相对较低的硬件环境（Nvidia 3060笔记本）测试截图（**16GB系统内存**和**6GB显存**），速度大约为1.35it/s。如今配备3060显卡的笔记本价格已大众化。

![image](https://github.com/lllyasviel/Fooocus/assets/19834515/938737a5-b105-4f19-b051-81356cb7c495)

我们发现最低的硬件环境仅需**4GB Nvidia GPU显存 (4GB VRAM)**和**8GB系统内存 (8GB RAM)**。这得益于微软的虚拟内存技术，大多数情况下，Windows会默认启用，无需调整。但某些用户会禁用它，如果你不确定是否已启用，或者 **当你看到控制台诸如"RuntimeError: CPUAllocator"这样的提示** ，你可以参照下方的操作步骤启用它：

<details>
<summary>点击这里展开图片详解启用虚拟内存。 </summary>

![image](https://github.com/lllyasviel/Fooocus/assets/19834515/2a06b130-fe9b-4504-94f1-2763be4476e9)


**如果持续看见"RuntimeError: CPUAllocator"的提示，请确保有至少虚拟内存所指定的硬盘有至少40GB以上的可用空间！**

</details>

如果你有类似的配置，但是达不到预期性能，你可以在本仓库提交一个issue请求。

### Colab

(最新测试于：2023年8月30日)

| Colab | Info
| --- | --- |
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lllyasviel/Fooocus/blob/main/colab.ipynb) | Fooocus Colab (官方版本)

有时候，Colab会报"you must restart the runtime in order to use newly installed XX"。遇到这个情况，你可以忽略它

感谢[camenduru](https://github.com/camenduru)提供的代码！

### Linux

执行下方命令

    git clone https://github.com/lllyasviel/Fooocus.git
    cd Fooocus
    conda env create -f environment.yaml
    conda activate fooocus
    pip install -r requirements_versions.txt

然后下载下列模型: 下载[sd_xl_base_1.0_0.9vae.safetensors from here](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0_0.9vae.safetensors)并放在 "Fooocus\models\checkpoints\sd_xl_base_1.0_0.9vae.safetensors"， 下载[sd_xl_refiner_1.0_0.9vae.safetensors from here](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0_0.9vae.safetensors)并放在"Fooocus\models\checkpoints\sd_xl_refiner_1.0_0.9vae.safetensors". **或者让Fooocus自动下载模型**，用下方命令启动:

    python launch.py

如果你想启动一个远程端口，可以加上下述参数，如：

    python launch.py --listen

### Mac/Windows(AMD GPUs)

即将提供 ...

## 一些“神操作”的说明
<a name="tech_list"></a>

下面的东西是软件内部工作原理，**用户无需额外做任何操作**。

（为减少歧义，不进行翻译，请自行阅读原文理解。）

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

## 高级功能

[点击这里浏览关于高级功能的说明](https://github.com/lllyasviel/Fooocus/discussions/117)

## 致谢

本项目混合了[Automatic1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui)和[ComfyUI](https://github.com/comfyanonymous/ComfyUI). (它们均为GPL授权许可.)

## 更新日志

The log is [这里](update_log.md).
