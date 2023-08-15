# Fooocus

![image](https://github.com/lllyasviel/Fooocus/assets/19834515/701e32be-f7d2-4be8-b8cc-8115394de406)

Fooocus是一款图像生成软件。

Fooocus是一种基于Stable Diffusion和Midjourney设计理念的重新思考：

* 受到了Stable Diffusion的启发，该软件是离线、开源和免费的。

* 受到了Midjourney的启发，不需要手动调整，用户只需专注于提示和图像。

Fooocus已经包含并自动化了 [许多内部优化和质量改进](#tech_list). 用户可以忘记所有那些复杂的技术参数，只需享受人与计算机之间的互动，以"探索思维的新媒介，扩展人类物种的想象力" `[1]`.

Fooocus简化了安装过程。在点击"下载"和生成第一张图像之间的步骤中，所需的鼠标点击次数严格限制在3次以下。最低的GPU内存要求是4GB（Nvidia）。

`[1]` David Holz, 2019.


## Download

### Windows

你可以直接使用以下链接下载Fooocus：

**[>>> 点击此处下载 <<<](https://github.com/lllyasviel/Fooocus/releases/download/release/Fooocus_win64_1-1-10.7z)**

下载完成后，请解压文件，并运行"run.bat"。

![image](https://github.com/lllyasviel/Fooocus/assets/19834515/c49269c4-c274-4893-b368-047c401cc58c)

第一次启动软件时，它会自动下载模型：

1. 它将从 [这里下载sd_xl_base_1.0_0.9vae.safetensors](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0_0.9vae.safetensors) 作为指定文件 "Fooocus\models\checkpoints\sd_xl_base_1.0_0.9vae.safetensors"。
2. 它将从 [这里下载sd_xl_refiner_1.0_0.9vae.safetensors](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0_0.9vae.safetensors) 作为指定文件"Fooocus\models\checkpoints\sd_xl_refiner_1.0_0.9vae.safetensors"。

![image](https://github.com/lllyasviel/Fooocus/assets/19834515/d386f817-4bd7-490c-ad89-c1e228c23447)

如果你已经有了这些文件，可以将它们复制到上述位置以加快安装速度。

下面是在一台配置相对较低的笔记本电脑上进行的测试，配备 **16GB系统内存** 和 **6GB VRAM**（Nvidia 3060笔记本）。在这台机器上的速度约为每次迭代1.35秒。非常令人印象深刻 - 现在的3060笔记本通常价格都是易于接受的。

![image](https://github.com/lllyasviel/Fooocus/assets/19834515/938737a5-b105-4f19-b051-81356cb7c495)

请注意，最低要求是 **4GB Nvidia GPU内存（4GB VRAM）**和 8GB系统内存（8GB RAM）。这需要使用微软的虚拟交换技术，在大多数情况下，Windows安装会自动启用它，因此通常无需进行任何操作。但如果你不确定，或者手动关闭了它（真的有人这样做吗？），你可以在这里启用它:

<details>
<summary>点击这里查看图像指引。 </summary>

![image](https://github.com/lllyasviel/Fooocus/assets/19834515/2a06b130-fe9b-4504-94f1-2763be4476e9)

</details>
如果您使用类似设备仍无法达到可接受的性能，请开启一个issue进行反馈。

### Colab

（最近测试日期 - 2023年8月14日）

| Colab | Info
| --- | --- |
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lllyasviel/Fooocus/blob/main/colab.ipynb) | Fooocus Colab（官方版本）

请注意，有时该Colab会显示"必须重新启动运行时才能使用新安装的XX"之类的消息。可以安全地忽略此消息。

感谢 [camenduru](https://github.com/camenduru)的代码

### Linux

命令行如下：

    git clone https://github.com/lllyasviel/Fooocus.git
    cd Fooocus
    conda env create -f environment.yaml
    conda activate fooocus
    pip install -r requirements_versions.txt

然后下载模型：[从此处下载sd_xl_base_1.0_0.9vae.safetensors](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0_0.9vae.safetensors)  作为文件"Fooocus\models\checkpoints\sd_xl_base_1.0_0.9vae.safetensors"，并[从此处下载sd_xl_refiner_1.0_0.9vae.safetensors](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0_0.9vae.safetensors)作为文件"Fooocus\models\checkpoints\sd_xl_refiner_1.0_0.9vae.safetensors"。**或者让Fooocus使用启动器自动下载模型**：

    python launch.py

如果您想打开远程端口，请使用以下命令：


    python launch.py --listen

### Mac/Windows(AMD GPUs)

即将推出...

## "隐藏"技巧清单
<a name="tech_list"></a>

下面这些内容已经包含在软件中，**用户无需对此进行任何操作**。

请注意，其中一些技巧（截至2023年8月11日）目前不可能在Automatic1111的界面或ComfyUI的节点系统中复现。即使其他软件使用相似的模型/流程，您也可以期待Fooocus提供更好的结果。


1. 单个k-sampler内的本地细化交换。优势在于细化模型现在可以重用从k-sampling中收集的基础模型动量（或ODE的历史参数），以实现更连贯的采样。在Automatic1111的高分辨率修复和ComfyUI的节点系统中，基础模型和细化模型使用了两个独立的k-sampler，这意味着动量很大程度上被浪费，采样的连续性被打破。Fooocus使用了其自己先进的k-diffusion采样，确保在细化设置中实现无缝、本地和连续的交换。（更新于8月13日：实际上，我几天前与Automatic1111讨论过这个问题，似乎“单个k-sampler内的本地细化交换”已经[合并]( https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12371) 到webui的dev分支中。太棒了！）
2. 负ADM引导。由于XL Base的最高分辨率级别没有交叉注意力，XL最高分辨率级别的正负信号在CFG采样过程中无法获得足够的对比度，导致结果在某些情况下看起来有点塑料或过于平滑。幸运的是，由于XL的最高分辨率级别仍然基于图像宽高比（ADM）进行约束，我们可以修改正/负面的adm来弥补最高分辨率级别中CFG对比度不足的问题。
3. 我们实现了["利用自注意引导提高扩散模型的样本质量](https://arxiv.org/pdf/2210.00939.pdf)第5.1节的经过精心调整的变体。权重设置得非常低，但这是Fooocus确保XL永远不会产生过于平滑或塑料外观的最终保证。这几乎可以消除所有XL偶尔生成过于平滑结果的情况，即使有负ADM引导也是如此。
4. 我们稍微修改了样式模板，并添加了“cinematic-default”。
5. 我们测试了“sd_xl_offset_example-lora_1.0.safetensors”，发现当lora权重低于0.5时，比没有lora的XL结果始终更好。
6. 采样器的参数经过精心调整。
7. 因为XL在生成分辨率上使用了位置编码，所以通过几个固定分辨率生成的图像看起来比从任意分辨率生成的图像更好（因为位置编码在处理训练中未见的整数时效果不佳）。这表明UI中的分辨率可能是硬编码以获得最佳结果。
8. 两个不同文本编码器的分开提示似乎是不必要的。基础模型和细化模型的分开提示可能有效，但效果是随机的，我们决定不实现这个功能。
9. DPM系列似乎非常适合XL，因为XL有时会生成过于平滑的纹理，而DPM系列有时会在纹理中生成过于密集的细节。它们的联合效果在人类感知中具有中性和吸引力。


## 致谢

代码库从[Automatic1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui)和[ComfyUI](https://github.com/comfyanonymous/ComfyUI)的奇怪混合开始。（他们都使用GPL许可证。）

## 更新日志

日志在 [这里](update_log.md).
