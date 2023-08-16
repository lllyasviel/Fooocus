# Fooocus

<img src="https://github.com/lllyasviel/Fooocus/assets/19834515/bcb0336b-5c79-4de2-b0cb-f7f68c753a88" width=100%>

Fooocus is an image generating software.

Fooocus is a rethinking of Stable Diffusion and Midjourney’s designs:

* Learned from Stable Diffusion, the software is offline, open source, and free.

* Learned from Midjourney, the manual tweaking is not needed, and users only need to focus on the prompts and images.

Fooocus has included and automated [lots of inner optimizations and quality improvements](#tech_list). Users can forget all those difficult technical parameters, and just enjoy the interaction between human and computer to "explore new mediums of thought and expanding the imaginative powers of the human species" `[1]`.

Fooocus has simplified the installation. Between pressing "download" and generating the first image, the number of needed mouse clicks is strictly limited to less than 3. Minimal GPU memory requirement is 4GB (Nvidia).

Fooocus also developed many "fooocus-only" features for advanced users to get perfect results. [Click here to browse the advanced features.](https://github.com/lllyasviel/Fooocus/discussions/117)

`[1]` David Holz, 2019.

## Download

### Windows

You can directly download Fooocus with:

**[>>> Click here to download <<<](https://github.com/lllyasviel/Fooocus/releases/download/release/Fooocus_win64_1-1-10.7z)**

After you download the file, please uncompress it, and then run the "run.bat".

![image](https://github.com/lllyasviel/Fooocus/assets/19834515/c49269c4-c274-4893-b368-047c401cc58c)

In the first time you launch the software, it will automatically download models:

1. It will download [sd_xl_base_1.0_0.9vae.safetensors from here](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0_0.9vae.safetensors) as the file "Fooocus\models\checkpoints\sd_xl_base_1.0_0.9vae.safetensors".
2. It will download [sd_xl_refiner_1.0_0.9vae.safetensors from here](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0_0.9vae.safetensors) as the file "Fooocus\models\checkpoints\sd_xl_refiner_1.0_0.9vae.safetensors".

![image](https://github.com/lllyasviel/Fooocus/assets/19834515/d386f817-4bd7-490c-ad89-c1e228c23447)

If you already have these files, you can copy them to the above locations to speed up installation.

Below is a test on a relatively low-end laptop with **16GB System RAM** and **6GB VRAM** (Nvidia 3060 laptop). The speed on this machine is about 1.35 seconds per iteration. Pretty impressive – nowadays laptops with 3060 are usually at very acceptable price.

![image](https://github.com/lllyasviel/Fooocus/assets/19834515/938737a5-b105-4f19-b051-81356cb7c495)

Note that the minimal requirement is **4GB Nvidia GPU memory (4GB VRAM)** and **8GB system memory (8GB RAM)**. This requires using Microsoft’s Virtual Swap technique, which is automatically enabled by your Windows installation in most cases, so you often do not need to do anything about it. However, if you are not sure, or if you manually turned it off (would anyone really do that?), you can enable it here:

<details>
<summary>Click here to the see the image instruction. </summary>

![image](https://github.com/lllyasviel/Fooocus/assets/19834515/2a06b130-fe9b-4504-94f1-2763be4476e9)

</details>

Please open an issue if you use similar devices but still cannot achieve acceptable performances.

### Colab

(Last tested - 2023 Aug 14)

| Colab | Info
| --- | --- |
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lllyasviel/Fooocus/blob/main/colab.ipynb) | Fooocus Colab (Official Version)

Note that sometimes this Colab will say like "you must restart the runtime in order to use newly installed XX". This can be safely ignored.

Thanks to [camenduru](https://github.com/camenduru)'s codes!

### Linux

The command lines are

    git clone https://github.com/lllyasviel/Fooocus.git
    cd Fooocus
    conda env create -f environment.yaml
    conda activate fooocus
    pip install -r requirements_versions.txt

Then download the models: download [sd_xl_base_1.0_0.9vae.safetensors from here](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0_0.9vae.safetensors) as the file "Fooocus\models\checkpoints\sd_xl_base_1.0_0.9vae.safetensors", and download [sd_xl_refiner_1.0_0.9vae.safetensors from here](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0_0.9vae.safetensors) as the file "Fooocus\models\checkpoints\sd_xl_refiner_1.0_0.9vae.safetensors". **Or let Fooocus automatically download the models** using the launcher:

    python launch.py

Or if you want to open a remote port, use

    python launch.py --listen

### Mac/Windows(AMD GPUs)

Coming soon ...

## List of "Hidden" Tricks
<a name="tech_list"></a>

<h3>Important Notes:</h3>
<p>Below things are already inside the software, and users do not need to do anything about these.</p>

<p>Note that some of these tricks are currently (2023 Aug 11) impossible to reproduce in Automatic1111's interface or ComfyUI's node system. You may expect better results from Fooocus than other software even when they use similar models/pipelines.</p>

<h3>Key Features:</h3>
<ul>
  <li><strong>Native Refiner Swap:</strong> Inside one single k-sampler. This advantage allows the refiner model to reuse the base model's momentum (or ODE's history parameters) collected from k-sampling for more coherent sampling.</li>
  <li><strong>Negative ADM Guidance:</strong> Adjusting ADM on the positive/negative side compensates for lack of CFG contrast in the highest resolution level of XL's base model, ensuring more natural results.</li>
  <li><strong>Improved Sample Quality:</strong> A variation of "Improving Sample Quality of Diffusion Models Using Self-Attention Guidance" is implemented, ensuring that XL avoids overly smooth or plastic appearances.</li>
  <li><strong>Style Template Modification:</strong> Style templates are tweaked, including the addition of "cinematic-default".</li>
  <li><strong>Optimized Samplers:</strong> Parameters of samplers are carefully tuned for optimal performance.</li>
  <li><strong>Resolution Impact:</strong> Fixed resolutions yield better results due to positional encoding for generation resolution. Suggests that UI resolutions may be hard coded for best results.</li>
  <li><strong>Unified Prompts:</strong> Separate prompts for two different text encoders deemed unnecessary, and separate prompts for base model and refiner are refrained from implementation.</li>
  <li><strong>DPM Family:</strong> XL and DPM family joint effect provides neutral and appealing results, balancing texture smoothness and detail density.</li>
</ul>

<p><em>Last Updated: August 16, 2023</em></p>

## Advanced Features

[Click here to browse the advanced features.](https://github.com/lllyasviel/Fooocus/discussions/117)

## Thanks

The codebase starts from an odd mixture of [Automatic1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui) and [ComfyUI](https://github.com/comfyanonymous/ComfyUI). (And they both use GPL license.)

## Update Log

The log is [here](update_log.md).
