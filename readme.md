# Fooocus - mashb1t's 1-Up Edition

The purpose of this fork is to add new features / fix bugs and contribute back to [Fooocus](https://github.com/lllyasviel/Fooocus).

As a collaborator & contributor of the Fooocus repository you can find me in almost every [issue](https://github.com/lllyasviel/Fooocus/issues), [pull request](https://github.com/lllyasviel/Fooocus/pulls), [discussion](https://github.com/lllyasviel/Fooocus/discussions) etc.

Sadly the creator of Fooocus has gone dark multiple times for an extended amount of time, which is why I took matters into my own hands.

![BillsUghGIF](https://github.com/mashb1t/Fooocus/assets/9307310/78c04e06-8ef0-4224-9c25-8f1bee9861de)

## Additional features included in this fork:
(mostly a reflection of [my PRs](https://github.com/lllyasviel/Fooocus/pulls/mashb1t))

* ✨ ~~https://github.com/lllyasviel/Fooocus/pull/958 - NSFW image censoring (config and UI)~~
* 🐛 ~~https://github.com/lllyasviel/Fooocus/pull/981 - prevent users from skipping/stopping other users tasks in queue (multi-user capabilities) + rework advanced_parameters (removal + PID handling)~~
* ✨ ~~https://github.com/lllyasviel/Fooocus/pull/985 - add list of 100 animals to wildcards~~
* ✨ ~~https://github.com/lllyasviel/Fooocus/pull/1013 - add advanced parameter for disable_intermediate_results (progress_gallery, prevents UI lag when generation is too fast)~~
* ✨ https://github.com/lllyasviel/Fooocus/pull/1039 - add prompt translation
* ✨ https://github.com/lllyasviel/Fooocus/pull/1043 - add lcm realtime canvas painting (not merged to main in this repository)
* ✨ ~~https://github.com/lllyasviel/Fooocus/pull/1167 - update model BluePencil XL v0.5 to v3.1.0~~
* ✨ ~~https://github.com/lllyasviel/Fooocus/pull/1570 - add preset selection to Gradio UI (session based)~~
* 🐛 ~~https://github.com/lllyasviel/Fooocus/pull/1578 - add workaround for changing prompt while generating~~
* ✨ https://github.com/lllyasviel/Fooocus/pull/1580 - add preset for SDXL Turbo (model DreamShaperXL_Turbo)
* ✨ ~~https://github.com/lllyasviel/Fooocus/pull/1616 - add config setting for default_max_image_number~~
* 🐛 ~~https://github.com/lllyasviel/Fooocus/pull/1668 - fix path_outputs directory creation if it doesn't exist~~
* ✨ show more details for each performance setting, e.g. steps
* ✨ ~~add default_overwrite_step handling for meta data and gradio (allows turbo preset switching to set default_overwrite_step correctly)~~
* ✨ ~~https://github.com/lllyasviel/Fooocus/pull/1762 - add style preview on mouseover~~
* 🐛 ~~https://github.com/lllyasviel/Fooocus/pull/1784 - correctly sort files, display deepest directory level first~~
* ✨ ~~https://github.com/lllyasviel/Fooocus/pull/1785 - update model Juggernaut XL v6 to v8~~
* ✨ https://github.com/lllyasviel/Fooocus/pull/1809 - reduce file size of preview images
* ✨ ~~https://github.com/lllyasviel/Fooocus/pull/1932 - use consistent file name in gradio~~
* ✨ ~~https://github.com/lllyasviel/Fooocus/pull/1863 - image extension support (png, jpg, webp)~~
* ✨ ~~https://github.com/lllyasviel/Fooocus/pull/1938 - automatically describe image on uov image upload if prompt is empty~~
* ✨ ~~https://github.com/lllyasviel/Fooocus/pull/1940 - meta data handling, schemes: Fooocus (json) and A1111 (plain text). Compatible with Civitai.~~
* ✨ ~~https://github.com/lllyasviel/Fooocus/pull/1979 - prevent outdated history log link after midnight~~
* ✨ https://github.com/lllyasviel/Fooocus/pull/2032 - add inpaint mask generation functionality using rembg, incl. segmentation support
* 🐛 ~~https://github.com/lllyasviel/Fooocus/pull/2332 - allow path_outputs to be outside of root dir~~
* ✨ ~~https://github.com/lllyasviel/Fooocus/pull/2415 - add performance sdxl lightning (4 steps)~~
* and many more (90+) are already merged, see [my PRs](https://github.com/lllyasviel/Fooocus/pulls/mashb1t)

✨ = new feature<br>
🐛 = bugfix<br>
~~abc~~ = merged

---

## Feature showcase

### Enhance - Automatic Image Upscaling + Enhancement

Similar to adetailer, but based on dynamic image detection instead of specific mask detection models. All with one click!

See [documentation](https://github.com/mashb1t/Fooocus/discussions/42).

| Original (generated)| Upscale (before) | `#1` Yellow Sundress | `#2` Hands replacement|
|--------|--------|--------|--------|
| ![image - 2024-06-23T154659 098](https://github.com/mashb1t/Fooocus/assets/9307310/260e6a0b-5f42-47a3-83fd-fc3f4310d3a7)| ![image - 2024-06-23T155614 010](https://github.com/mashb1t/Fooocus/assets/9307310/09c9d620-08d8-4a3f-b99e-dddf9b653234)| ![image - 2024-06-23T155852 562](https://github.com/mashb1t/Fooocus/assets/9307310/c31b210c-b846-47c2-a3f6-db4e3241401f) | ![image - 2024-06-23T161011 491](https://github.com/mashb1t/Fooocus/assets/9307310/8a5d1ff8-cf60-4cf2-85f7-065cf5ccaf8d)| 

---

### https://github.com/lllyasviel/Fooocus/pull/2032 - Automated Mask Generation + Mask Prompting

https://github.com/mashb1t/Fooocus/assets/9307310/b69bf607-128b-48a6-a248-fa2b09218fe7

Videos by [@rayronvictor](https://github.com/rayronvictor)

<details><summary>Mask generation by cloth category</summary>
<p>

https://github.com/mashb1t/Fooocus/assets/9307310/204a01f6-63af-4fd2-bd92-76e176849f19

</p>
</details> 

<details><summary>Mask generation by prompt</summary>
<p>

https://github.com/mashb1t/Fooocus/assets/9307310/b69bf607-128b-48a6-a248-fa2b09218fe7

</p>
</details> 

---

### https://github.com/lllyasviel/Fooocus/pull/1940 - Metadata Handling - Compatible with Civitai & A1111
This feature offers activatable metadata persistency in images for both a Fooocus (json) and A1111 (plain text) meta data scheme, where the latter is 100% compatible with A1111 and Civitai, but can not be used to reproduce the image outside of Fooocus, as there are so many improvements and special things happening in Fooocus it's just not applicable anywhere else.
- Supports metadata for PNG (PngInfo) + JPG and WebP (both EXIF).
- Save & restore configurations directly from images
- You can also configure a copyright / creator tag

![Screenshot 2024-01-29 at 15 13 17](https://github.com/lllyasviel/Fooocus/assets/9307310/6b7df4eb-feb3-46ee-bf09-f336be63b625)

<details><summary>Gradio (setting in Developer Debug Mode)</summary>
<p>

Default is Fooocus Scheme
![image](https://github.com/lllyasviel/Fooocus/assets/9307310/ae529db2-f5d1-4725-9735-3036b50020b7)

</p>
</details> 

<details><summary>Config options</summary>
<p>

    "default_save_metadata_to_images": true,
    "default_metadata_scheme": "a1111",
    "metadata_created_by": "mashb1t"

</p>
</details> 

<details><summary>Arg --disable-metadata</summary>
<p>

 `--disable-metadata` completely prevents metadata processing and output in Gradio

</p>
</details> 


<details><summary>Metadata Reader</summary>
<p>

1. open Image Input > Metadata tab
2. drag & Drop image to image upload
3. automatic preview of image metadata
4. apply metadata to Gradio inputs on button click

Fooocus scheme
![Screenshot 2024-01-29 at 15 13 17](https://github.com/lllyasviel/Fooocus/assets/9307310/6b7df4eb-feb3-46ee-bf09-f336be63b625)

A1111 scheme
![Screenshot 2024-01-29 at 15 09 52](https://github.com/lllyasviel/Fooocus/assets/9307310/1ee3c030-2b6d-41cb-9f88-40df6452df15)

</p>
</details> 


<details><summary>Metadata in files</summary>
<p>

Speed Fooocus scheme
![image](https://github.com/lllyasviel/Fooocus/assets/9307310/c556b2ed-0e0b-4117-9bda-9508fb3e0d96)

LCM A1111 scheme (yes, with negative prompt, because it technically exists but doesn't have an influence)
![image](https://github.com/lllyasviel/Fooocus/assets/9307310/322de4e1-00d7-428a-84db-64e4a9a5637e)

Speed A1111 scheme
![image](https://github.com/lllyasviel/Fooocus/assets/9307310/3b49627a-4e3f-4d13-b795-04db2743baab)

</p>
</details> 

<details><summary>Civitai</summary>
<p>

Speed Fooocus scheme
![image](https://github.com/lllyasviel/Fooocus/assets/9307310/1d31af41-d5a7-4f86-b89c-3aeeb1733417)

LCM A1111 scheme
![image](https://github.com/lllyasviel/Fooocus/assets/9307310/d91a4b71-ffe1-430b-8db0-0a5b0662d48f)

![image](https://github.com/lllyasviel/Fooocus/assets/9307310/e0e93b0e-30ee-4263-95de-05d56ff7a885)

Speed A1111 scheme
![image](https://github.com/lllyasviel/Fooocus/assets/9307310/1cf4b6d6-775d-4d68-99f2-a7dea34241b3)

![image](https://github.com/lllyasviel/Fooocus/assets/9307310/d55c6b6c-08e1-4fcb-a22e-39d45be1c0c2)


</p>
</details>

---

<div align=center>
<img src="https://github.com/lllyasviel/Fooocus/assets/19834515/483fb86d-c9a2-4c20-997c-46dafc124f25">
</div>

# Fooocus

[>>> Click Here to Install Fooocus <<<](#download)

Fooocus is an image generating software (based on [Gradio](https://www.gradio.app/) <a href='https://github.com/gradio-app/gradio'><img src='https://img.shields.io/github/stars/gradio-app/gradio'></a>).

Fooocus presents a rethinking of image generator designs. The software is offline, open source, and free, while at the same time, similar to many online image generators like Midjourney, the manual tweaking is not needed, and users only need to focus on the prompts and images. Fooocus has also simplified the installation: between pressing "download" and generating the first image, the number of needed mouse clicks is strictly limited to less than 3. Minimal GPU memory requirement is 4GB (Nvidia).

**Recently many fake websites exist on Google when you search “fooocus”. Do not trust those – here is the only official source of Fooocus.**

# Project Status: Limited Long-Term Support (LTS) with Bug Fixes Only

The Fooocus project, built entirely on the **Stable Diffusion XL** architecture, is now in a state of limited long-term support (LTS) with bug fixes only. As the existing functionalities are considered as nearly free of programmartic issues (Thanks to [mashb1t](https://github.com/mashb1t)'s huge efforts), future updates will focus exclusively on addressing any bugs that may arise. 

**There are no current plans to migrate to or incorporate newer model architectures.** However, this may change during time with the development of open-source community. For example, if the community converge to one single dominant method for image generation (which may really happen in half or one years given the current status), Fooocus may also migrate to that exact method.

For those interested in utilizing newer models such as **Flux**, we recommend exploring alternative platforms such as [WebUI Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge) (also from us), [ComfyUI/SwarmUI](https://github.com/comfyanonymous/ComfyUI). Additionally, several [excellent forks of Fooocus](https://github.com/lllyasviel/Fooocus?tab=readme-ov-file#forks) are available for experimentation.

Again, recently many fake websites exist on Google when you search “fooocus”. Do **NOT** get Fooocus from those websites – this page is the only official source of Fooocus. We never have any website like such as “fooocus.com”, “fooocus.net”, “fooocus.co”, “fooocus.ai”, “fooocus.org”, “fooocus.pro”, “fooocus.one”. Those websites are ALL FAKE. **They have ABSOLUTLY no relationship to us. Fooocus is a 100% non-commercial offline open-source software.**

# Features

Below is a quick list using Midjourney's examples:

| Midjourney | Fooocus |
| - | - |
| High-quality text-to-image without needing much prompt engineering or parameter tuning. <br> (Unknown method) | High-quality text-to-image without needing much prompt engineering or parameter tuning. <br> (Fooocus has an offline GPT-2 based prompt processing engine and lots of sampling improvements so that results are always beautiful, no matter if your prompt is as short as “house in garden” or as long as 1000 words) |
| V1 V2 V3 V4 | Input Image -> Upscale or Variation -> Vary (Subtle) / Vary (Strong)|
| U1 U2 U3 U4 | Input Image -> Upscale or Variation -> Upscale (1.5x) / Upscale (2x) |
| Inpaint / Up / Down / Left / Right (Pan) | Input Image -> Inpaint or Outpaint -> Inpaint / Up / Down / Left / Right <br> (Fooocus uses its own inpaint algorithm and inpaint models so that results are more satisfying than all other software that uses standard SDXL inpaint method/model) |
| Image Prompt | Input Image -> Image Prompt <br> (Fooocus uses its own image prompt algorithm so that result quality and prompt understanding are more satisfying than all other software that uses standard SDXL methods like standard IP-Adapters or Revisions) |
| --style | Advanced -> Style |
| --stylize | Advanced -> Advanced -> Guidance |
| --niji | [Multiple launchers: "run.bat", "run_anime.bat", and "run_realistic.bat".](https://github.com/lllyasviel/Fooocus/discussions/679) <br> Fooocus support SDXL models on Civitai <br> (You can google search “Civitai” if you do not know about it) |
| --quality | Advanced -> Quality |
| --repeat | Advanced -> Image Number |
| Multi Prompts (::) | Just use multiple lines of prompts |
| Prompt Weights | You can use " I am (happy:1.5)". <br> Fooocus uses A1111's reweighting algorithm so that results are better than ComfyUI if users directly copy prompts from Civitai. (Because if prompts are written in ComfyUI's reweighting, users are less likely to copy prompt texts as they prefer dragging files) <br> To use embedding, you can use "(embedding:file_name:1.1)" |
| --no | Advanced -> Negative Prompt |
| --ar | Advanced -> Aspect Ratios |
| InsightFace | Input Image -> Image Prompt -> Advanced -> FaceSwap |
| Describe | Input Image -> Describe |

Below is a quick list using LeonardoAI's examples:

| LeonardoAI | Fooocus |
| - | - |
| Prompt Magic | Advanced -> Style -> Fooocus V2 |
| Advanced Sampler Parameters (like Contrast/Sharpness/etc) | Advanced -> Advanced -> Sampling Sharpness / etc |
| User-friendly ControlNets | Input Image -> Image Prompt -> Advanced |

Also, [click here to browse the advanced features.](https://github.com/lllyasviel/Fooocus/discussions/117)

# Download

### Windows

You can directly download Fooocus with:

**[>>> Click here to download <<<](https://github.com/mashb1t/Fooocus/releases/download/v2.6.0/Fooocus_win64_2-6-0.7z)**

After you download the file, please uncompress it and then run the "run.bat".

![image](https://github.com/lllyasviel/Fooocus/assets/19834515/c49269c4-c274-4893-b368-047c401cc58c)

The first time you launch the software, it will automatically download models:

1. It will download [default models](#models) to the folder "Fooocus\models\checkpoints" given different presets. You can download them in advance if you do not want automatic download.
2. Note that if you use inpaint, at the first time you inpaint an image, it will download [Fooocus's own inpaint control model from here](https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v26.fooocus.patch) as the file "Fooocus\models\inpaint\inpaint_v26.fooocus.patch" (the size of this file is 1.28GB).

After Fooocus 2.1.60, you will also have `run_anime.bat` and `run_realistic.bat`. They are different model presets (and require different models, but they will be automatically downloaded). [Check here for more details](https://github.com/lllyasviel/Fooocus/discussions/679).

After Fooocus 2.3.0 you can also switch presets directly in the browser. Keep in mind to add these arguments if you want to change the default behavior:
* Use `--disable-preset-selection` to disable preset selection in the browser.
* Use `--always-download-new-model` to download missing models on preset switch. Default is fallback to `previous_default_models` defined in the corresponding preset, also see terminal output.

![image](https://github.com/lllyasviel/Fooocus/assets/19834515/d386f817-4bd7-490c-ad89-c1e228c23447)

If you already have these files, you can copy them to the above locations to speed up installation.

Note that if you see **"MetadataIncompleteBuffer" or "PytorchStreamReader"**, then your model files are corrupted. Please download models again.

Below is a test on a relatively low-end laptop with **16GB System RAM** and **6GB VRAM** (Nvidia 3060 laptop). The speed on this machine is about 1.35 seconds per iteration. Pretty impressive – nowadays laptops with 3060 are usually at very acceptable price.

![image](https://github.com/lllyasviel/Fooocus/assets/19834515/938737a5-b105-4f19-b051-81356cb7c495)

Besides, recently many other software report that Nvidia driver above 532 is sometimes 10x slower than Nvidia driver 531. If your generation time is very long, consider download [Nvidia Driver 531 Laptop](https://www.nvidia.com/download/driverResults.aspx/199991/en-us/) or [Nvidia Driver 531 Desktop](https://www.nvidia.com/download/driverResults.aspx/199990/en-us/).

Note that the minimal requirement is **4GB Nvidia GPU memory (4GB VRAM)** and **8GB system memory (8GB RAM)**. This requires using Microsoft’s Virtual Swap technique, which is automatically enabled by your Windows installation in most cases, so you often do not need to do anything about it. However, if you are not sure, or if you manually turned it off (would anyone really do that?), or **if you see any "RuntimeError: CPUAllocator"**, you can enable it here:

<details>
<summary>Click here to see the image instructions. </summary>

![image](https://github.com/lllyasviel/Fooocus/assets/19834515/2a06b130-fe9b-4504-94f1-2763be4476e9)

**And make sure that you have at least 40GB free space on each drive if you still see "RuntimeError: CPUAllocator" !**

</details>

Please open an issue if you use similar devices but still cannot achieve acceptable performances.

Note that the [minimal requirement](#minimal-requirement) for different platforms is different.

See also the common problems and troubleshoots [here](troubleshoot.md).

### Switching from Fooocus to this fork

1. open a terminal in your Fooocus folder (the one with your config.txt)
2. execute `git status`. You should see the following:
    ```
    On branch main
    Your branch is up to date with 'origin/main'.
    
    nothing to commit, working tree clean
    ```
    If not, execute `git reset --hard origin/main` and check `git status` again.
3. execute
    ```
    git remote set-url origin https://github.com/mashb1t/Fooocus.git
    git reset --hard origin/main
    git pull
    ```
6. activate your venv (not necessary when installed from 7z) and update your python packages depending on your environment (7z, venv, conda, etc.)

   Example for Windows (7z): `..\python_embeded\python.exe -m pip install -r "requirements_versions.txt"`
8. start Fooocus either by opening the run.bat or corresponding entrypoint (same as before)

OR

Windows: download the [7z file](#download), extract it and run `run.bat`. You may want to copy over already downloaded checkpoints / LoRAs / etc.

### Colab

(Last tested - 2024 Aug 12 by [mashb1t](https://github.com/mashb1t))

| Colab | Info
| --- | --- |
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mashb1t/Fooocus/blob/main/fooocus_colab.ipynb) | Fooocus Official

In Colab, you can modify the last line to `!python entry_with_update.py --share --always-high-vram` or `!python entry_with_update.py --share --always-high-vram --preset anime` or `!python entry_with_update.py --share --always-high-vram --preset realistic` for Fooocus Default/Anime/Realistic Edition.

You can also change the preset in the UI. Please be aware that this may lead to timeouts after 60 seconds. If this is the case, please wait until the download has finished, change the preset to initial and back to the one you've selected or reload the page.

Note that this Colab will disable refiner by default because Colab free's resources are relatively limited (and some "big" features like image prompt may cause free-tier Colab to disconnect). We make sure that basic text-to-image is always working on free-tier Colab.

Using `--always-high-vram` shifts resource allocation from RAM to VRAM and achieves the overall best balance between performance, flexibility and stability on the default T4 instance. Please find more information [here](https://github.com/lllyasviel/Fooocus/pull/1710#issuecomment-1989185346).

Thanks to [camenduru](https://github.com/camenduru) for the template!

### Linux (Using Anaconda)

If you want to use Anaconda/Miniconda, you can

    git clone https://github.com/lllyasviel/Fooocus.git
    cd Fooocus
    conda env create -f environment.yaml
    conda activate fooocus
    pip install -r requirements_versions.txt

Then download the models: download [default models](#models) to the folder "Fooocus\models\checkpoints". **Or let Fooocus automatically download the models** using the launcher:

    conda activate fooocus
    python entry_with_update.py

Or, if you want to open a remote port, use

    conda activate fooocus
    python entry_with_update.py --listen

Use `python entry_with_update.py --preset anime` or `python entry_with_update.py --preset realistic` for Fooocus Anime/Realistic Edition.

### Linux (Using Python Venv)

Your Linux needs to have **Python 3.10** installed, and let's say your Python can be called with the command **python3** with your venv system working; you can

    git clone https://github.com/lllyasviel/Fooocus.git
    cd Fooocus
    python3 -m venv fooocus_env
    source fooocus_env/bin/activate
    pip install -r requirements_versions.txt

See the above sections for model downloads. You can launch the software with:

    source fooocus_env/bin/activate
    python entry_with_update.py

Or, if you want to open a remote port, use

    source fooocus_env/bin/activate
    python entry_with_update.py --listen

Use `python entry_with_update.py --preset anime` or `python entry_with_update.py --preset realistic` for Fooocus Anime/Realistic Edition.

### Linux (Using native system Python)

If you know what you are doing, and your Linux already has **Python 3.10** installed, and your Python can be called with the command **python3** (and Pip with **pip3**), you can

    git clone https://github.com/lllyasviel/Fooocus.git
    cd Fooocus
    pip3 install -r requirements_versions.txt

See the above sections for model downloads. You can launch the software with:

    python3 entry_with_update.py

Or, if you want to open a remote port, use

    python3 entry_with_update.py --listen

Use `python entry_with_update.py --preset anime` or `python entry_with_update.py --preset realistic` for Fooocus Anime/Realistic Edition.

### Linux (AMD GPUs)

Note that the [minimal requirement](#minimal-requirement) for different platforms is different.

Same with the above instructions. You need to change torch to the AMD version

    pip uninstall torch torchvision torchaudio torchtext functorch xformers 
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6

AMD is not intensively tested, however. The AMD support is in beta.

Use `python entry_with_update.py --preset anime` or `python entry_with_update.py --preset realistic` for Fooocus Anime/Realistic Edition.

### Windows (AMD GPUs)

Note that the [minimal requirement](#minimal-requirement) for different platforms is different.

Same with Windows. Download the software and edit the content of `run.bat` as:

    .\python_embeded\python.exe -m pip uninstall torch torchvision torchaudio torchtext functorch xformers -y
    .\python_embeded\python.exe -m pip install torch-directml
    .\python_embeded\python.exe -s Fooocus\entry_with_update.py --directml
    pause

Then run the `run.bat`.

AMD is not intensively tested, however. The AMD support is in beta.

For AMD, use `.\python_embeded\python.exe entry_with_update.py --directml --preset anime` or `.\python_embeded\python.exe entry_with_update.py --directml --preset realistic` for Fooocus Anime/Realistic Edition.

### Mac

Note that the [minimal requirement](#minimal-requirement) for different platforms is different.

Mac is not intensively tested. Below is an unofficial guideline for using Mac. You can discuss problems [here](https://github.com/lllyasviel/Fooocus/pull/129).

You can install Fooocus on Apple Mac silicon (M1 or M2) with macOS 'Catalina' or a newer version. Fooocus runs on Apple silicon computers via [PyTorch](https://pytorch.org/get-started/locally/) MPS device acceleration. Mac Silicon computers don't come with a dedicated graphics card, resulting in significantly longer image processing times compared to computers with dedicated graphics cards.

1. Install the conda package manager and pytorch nightly. Read the [Accelerated PyTorch training on Mac](https://developer.apple.com/metal/pytorch/) Apple Developer guide for instructions. Make sure pytorch recognizes your MPS device.
1. Open the macOS Terminal app and clone this repository with `git clone https://github.com/lllyasviel/Fooocus.git`.
1. Change to the new Fooocus directory, `cd Fooocus`.
1. Create a new conda environment, `conda env create -f environment.yaml`.
1. Activate your new conda environment, `conda activate fooocus`.
1. Install the packages required by Fooocus, `pip install -r requirements_versions.txt`.
1. Launch Fooocus by running `python entry_with_update.py`. (Some Mac M2 users may need `python entry_with_update.py --disable-offload-from-vram` to speed up model loading/unloading.) The first time you run Fooocus, it will automatically download the Stable Diffusion SDXL models and will take a significant amount of time, depending on your internet connection.

Use `python entry_with_update.py --preset anime` or `python entry_with_update.py --preset realistic` for Fooocus Anime/Realistic Edition.

### Docker

See [docker.md](docker.md)

### Download Previous Version

See the guidelines [here](https://github.com/lllyasviel/Fooocus/discussions/1405).

## Minimal Requirement

Below is the minimal requirement for running Fooocus locally. If your device capability is lower than this spec, you may not be able to use Fooocus locally. (Please let us know, in any case, if your device capability is lower but Fooocus still works.)

| Operating System  | GPU                          | Minimal GPU Memory           | Minimal System Memory     | [System Swap](troubleshoot.md) | Note                                                                       |
|-------------------|------------------------------|------------------------------|---------------------------|--------------------------------|----------------------------------------------------------------------------|
| Windows/Linux     | Nvidia RTX 4XXX              | 4GB                          | 8GB                       | Required                       | fastest                                                                    |
| Windows/Linux     | Nvidia RTX 3XXX              | 4GB                          | 8GB                       | Required                       | usually faster than RTX 2XXX                                               |
| Windows/Linux     | Nvidia RTX 2XXX              | 4GB                          | 8GB                       | Required                       | usually faster than GTX 1XXX                                               |
| Windows/Linux     | Nvidia GTX 1XXX              | 8GB (&ast; 6GB uncertain)    | 8GB                       | Required                       | only marginally faster than CPU                                            |
| Windows/Linux     | Nvidia GTX 9XX               | 8GB                          | 8GB                       | Required                       | faster or slower than CPU                                                  |
| Windows/Linux     | Nvidia GTX < 9XX             | Not supported                | /                         | /                              | /                                                                          |
| Windows           | AMD GPU                      | 8GB    (updated 2023 Dec 30) | 8GB                       | Required                       | via DirectML (&ast; ROCm is on hold), about 3x slower than Nvidia RTX 3XXX |
| Linux             | AMD GPU                      | 8GB                          | 8GB                       | Required                       | via ROCm, about 1.5x slower than Nvidia RTX 3XXX                           |
| Mac               | M1/M2 MPS                    | Shared                       | Shared                    | Shared                         | about 9x slower than Nvidia RTX 3XXX                                       |
| Windows/Linux/Mac | only use CPU                 | 0GB                          | 32GB                      | Required                       | about 17x slower than Nvidia RTX 3XXX                                      |

&ast; AMD GPU ROCm (on hold): The AMD is still working on supporting ROCm on Windows.

&ast; Nvidia GTX 1XXX 6GB uncertain: Some people report 6GB success on GTX 10XX, but some other people report failure cases.

*Note that Fooocus is only for extremely high quality image generating. We will not support smaller models to reduce the requirement and sacrifice result quality.*

## Troubleshoot

See the common problems [here](troubleshoot.md).

## Default Models
<a name="models"></a>

Given different goals, the default models and configs of Fooocus are different:

| Task      | Windows | Linux args | Main Model                  | Refiner | Config                                                                         |
|-----------| --- | --- |-----------------------------| --- |--------------------------------------------------------------------------------|
| General   | run.bat |  | juggernautXL_v8Rundiffusion | not used | [here](https://github.com/lllyasviel/Fooocus/blob/main/presets/default.json)   |
| Realistic | run_realistic.bat | --preset realistic | realisticStockPhoto_v20     | not used | [here](https://github.com/lllyasviel/Fooocus/blob/main/presets/realistic.json) |
| Anime     | run_anime.bat | --preset anime | animaPencilXL_v500          | not used | [here](https://github.com/lllyasviel/Fooocus/blob/main/presets/anime.json)     |

Note that the download is **automatic** - you do not need to do anything if the internet connection is okay. However, you can download them manually if you (or move them from somewhere else) have your own preparation.

## UI Access and Authentication
In addition to running on localhost, Fooocus can also expose its UI in two ways: 
* Local UI listener: use `--listen` (specify port e.g. with `--port 8888`). 
* API access: use `--share` (registers an endpoint at `.gradio.live`).

In both ways the access is unauthenticated by default. You can add basic authentication by creating a file called `auth.json` in the main directory, which contains a list of JSON objects with the keys `user` and `pass` (see example in [auth-example.json](./auth-example.json)).

## List of "Hidden" Tricks
<a name="tech_list"></a>

<details>
<summary>Click to see a list of tricks. Those are based on SDXL and are not very up-to-date with latest models.</summary>

1. GPT2-based [prompt expansion as a dynamic style "Fooocus V2".](https://github.com/lllyasviel/Fooocus/discussions/117#raw) (similar to Midjourney's hidden pre-processing and "raw" mode, or the LeonardoAI's Prompt Magic).
2. Native refiner swap inside one single k-sampler. The advantage is that the refiner model can now reuse the base model's momentum (or ODE's history parameters) collected from k-sampling to achieve more coherent sampling. In Automatic1111's high-res fix and ComfyUI's node system, the base model and refiner use two independent k-samplers, which means the momentum is largely wasted, and the sampling continuity is broken. Fooocus uses its own advanced k-diffusion sampling that ensures seamless, native, and continuous swap in a refiner setup. (Update Aug 13: Actually, I discussed this with Automatic1111 several days ago, and it seems that the “native refiner swap inside one single k-sampler” is [merged]( https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/12371) into the dev branch of webui. Great!)
3. Negative ADM guidance. Because the highest resolution level of XL Base does not have cross attentions, the positive and negative signals for XL's highest resolution level cannot receive enough contrasts during the CFG sampling, causing the results to look a bit plastic or overly smooth in certain cases. Fortunately, since the XL's highest resolution level is still conditioned on image aspect ratios (ADM), we can modify the adm on the positive/negative side to compensate for the lack of CFG contrast in the highest resolution level. (Update Aug 16, the IOS App [Draw Things](https://apps.apple.com/us/app/draw-things-ai-generation/id6444050820) will support Negative ADM Guidance. Great!)
4. We implemented a carefully tuned variation of Section 5.1 of ["Improving Sample Quality of Diffusion Models Using Self-Attention Guidance"](https://arxiv.org/pdf/2210.00939.pdf). The weight is set to very low, but this is Fooocus's final guarantee to make sure that the XL will never yield an overly smooth or plastic appearance (examples [here](https://github.com/lllyasviel/Fooocus/discussions/117#sharpness)). This can almost eliminate all cases for which XL still occasionally produces overly smooth results, even with negative ADM guidance. (Update 2023 Aug 18, the Gaussian kernel of SAG is changed to an anisotropic kernel for better structure preservation and fewer artifacts.)
5. We modified the style templates a bit and added the "cinematic-default".
6. We tested the "sd_xl_offset_example-lora_1.0.safetensors" and it seems that when the lora weight is below 0.5, the results are always better than XL without lora.
7. The parameters of samplers are carefully tuned.
8. Because XL uses positional encoding for generation resolution, images generated by several fixed resolutions look a bit better than those from arbitrary resolutions (because the positional encoding is not very good at handling int numbers that are unseen during training). This suggests that the resolutions in UI may be hard coded for best results.
9. Separated prompts for two different text encoders seem unnecessary. Separated prompts for the base model and refiner may work, but the effects are random, and we refrain from implementing this.
10. The DPM family seems well-suited for XL since XL sometimes generates overly smooth texture, but the DPM family sometimes generates overly dense detail in texture. Their joint effect looks neutral and appealing to human perception.
11. A carefully designed system for balancing multiple styles as well as prompt expansion.
12. Using automatic1111's method to normalize prompt emphasizing. This significantly improves results when users directly copy prompts from civitai.
13. The joint swap system of the refiner now also supports img2img and upscale in a seamless way.
14. CFG Scale and TSNR correction (tuned for SDXL) when CFG is bigger than 10.
</details>

## Customization

After the first time you run Fooocus, a config file will be generated at `Fooocus\config.txt`. This file can be edited to change the model path or default parameters.

For example, an edited `Fooocus\config.txt` (this file will be generated after the first launch) may look like this:

```json
{
    "path_checkpoints": "D:\\Fooocus\\models\\checkpoints",
    "path_loras": "D:\\Fooocus\\models\\loras",
    "path_embeddings": "D:\\Fooocus\\models\\embeddings",
    "path_vae_approx": "D:\\Fooocus\\models\\vae_approx",
    "path_upscale_models": "D:\\Fooocus\\models\\upscale_models",
    "path_inpaint": "D:\\Fooocus\\models\\inpaint",
    "path_controlnet": "D:\\Fooocus\\models\\controlnet",
    "path_clip_vision": "D:\\Fooocus\\models\\clip_vision",
    "path_fooocus_expansion": "D:\\Fooocus\\models\\prompt_expansion\\fooocus_expansion",
    "path_outputs": "D:\\Fooocus\\outputs",
    "default_model": "realisticStockPhoto_v10.safetensors",
    "default_refiner": "",
    "default_loras": [["lora_filename_1.safetensors", 0.5], ["lora_filename_2.safetensors", 0.5]],
    "default_cfg_scale": 3.0,
    "default_sampler": "dpmpp_2m",
    "default_scheduler": "karras",
    "default_negative_prompt": "low quality",
    "default_positive_prompt": "",
    "default_styles": [
        "Fooocus V2",
        "Fooocus Photograph",
        "Fooocus Negative"
    ]
}
```

Many other keys, formats, and examples are in `Fooocus\config_modification_tutorial.txt` (this file will be generated after the first launch).

Consider twice before you really change the config. If you find yourself breaking things, just delete `Fooocus\config.txt`. Fooocus will go back to default.

A safer way is just to try "run_anime.bat" or "run_realistic.bat" - they should already be good enough for different tasks.

~Note that `user_path_config.txt` is deprecated and will be removed soon.~ (Edit: it is already removed.)

### All CMD Flags

```
entry_with_update.py  [-h] [--listen [IP]] [--port PORT]
                      [--disable-header-check [ORIGIN]]
                      [--web-upload-size WEB_UPLOAD_SIZE]
                      [--hf-mirror HF_MIRROR]
                      [--external-working-path PATH [PATH ...]]
                      [--output-path OUTPUT_PATH]
                      [--temp-path TEMP_PATH] [--cache-path CACHE_PATH]
                      [--in-browser] [--disable-in-browser]
                      [--gpu-device-id DEVICE_ID]
                      [--async-cuda-allocation | --disable-async-cuda-allocation]
                      [--disable-attention-upcast]
                      [--all-in-fp32 | --all-in-fp16]
                      [--unet-in-bf16 | --unet-in-fp16 | --unet-in-fp8-e4m3fn | --unet-in-fp8-e5m2]
                      [--vae-in-fp16 | --vae-in-fp32 | --vae-in-bf16]
                      [--vae-in-cpu]
                      [--clip-in-fp8-e4m3fn | --clip-in-fp8-e5m2 | --clip-in-fp16 | --clip-in-fp32]
                      [--directml [DIRECTML_DEVICE]]
                      [--disable-ipex-hijack]
                      [--preview-option [none,auto,fast,taesd]]
                      [--attention-split | --attention-quad | --attention-pytorch]
                      [--disable-xformers]
                      [--always-gpu | --always-high-vram | --always-normal-vram | --always-low-vram | --always-no-vram | --always-cpu [CPU_NUM_THREADS]]
                      [--always-offload-from-vram]
                      [--pytorch-deterministic] [--disable-server-log]
                      [--debug-mode] [--is-windows-embedded-python]
                      [--disable-server-info] [--multi-user] [--share]
                      [--preset PRESET] [--disable-preset-selection]
                      [--language LANGUAGE]
                      [--disable-offload-from-vram] [--theme THEME]
                      [--disable-image-log] [--disable-analytics]
                      [--disable-metadata] [--disable-preset-download]
                      [--disable-enhance-output-sorting]
                      [--enable-auto-describe-image]
                      [--always-download-new-model]
                      [--rebuild-hash-cache [CPU_NUM_THREADS]]
```

## Inline Prompt Features

### Wildcards

Example prompt: `__color__ flower`

Processed for positive and negative prompt.

Selects a random wildcard from a predefined list of options, in this case the `wildcards/color.txt` file. 
The wildcard will be replaced with a random color (randomness based on seed). 
You can also disable randomness and process a wildcard file from top to bottom by enabling the checkbox `Read wildcards in order` in Developer Debug Mode.

Wildcards can be nested and combined, and multiple wildcards can be used in the same prompt (example see `wildcards/color_flower.txt`).

### Array Processing

Example prompt: `[[red, green, blue]] flower`

Processed only for positive prompt.

Processes the array from left to right, generating a separate image for each element in the array. In this case 3 images would be generated, one for each color.
Increase the image number to 3 to generate all 3 variants.

Arrays can not be nested, but multiple arrays can be used in the same prompt.
Does support inline LoRAs as array elements!

### Inline LoRAs

Example prompt: `flower <lora:sunflowers:1.2>`

Processed only for positive prompt.

Applies a LoRA to the prompt. The LoRA file must be located in the `models/loras` directory.

## Advanced Features

[Click here to browse the advanced features.](https://github.com/lllyasviel/Fooocus/discussions/117)

## Forks

Below are some Forks to Fooocus:

| Fooocus' forks |
| - |
| [fenneishi/Fooocus-Control](https://github.com/fenneishi/Fooocus-Control) </br>[runew0lf/RuinedFooocus](https://github.com/runew0lf/RuinedFooocus) </br> [MoonRide303/Fooocus-MRE](https://github.com/MoonRide303/Fooocus-MRE) </br> [metercai/SimpleSDXL](https://github.com/metercai/SimpleSDXL) </br> [mashb1t/Fooocus](https://github.com/mashb1t/Fooocus) </br> and so on ... |

## Thanks

Many thanks to [twri](https://github.com/twri) and [3Diva](https://github.com/3Diva) and [Marc K3nt3L](https://github.com/K3nt3L) for creating additional SDXL styles available in Fooocus. 

The project starts from a mixture of [Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) and [ComfyUI](https://github.com/comfyanonymous/ComfyUI) codebases.

Also, thanks [daswer123](https://github.com/daswer123) for contributing the Canvas Zoom!

## Update Log

The log is [here](update_log.md).

## Localization/Translation/I18N

You can put json files in the `language` folder to translate the user interface.

For example, below is the content of `Fooocus/language/example.json`:

```json
{
  "Generate": "生成",
  "Input Image": "入力画像",
  "Advanced": "고급",
  "SAI 3D Model": "SAI 3D Modèle"
}
```

If you add `--language example` arg, Fooocus will read `Fooocus/language/example.json` to translate the UI.

For example, you can edit the ending line of Windows `run.bat` as

    .\python_embeded\python.exe -s Fooocus\entry_with_update.py --language example

Or `run_anime.bat` as

    .\python_embeded\python.exe -s Fooocus\entry_with_update.py --language example --preset anime

Or `run_realistic.bat` as

    .\python_embeded\python.exe -s Fooocus\entry_with_update.py --language example --preset realistic

For practical translation, you may create your own file like `Fooocus/language/jp.json` or `Fooocus/language/cn.json` and then use flag `--language jp` or `--language cn`. Apparently, these files do not exist now. **We need your help to create these files!**

Note that if no `--language` is given and at the same time `Fooocus/language/default.json` exists, Fooocus will always load `Fooocus/language/default.json` for translation. By default, the file `Fooocus/language/default.json` does not exist.
