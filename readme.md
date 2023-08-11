# Fooocus

<img src="https://raw.githubusercontent.com/lllyasviel/misc_files/main/202308/fsm2.png" width=100%>

Fooocus is an image generating software.

Fooocus is a rethinking of Stable Diffusion and Midjourney’s designs:

* Learned from Stable Diffusion, the software is offline, open source, and free.

* Learned from Midjourney, the manual tweaking is not needed, and users only need to focus on the prompts and images.

Fooocus has automated [lots of inner optimizations and quality improvements](tech_list). Users can forget everything about technical parameters, and just enjoy the interaction between human and computer to "explore new mediums of thought and expanding the imaginative powers of the human species" `[1]`.

Fooocus has simplified the installation. Between pressing "download" and generating the first image, the number of needed mouse clicks is strictly limited to less than 5. Minimal GPU memory requirement is 4GB (Nvidia).

`[1]` Midjourney About, David Holz, 2020.


## Download

### Windows

**[>>> Click here to download <<<](https://github.com/lllyasviel/Fooocus/releases/download/release/Fooocus_win64_1-1-10.7z)**

After you download the file, please uncompress it, and then run the "run.bat".

In the first time you launch the software, it will automatically download models:

1. It will download [sd_xl_base_1.0.safetensors from here](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors) as the file "Fooocus\models\checkpoints\sd_xl_base_1.0.safetensors".
2. It will download [sd_xl_refiner_1.0.safetensors from here](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/resolve/main/sd_xl_refiner_1.0.safetensors) as the file "Fooocus\models\checkpoints\sd_xl_refiner_1.0.safetensors".

If you already have these files, you can copy them to the above locations to speed up installation.

### Linux and Mac

Coming soon ...

## List of Tricks Used in Improving the Result
<a name="tech_list"></a>

123

## Thanks

The codebase starts from an odd mixture of [Automatic1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui) and [ComfyUI](https://github.com/comfyanonymous/ComfyUI).
