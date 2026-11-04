Below are many common problems that people encountered:

---

## Quick Troubleshooting Checklist

Work through this checklist when Fooocus fails to start or behaves unexpectedly:

**1. Process & port checks**
- Windows: `Task Manager` → End any existing `python` processes named `python_embeded/python` or `launch.py`, then restart Fooocus. Do not run two Fooocus instances at the same time.
- Port in use: another instance may already be running. Check that nothing else is listening on the Fooocus port before relaunching.

**2. Console & logs**
- Do not close the black console window while Fooocus is running — closing it kills the server.
- If the console closes instantly with no error, look for a `.log` file or run `run.bat` from a terminal to see the output directly.
- Copy the last 20 lines of the console output, including the exact error text, when opening an issue.

**3. Installation & environment**
- Use **Python 3.10** (via the bundled `python_embeded`). Other versions cause dependency errors.
- In `vc` mode, delete `python_embeded` entirely before re-installing. A partial folder causes broken imports.
- DLL errors / missing MSVCRTD: reinstall the Microsoft Visual C++ Redistributable (x64) from Microsoft's site. Conda users should run `conda update --all` for the environment.

**4. OpenMP workaround (Windows startup crash)**
- If Fooocus crashes on launch with an OpenMP runtime error, set the environment variable `KMP_DUPLICATE_LIB_OK=TRUE` before running `run.bat`.

**5. Restart & run guidance**
- Restart your computer after any swap or driver change so that swap and driver settings take effect.
- Run `run.bat` from the Fooocus installation folder (not a shortcut) so that relative paths resolve correctly.
- If `run.bat` does nothing, run it from a terminal to see the error output captured by the window.

---

### RuntimeError: CPUAllocator

See also the section: **System Swap**

### Model loaded, then paused, then nothing happens

See also the section: **System Swap**

### Segmentation Fault

See also the section: **System Swap**

### Aborted

See also the section: **System Swap**

### core dumped

See also the section: **System Swap**

### Killed

See also the section: **System Swap**

### ^C, then quit

See also the section: **System Swap**

### adm 2816, then stuck

See also the section: **System Swap**

### Connection errored out

See also the section: **System Swap**

### Error 1006

See also the section: **System Swap**

### WinError 10060

See also the section: **System Swap**

### Read timed out

See also the section: **System Swap**

### No error, but the console close in a flash. Cannot find any error.

See also the section: **System Swap**

### Model loading is extremely slow (more than 1 minute)

See also the section: **System Swap**

### System Swap

All above problems are caused by the fact that you do not have enough System Swap.

Please make sure that you have at least 40GB System Swap. In fact, it does not need so much Swap, but 40Gb should be safe for you to run Fooocus in 100% success.

(If you have more than 64GB RAM, then *perhaps* you do not need any System Swap, but we are not exactly sure about this.)

Also, if your system swap is on HDD, the speed of model loading will be very slow. Please try best to put system swap on SSD.

If you are using Linux/Mac, please follow your provider's instructions to set Swap Space. Herein, the "provider" refers to Ubuntu official, CentOS official, Mac official, etc.

If you are using Windows, you can set Swap here:

![swap](https://github.com/lllyasviel/Fooocus/assets/19834515/2a06b130-fe9b-4504-94f1-2763be4476e9)

If you use both HDD and SSD, you *may* test some settings on the above step 7 to try best to put swap area on SSD, so that the speed of model loading will be faster.

**Important: Microsoft Windows 10/11 by default automate system swap for you so that you do not need to touch this dangerous setting. If you do not have enough system swap, just make sure that you have at least 40GB free space on each disk.** The Microsoft Windows 10/11 will automatically make swap areas for you.

Also, if you obtain Microsoft Windows 10/11 from some unofficial Chinese or Russian provider, they may have modified the default setting of system swap to advertise some "Enhanced Windows 10/11" (but actually they are just making things worse rather than improve things). In those cases, you may need to manually check if your system swap setting is consistent to the above screenshot.

Finally, note that you need to restart computer to activate any changes in system swap.

### MetadataIncompleteBuffer

See also the section: **Model corrupted**

### PytorchStreamReader failed

See also the section: **Model corrupted**

### Model corrupted

If you see Model Corrupted, then your model is corrupted. Fooocus will re-download corrupted models for you if your internet connection is good. Otherwise, you may also manually download models. You can find model url and their local location in the console each time a model download is requested.

### UserWarning: The operator 'aten::std_mean.correction' is not currently supported on the DML

This is a warning that you can ignore.

### Torch not compiled with CUDA enabled

You are not following the official installation guide. 

Please do not trust those wrong tutorials on the internet, and please only trust the official installation guide. 

### subprocess-exited-with-error

Please use python 3.10

Also, you are not following the official installation guide. 

Please do not trust those wrong tutorials on the internet, and please only trust the official installation guide. 

### SSL: CERTIFICATE_VERIFY_FAILED

Are you living in China? If yes, please consider turn off VPN, and/or try to download models manually.

If you get this error elsewhere in the world, then you may need to look at [this search](https://www.google.com/search?q=SSL+Certificate+Error). We cannot give very specific guide to fix this since the cause can vary a lot.

### CUDA kernel errors might be asynchronously reported at some other API call

A very small amount of devices does have this problem. The cause can be complicated but usually can be resolved after following these steps:

1. Make sure that you are using official version and latest version installed from [here](https://github.com/lllyasviel/Fooocus#download). (Some forks and other versions are more likely to cause this problem.)
2. Upgrade your Nvidia driver to the latest version. (Usually the version of your Nvidia driver should be 53X, not 3XX or 4XX.)
3. If things still do not work, then perhaps it is a problem with CUDA 12. You can use CUDA 11 and Xformers to try to solve this problem. We have prepared all files for you, and please do NOT install any CUDA or other environment on you own. The only one official way to do this is: (1) Backup and delete your `python_embeded` folder (near the `run.bat`); (2) Download the "previous_old_xformers_env.7z" from the [release page](https://github.com/lllyasviel/Fooocus/releases/tag/release), decompress it, and put the newly extracted `python_embeded` folder near your `run.bat`; (3) run Fooocus.
4. If it still does not work, please open an issue for us to take a look.

### Found no NVIDIA driver on your system

Please upgrade your Nvidia Driver. 

If you are using AMD, please follow official installation guide.

### NVIDIA driver too old

Please upgrade your Nvidia Driver.

### I am using Mac, the speed is very slow.

Some MAC users may need `--disable-offload-from-vram` to speed up model loading.

Besides, the current support for MAC is very experimental, and we encourage users to also try Diffusionbee or Drawingthings: they are developed only for MAC.

### I am using Nvidia with 8GB VRAM, I get CUDA Out Of Memory

It is a BUG. Please let us know as soon as possible. Please make an issue. See also [minimal requirements](https://github.com/lllyasviel/Fooocus/tree/main?tab=readme-ov-file#minimal-requirement).

### I am using Nvidia with 6GB VRAM, I get CUDA Out Of Memory

It is very likely a BUG. Please let us know as soon as possible. Please make an issue. See also [minimal requirements](https://github.com/lllyasviel/Fooocus/tree/main?tab=readme-ov-file#minimal-requirement).

### I am using Nvidia with 4GB VRAM with Float16 support, like RTX 3050, I get CUDA Out Of Memory

It is a BUG. Please let us know as soon as possible. Please make an issue. See also [minimal requirements](https://github.com/lllyasviel/Fooocus/tree/main?tab=readme-ov-file#minimal-requirement).

### I am using Nvidia with 4GB VRAM without Float16 support, like GTX 960, I get CUDA Out Of Memory

Supporting GPU with 4GB VRAM without fp16 is extremely difficult, and you may not be able to use SDXL. However, you may still make an issue and let us know. You may try SD1.5 in Automatic1111 or other software for your device. See also [minimal requirements](https://github.com/lllyasviel/Fooocus/tree/main?tab=readme-ov-file#minimal-requirement).

### I am using AMD GPU on Windows, I get CUDA Out Of Memory

Current AMD support is very experimental for Windows. If you see this, then perhaps you cannot use Fooocus on this device on Windows.

However, if you re able to run SDXL on this same device on any other software, please let us know immediately, and we will support it as soon as possible. If no other software can enable your device to run SDXL on Windows, then we also do not have much to help.

Besides, the AMD support on Linux is slightly better because it will use ROCM. You may also try it if you are willing to change OS to linux. See also [minimal requirements](https://github.com/lllyasviel/Fooocus/tree/main?tab=readme-ov-file#minimal-requirement).

### I am using AMD GPU on Linux, I get CUDA Out Of Memory

Current AMD support for Linux is better than that for Windows, but still, very experimental. However, if you re able to run SDXL on this same device on any other software, please let us know immediately, and we will support it as soon as possible. If no other software can enable your device to run SDXL on Windows, then we also do not have much to help. See also [minimal requirements](https://github.com/lllyasviel/Fooocus/tree/main?tab=readme-ov-file#minimal-requirement).

### I tried flags like --lowvram or --gpu-only or --bf16 or so on, and things are not getting any better?

Please remove these flags if you are mislead by some wrong tutorials. In most cases these flags are making things worse and introducing more problems.

### Fooocus suddenly becomes very slow and I have not changed anything

Are you accidentally running two Fooocus at the same time?
