Below are many common problems that people encountered:

### RuntimeError: CPUAllocator

See also the section: **System Swap**

### Model loaded, then paused, then nothing happens

See also the section: **System Swap**

### Segmentation Fault

See also the section: **System Swap**

### Killed

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

![swap](https://private-user-images.githubusercontent.com/19834515/260322660-2a06b130-fe9b-4504-94f1-2763be4476e9.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTEiLCJleHAiOjE3MDIyNTE5NzMsIm5iZiI6MTcwMjI1MTY3MywicGF0aCI6Ii8xOTgzNDUxNS8yNjAzMjI2NjAtMmEwNmIxMzAtZmU5Yi00NTA0LTk0ZjEtMjc2M2JlNDQ3NmU5LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFJV05KWUFYNENTVkVINTNBJTJGMjAyMzEyMTAlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjMxMjEwVDIzNDExM1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTc2NTg3MWFmMmI2MTcwNzc2NDc2NjkyZTRjN2Q2N2Q3MTBkZDEyNmQ3MTY3ZjU4NGVjMzQ5MDQ0ZjAyMzEzMjcmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.z9ztLn3l-Tmy98pLTgKwvGayhJchT-roB4dgMrjHvro)

If you use both HDD and SSD, you *may* test some settings on the above step 7 to try best to put swap area on SSD, so that the speed of model loading will be faster.

**Important: Microsoft Windows 10/11 by default automate system swap for you so that you do not need to touch this dangerous setting. If you do not have enough system swap, just make sure that you have at least 40GB free space on each disk.** The Microsoft Windows 10/11 will automatically make swap areas for you.

Also, if you obtain Microsoft Windows 10/11 from some unofficial Chinese or Russian provider, they may have modified the default setting of system swap to advertise some "Enhanced Windows 10/11" (but actually they are just making things worse rather than improve things). In those cases, you may need to manually check if your system swap setting is consistent to the above screenshot.

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

This problem is fixed two months ago. Please make sure that you are using the latest version of Fooocus (try fresh install).

If it still does not work, try to upgrade your Nvidia driver.

If it still does not work, open an issue with full log, and we will take a look.


### Found no NVIDIA driver on your system

Please upgrade your Nvidia Driver. 

If you are using AMD, please follow official installation guide.

### NVIDIA driver too old

Please upgrade your Nvidia Driver.

### I am using Mac, the speed is very slow.

Some MAC users may need `--enable-smart-memory` to speed up model loading.

Besides, the current support for MAC is very experimental, and we encourage users to also try Diffusionbee or Drawingthings: they are developed only for MAC.

### I am using Nvidia with 8GB VRAM, I get CUDA Out Of Memory

It is a BUG. Please let us know as soon as possible. Please make an issue.

### I am using Nvidia with 6GB VRAM, I get CUDA Out Of Memory

It is a BUG. Please let us know as soon as possible. Please make an issue.

### I am using Nvidia with 4GB VRAM with Float16 support, like RTX 3050, I get CUDA Out Of Memory

It is a BUG. Please let us know as soon as possible. Please make an issue.

### I am using Nvidia with 4GB VRAM without Float16 support, like GTX 960, I get CUDA Out Of Memory

Supporting GPU with 4GB VRAM without fp16 is extremely difficult, and you may not be able to use SDXL. However, you may still make an issue and let us know. You may try SD1.5 in Automatic1111 or other software for your device.

### I am using AMD GPU on Windows, I get CUDA Out Of Memory

Current AMD support is very experimental for Windows. If you see this, then perhaps you cannot use Fooocus on this device on Windows.

However, if you re able to run SDXL on this same device on any other software, please let us know immediately, and we will support it as soon as possible. If no other software can enable your device to run SDXL on Windows, then we also do not have much to help.

Besides, the AMD support on Linux is slightly better because it will use ROCM. You may also try it if you are willing to change OS to linux.

### I am using AMD GPU on Linux, I get CUDA Out Of Memory

Current AMD support for Linux is better than that for Windows, but still, very experimental. However, if you re able to run SDXL on this same device on any other software, please let us know immediately, and we will support it as soon as possible. If no other software can enable your device to run SDXL on Windows, then we also do not have much to help.

### I tried flags like --lowvram or --gpu-only or --bf16 or so on, and things are not getting any better?

Please remove these flags if you are mislead by some wrong tutorials. In most cases these flags are making things worse and introducing more problems.
