(The update of Fooocus will be paused for a period of time for AUTOMATIC1111 sd-webui 1.6.X, and some features will also be implemented as webui extensions)

### 1.0.40

* Behaviors reverted to 1.0.36 again (refiner steps). The 1.0.36 is too perfect and too typical; beating 1.0.36 is just impossible.

### 1.0.39

* Reverted unstable changes between 1.0.37 and 1.0.38 .
* Increased refiner steps to half of sampling steps.

### 1.0.36

* Change gaussian kernel to anisotropic kernel.

### 1.0.34

* Random seed restoring.

### 1.0.33

* Hide items in log when images are removed.

### 1.0.32

* Fooocus private log

### 1.0.31

* Fix typo and UI.

### 1.0.29

* Added "Advanced->Advanced->Advanced" block for future development.

### 1.0.29

* Fix overcook problem in 1.0.28

### 1.0.28

* SAG implemented

### 1.0.27

* Fix small problem in textbox css 

### 1.0.25

* support sys.argv --listen --share --port

### 1.0.24

* Taller input textbox.

### 1.0.23

* Added some hints on linux after UI start so users know the App does not fail.

### 1.0.20

* Support linux.

### 1.0.20

* Speed-up text encoder.

### 1.0.20

* Re-write UI to use async codes: (1) for faster start, and (2) for better live preview.
* Removed opencv dependency
* Plan to support Linux soon

### 1.0.19

* Unlock to allow changing model.

### 1.0.17

* Change default model to SDXL-1.0-vae-0.9. (This means the models will be downloaded again, but we should do it as early as possible so that all new users only need to download once. Really sorry for day-0 users. But frankly this is not too late considering that the project is just publicly available in less than 24 hours - if it has been a week then we will prefer more lightweight tricks to update.)

### 1.0.16

* Implemented "Fooocus/outputs" folder for saving user results.
* Ignored cv2 errors when preview fails.
* Mentioned future AMD support in Readme.
* Created this log.

### 1.0.15

Publicly available.

### 1.0.0

Initial Version.
