# 💬 [lllyasviel Fooocus](https://github.com/lllyasviel/Fooocus) on AMD ROCm 📷
- Make sure that your system was fresh (ubuntu 22.04)
- Computer will restart during the AMD driver install.
- Prepare AMD ROCm Driver install
```
sudo usermod -a -G render,video $LOGNAME
wget https://raw.githubusercontent.com/lllyasviel/Fooocus/main/AMD/amd_drivers.sh
sudo chmod 777 amd_drivers.sh
sudo ./amd_drivers.sh
```
- For RDNA and RDNA 2 cards RX5000 RX6000:
```
export HSA_OVERRIDE_GFX_VERSION=10.3.0
```
- For RDNA 3 cards RX7000:
```
export HSA_OVERRIDE_GFX_VERSION=11.0.0
```
- install Python
```
wget https://raw.githubusercontent.com/lllyasviel/Fooocus/main/AMD/install_python_conda.sh
sudo chmod 777 install_python_conda.sh
sudo ./install_python_conda.sh
```
- Install and Start lllyasviel/Fooocus
```
git clone --branch 2.3.1 https://github.com/lllyasviel/Fooocus.git
cd Fooocus
source fooocus_env/bin/activate
pip3 install -r requirements_versions.txt
pip3 install torch==2.1.2+rocm5.6 torchvision==0.16.2+rocm5.6 --extra-index-url https://download.pytorch.org/whl/rocm5.6
python entry_with_update.py --listen --attention-split
```
- Restart Fooocus
```
cd Fooocus
export HSA_OVERRIDE_GFX_VERSION=11.0.0
source fooocus_env/bin/activate
python entry_with_update.py --listen --attention-split
```
To see a prompt from your GPU usage.
```
watch /opt/rocm-6.0.2/bin/rocm-smi
```
