#!/bin/bash
sudo apt update -y && sudo apt full-upgrade -y
sudo apt install "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"
sudo usermod -a -G render,video $LOGNAME
wget https://repo.radeon.com/amdgpu-install/6.0.2/ubuntu/jammy/amdgpu-install_6.0.60002-1_all.deb
# to Ubuntu 20.04: wget https://repo.radeon.com/amdgpu-install/6.0.2/ubuntu/focal/amdgpu-install_6.0.60002-1_all.deb
sudo apt install ./amdgpu-install_6.0.60002-1_all.deb
sudo apt update
sudo apt install amdgpu-dkms -y
sudo apt install rocm -y
sudo reboot now
