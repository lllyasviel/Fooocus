#!/bin/bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
sudo apt install python3 -y
sudo apt-get install python-is-python3 -y
sudo apt install python3.11-venv -y
sudo apt-get install git -y
sudo apt-get install wget -y
