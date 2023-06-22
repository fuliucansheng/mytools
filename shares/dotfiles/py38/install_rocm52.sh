#!/bin/bash - 
#===============================================================================
#
#          FILE: rocm52.py38.ds.sh
# 
#         USAGE: ./rocm52.py38.ds.sh 
# 
#   DESCRIPTION: 
# 
#       OPTIONS: ---
#  REQUIREMENTS: ---
#          BUGS: ---
#         NOTES: ---
#        AUTHOR: fuliucansheng (fuliu), fuliucansheng@gmail.com
#  ORGANIZATION: 
#       CREATED: 06/17/2023 00:14
#      REVISION:  ---
#===============================================================================

set -o nounset                              # Treat unset variables as an error

sudo apt update

sudo apt install -y python3.8-dev python3-pip
sudo apt install -y ninja-build

curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
sudo python3.8 get-pip.py

sudo rm -rf get-pip.py

sudo pip3.8 install torch==1.13.1+rocm5.2 torchvision==0.14.1+rocm5.2 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/rocm5.2

sudo pip3.8 install deepspeed ninja

