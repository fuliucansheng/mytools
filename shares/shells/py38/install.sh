#!/bin/bash - 
#===============================================================================
#
#          FILE: py38.sh
# 
#         USAGE: ./py38.sh 
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

curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
sudo python3.8 get-pip.py

sudo rm -rf get-pip.py

