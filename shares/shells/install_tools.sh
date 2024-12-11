#!/bin/bash -
#===============================================================================
#
#          FILE: install_ubuntu.sh
#
#         USAGE: ./install_ubuntu.sh
#
#   DESCRIPTION:
#
#       OPTIONS: ---
#  REQUIREMENTS: ---
#          BUGS: ---
#         NOTES: ---
#        AUTHOR: fuliucansheng
#  ORGANIZATION:
#       CREATED: 04/30/2022 22:39
#      REVISION:  ---
#===============================================================================

set -o nounset                              # Treat unset variables as an error

sudo apt update
sudo apt install -y htop vim tmux zsh

pip3 install jupyterlab gpustat
mkdir -p ~/.jupyter
wget https://raw.githubusercontent.com/fuliucansheng/mytools/master/shares/shells/jupyter/config.json -O ~/.jupyter/jupyter_server_config.json
