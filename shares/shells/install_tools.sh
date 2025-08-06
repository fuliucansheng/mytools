#!/bin/bash -
#===============================================================================
#
#          FILE: install_tools.sh
#
#         USAGE: ./install_tools.sh
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

pip3 install jupyterlab gpustat
mkdir -p ~/.jupyter
wget https://raw.githubusercontent.com/fuliucansheng/mytools/master/shares/shells/jupyter/config.json -O ~/.jupyter/jupyter_server_config.json
