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

# fzf
git clone --depth 1 https://github.com/junegunn/fzf.git ~/.fzf
~/.fzf/install

# tmux conf
cd
git clone https://github.com/gpakosz/.tmux.git
ln -s -f .tmux/.tmux.conf
cp .tmux/.tmux.conf.local .

# vimrc
wget https://raw.githubusercontent.com/fuliucansheng/mytools/master/shares/dotfiles/vim/vimrc -q -O ~/.vimrc

# oh my zsh
sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"


