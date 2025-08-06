#!/bin/bash -
#===============================================================================
#
#          FILE: install_base.sh
#
#         USAGE: ./install_base.sh
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

set -o nounset  # Treat unset variables as an error
set -o errexit  # Exit on any error

# 检测平台：macOS 使用 brew，Linux 使用 apt
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS"
    command -v brew >/dev/null 2>&1 || {
        echo >&2 "Homebrew not installed. Please install it first: https://brew.sh/"
        exit 1
    }

    echo "Updating brew..."
    brew update

    echo "Installing packages with brew..."
    brew install -y htop vim tmux zsh wget
else
    echo "Detected Linux"
    if ! command -v apt >/dev/null 2>&1; then
        echo >&2 "apt not found. This script only supports apt-based Linux distros."
        exit 1
    fi

    echo "Updating apt..."
    sudo apt update

    echo "Installing packages with apt..."
    sudo apt install -y htop vim tmux zsh wget curl git
fi

# 安装 fzf
if [ ! -d "$HOME/.fzf" ]; then
    echo "Installing fzf..."
    git clone --depth 1 https://github.com/junegunn/fzf.git ~/.fzf
    ~/.fzf/install --all
else
    echo "fzf already installed."
fi

# 设置 tmux 配置
if [ ! -d "$HOME/.tmux" ]; then
    echo "Cloning tmux config..."
    git clone https://github.com/gpakosz/.tmux.git ~/.tmux
    ln -s -f ~/.tmux/.tmux.conf ~/.tmux.conf
    cp ~/.tmux/.tmux.conf.local ~/.tmux.conf.local
else
    echo "tmux config already exists."
fi

# 安装 vim 配置
echo "Installing .vimrc..."
wget -q https://raw.githubusercontent.com/fuliucansheng/mytools/master/shares/shells/vim/vimrc -O ~/.vimrc

# 安装 oh-my-zsh
if [ ! -d "$HOME/.oh-my-zsh" ]; then
    echo "Installing oh-my-zsh..."
    sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
else
    echo "oh-my-zsh already installed."
fi
