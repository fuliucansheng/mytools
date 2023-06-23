---
shortTitle: Ubuntu常用脚本
icon: feather-pointed
---

# Ubuntu 常用脚本

本文档提供了一些在 Ubuntu 上常用的脚本，用于配置和设置环境。

## Shell 配置

#### 基础配置

可以运行以下命令来进行基础的 Shell 配置：

```bash
sh -c "$(curl -fsSL https://raw.githubusercontent.com/fuliucansheng/mytools/master/shares/dotfiles/install_ubuntu.sh)"
```

这个命令将下载一个脚本并执行，用于配置 Shell 的一些常用设置。

#### Make GPU Active

如果你需要使 GPU 不休眠，可以使用以下命令：

```bash
wget https://raw.githubusercontent.com/fuliucansheng/mytools/master/shares/dotfiles/gpu/gpu.py
```

```bash
python gpu.py
```

这些命令将下载一个 Python 脚本并运行，使 GPU 不休眠。

## Python3.8 环境

#### 基础配置

如果你想配置 Python3.8 环境，可以运行以下命令：

```bash
sh -c "$(curl -fsSL https://raw.githubusercontent.com/fuliucansheng/mytools/master/shares/dotfiles/py38/install.sh)"
```

这个命令将下载一个脚本并执行，用于配置 Python3.8 环境的一些基本设置。

#### ROCM5.2 配置

如果你需要在 ROCM5.2 环境里配置 Python3.8 环境，可以运行以下命令：

```bash
sh -c "$(curl -fsSL https://raw.githubusercontent.com/fuliucansheng/mytools/master/shares/dotfiles/py38/install_rocm52.sh)"
```

这个命令将下载一个脚本并执行，用于在 ROCM5.2 环境配置 Python3.8 的一些设置。
