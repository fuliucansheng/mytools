---
shortTitle: Ubuntu常用脚本
---

# Ubuntu常用脚本

## Shell配置

#### 基础配置

```bash
sh -c "$(curl -fsSL https://raw.githubusercontent.com/fuliucansheng/mytools/master/shares/dotfiles/install_ubuntu.sh)"
```

#### Make GPU Active

```bash
wget https://raw.githubusercontent.com/fuliucansheng/mytools/master/shares/dotfiles/gpu/gpu.py
```

```bash
python gpu.py
```

## Python3.8环境配置


#### 基础配置

```bash
sh -c "$(curl -fsSL https://raw.githubusercontent.com/fuliucansheng/mytools/master/shares/dotfiles/py38/install.sh)"
```

#### ROCM5.2环境配置

```bash
sh -c "$(curl -fsSL https://raw.githubusercontent.com/fuliucansheng/mytools/master/shares/dotfiles/py38/install_rocm52.sh)"
```
