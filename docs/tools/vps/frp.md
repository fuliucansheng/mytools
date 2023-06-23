---
shortTitle: FRP 内网穿透
title: FRP 内网穿透配置
icon: feather
---

&nbsp;&nbsp;&nbsp;&nbsp;内网穿透是一种网络技术，它允许将位于私有网络（内网）中的设备或服务暴露给公共网络（互联网），从而可以通过公共网络访问这些设备或服务。

## 准备软件

首先，从这里下载最新的 [FRP](https://github.com/fatedier/frp/releases)

## Server 端

在服务器端执行以下步骤来配置和运行FRP。

1. 准备配置文件

```bash
wget https://raw.githubusercontent.com/fuliucansheng/mytools/master/shares/vps/frp/server.ini
```

这个命令将下载一个配置文件 server.ini。

2. 更新监听的端口

使用文本编辑器打开 server.ini 文件，将 `8876` 替换为你自己想要监听的端口。

3. 运行命令

```bash
./frps -c ./server.ini
```

这个命令将使用配置文件 server.ini 启动FRP服务器端。

## Client 端

在客户端执行以下步骤来配置和运行FRP。

1. 准备配置文件

```bash
wget https://raw.githubusercontent.com/fuliucansheng/mytools/master/shares/vps/frp/client.ini
```

这个命令将下载一个配置文件 client.ini。

2. 更新IP和端口

使用文本编辑器打开 client.ini 文件，将 `8876` 替换为服务器端的IP和端口。在默认的配置文件中，已经包含了用于SSH代理的配置示例，你可以通过修改 `remote_port` 来更改远程服务器代理本地机器SSH的端口。

3. 运行命令

```bash
./frpc -c ./client.ini
```

这个命令将使用配置文件 client.ini 启动FRP客户端。
