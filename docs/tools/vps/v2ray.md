---
shortTitle: V2ray 翻墙
title: V2ray 翻墙配置
icon: feather
---

## Server 端

在服务器端使用以下步骤安装和配置v2ray。

1. 安装v2ray

```bash
wget https://raw.githubusercontent.com/v2fly/fhs-install-v2ray/master/install-release.sh
```

这个命令将下载一个脚本并执行，用于安装v2ray。

2. 准备配置文件

```bash
wget https://raw.githubusercontent.com/fuliucansheng/mytools/master/shares/vps/v2ray/server.json
```

这个命令将下载一个配置文件，用于配置v2ray的服务器端。

3. 更新配置文件中的 UUID：

使用文本编辑器打开 server.json 文件，将 `5887937f-xxxx-xxxx-xxxx-648ff02f3029` 替换为你自己的 UUID。

4. 运行代理

```bash
v2ray run ./server.json
```

这个命令将使用配置文件启动v2ray服务器端代理。

## Nginx 配置

以下是配置Nginx服务器的步骤。

1. 将PFX证书转换为CRT/KEY

```bash
openssl pkcs12 -in ./fuliucansheng.com.pfx -clcerts -nokeys -out server.crt
openssl pkcs12 -in ./fuliucansheng.com.pfx -nocerts -nodes -out server.key
```

这些命令将从PFX证书文件中提取出CRT和KEY文件。

2. 验证转换后的证书

```bash
openssl s_server -www -accept 443 -cert server.crt -key server.key
```

这个命令将使用转换后的证书启动一个临时的HTTPS服务器，以验证证书是否有效。

3. 更新Nginx配置

   - 下载nginx.conf

   ```bash
   wget https://raw.githubusercontent.com/fuliucansheng/mytools/master/shares/vps/v2ray/nginx.conf
   ```

   - 使用文本编辑器打开 nginx.conf 文件。
     - 更新 ssl_certificate 和 ssl_certificate_key 为你自己的证书路径。
     - 更新 server_name 为你自己的域名。

4. 更新Nginx配置并重启Nginx：

```bash
nginx -s reload
```

这个命令将重新加载Nginx配置并重启Nginx服务器。

## Cloudflare 代理网站

登录到 Cloudflare 并配置你自己的网站，以使用 Cloudflare 的代理功能。

## Client 端

1. 下载安装V2rayN
2. 准备配置文件

```bash
wget https://raw.githubusercontent.com/fuliucansheng/mytools/master/shares/vps/v2ray/client.json
```

这个命令将下载一个配置文件，用于配置v2ray的客户端。

3. 更新配置文件中的 address, port 和 id：使用文本编辑器打开 client.json 文件，将 address, port 和 id 替换为你自己的服务器地址、端口和UUID。
4. 导入并启动代理：将修改后的 client.json 文件导入到V2rayN中，并启动代理功能。
