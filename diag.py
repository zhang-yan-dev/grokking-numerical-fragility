"""诊断：区分是「通用证书问题」还是「GitHub 被中间人拦截」"""
import socket
import ssl
import sys

import certifi
import requests

print("Python:", sys.version.split()[0])
print("certifi 证书包:", certifi.where())
print()

targets = [
    "https://www.baidu.com",
    "https://pypi.tuna.tsinghua.edu.cn",
    "https://github.com",
    "https://api.github.com",
]

print("=== 各站点 HTTPS 连通性 ===")
for url in targets:
    try:
        r = requests.get(url, timeout=15)
        print(f"  [通] {url}  ->  {r.status_code}")
    except Exception as e:
        print(f"  [断] {url}  ->  {type(e).__name__}: {str(e)[:100]}")
print()


def peek(host, verify):
    """抓取对方实际出示的证书签发者"""
    ctx = (
        ssl.create_default_context(cafile=certifi.where())
        if verify
        else ssl._create_unverified_context()
    )
    with socket.create_connection((host, 443), timeout=15) as sock:
        with ctx.wrap_socket(sock, server_hostname=host) as ss:
            return ss.getpeercert()


print("=== github.com 实际出示的证书 ===")
for label, verify in (("验证模式", True), ("不验证模式", False)):
    try:
        cert = peek("github.com", verify)
        issuer = dict(x[0] for x in cert.get("issuer", []))
        subject = dict(x[0] for x in cert.get("subject", []))
        print(f"  {label} 签发者: {issuer.get('organizationName')} / CN={issuer.get('commonName')}")
        print(f"  {label} 主体  : CN={subject.get('commonName')}")
    except Exception as e:
        print(f"  {label} 失败: {type(e).__name__}: {str(e)[:120]}")
