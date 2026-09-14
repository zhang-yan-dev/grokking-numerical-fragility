"""抓取 github.com 握手时实际出示的证书，识别中间人"""
import socket
import ssl

from cryptography import x509
from cryptography.hazmat.backends import default_backend

ctx = ssl._create_unverified_context()
with socket.create_connection(("github.com", 443), timeout=15) as sock:
    with ctx.wrap_socket(sock, server_hostname="github.com") as ss:
        der = ss.getpeercert(binary_form=True)
        print("TLS 版本:", ss.version())
        print("加密套件:", ss.cipher()[0])

cert = x509.load_der_x509_certificate(der, default_backend())


def name(n):
    parts = []
    for attr in n:
        parts.append(f"{attr.oid._name}={attr.value}")
    return " | ".join(parts)


print()
print("=== 对方出示的证书 ===")
print("主体 (Subject):", name(cert.subject))
print("签发者 (Issuer):", name(cert.issuer))
print("有效期:", cert.not_valid_before_utc, "->", cert.not_valid_after_utc)
print("序列号:", hex(cert.serial_number))

try:
    san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
    print("SAN:", san.value.get_values_for_type(x509.DNSName)[:10])
except Exception as e:
    print("SAN: 读取失败", e)

print()
print("=== 判断 ===")
issuer_cn = cert.issuer.get_attributes_for_oid(x509.NameOID.COMMON_NAME)
issuer_cn = issuer_cn[0].value if issuer_cn else "?"
real = any(k in issuer_cn for k in ("Sectigo", "DigiCert", "Let's Encrypt", "GlobalSign", "ISRG"))
if real:
    print("签发者是公认 CA，不像是中间人。问题可能在别处。")
else:
    print(f"签发者 CN = {issuer_cn}")
    print(">>> 这不是 GitHub 的正常签发者，确认存在中间人 (TLS 拦截) <<<")
