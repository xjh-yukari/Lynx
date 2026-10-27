"""仅连接已验证的公网地址；DNS 解析结果直接交连接器，避免二次解析绕过。"""

import asyncio
import ipaddress
import re
import socket

from aiohttp.abc import AbstractResolver
from yarl import URL

from lynx_agent.web.errors import UnsafeURLError


def check_ip(value: str) -> None:
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        raise UnsafeURLError("解析结果不是有效 IP，访问已拒绝") from None
    if isinstance(address, ipaddress.IPv6Address):
        if address.ipv4_mapped:
            address = address.ipv4_mapped
        elif address.sixtofour or address.teredo:
            raise UnsafeURLError("不允许访问 IPv6 转换隧道地址")
    if not address.is_global or address.is_multicast or address.is_reserved or address.is_unspecified:
        raise UnsafeURLError("禁止访问私网、本机、保留地址或元数据服务；不要换工具绕过")


def public_url(raw: str) -> URL:
    if len(raw) > 4096 or any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in raw) or "\\" in raw:
        raise UnsafeURLError("URL 包含非法字符或长度超限")
    try:
        url = URL(raw)
        host = (url.raw_host or "").lower().rstrip(".")
        if url.scheme not in {"http", "https"} or not host:
            raise UnsafeURLError("只允许 HTTP/HTTPS 公网地址")
        if url.user is not None or url.password is not None:
            raise UnsafeURLError("不允许包含账号密码的 URL")
        if url.port not in {80, 443}:
            raise UnsafeURLError("第一版仅允许公网 80/443 端口")
        if "%" in host or host == "localhost" or host.endswith((".localhost", ".local", ".internal", ".lan", ".home")):
            raise UnsafeURLError("不允许访问本地或内部域名")
        try:
            ipaddress.ip_address(host)
        except ValueError:
            if "." not in host or re.fullmatch(r"(?:0[xX][0-9a-fA-F]+|[0-9]+)(?:\.(?:0[xX][0-9a-fA-F]+|[0-9]+))*", host):
                raise UnsafeURLError("不允许内部名称或非标准数字 IP 表达") from None
        else:
            check_ip(host)
        return url.with_host(host).with_fragment(None)
    except (ValueError, UnicodeError):
        raise UnsafeURLError("URL 格式不合法，访问已拒绝") from None


class PublicResolver(AbstractResolver):
    async def resolve(self, host, port=0, family=socket.AF_INET):
        try:
            infos = await asyncio.get_running_loop().getaddrinfo(host, port, family=family,
                                                                type=socket.SOCK_STREAM, proto=socket.IPPROTO_TCP)
        except OSError:
            raise UnsafeURLError("域名解析失败，未建立连接") from None
        if not infos:
            raise UnsafeURLError("域名没有可用公网地址")
        results = []
        for address_family, _, proto, _, sockaddr in infos:
            check_ip(sockaddr[0])
            results.append({"hostname": host, "host": sockaddr[0], "port": port,
                            "family": address_family, "proto": proto, "flags": socket.AI_NUMERICHOST})
        return results

    async def close(self):
        pass
