"""Security perimeter module providing dual-stack SSRF shielding, input limits, and prompt isolation."""

import html
import ipaddress
import re
import socket
from typing import List, Tuple
from urllib.parse import urljoin, urlparse
from unittest.mock import patch
import requests


class SecurityValidationError(ValueError):
    """Raised when an input, URL, or response violates security constraints."""
    pass


# Blocked IPv4 networks (Carrier-grade NAT, private, loopback, link-local, etc.)
BLOCKED_IPV4_NETWORKS = [
    ipaddress.IPv4Network("0.0.0.0/8"),
    ipaddress.IPv4Network("10.0.0.0/8"),
    ipaddress.IPv4Network("100.64.0.0/10"),
    ipaddress.IPv4Network("127.0.0.0/8"),
    ipaddress.IPv4Network("169.254.0.0/16"),
    ipaddress.IPv4Network("172.16.0.0/12"),
    ipaddress.IPv4Network("192.0.0.0/24"),
    ipaddress.IPv4Network("192.0.2.0/24"),
    ipaddress.IPv4Network("192.88.99.0/24"),
    ipaddress.IPv4Network("192.168.0.0/16"),
    ipaddress.IPv4Network("198.18.0.0/15"),
    ipaddress.IPv4Network("198.51.100.0/24"),
    ipaddress.IPv4Network("203.0.113.0/24"),
    ipaddress.IPv4Network("224.0.0.0/4"),
    ipaddress.IPv4Network("240.0.0.0/4"),
    ipaddress.IPv4Network("255.255.255.255/32"),
]

# Blocked IPv6 networks (Unique Local, link-local, loopback, etc.)
BLOCKED_IPV6_NETWORKS = [
    ipaddress.IPv6Network("::/128"),
    ipaddress.IPv6Network("::1/128"),
    ipaddress.IPv6Network("::ffff:0:0/96"),  # IPv4-mapped
    ipaddress.IPv6Network("64:ff9b::/96"),
    ipaddress.IPv6Network("100::/64"),
    ipaddress.IPv6Network("2001::/23"),
    ipaddress.IPv6Network("2001:db8::/32"),
    ipaddress.IPv6Network("2002::/16"),  # 6to4
    ipaddress.IPv6Network("fc00::/7"),   # Unique Local Address (ULA)
    ipaddress.IPv6Network("fe80::/10"),  # Link-Local
    ipaddress.IPv6Network("ff00::/8"),   # Multicast
]


def is_ip_blocked(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Check if an IPv4 or IPv6 address belongs to a non-public, loopback, or private range."""
    # Standard properties check
    if (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    ):
        return True

    # Check against explicit IPv4 networks
    if isinstance(ip, ipaddress.IPv4Address):
        for net in BLOCKED_IPV4_NETWORKS:
            if ip in net:
                return True

    # Check against explicit IPv6 networks
    if isinstance(ip, ipaddress.IPv6Address):
        # Unpack IPv4-mapped addresses
        if ip.ipv4_mapped:
            return is_ip_blocked(ip.ipv4_mapped)
        if ip.sixtofour:
            return is_ip_blocked(ip.sixtofour)
        for net in BLOCKED_IPV6_NETWORKS:
            if ip in net:
                return True

    return False


def resolve_and_validate_hostname(hostname: str, port: int) -> List[Tuple]:
    """Resolve hostname via DNS for both IPv4 and IPv6, validating that every resolved IP is public."""
    if not hostname or hostname.strip().lower() == "localhost":
        raise SecurityValidationError(f"Access to hostname '{hostname}' is blocked.")

    try:
        addr_info = socket.getaddrinfo(
            hostname,
            port,
            proto=socket.IPPROTO_TCP,
            flags=socket.AI_ADDRCONFIG,
        )
    except socket.gaierror as exc:
        raise SecurityValidationError(f"Could not resolve hostname '{hostname}': {exc}") from exc

    if not addr_info:
        raise SecurityValidationError(f"No DNS records returned for '{hostname}'.")

    validated_addrs = []
    for family, socktype, proto, canonname, sockaddr in addr_info:
        ip_str = sockaddr[0]
        try:
            ip_obj = ipaddress.ip_address(ip_str)
        except ValueError as exc:
            raise SecurityValidationError(f"Invalid resolved IP '{ip_str}' for '{hostname}'.") from exc

        if is_ip_blocked(ip_obj):
            raise SecurityValidationError(
                f"Resolved IP '{ip_str}' for host '{hostname}' is a blocked/private/loopback address."
            )
        validated_addrs.append((family, socktype, proto, canonname, sockaddr))

    return validated_addrs


def validate_safe_url(url: str) -> Tuple[str, List[Tuple]]:
    """Validate URL scheme and ensure all resolved destination IPs are strictly safe and public."""
    if not url or not isinstance(url, str):
        raise SecurityValidationError("URL must be a non-empty string.")

    parsed = urlparse(url.strip())
    if parsed.scheme.lower() not in ("http", "https"):
        raise SecurityValidationError(f"Blocked URL scheme '{parsed.scheme}'. Only http and https are permitted.")

    hostname = parsed.hostname
    if not hostname:
        raise SecurityValidationError("URL must have a valid hostname.")

    port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    validated_addrs = resolve_and_validate_hostname(hostname, port)
    return hostname, validated_addrs


def safe_http_get(
    url: str,
    timeout: int = 10,
    max_bytes: int = 1024 * 1024,
    max_redirects: int = 5,
    session: requests.Session = None,
) -> requests.Response:
    """Fetch an HTTP/HTTPS URL with strict post-DNS SSRF validation, redirect revalidation, and size limits.

    Ensures:
    1. Only http/https schemes allowed.
    2. Every resolved IP address (IPv4 and IPv6) is validated before connecting.
    3. DNS rebinding / TOCTOU gap is prevented by pinning socket resolution to the validated addresses.
    4. Full TLS certificate verification (verify=True) and SNI are maintained with no weakening.
    5. Relative and absolute redirects are re-validated through the full SSRF pipeline before following.
    6. Response body size is enforced to prevent resource exhaustion (max 1MB).
    """
    req_session = session or requests.Session()
    current_url = url
    redirect_count = 0

    original_getaddrinfo = socket.getaddrinfo

    while True:
        # Step 1: Validate URL and resolve DNS
        hostname, validated_addrs = validate_safe_url(current_url)

        # Step 2: Bind DNS resolution strictly to the validated addresses to prevent DNS rebinding
        def pinned_getaddrinfo(host, port, *args, **kwargs):
            if host == hostname:
                return validated_addrs
            return original_getaddrinfo(host, port, *args, **kwargs)

        with patch("socket.getaddrinfo", side_effect=pinned_getaddrinfo):
            # Strict HTTPS verification: verify=True is mandatory
            response = req_session.get(
                current_url,
                timeout=timeout,
                stream=True,
                allow_redirects=False,
                verify=True,
                headers={"User-Agent": "AIMitra/1.0 (Personal AI News Assistant)"},
            )

        # Step 3: Handle HTTP redirects securely
        if response.is_redirect or response.status_code in (301, 302, 303, 307, 308):
            redirect_count += 1
            if redirect_count > max_redirects:
                raise SecurityValidationError(f"Exceeded maximum redirect limit of {max_redirects}.")

            location = response.headers.get("Location")
            if not location:
                raise SecurityValidationError("Redirect response missing Location header.")

            # Safely resolve relative redirects against current URL
            current_url = urljoin(current_url, location)
            response.close()
            continue

        # Step 4: Stream response content and enforce max_bytes limit
        content_buffer = bytearray()
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                content_buffer.extend(chunk)
                if len(content_buffer) > max_bytes:
                    response.close()
                    raise SecurityValidationError(f"Response size exceeded configured limit of {max_bytes} bytes.")

        # Cache fully buffered content into response._content
        response._content = bytes(content_buffer)
        return response


# ==============================================================================
# Input Length Enforcement
# ==============================================================================

def enforce_user_message_length(text: str, max_chars: int = 4000) -> str:
    """Rejects or safely truncates input above the configured maximum length (user messages: 4,000 chars)."""
    if not text:
        return ""
    if len(text) > max_chars:
        return text[:max_chars]
    return text


def enforce_article_content_length(text: str, max_chars: int = 15000) -> str:
    """Rejects or safely truncates input above the configured maximum length (article content: 15,000 chars)."""
    if not text:
        return ""
    if len(text) > max_chars:
        return text[:max_chars]
    return text


# ==============================================================================
# Prompt Injection Defense & Untrusted Content Containment
# ==============================================================================

def wrap_untrusted_content(content: str, source_name: str) -> str:
    """Wrap untrusted external text in boundary markers with attribute escaping.

    - Escapes source_name attribute to prevent quote or attribute injection.
    - Neutralizes nested closing tags </untrusted_external_content>.
    - Establishes that external content is strictly untrusted data.
    """
    safe_source = html.escape(source_name.strip() if source_name else "unknown", quote=True)
    # Neutralize any attempts to prematurely close the boundary tag inside content
    sanitized_content = re.sub(
        r"</\s*untrusted_external_content\s*>",
        "[NESTED_TAG_NEUTRALIZED]",
        content,
        flags=re.IGNORECASE,
    )
    return (
        f'<untrusted_external_content source="{safe_source}">\n'
        f"{sanitized_content}\n"
        f"</untrusted_external_content>"
    )

