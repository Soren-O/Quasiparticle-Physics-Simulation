"""Shared host policy for the intentionally local qpsim web UI."""

from __future__ import annotations

import ipaddress


def is_loopback_host(host: str) -> bool:
    """Return whether ``host`` is localhost or a loopback IP literal."""
    if host.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False
