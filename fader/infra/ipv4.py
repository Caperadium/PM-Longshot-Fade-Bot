"""infra/ipv4.py

Force all outbound HTTP in this process onto IPv4.

The Polymarket CLOB/Gamma hosts are Cloudflare-fronted and now serve AAAA
(IPv6) records. On hosts whose IPv6 egress to Cloudflare is unroutable
(e.g. some OVH VPS), getaddrinfo returns the IPv6 address first and every
connection attempt stalls on the dead route until it times out before
falling back to IPv4 -- this breaks the WS opening handshake outright and
slows every REST call.

force_ipv4() patches urllib3's address-family selector so requests/urllib3
(and py-clob-client, which uses requests under the hood) only ever resolve
to IPv4. The WS client forces IPv4 separately via
websockets.connect(family=AF_INET).

Process-scoped only -- no system/global config is touched.
"""

from __future__ import annotations

import logging
import socket

logger = logging.getLogger(__name__)


def force_ipv4() -> None:
    """Make urllib3 resolve only IPv4 addresses for the rest of this process.

    Idempotent and best-effort: if urllib3's internals ever change, log a
    warning rather than crash the caller.
    """
    try:
        import urllib3.util.connection as urllib3_conn

        urllib3_conn.allowed_gai_family = lambda: socket.AF_INET
        logger.info("Forced IPv4 for all urllib3/requests HTTP in this process")
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(
            f"force_ipv4: could not patch urllib3 ({e}); IPv6 may be attempted"
        )
