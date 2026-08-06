"""Bearer tokens for the paired Machine A / Machine B translation link.

Pairing already proves who the other machine is: B shows a six-digit code, A
sends it back, and B checks it in constant time. What B did with that proof was
store the *IP address* — and from then on anything arriving from that address
was a paired client. On a DHCP network that is a lease away from being someone
else: the address moves, and a different device silently inherits a trust
nobody granted it. It also made unpairing hollow, since an unpaired machine
that kept its address could simply be re-added by the next hand-edit of config.

So pairing now hands A a secret, and A presents it on every call. Trust follows
the machine rather than the address, and revoking it deletes something.

The upgrade is designed to happen without an operator: a machine that is
already paired by IP and holds no token yet may claim one (:func:`can_claim`),
which is exactly the trust it already had. The moment it does, the IP path
closes for it — :func:`authorize` stops accepting an un-authenticated request
from an address that has a token on file. A machine still on an older build
keeps working on the IP path until it updates.

B stores only the fingerprint of each token, so its config cannot hand out
anyone's credentials; A stores the token itself, because it has to send it.

Stdlib-only and free of Flask, so the rules can be tested directly.
"""

import hashlib
import secrets
from typing import Dict, Iterable, Mapping, NamedTuple, Optional, Tuple

#: Bytes of randomness in a token. 32 bytes is far past guessable, and the
#: url-safe encoding survives being written into a JSON config by hand.
_TOKEN_BYTES = 32


def mint_token() -> str:
    """A fresh secret for one paired client."""
    return secrets.token_urlsafe(_TOKEN_BYTES)


def fingerprint(token: str) -> str:
    """What B stores for a token: its SHA-256, hex.

    Never reversible to the token, so a leaked config on B does not let anyone
    speak as a paired machine.
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def parse_bearer(header: Optional[str]) -> Optional[str]:
    """The token out of an ``Authorization: Bearer <token>`` header, or None."""
    if not header:
        return None
    parts = header.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    token = parts[1].strip()
    return token or None


class Decision(NamedTuple):
    """The outcome of checking one request from a would-be paired machine."""

    authorized: bool
    #: The address the matched token was last seen at, when it has moved. The
    #: caller updates its records so back-links and displays follow the machine.
    rebind_from: Optional[str] = None
    #: Why, for logging — never returned to the caller over HTTP.
    reason: str = ""


def authorize(
    tokens: Mapping[str, str],
    trusted_ips: Iterable[str],
    ip: Optional[str],
    presented: Optional[str],
) -> Decision:
    """Decide whether a request may act as the paired Machine A.

    ``tokens`` maps fingerprint -> the address that token was last used from.

    A presented token is the whole answer: known means yes from any address,
    unknown means no. An unknown token is never quietly downgraded to the
    address check — a machine holding a revoked secret is exactly the case this
    exists to refuse, and one whose token B has forgotten recovers by claiming a
    new one rather than by being let in anyway.

    With no token presented, the address still works — but only for an address
    that has no token on file. That is what makes the upgrade safe to do
    silently: an old build keeps working, and the moment a machine claims a
    token its address stops being a way in.
    """
    if presented:
        fp = fingerprint(presented)
        bound = tokens.get(fp)
        if bound is None:
            return Decision(False, reason="unknown token")
        if bound != ip:
            return Decision(True, rebind_from=bound, reason="token, moved address")
        return Decision(True, reason="token")

    if not ip or ip not in set(trusted_ips):
        return Decision(False, reason="not a trusted address")
    if ip in set(tokens.values()):
        return Decision(False, reason="address has a token; one must be presented")
    return Decision(True, reason="trusted address, no token yet")


def can_claim(tokens: Mapping[str, str], trusted_ips: Iterable[str], ip: Optional[str],
              grace_ips: Iterable[str] = ()) -> bool:
    """Whether ``ip`` may be handed a token without any other proof.

    True for an address that pairing already trusted and that holds no token yet
    — the same trust it has been exercising all along, exchanged for something
    better. Once it holds one, a further token has to be authorized by
    presenting the current one.

    ``grace_ips`` is the exception, and it exists for one failure: B issues a
    token and A never manages to store it (a write that failed, a process killed
    between the response and the save). A then has no token, and its address no
    longer works, so it can neither talk nor ask again — a pairing that only an
    operator at B's keyboard could rescue, mid-service. For a short window after
    issuing, the address may therefore claim again. The window is the whole
    exposure, and it is one A retries inside within a heartbeat.
    """
    if not ip or ip not in set(trusted_ips):
        return False
    return ip not in set(tokens.values()) or ip in set(grace_ips)


def bind(tokens: Mapping[str, str], ip: str, token: str) -> Dict[str, str]:
    """``tokens`` with ``token`` recorded against ``ip``, replacing that IP's old one."""
    updated = {fp: bound for fp, bound in tokens.items() if bound != ip}
    updated[fingerprint(token)] = ip
    return updated


def rebind(tokens: Mapping[str, str], old_ip: str, new_ip: str) -> Dict[str, str]:
    """``tokens`` with every entry for ``old_ip`` moved to ``new_ip``."""
    return {fp: (new_ip if bound == old_ip else bound) for fp, bound in tokens.items()}


def forget(tokens: Mapping[str, str], ip: str) -> Dict[str, str]:
    """``tokens`` without any entry for ``ip`` — what unpairing revokes."""
    return {fp: bound for fp, bound in tokens.items() if bound != ip}


def issue(tokens: Mapping[str, str], ip: str) -> Tuple[Dict[str, str], str]:
    """Mint a token for ``ip`` and return ``(updated tokens, the token)``."""
    token = mint_token()
    return bind(tokens, ip, token), token
