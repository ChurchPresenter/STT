"""Whether a control-surface request may be honoured.

A stream deck button, a show-control cue and a line in a run-of-show document all want the
same thing: one URL that does one thing, fetched with no ceremony. The simplest button
Bitfocus Companion offers is an HTTP GET, and every route here that changes something is
POST-only, so that button gets a 405.

Relaxing the method is not free. A GET that changes state can be fired by any page a browser
on the network happens to load — an image tag pointing at it is enough — and the operator's
own browser is authorised by an ambient cookie or by being on the whitelist. POST-only is
currently the only thing standing between the phase timeline and a drive-by.

So the relaxation is paired with a narrowing: a GET-capable control route is honoured only
when the request carries the access token explicitly. A page that a browser was tricked into
loading cannot know the token, while a button configured once by the person who owns the
machine has it in its URL forever. The ambient credentials that make CSRF possible are
exactly the ones that stop working here.

Stdlib-only and framework-free: the caller passes in what it knows and gets back either a
reason to refuse or None.
"""

import secrets
from typing import Any, Mapping, Optional

# The parameter a control surface puts the token in, matching the existing ?key= the
# caption display and OBS browser sources already use.
KEY_PARAM = "key"

REFUSE_NO_TOKEN = ("This machine has no access token, so control links are disabled. "
                   "Set one in Server Settings, then put it in the button's URL as "
                   "?key=...")
REFUSE_MISSING_KEY = ("A control link must carry the access token as ?key=... — a session "
                      "cookie or a whitelisted address is not enough for a link that can "
                      "be triggered by any page.")
REFUSE_BAD_KEY = "That access token is not this machine's."


def refuse_reason(params: Mapping[str, Any], *, token: str,
                  method: str = "GET") -> Optional[str]:
    """Why this control request must not be honoured, or None if it may be.

    The token is compared with :func:`secrets.compare_digest`, so a wrong one takes the same
    time to reject however much of it is right.

    ``method`` is accepted so a POST can keep behaving as it always has: it arrives through
    the routes that already exist, with their own authorisation, and this is only the extra
    condition that makes the GET form safe.
    """
    if str(method or "").upper() == "POST":
        return None
    if not token:
        # Never "no token means everyone": an installation that has not set one has not
        # opted into control links at all.
        return REFUSE_NO_TOKEN
    provided = str(params.get(KEY_PARAM) or "")
    if not provided:
        return REFUSE_MISSING_KEY
    if not secrets.compare_digest(provided, str(token)):
        return REFUSE_BAD_KEY
    return None


def allowed(params: Mapping[str, Any], *, token: str, method: str = "GET") -> bool:
    """``refuse_reason`` as a plain yes/no."""
    return refuse_reason(params, token=token, method=method) is None
