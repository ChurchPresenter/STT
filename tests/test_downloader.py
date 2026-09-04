"""download_url_to_file: retry/resume/cancel downloader (stt/downloads.py)."""

import hashlib
import http.server
import os
import shutil
import socketserver
import threading
import time

import pytest

from stt import downloads

CONTENT = b"x" * 300_000


class _Handler(http.server.BaseHTTPRequestHandler):
    """/fast, /slow, and /truncated — a transfer that stops half way.

    /truncated announces the real Content-Length and then closes after half the
    body, which is what an interrupted download actually looks like on the wire.
    Byte ranges are honoured so the resume path (`wget -c` / `curl -C -`) is
    exercised the way huggingface.co exercises it, rather than being silently
    downgraded to a restart.
    """

    def do_GET(self):
        start = 0
        rng = self.headers.get("Range", "")
        if rng.startswith("bytes="):
            try:
                start = int(rng.split("=", 1)[1].split("-", 1)[0])
            except ValueError:
                start = 0
        body = CONTENT[start:]
        self.send_response(206 if start else 200)
        if start:
            self.send_header("Content-Range", f"bytes {start}-{len(CONTENT) - 1}/{len(CONTENT)}")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            if self.path == "/slow":
                for i in range(0, len(body), 10_000):
                    self.wfile.write(body[i:i + 10_000])
                    time.sleep(0.2)
            elif self.path == "/truncated":
                # Promise the whole file, deliver half, hang up.
                self.wfile.write(body[:len(body) // 2])
                self.close_connection = True
            else:
                self.wfile.write(body)
        except BrokenPipeError:
            pass  # client (or terminated wget) went away — expected in cancel tests

    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def http_server():
    srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


class _NoToolShutil:
    """Force the pure-Python fallback path."""

    @staticmethod
    def which(name):
        return None


@pytest.fixture
def no_tools(monkeypatch):
    monkeypatch.setattr(downloads, "shutil", _NoToolShutil)


def _downloader(shutil_impl=None):
    return downloads.download_url_to_file


def test_python_fallback_ok(http_server, tmp_path, no_tools):
    dl = _downloader()
    dest = tmp_path / "f.bin"
    assert dl(f"{http_server}/fast", str(dest)) == "ok"
    assert dest.stat().st_size == len(CONTENT)


def test_python_fallback_cancel(http_server, tmp_path, no_tools):
    dl = _downloader()
    assert dl(f"{http_server}/fast", str(tmp_path / "c.bin"), cancel_check=lambda: True) == "cancelled"


def test_python_fallback_raises_after_retries(http_server, tmp_path, no_tools):
    dl = _downloader()
    bad_url = http_server.rsplit(":", 1)[0] + ":1/nope"  # nothing listens on port 1
    with pytest.raises(Exception, match="Failed to download"):
        dl(bad_url, str(tmp_path / "f.bin"), max_attempts=1)


needs_tool = pytest.mark.skipif(
    not (shutil.which("wget") or shutil.which("curl")),
    reason="neither wget nor curl installed",
)


@needs_tool
def test_subprocess_path_ok(http_server, tmp_path):
    dl = _downloader()
    dest = tmp_path / "a.bin"
    assert dl(f"{http_server}/fast", str(dest)) == "ok"
    assert dest.stat().st_size == len(CONTENT)


@needs_tool
def test_subprocess_path_cancel_mid_download(http_server, tmp_path):
    dl = _downloader()
    cancel = {"flag": False}
    threading.Timer(1.0, lambda: cancel.update(flag=True)).start()
    t0 = time.time()
    outcome = dl(f"{http_server}/slow", str(tmp_path / "b.bin"), cancel_check=lambda: cancel["flag"])
    assert outcome == "cancelled"
    assert time.time() - t0 < 4, "cancel must terminate the subprocess promptly"


# --- staging: an interrupted transfer must not look like a complete one ------
#
# Writing straight to the destination is what made a truncated file
# indistinguishable from a whole one: it listed as downloaded, a re-download
# skipped it as "already exists", and a loader eventually threw or hung on it.
# Everything below is about the file that is *not* at dest_path.

def _part(path):
    return str(path) + ".part"


class TestStaging:
    def test_nothing_lands_at_the_destination_until_it_verifies(self, http_server, tmp_path, no_tools):
        dest = tmp_path / "model.bin"
        with pytest.raises(Exception, match="Failed to download"):
            downloads.download_url_to_file(
                f"{http_server}/fast", str(dest), max_attempts=1,
                expected_size=len(CONTENT) + 1,  # we were promised one more byte
            )
        assert not dest.exists(), "a transfer that did not verify must not claim the real name"

    def test_a_wrong_size_transfer_does_not_leave_a_resumable_part(self, http_server, tmp_path, no_tools):
        """Resuming bytes we already know are wrong would append to garbage."""
        dest = tmp_path / "model.bin"
        with pytest.raises(Exception, match="Failed to download"):
            downloads.download_url_to_file(
                f"{http_server}/fast", str(dest), max_attempts=1, expected_size=len(CONTENT) + 1,
            )
        assert not os.path.exists(_part(dest))

    def test_expected_size_match_promotes(self, http_server, tmp_path, no_tools):
        dest = tmp_path / "model.bin"
        assert downloads.download_url_to_file(
            f"{http_server}/fast", str(dest), expected_size=len(CONTENT)
        ) == "ok"
        assert dest.stat().st_size == len(CONTENT)
        assert not os.path.exists(_part(dest))

    def test_sha_mismatch_is_rejected(self, http_server, tmp_path, no_tools):
        dest = tmp_path / "model.bin"
        with pytest.raises(Exception, match="Failed to download"):
            downloads.download_url_to_file(
                f"{http_server}/fast", str(dest), max_attempts=1,
                expected_size=len(CONTENT), expected_sha256="0" * 64,
            )
        assert not dest.exists()

    def test_sha_match_promotes(self, http_server, tmp_path, no_tools):
        dest = tmp_path / "model.bin"
        digest = hashlib.sha256(CONTENT).hexdigest()
        assert downloads.download_url_to_file(
            f"{http_server}/fast", str(dest), expected_size=len(CONTENT), expected_sha256=digest,
        ) == "ok"

    def test_cancel_leaves_no_destination_file(self, http_server, tmp_path, no_tools):
        dest = tmp_path / "model.bin"
        assert downloads.download_url_to_file(
            f"{http_server}/fast", str(dest), cancel_check=lambda: True
        ) == "cancelled"
        assert not dest.exists()

    @needs_tool
    def test_subprocess_truncated_transfer_leaves_a_resumable_part(self, http_server, tmp_path):
        """The real shape of the bug: the connection drops mid-file.

        The destination must stay empty, and the bytes that did arrive must
        survive as a .part so the next attempt resumes instead of restarting —
        which is more than the old code managed, since its retries all pointed at
        the final name and a later *call* saw a complete-looking file.
        """
        dest = tmp_path / "model.bin"
        with pytest.raises(Exception, match="Failed to download"):
            downloads.download_url_to_file(
                f"{http_server}/truncated", str(dest), max_attempts=1, log=lambda _m: None,
            )
        assert not dest.exists()
        assert os.path.exists(_part(dest))
        assert 0 < os.path.getsize(_part(dest)) < len(CONTENT)

    @needs_tool
    def test_a_second_call_resumes_the_part_and_completes(self, http_server, tmp_path):
        dest = tmp_path / "model.bin"
        with pytest.raises(Exception, match="Failed to download"):
            downloads.download_url_to_file(
                f"{http_server}/truncated", str(dest), max_attempts=1, log=lambda _m: None,
            )
        partial = os.path.getsize(_part(dest))
        assert downloads.download_url_to_file(
            f"{http_server}/fast", str(dest), expected_size=len(CONTENT)
        ) == "ok"
        assert dest.read_bytes() == CONTENT, f"resume from {partial} bytes produced the wrong file"
        assert not os.path.exists(_part(dest))

    @needs_tool
    def test_subprocess_path_rejects_a_wrong_expected_size(self, http_server, tmp_path):
        dest = tmp_path / "model.bin"
        with pytest.raises(Exception, match="incomplete transfer"):
            downloads.download_url_to_file(
                f"{http_server}/fast", str(dest), max_attempts=1,
                expected_size=len(CONTENT) + 1, log=lambda _m: None,
            )
        assert not dest.exists()
