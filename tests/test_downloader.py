"""download_url_to_file: retry/resume/cancel downloader (stt/downloads.py)."""

import hashlib
import http.server
import os
import pathlib
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
        if self.path == "/norange":
            # A server that ignores Range and sends the whole file with a 200.
            # Resuming against one would concatenate two overlapping prefixes.
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


class TestPurePythonResume:
    """The fallback used when neither wget nor curl is installed.

    It opened the part file "wb" and asked for the whole URL every time, so it
    restarted from zero on every attempt and every retry — while the retry log
    announced that it would resume. A large model could therefore never finish
    on a machine without either tool, however many times the user pressed
    Download.
    """

    def test_an_interrupted_transfer_leaves_a_part_to_resume(self, http_server, no_tools, tmp_path):
        dest = tmp_path / "model.bin"
        with pytest.raises(Exception, match="Failed to download"):
            downloads.download_url_to_file(
                f"{http_server}/truncated", str(dest), max_attempts=1, log=lambda m: None)

        part = pathlib.Path(_part(dest))
        assert part.exists(), "a transport failure must keep what arrived"
        assert 0 < part.stat().st_size < len(CONTENT)

    def test_a_second_call_continues_rather_than_starting_over(self, http_server, no_tools, tmp_path):
        dest = tmp_path / "model.bin"
        with pytest.raises(Exception, match="Failed to download"):
            downloads.download_url_to_file(
                f"{http_server}/truncated", str(dest), max_attempts=1, log=lambda m: None)
        part = pathlib.Path(_part(dest))
        first = part.stat().st_size

        downloads.download_url_to_file(
            f"{http_server}/fast", str(dest), max_attempts=1, log=lambda m: None)

        assert dest.read_bytes() == CONTENT
        assert first > 0, "the resume must have had something to build on"

    def test_a_server_that_ignores_the_range_starts_over_rather_than_corrupting(
        self, http_server, no_tools, tmp_path
    ):
        """A 200 to a ranged request means the whole file is coming.

        Appending it to what we already had would produce a file of the right
        name and the wrong bytes — worse than restarting.
        """
        dest = tmp_path / "model.bin"
        part = pathlib.Path(_part(dest))
        part.write_bytes(b"y" * 50_000)

        downloads.download_url_to_file(
            f"{http_server}/norange", str(dest), max_attempts=1, log=lambda m: None)

        assert dest.read_bytes() == CONTENT, "the stale prefix must not survive"


class TestCurlHasNoTotalTimeCap:
    """--max-time capped the whole transfer, not its idle time.

    3 GB inside 600s demands ~5 MB/s sustained, so on the slow connections that
    resume exists for, every attempt was killed on the clock regardless of how
    well it was going — and no number of retries could ever finish the file.
    """

    def _argv(self, monkeypatch, tmp_path):
        seen = {}

        class _Curl:
            @staticmethod
            def which(name):
                return "/usr/bin/curl" if name == "curl" else None

        class _Proc:
            returncode = 1

            def poll(self):
                return 1

        def fake_popen(cmd, **kwargs):
            seen["cmd"] = cmd
            return _Proc()

        monkeypatch.setattr(downloads, "shutil", _Curl)
        # subprocess is imported inside download_url_to_file, so patch the
        # stdlib module itself rather than an attribute of downloads.
        monkeypatch.setattr("subprocess.Popen", fake_popen)
        with pytest.raises(Exception, match="Failed to download"):
            downloads.download_url_to_file(
                "http://example.invalid/model.bin", str(tmp_path / "model.bin"),
                max_attempts=1, log=lambda m: None)
        return seen["cmd"]

    def test_no_cap_on_total_transfer_time(self, monkeypatch, tmp_path):
        assert "--max-time" not in self._argv(monkeypatch, tmp_path)

    def test_it_aborts_on_a_stalled_transfer_instead(self, monkeypatch, tmp_path):
        argv = self._argv(monkeypatch, tmp_path)
        assert "--speed-time" in argv and "--speed-limit" in argv
        assert argv[argv.index("--speed-time") + 1] == str(downloads.STALL_SECONDS)
        assert argv[argv.index("--speed-limit") + 1] == str(downloads.STALL_BYTES_PER_S)

    def test_the_stall_threshold_is_slow_enough_to_let_a_bad_link_finish(self):
        """Guards the numbers themselves: this is the bug, restated as a test."""
        assert downloads.STALL_BYTES_PER_S <= 8 * 1024, "must tolerate a very slow link"
        assert downloads.STALL_SECONDS >= 60, "a brief pause is not a stall"


class TestWhisperCheckpointIsStaged:
    """The .pt download used to write straight to its final name.

    A process killed mid-transfer left a truncated checkpoint under the real name.
    whisper.load_model then notices the bad checksum and re-downloads it with
    urllib.request.urlopen(url) and no timeout — inside the transcription worker.
    Staging means the real name never holds bytes that failed their checksum.
    """

    def test_the_monolith_routes_it_through_the_staged_downloader(self):
        source = pathlib.Path("speech_to_text.py").read_text(encoding="utf-8")
        assert "_downloads.download_url_to_file(\n" in source
        assert "expected_sha256=expected_sha256," in source, (
            "the checkpoint's sha256 is in its URL; the staged path must verify it"
        )

    def test_a_wrong_checksum_never_reaches_the_final_name(self, http_server, tmp_path):
        dest = tmp_path / "model.pt"
        with pytest.raises(Exception, match="Failed to download"):
            downloads.download_url_to_file(
                f"{http_server}/fast", str(dest), max_attempts=1, log=lambda m: None,
                expected_sha256="0" * 64,
            )
        assert not dest.exists(), "a checkpoint that failed its checksum must not be promoted"

    def test_a_correct_checksum_is_promoted(self, http_server, tmp_path):
        dest = tmp_path / "model.pt"
        downloads.download_url_to_file(
            f"{http_server}/fast", str(dest), max_attempts=1, log=lambda m: None,
            expected_sha256=hashlib.sha256(CONTENT).hexdigest(),
        )
        assert dest.read_bytes() == CONTENT
