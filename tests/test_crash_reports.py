"""What crash reports carry off the machine, and which ones never leave."""

from stt.crash_reports import is_websocket_handover, scrub_event


def _handover_event(exc_type="ConnectionError", filename="engineio/async_drivers/_websocket_wsgi.py"):
    """An event shaped like the ones a browser opening the live page produces:
    engineio raises out of the WSGI app to hand the socket to werkzeug, and the
    Flask integration reports the escape as an unhandled error."""
    return {
        "exception": {
            "values": [{
                "type": exc_type,
                "value": "",
                "stacktrace": {"frames": [
                    {"function": "__call__", "module": "flask.app", "filename": "flask/app.py"},
                    {"function": "handle_request", "module": "engineio.server", "filename": "engineio/server.py"},
                    {"function": "__call__", "filename": filename},
                ]},
            }],
        },
        "request": {"url": "http://127.0.0.1:8080/socket.io/", "method": "GET"},
    }


def test_websocket_upgrade_signal_is_dropped():
    assert scrub_event(_handover_event(), None) is None


def test_gunicorn_upgrade_signal_is_dropped():
    assert scrub_event(_handover_event(exc_type="StopIteration"), None) is None


def test_windows_paths_are_matched():
    win = r"C:\stt\.venv\Lib\site-packages\engineio\async_drivers\_websocket_wsgi.py"
    assert is_websocket_handover(_handover_event(filename=win))


def test_dotted_module_name_is_matched():
    event = _handover_event(filename="")
    event["exception"]["values"][-1]["stacktrace"]["frames"][-1] = {
        "function": "__call__", "module": "engineio.async_drivers._websocket_wsgi",
    }
    assert is_websocket_handover(event)


def test_real_connection_error_elsewhere_is_kept():
    """The exception type alone must not be enough — a peer request or a model
    download failing with ConnectionError is a report we want."""
    event = _handover_event()
    event["exception"]["values"][-1]["stacktrace"]["frames"][-1] = {
        "function": "peer_request", "module": "urllib.request", "filename": "urllib/request.py",
    }
    assert not is_websocket_handover(event)
    assert scrub_event(event, None) is not None


def test_engineio_frame_with_other_exception_is_kept():
    assert not is_websocket_handover(_handover_event(exc_type="ValueError"))


def test_event_without_exception_is_kept():
    assert scrub_event({"request": {"url": "http://x/"}}, None) is not None
    assert not is_websocket_handover({"exception": {"values": []}})


def test_request_body_query_and_headers_are_stripped():
    event = {"request": {
        "url": "http://x/api/translate",
        "method": "POST",
        "data": {"text": "verbatim congregation speech"},
        "query_string": "key=secret-access-token",
        "headers": {"Cookie": "session=abc"},
        "cookies": {"session": "abc"},
        "env": {"REMOTE_ADDR": "192.168.2.62"},
    }}
    scrubbed = scrub_event(event, None)
    assert set(scrubbed["request"]) == {"url", "method"}


def test_argv_is_stripped():
    event = {"extra": {"sys.argv": ["/Users/someone/.stt/app/speech_to_text.py"], "keep": 1}}
    assert scrub_event(event, None)["extra"] == {"keep": 1}


def test_subprocess_span_descriptions_lose_their_arguments():
    event = {"spans": [
        {"op": "subprocess", "description": "ffmpeg -f avfoundation -i :Scarlett -y /Users/someone/service.wav",
         "data": {"cwd": "/Users/someone"}},
        {"op": "db.query", "description": "SELECT 1"},
    ]}
    spans = scrub_event(event, None)["spans"]
    assert spans[0]["description"] == "ffmpeg"
    assert "data" not in spans[0]
    assert spans[1]["description"] == "SELECT 1"  # untouched
