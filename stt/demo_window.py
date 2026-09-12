"""The demo's control window: what is running, on which port, and how to stop it.

The shipped demo is a windowed application on macOS and Windows — no console, no
watchdog, and every server-lifecycle route (``/api/restart``,
``/api/server/restart``) deliberately answered "not available in the demo" by
:mod:`stt.demo_api`. That left Activity Monitor as the only way to quit it and no
way at all to move it off the port :func:`stt.demo_mode.pick_port` happened to
choose. This is the missing half: a small always-open window, modelled on the
watchdog's ``GuiWindow``, holding the three controls a visitor actually needs.

Tk owns the main thread on macOS, so :func:`run` *is* the demo's main loop — the
web server already runs on its own thread beside it.

The decisions are pure functions at the top of the file and are unit-tested; the
Tk shell below them is kept thin deliberately, because it cannot be.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Callable, Dict, List, MutableMapping, Optional, Sequence, Tuple

from stt import demo_mode

# Ports below 1024 need root on POSIX, and a demo must never be the reason someone
# types a password. The upper bound is the protocol's.
MIN_PORT = 1024
MAX_PORT = 65535


def browser_url(port: int, host: str = "127.0.0.1") -> str:
    """The address the window opens and displays. Loopback even though the server
    binds 0.0.0.0 — the person reading the window is at the machine."""
    return f"http://{host}:{port}/"


def status_text(running: bool, port: int, host: str = "127.0.0.1") -> str:
    """The one-line status the window shows for the web server.

    ``host`` defaults to loopback but callers pass the LAN address when one is known —
    the server binds 0.0.0.0 and the point of the demo is to be opened from a phone or
    a second screen, so showing the address nobody but this machine can use isn't
    useful.
    """
    return f"● Running — {browser_url(port, host)}" if running else "● Stopped"


def validate_port(raw: str) -> Tuple[Optional[int], str]:
    """Parse a typed port, returning ``(port, "")`` or ``(None, reason)``.

    A reason rather than an exception because the window shows it next to the field:
    the point is to tell someone what to type instead.
    """
    text = (raw or "").strip()
    if not text:
        return None, "Enter a port number."
    try:
        value = int(text)
    except ValueError:
        return None, f"{text!r} is not a number."
    if value < MIN_PORT:
        return None, f"Ports below {MIN_PORT} need administrator rights."
    if value > MAX_PORT:
        return None, f"The highest port is {MAX_PORT}."
    return value, ""


def _strip_flag(args: Sequence[str], flag: str) -> List[str]:
    """``args`` with every occurrence of ``flag`` (and ``flag=value``) removed.

    A relaunch appends nothing — it sets the environment — so leaving the old flag
    in place would let it win over whatever the visitor just picked.
    """
    kept: List[str] = []
    skip = False
    for item in args:
        if skip:
            skip = False
            continue
        if item == flag:
            skip = True
            continue
        if item.startswith(flag + "="):
            continue
        kept.append(item)
    return kept


def strip_port_flag(args: Sequence[str]) -> List[str]:
    """``args`` with every ``--port`` (and ``--port=N``) removed."""
    return _strip_flag(args, "--port")


def strip_session_flag(args: Sequence[str]) -> List[str]:
    """``args`` with every ``--session`` (and ``--session=PATH``) removed."""
    return _strip_flag(args, "--session")


def _relaunch_command(rest: List[str], argv: Sequence[str], executable: str,
                      frozen: bool, environ: Optional[MutableMapping[str, str]],
                      env_key: str, env_value: str) -> Tuple[List[str], Dict[str, str]]:
    """Shared shape of a re-exec: argv from ``rest``, environment from ``environ``
    (or the live process) with ``env_key`` set.

    Frozen, ``sys.executable`` is the demo binary itself and ``argv[0]`` is the same
    path, so the script argument must not be repeated. From source the interpreter
    and the script are two separate words.
    """
    command = [executable, *rest] if frozen else [executable, argv[0], *rest]
    env = dict(os.environ if environ is None else environ)
    env[env_key] = env_value
    return command, env


def relaunch_command(port: int, argv: Sequence[str], executable: str,
                     frozen: bool,
                     environ: Optional[MutableMapping[str, str]] = None,
                     ) -> Tuple[List[str], Dict[str, str]]:
    """Argv and environment to re-exec this demo on ``port``.

    Restarting is the honest way to change the port: it is baked into the demo's
    ``config.json`` by :func:`stt.demo_mode.write_config` before the server ever
    reads it, so nothing short of a new process rebinds.
    """
    rest = strip_port_flag(list(argv)[1:])
    # The flag was stripped from argv, so the variable is the only instruction left.
    return _relaunch_command(rest, argv, executable, frozen, environ,
                             demo_mode.ENV_PORT, str(port))


def relaunch_command_for_session(session_path: str, argv: Sequence[str], executable: str,
                                 frozen: bool,
                                 environ: Optional[MutableMapping[str, str]] = None,
                                 ) -> Tuple[List[str], Dict[str, str]]:
    """Argv and environment to re-exec this demo playing ``session_path``.

    Same shape as :func:`relaunch_command`: the recording is picked at startup
    (``stt.demo_mode.ensure_session``/``requested_session``), so switching it is
    another restart. Because ``os.execve`` replaces the process environment outright,
    a later port change (whose ``relaunch_command`` copies ``os.environ``) still
    carries this session forward without being told to.
    """
    rest = strip_session_flag(list(argv)[1:])
    return _relaunch_command(rest, argv, executable, frozen, environ,
                             "STT_DEMO_DB", session_path)


def sessions_dir(executable_dir: Optional[str], data_dir: str) -> str:
    """Where the "Open Sessions Folder" button points.

    The drop-in folder beside the shipped executable when there is one (what the
    README tells someone to use); the demo's own data dir otherwise — a dev run from
    source has no ``executable_dir``, but ``data_dir`` (``~/.stt-demo``) always exists.
    """
    base = executable_dir if executable_dir else data_dir
    return os.path.join(base, demo_mode.SESSIONS_DIR_NAME)


def open_folder_command(path: str, platform: str = sys.platform) -> Optional[List[str]]:
    """The argv to open ``path`` in the OS file manager, or ``None`` on Windows.

    Windows has no single-purpose "open a folder" executable to shell out to —
    ``os.startfile`` is the documented way, and it isn't a subprocess call, so callers
    use it directly instead of this argv when ``platform == "win32"``.
    """
    if platform == "darwin":
        return ["open", path]
    if platform == "win32":
        return None
    return ["xdg-open", path]


# --- the Tk shell ----------------------------------------------------------


# Forces the window on ("1") or off ("0"). Off is what a smoke test, a CI run or a
# demo started by a service manager wants: those have no window server, and asking
# for one there does not raise — it kills the process.
ENV_WINDOW = "STT_DEMO_WINDOW"


def _probe_window() -> bool:
    """Open a real, visible Tk window and pump it once, then take it down.

    Deliberately not the watchdog's withdrawn-root probe. A withdrawn root succeeds
    in contexts where showing a window then aborts the process, so the cheap probe
    answered yes and the demo died a moment later — measured as SIGSEGV when the demo
    was started detached from a GUI session. The probe has to do what the window does
    or it is not testing the thing that fails.
    """
    import tkinter as tk

    root = tk.Tk()
    try:
        root.geometry("1x1+0+0")
        root.deiconify()
        root.update()
    finally:
        root.destroy()
    return True


def _forked_probe() -> bool:
    """Run :func:`_probe_window` in a child, so a crash costs a process, not the demo.

    Tk failing to reach a window server does not raise on macOS: it aborts, and no
    ``except`` in this process can see that coming. A child is the only way to ask
    the question and still be running afterwards. The child touches nothing before
    ``os._exit``, so the parent's buffers and atexit handlers are untouched.
    """
    if sys.platform == "win32":  # no fork; the caller never takes this branch there
        return _probe_window()
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:  # child
        try:
            os.close(read_fd)
            ok = _probe_window()
            os.write(write_fd, b"1" if ok else b"0")
        except BaseException:
            try:
                os.write(write_fd, b"0")
            except OSError:
                pass
        finally:
            os._exit(0)
    os.close(write_fd)
    try:
        answer = os.read(read_fd, 1)
    except OSError:
        answer = b""
    finally:
        os.close(read_fd)
    try:
        _, status = os.waitpid(pid, 0)
    except OSError:
        return False
    # A crash reaches us as a signal or a non-zero exit, never as an exception.
    if os.WIFSIGNALED(status) or os.WEXITSTATUS(status) != 0:
        return False
    return answer == b"1"


def forced_window(environ: Optional[MutableMapping[str, str]] = None) -> Optional[bool]:
    """``True``/``False`` if STT_DEMO_WINDOW settles it, ``None`` to probe."""
    env = os.environ if environ is None else environ
    raw = str(env.get(ENV_WINDOW, "")).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return None


_last_probe_error: Optional[str] = None


def last_probe_error() -> Optional[str]:
    """Why the most recent :func:`display_available` said no, or ``None`` if it
    said yes.

    The demo is windowed on Windows and macOS, so ``print`` has nowhere to go —
    a windowed PyInstaller build has no console and silently discards writes to
    it. Callers that want the reason on disk (see ``speech_to_text.py``'s DEMO
    startup) read this instead of relying on stdout.
    """
    return _last_probe_error


def display_available() -> bool:
    """Whether a Tk window can actually be opened here.

    Probed in a child process where fork exists, because the failure mode is a crash
    rather than an exception. Windows has no fork, and there the demo is a windowed
    .exe with a desktop under it, so the probe runs in place.
    """
    global _last_probe_error
    forced = forced_window()
    if forced is not None:
        _last_probe_error = None if forced else f"{ENV_WINDOW} forced off"
        return forced
    try:
        ok = _probe_window() if sys.platform == "win32" else _forked_probe()
    except Exception as exc:
        _last_probe_error = f"{type(exc).__name__}: {exc}"
        return False
    _last_probe_error = None if ok else "probe did not report a display"
    return ok


# A fixed light palette for the window, deliberately not following the system's
# light/dark setting — see the comment in DemoWindow.__init__ for why.
_BG = "#f0f0f0"
_FG = "#000000"


class DemoWindow:
    """Status, port, session, transcription and quit. The demo itself is the web UI —
    this is only the controls that UI has no way to expose."""

    def __init__(self, port: int, session_path: str, version: str,
                 on_quit: Callable[[], None],
                 on_set_port: Callable[[int], None],
                 status_probe: Callable[[], bool],
                 lan: Optional[str] = None,
                 sessions_path: str = "",
                 sessions: Sequence[str] = (),
                 on_select_session: Optional[Callable[[str], None]] = None,
                 on_start: Optional[Callable[[], None]] = None,
                 on_stop: Optional[Callable[[], None]] = None,
                 transcription_status_probe: Optional[Callable[[], bool]] = None,
                 ) -> None:
        import tkinter as tk

        self._tk = tk
        self._port = port
        self._session = session_path
        self._on_quit = on_quit
        self._on_set_port = on_set_port
        self._status_probe = status_probe
        self._lan = lan
        self._sessions_path = sessions_path
        # Basename -> full path, same pattern as the watchdog GuiWindow's microphone
        # menu — Tk shows the short label, the callback gets the real path.
        self._session_map: Dict[str, str] = {
            os.path.basename(path): path for path in sessions
        }
        self._on_select_session = on_select_session
        self._on_start = on_start
        self._on_stop = on_stop
        self._transcription_status_probe = transcription_status_probe
        self.root = tk.Tk()
        self.root.title(f"STT Demo v{version}")
        self.root.config(menu=tk.Menu(self.root))  # drop Tk's stock macOS menubar
        self.root.resizable(False, False)
        self.root.minsize(380, 1)
        self.root.protocol("WM_DELETE_WINDOW", self._quit)
        # Plain tk.Label/tk.Frame text goes invisible in macOS Dark Mode: Tk resolves
        # their foreground against the system's dynamic window-background color, and
        # that resolution silently breaks for these (non-ttk) widgets — confirmed on
        # this machine, where "Web UI:", the status line and every other Label simply
        # didn't paint, while native Button/OptionMenu/Entry chrome was unaffected.
        # Pinning to a fixed light palette here sidesteps the bug instead of chasing
        # Tk's dark-mode color resolution.
        self.root.configure(bg=_BG)
        self._build()
        self.root.after(500, self._poll)

    # -- widgets ------------------------------------------------------------

    def _build(self) -> None:
        tk = self._tk
        pad = {"padx": 12, "pady": 4}

        row = tk.Frame(self.root, bg=_BG)
        row.pack(fill="x", **pad)
        tk.Label(row, text="Web UI:", width=11, anchor="w", bg=_BG, fg=_FG).pack(side="left")
        self._status_lbl = tk.Label(
            row, text=status_text(False, self._port, self._lan or "127.0.0.1"),
            fg="red", bg=_BG, font=("", 10, "bold"))
        self._status_lbl.pack(side="left", anchor="w")

        row = tk.Frame(self.root, bg=_BG)
        row.pack(fill="x", **pad)
        tk.Label(row, text="Transcription:", width=11, anchor="w", bg=_BG, fg=_FG).pack(side="left")
        self._transcription_lbl = tk.Label(row, text="● Stopped", fg="red", bg=_BG,
                                           font=("", 10, "bold"))
        self._transcription_lbl.pack(side="left", expand=True, anchor="w")
        self._transcription_btn = tk.Button(
            row, text="Start", width=8, command=self._on_toggle_transcription)
        self._transcription_btn.pack(side="right")

        row = tk.Frame(self.root, bg=_BG)
        row.pack(fill="x", **pad)
        tk.Label(row, text="Replaying:", width=11, anchor="w", bg=_BG, fg=_FG).pack(side="left")
        self._session_var = tk.StringVar(value=os.path.basename(self._session))
        session_menu = tk.OptionMenu(row, self._session_var,
                                     *(self._session_map or {self._session_var.get(): ""}),
                                     command=self._on_select_session_changed)
        session_menu.config(width=20, font=("", 9))
        session_menu.pack(side="left")

        row = tk.Frame(self.root, bg=_BG)
        row.pack(fill="x", **pad)
        tk.Label(row, text="Sessions:", width=11, anchor="w", bg=_BG, fg=_FG).pack(side="left")
        tk.Button(row, text="Open Folder", command=self._open_sessions).pack(side="left")

        tk.Frame(self.root, height=1, bg="#cccccc").pack(fill="x", padx=12, pady=4)

        row = tk.Frame(self.root, bg=_BG)
        row.pack(fill="x", **pad)
        tk.Label(row, text="Port:", width=11, anchor="w", bg=_BG, fg=_FG).pack(side="left")
        self._port_var = tk.StringVar(value=str(self._port))
        tk.Entry(row, textvariable=self._port_var, width=8).pack(side="left")
        tk.Button(row, text="Apply", width=8, command=self._apply_port).pack(side="right")

        self._hint_lbl = tk.Label(
            self.root,
            text="Changing the port restarts the demo; the replay starts again.",
            fg="gray", bg=_BG, font=("", 8), anchor="w", justify="left", wraplength=340)
        self._hint_lbl.pack(fill="x", padx=12)

        row = tk.Frame(self.root, bg=_BG)
        row.pack(fill="x", padx=12, pady=(10, 12))
        tk.Button(row, text="Open in Browser",
                  command=self._open_browser).pack(side="left")
        tk.Button(row, text="Quit Demo", command=self._quit).pack(side="right")

    # -- actions ------------------------------------------------------------

    def _open_browser(self) -> None:
        import webbrowser

        try:
            webbrowser.open(browser_url(self._port))
        except Exception:
            pass

    def _open_sessions(self) -> None:
        try:
            os.makedirs(self._sessions_path, exist_ok=True)
            if sys.platform == "win32":
                os.startfile(self._sessions_path)  # type: ignore[attr-defined]
                return
            command = open_folder_command(self._sessions_path)
            if command:
                import subprocess

                subprocess.Popen(command)
        except Exception:
            pass

    def _apply_port(self) -> None:
        port, reason = validate_port(self._port_var.get())
        if port is None:
            self._hint_lbl.config(text=reason, fg="red")
            return
        if port == self._port:
            self._hint_lbl.config(text="Already running on that port.", fg="gray")
            return
        self._hint_lbl.config(text=f"Restarting on port {port}…", fg="gray")
        self.root.update_idletasks()
        self._on_set_port(port)

    def _on_select_session_changed(self, label: str) -> None:
        from tkinter import messagebox

        path = self._session_map.get(label)
        if not path or path == self._session or self._on_select_session is None:
            return
        if not messagebox.askokcancel(
                "Switch recording",
                "Switching recordings restarts the demo — continue?"):
            self._session_var.set(os.path.basename(self._session))
            return
        self._on_select_session(path)

    def _on_toggle_transcription(self) -> None:
        running = self._transcription_running()
        if running:
            if self._on_stop is not None:
                self._on_stop()
        else:
            if self._on_start is not None:
                self._on_start()

    def _transcription_running(self) -> bool:
        if self._transcription_status_probe is None:
            return False
        try:
            return bool(self._transcription_status_probe())
        except Exception:
            return False

    def _quit(self) -> None:
        from tkinter import messagebox

        if not messagebox.askokcancel("Quit", "Stop the demo?"):
            return
        try:
            self.root.destroy()
        except Exception:
            pass
        self._on_quit()

    # -- polling ------------------------------------------------------------

    def _poll(self) -> None:
        try:
            running = bool(self._status_probe())
        except Exception:
            running = False
        self._status_lbl.config(
            text=status_text(running, self._port, self._lan or "127.0.0.1"),
            fg="green" if running else "red")

        transcribing = self._transcription_running()
        self._transcription_lbl.config(
            text="● Running" if transcribing else "● Stopped",
            fg="green" if transcribing else "red")
        self._transcription_btn.config(text="Stop" if transcribing else "Start")

        self.root.after(1000, self._poll)

    def mainloop(self) -> None:
        self.root.mainloop()


def run(port: int, session_path: str, version: str,
        on_quit: Callable[[], None],
        on_set_port: Callable[[int], None],
        status_probe: Callable[[], bool],
        lan: Optional[str] = None,
        sessions_path: str = "",
        sessions: Sequence[str] = (),
        on_select_session: Optional[Callable[[str], None]] = None,
        on_start: Optional[Callable[[], None]] = None,
        on_stop: Optional[Callable[[], None]] = None,
        transcription_status_probe: Optional[Callable[[], bool]] = None,
        ) -> Any:
    """Open the control window and block until it closes."""
    window = DemoWindow(port, session_path, version, on_quit, on_set_port, status_probe,
                        lan=lan, sessions_path=sessions_path, sessions=sessions,
                        on_select_session=on_select_session, on_start=on_start,
                        on_stop=on_stop,
                        transcription_status_probe=transcription_status_probe)
    window.mainloop()
    return window
