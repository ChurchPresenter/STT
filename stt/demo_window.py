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


def status_text(running: bool, port: int) -> str:
    """The one-line status the window shows for the web server."""
    return f"● Running — {browser_url(port)}" if running else "● Stopped"


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


def strip_port_flag(args: Sequence[str]) -> List[str]:
    """``args`` with every ``--port`` (and ``--port=N``) removed.

    A relaunch appends nothing — it sets the environment — so leaving the old flag
    in place would let it win over the port the visitor just typed.
    """
    kept: List[str] = []
    skip = False
    for item in args:
        if skip:
            skip = False
            continue
        if item == "--port":
            skip = True
            continue
        if item.startswith("--port="):
            continue
        kept.append(item)
    return kept


def relaunch_command(port: int, argv: Sequence[str], executable: str,
                     frozen: bool,
                     environ: Optional[MutableMapping[str, str]] = None,
                     ) -> Tuple[List[str], Dict[str, str]]:
    """Argv and environment to re-exec this demo on ``port``.

    Restarting is the honest way to change the port: it is baked into the demo's
    ``config.json`` by :func:`stt.demo_mode.write_config` before the server ever
    reads it, so nothing short of a new process rebinds.

    Frozen, ``sys.executable`` is the demo binary itself and ``argv[0]`` is the same
    path, so the script argument must not be repeated. From source the interpreter
    and the script are two separate words.
    """
    rest = strip_port_flag(list(argv)[1:])
    command = [executable, *rest] if frozen else [executable, argv[0], *rest]
    env = dict(os.environ if environ is None else environ)
    env[demo_mode.ENV_PORT] = str(port)
    # The flag was stripped from argv, so the variable is the only instruction left.
    return command, env


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


def display_available() -> bool:
    """Whether a Tk window can actually be opened here.

    Probed in a child process where fork exists, because the failure mode is a crash
    rather than an exception. Windows has no fork, and there the demo is a windowed
    .exe with a desktop under it, so the probe runs in place.
    """
    forced = forced_window()
    if forced is not None:
        return forced
    try:
        if hasattr(os, "fork"):
            return _forked_probe()
        return _probe_window()
    except Exception:
        return False


class DemoWindow:
    """Status, port and quit. Nothing else — the demo itself is the web UI."""

    def __init__(self, port: int, session_path: str, version: str,
                 on_quit: Callable[[], None],
                 on_set_port: Callable[[int], None],
                 status_probe: Callable[[], bool]) -> None:
        import tkinter as tk

        self._tk = tk
        self._port = port
        self._session = os.path.basename(session_path)
        self._on_quit = on_quit
        self._on_set_port = on_set_port
        self._status_probe = status_probe
        self.root = tk.Tk()
        self.root.title(f"STT Demo v{version}")
        self.root.config(menu=tk.Menu(self.root))  # drop Tk's stock macOS menubar
        self.root.resizable(False, False)
        self.root.minsize(380, 1)
        self.root.protocol("WM_DELETE_WINDOW", self._quit)
        self._build()
        self.root.after(500, self._poll)

    # -- widgets ------------------------------------------------------------

    def _build(self) -> None:
        tk = self._tk
        pad = {"padx": 12, "pady": 4}

        row = tk.Frame(self.root)
        row.pack(fill="x", **pad)
        tk.Label(row, text="Web UI:", width=11, anchor="w").pack(side="left")
        self._status_lbl = tk.Label(row, text=status_text(False, self._port),
                                    fg="red", font=("", 10, "bold"))
        self._status_lbl.pack(side="left", anchor="w")

        row = tk.Frame(self.root)
        row.pack(fill="x", **pad)
        tk.Label(row, text="Replaying:", width=11, anchor="w").pack(side="left")
        tk.Label(row, text=self._session, fg="gray", anchor="w").pack(side="left")

        tk.Frame(self.root, height=1, bg="#cccccc").pack(fill="x", padx=12, pady=4)

        row = tk.Frame(self.root)
        row.pack(fill="x", **pad)
        tk.Label(row, text="Port:", width=11, anchor="w").pack(side="left")
        self._port_var = tk.StringVar(value=str(self._port))
        tk.Entry(row, textvariable=self._port_var, width=8).pack(side="left")
        tk.Button(row, text="Apply", width=8, command=self._apply_port).pack(side="right")

        self._hint_lbl = tk.Label(
            self.root,
            text="Changing the port restarts the demo; the replay starts again.",
            fg="gray", font=("", 8), anchor="w", justify="left", wraplength=340)
        self._hint_lbl.pack(fill="x", padx=12)

        row = tk.Frame(self.root)
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
        self._status_lbl.config(text=status_text(running, self._port),
                                fg="green" if running else "red")
        self.root.after(1000, self._poll)

    def mainloop(self) -> None:
        self.root.mainloop()


def run(port: int, session_path: str, version: str,
        on_quit: Callable[[], None],
        on_set_port: Callable[[int], None],
        status_probe: Callable[[], bool]) -> Any:
    """Open the control window and block until it closes."""
    window = DemoWindow(port, session_path, version, on_quit, on_set_port, status_probe)
    window.mainloop()
    return window
