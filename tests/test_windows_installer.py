"""The Windows installer's start-at-logon task must name a program that exists.

schtasks takes the task's whole command line as the single /TR argument, so a
program path containing a space needs quotes of its own inside it. Without them,
every install under C:\\Program Files registered "C:\\Program" as the program and
"Files\\STT\\STT.exe --headless" as its arguments: the task failed at every logon
with 0x80070002, and so did the /Run that restarts STT after a silent update.
Found on a fresh Windows 11 install.
"""

import re
from pathlib import Path

SETUP_ISS = Path(__file__).resolve().parent.parent / "installer" / "windows" / "setup.iss"
APP = r"C:\Program Files\STT"


def windows_argv(cmdline):
    """Split a command line the way CommandLineToArgvW / the MS C runtime do.

    2n backslashes before a quote give n backslashes and toggle quoting; 2n+1
    give n backslashes and a literal quote; other backslashes are literal.
    """
    args, cur, quoted, have, i = [], [], False, False, 0
    while i < len(cmdline):
        c = cmdline[i]
        if c == "\\":
            n = 0
            while i < len(cmdline) and cmdline[i] == "\\":
                n += 1
                i += 1
            if i < len(cmdline) and cmdline[i] == '"':
                cur.append("\\" * (n // 2))
                if n % 2:
                    cur.append('"')
                    i += 1
            else:
                cur.append("\\" * n)
            have = True
            continue
        if c == '"':
            quoted = not quoted
            have = True
        elif c in " \t" and not quoted:
            if have:
                args.append("".join(cur))
                cur, have = [], False
        else:
            cur.append(c)
            have = True
        i += 1
    if have:
        args.append("".join(cur))
    return args


def create_task_argv():
    line = next(ln for ln in SETUP_ISS.read_text(encoding="utf-8").splitlines()
                if ln.startswith('Filename: "schtasks.exe"') and "/Create" in ln)
    params = re.search(r'Parameters: "((?:[^"]|"")*)"', line).group(1)
    return windows_argv(params.replace('""', '"').replace("{app}", APP))


def test_the_task_runs_the_installed_exe():
    argv = create_task_argv()
    tr = argv[argv.index("/TR") + 1]
    # What Task Scheduler stores: the quoted program, then its arguments.
    assert tr == rf'"{APP}\STT.exe" --headless'


def test_the_splitter_matches_the_documented_rules():
    assert windows_argv(r'a "b c" d') == ["a", "b c", "d"]
    assert windows_argv(r'"\"x y\" z"') == ['"x y" z']
    assert windows_argv(r'a\\b') == [r"a\\b"]
