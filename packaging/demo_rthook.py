"""PyInstaller runtime hook: this build is a demo.

Runs before speech_to_text.py is imported, which is what matters — the module decides
its data directory, its port and whether to start a transcription worker at import
time, and all three depend on this flag.

Set with setdefault so a developer can still force it off to test the same bundle as
a normal server.
"""

import os

os.environ.setdefault("STT_DEMO", "1")
