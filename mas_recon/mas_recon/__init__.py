"""mas-recon: point it at a running multi-agent system, get back the
real agent topology and silent-failure report from an actual run -- not a
static guess at the source code.

    from mas_recon import start_capture

    handle = start_capture()
    ... run your agent exactly as you already do ...
    handle.stop()   # writes the trace, prints the report
"""

from .capture import CaptureHandle, UnsupportedFrameworkError, detect_framework, start_capture

__all__ = [
    "start_capture",
    "detect_framework",
    "CaptureHandle",
    "UnsupportedFrameworkError",
]

__version__ = "0.1.0"
