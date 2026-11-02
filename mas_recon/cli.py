"""`mas-recon analyze <file>` -- the command a BD person or a prospect's
engineer runs after a capture to see the report. Thin wrapper; all the logic
is in analyzer.run_and_print so capture.py's automatic post-run report and
this command never drift apart."""

from __future__ import annotations

from .analyzer import main as _main


def main() -> None:
    _main()


if __name__ == "__main__":
    main()
