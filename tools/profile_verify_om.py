#!/usr/bin/env python3
"""Compatibility entry: the shared OM profiler, defaulting to one verify."""
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profile_om import (
    main as _main, parser as _parser, prompt_and_eos, run_profile, summarize, token_csv,
)


def parser():
    return _parser(default_stage="verify")


def main(argv=None):
    return _main(argv, default_stage="verify")


if __name__ == "__main__":
    raise SystemExit(main())
