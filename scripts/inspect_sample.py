"""Convenience wrapper around ``chart2data inspect``."""

from chart2data.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["inspect", *__import__("sys").argv[1:]]))
