"""Convenience wrapper around ``chart2data generate``."""

from chart2data.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["generate", *__import__("sys").argv[1:]]))
