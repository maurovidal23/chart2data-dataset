"""Convenience wrapper around ``chart2data validate``."""

from chart2data.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["validate", *__import__("sys").argv[1:]]))
