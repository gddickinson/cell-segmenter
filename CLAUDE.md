# Claude collaboration notes for `cell-segmenter`

This project was built with substantial assistance from **Claude (Anthropic)**, primarily via **Claude Code**.

## Global preferences

George's global Claude preferences live in `~/.claude/CLAUDE.md` and apply to every project. Highlights:

- Keep all source files under 500 lines; split into focused modules if approaching that limit.
- For Python projects, maintain an `INTERFACE.md` at the project root as a navigation map of modules, classes, and key functions.
- Prefer many small, focused files over fewer large ones.
- Group related functionality; extract reusable logic into utility modules.

## Project navigation

See [`INTERFACE.md`](./INTERFACE.md) for a navigation map of this repository's modules and how they connect. Read `INTERFACE.md` before opening individual source files.

## Working with Claude on this repo

When directing Claude Code on this codebase, follow the global rules above, prefer making targeted edits via the `Edit` tool over rewriting whole files, and update `INTERFACE.md` whenever the project structure changes.

Project-specific conventions, build instructions, and lab-specific notes should be added below as the project evolves.

@INTERFACE.md

## Tests and web app
- Tests: `python -m pytest -q tests` (Qt tests run offscreen; no PyTorch needed except for the CNN itself).
- Web app: `python run_web.py` (port 8146). Static site: `python scripts/build_static.py --out site`.
- After changing `models/features.py`, `utils/object_stats.py` or `utils/synthetic_stack.py`, run
  `python scripts/make_web_fixtures.py` and the node tests in `cell_segmenter/web/static/*.test.mjs`: the JS ports
  must keep matching Python.
- Keep every file under 500 lines; update INTERFACE.md and SESSION_LOG.md.
