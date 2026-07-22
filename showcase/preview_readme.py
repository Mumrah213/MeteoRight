#!/usr/bin/env python3
"""Render a Markdown file exactly as GitHub will, into a local HTML file.

Uses GitHub's own /markdown API (via `gh`), so the result is the real
renderer — not an approximation. Local images are inlined as data URIs and
GitHub's README stylesheet is embedded, so the output opens offline and
looks like the repo page.

Usage:
    python showcase/preview_readme.py [FILE] [-o OUT] [--open]

Requires `gh` to be authenticated (`gh auth status`).
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import mimetypes
import re
import subprocess
import sys
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "README.md"
DEFAULT_OUTPUT = REPO_ROOT / "readme-preview.html"

# GitHub renders READMEs in a ~1012px column with this type stack.
PAGE_CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body {
  margin: 0; padding: 32px 16px 96px;
  background: var(--page);
  color: var(--fg);
  font: 16px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans",
        Helvetica, Arial, sans-serif;
}
:root {
  --page: #ffffff; --fg: #1f2328; --muted: #59636e; --border: #d1d9e0;
  --canvas-subtle: #f6f8fa; --link: #0969da; --accent: #0969da;
}
@media (prefers-color-scheme: dark) {
  :root {
    --page: #0d1117; --fg: #e6edf3; --muted: #9198a1; --border: #3d444d;
    --canvas-subtle: #151b23; --link: #4493f8; --accent: #4493f8;
  }
}
.wrap { max-width: 1012px; margin: 0 auto; }
.filebar {
  border: 1px solid var(--border); border-bottom: 0;
  border-radius: 6px 6px 0 0; padding: 16px;
  background: var(--page); font-weight: 600; font-size: 14px;
  display: flex; align-items: center; gap: 8px;
}
.filebar .hint { font-weight: 400; color: var(--muted); margin-left: auto; font-size: 12px; }
.markdown-body {
  border: 1px solid var(--border); border-radius: 0 0 6px 6px;
  padding: 32px; background: var(--page);
}
.markdown-body > *:first-child { margin-top: 0 !important; }
.markdown-body h1, .markdown-body h2 {
  padding-bottom: .3em; border-bottom: 1px solid var(--border); margin: 24px 0 16px;
  font-weight: 600; line-height: 1.25;
}
.markdown-body h1 { font-size: 2em; } .markdown-body h2 { font-size: 1.5em; }
.markdown-body h3 { font-size: 1.25em; font-weight: 600; margin: 24px 0 16px; }
.markdown-body p, .markdown-body ul, .markdown-body ol, .markdown-body blockquote,
.markdown-body table, .markdown-body pre { margin: 0 0 16px; }
.markdown-body ul, .markdown-body ol { padding-left: 2em; }
.markdown-body li + li { margin-top: .25em; }
.markdown-body a { color: var(--link); text-decoration: none; }
.markdown-body a:hover { text-decoration: underline; }
.markdown-body code:not(pre code) {
  background: var(--canvas-subtle); padding: .2em .4em; border-radius: 6px;
  font-size: 85%; font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, monospace;
}
.markdown-body pre {
  background: var(--canvas-subtle); padding: 16px; border-radius: 6px;
  overflow: auto; font-size: 85%; line-height: 1.45;
}
.markdown-body pre code {
  font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, monospace;
  background: none; padding: 0;
}
.markdown-body img { max-width: 100%; border-radius: 6px; }
.markdown-body table { border-collapse: collapse; display: block; overflow: auto; width: max-content;
  max-width: 100%; }
.markdown-body th, .markdown-body td { border: 1px solid var(--border); padding: 6px 13px; }
.markdown-body tr:nth-child(2n) { background: var(--canvas-subtle); }
.markdown-body blockquote {
  padding: 0 1em; color: var(--muted); border-left: .25em solid var(--border);
}
.markdown-body hr { height: .25em; background: var(--border); border: 0; margin: 24px 0; }
"""


def render_via_github(text: str, repo: str | None, *, gfm: bool = False) -> str:
    """Render Markdown through GitHub's API using the gh CLI.

    Defaults to ``mode=markdown``, which is how GitHub renders a README file.
    ``mode=gfm`` is the *comment* flavour: it turns every single newline into
    a hard ``<br>``, so hard-wrapped prose renders with breaks a README would
    not show. Pass ``gfm=True`` only to preview text destined for an issue or
    pull-request comment.
    """
    payload: dict[str, str] = {"text": text, "mode": "gfm" if gfm else "markdown"}
    if gfm and repo:
        payload["context"] = repo

    result = subprocess.run(
        ["gh", "api", "-X", "POST", "/markdown", "--input", "-"],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "gh api /markdown failed")
    return result.stdout


def current_repo() -> str | None:
    """Return owner/name for the current repo, or None if unavailable."""
    result = subprocess.run(
        ["gh", "repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or None if result.returncode == 0 else None


def inline_images(rendered: str, base_dir: Path) -> tuple[str, int, list[str]]:
    """Replace relative <img src> with data URIs so the page works offline."""
    inlined = 0
    missing: list[str] = []

    def replace(match: re.Match[str]) -> str:
        nonlocal inlined
        src = match.group(1)
        if src.startswith(("http://", "https://", "data:")):
            return match.group(0)

        path = (base_dir / src).resolve()
        if not path.is_file():
            missing.append(src)
            return match.group(0)

        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        inlined += 1
        return f'src="data:{mime};base64,{encoded}"'

    return re.sub(r'src="([^"]+)"', replace, rendered), inlined, missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("file", nargs="?", default=DEFAULT_INPUT, type=Path)
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT, type=Path)
    parser.add_argument("--open", action="store_true", help="Open in the default browser")
    parser.add_argument(
        "--comment",
        action="store_true",
        help=(
            "Render in GitHub's comment flavour (mode=gfm): single newlines "
            "become hard breaks and #123 autolinks. Use for issue/PR text, "
            "not for a README."
        ),
    )
    args = parser.parse_args()

    source: Path = args.file
    if not source.is_file():
        print(f"No such file: {source}", file=sys.stderr)
        return 1

    repo = current_repo() if args.comment else None
    try:
        rendered = render_via_github(source.read_text(), repo, gfm=args.comment)
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"GitHub render failed: {exc}", file=sys.stderr)
        print("Is `gh` installed and authenticated? Try: gh auth status", file=sys.stderr)
        return 1

    rendered, inlined, missing = inline_images(rendered, source.parent)

    page = (
        "<!doctype html>\n<html><head><meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{html.escape(source.name)} — preview</title>"
        f"<style>{PAGE_CSS}</style></head><body><div class='wrap'>"
        f"<div class='filebar'>{html.escape(source.name)}"
        f"<span class='hint'>rendered by GitHub · "
        f"{'comment flavour (gfm)' if args.comment else 'README flavour'}</span></div>"
        f"<article class='markdown-body'>{rendered}</article>"
        "</div></body></html>\n"
    )
    args.output.write_text(page)

    print(f"Wrote {args.output}  ({inlined} image(s) inlined)")
    if missing:
        print(f"  Missing images ({len(missing)}):", file=sys.stderr)
        for src in missing:
            print(f"    {src}", file=sys.stderr)
    if args.open:
        webbrowser.open(args.output.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
