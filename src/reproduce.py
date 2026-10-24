"""Execute both notebooks and write the HTML pages. Network is not used."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import nbformat
from bs4 import BeautifulSoup
from nbclient import NotebookClient
from nbconvert import HTMLExporter

PAGES = (
    ("kansalaisaloite_en.ipynb", "en", "Daily signatures on kansalaisaloite.fi"),
    ("kansalaisaloite.ipynb", "fi", "Verkkoallekirjoitukset palvelussa kansalaisaloite.fi"),
)


def _finish(raw: str, lang: str, title: str) -> str:
    page = BeautifulSoup(raw, "html.parser")
    if page.html is not None:
        page.html["lang"] = lang
    if page.title is not None:
        page.title.string = title
    return str(page)


def _execute(root: Path, name: str, lang: str, title: str, publish: bool) -> None:
    notebook = nbformat.read(root / name, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    NotebookClient(
        notebook,
        timeout=2400,
        kernel_name="kansalaisaloite-analysis",
        resources={"metadata": {"path": str(root)}},
    ).execute()
    out = root / "reports"
    out.mkdir(exist_ok=True)
    stem = name.removesuffix(".ipynb")
    nbformat.write(notebook, out / name)
    exporter = HTMLExporter(template_name="lab")
    exporter.exclude_input = True
    html, _resources = exporter.from_notebook_node(notebook)
    page = _finish(html, lang, title)
    (out / f"{stem}.html").write_text(page, encoding="utf-8")
    if publish:
        nbformat.write(notebook, root / name)
        (root / f"{stem}.html").write_text(page, encoding="utf-8")
    print(f"Wrote {out / name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boot", type=int, default=None, help="Initiative-cluster bootstrap draws. Default is the analysis default.")
    parser.add_argument("--no-publish", action="store_true", help="Write only under reports/.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    if args.boot is not None:
        os.environ["ANALYSIS_BOOT"] = str(args.boot)
    subprocess.check_call(
        [sys.executable, "-m", "ipykernel", "install", "--sys-prefix", "--name", "kansalaisaloite-analysis"],
        cwd=root,
    )
    for name, lang, title in PAGES:
        _execute(root, name, lang, title, publish=not args.no_publish)


if __name__ == "__main__":
    main()
