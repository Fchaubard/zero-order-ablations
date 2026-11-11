#!/usr/bin/env python3
"""Render paper figures as cropped PNGs for the README and the blog post (assets/figure1.png, assets/figure2.png).

    python scripts/readme_figures.py            # needs pdflatex and PyMuPDF (pip install pymupdf)

Each figure is compiled alone with the paper's preamble (so fonts, colours and data are the paper's), then
cropped to its drawn content at 300 dpi on a white background.
"""
import os
import subprocess
import sys
import tempfile

import pymupdf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "paper")
FIGURES = {"figure1.png": "figures/fig_overall", "figure2.png": "figures/fig_scale"}


def render(src: str, out: str, tmp: str):
    tex = open(os.path.join(PAPER, "main.tex")).read()
    preamble = tex[: tex.index("\\begin{document}")]
    doc = preamble + "\\pagestyle{empty}\n\\begin{document}\n\\begin{figure}[t]\\centering\\input{" + src + "}\\end{figure}\n\\end{document}\n"
    path = os.path.join(tmp, "fig.tex")
    open(path, "w").write(doc)
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "-output-directory", tmp, path],
                   cwd=PAPER, check=True, stdout=subprocess.DEVNULL)
    page = pymupdf.open(os.path.join(tmp, "fig.pdf"))[0]
    r = None
    for d in page.get_drawings():
        r = d["rect"] if r is None else r | d["rect"]
    for b in page.get_text("dict")["blocks"]:
        for line in b.get("lines", []):
            for s in line["spans"]:
                if s["text"].strip():
                    r = r | pymupdf.Rect(s["bbox"])
    r = pymupdf.Rect(r.x0 - 6, r.y0 - 6, r.x1 + 6, r.y1 + 6)
    page.get_pixmap(clip=r, dpi=300, alpha=False).save(out)
    print("wrote", out)


def main():
    os.makedirs(os.path.join(ROOT, "assets"), exist_ok=True)
    names = sys.argv[1:] or list(FIGURES)
    with tempfile.TemporaryDirectory() as tmp:
        for name in names:
            render(FIGURES[name], os.path.join(ROOT, "assets", name), tmp)


if __name__ == "__main__":
    main()
