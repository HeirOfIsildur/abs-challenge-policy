"""Submission lint: no internal or meta text may reach a compiled PDF.

usage (from paper/): python lint.py [--submission]
Checks the text of main.pdf and supplement.pdf (pdftotext) and their metadata (pdfinfo):
  * meta words: referee, reviewer, first pass, kernel, HONZA-, draft, working paper, pending, TODO, ??;
  * software version strings (Mathematica 15.0, $Version, Mac OS X, ...) outside the main text's Reproducibility
    section, which names the Mathematica version once;
  * PDF title and author set and free of draft markers.
--submission also fails on any \\todo left in main.tex or supplement.tex.
Exit status 1 on any hit, with the offending lines printed."""
import re, subprocess, sys

META = re.compile(r"referee|reviewer|first[ -]pass|kernel|HONZA-|\bdraft\b|working paper|\bpending\b|\bTODO\b|\?\?", re.I)
VERSION = re.compile(r"Mathematica\s*\d|\$Version|Mac OS X|\b(?:Python|JAX|NumPy)\s+\d+\.\d+", re.I)


def text(pdf):
    return subprocess.run(["pdftotext", "-layout", pdf, "-"], check=True, capture_output=True, text=True).stdout


def info(pdf):
    out = subprocess.run(["pdfinfo", pdf], check=True, capture_output=True, text=True).stdout
    return dict(l.split(":", 1) for l in out.splitlines() if ":" in l)


def repro_span(t):
    """Character span of the main text's Reproducibility section: from its heading to the references."""
    a = re.search(r"^\s*\d+\.\s+Reproducibility", t, re.M | re.I)
    b = re.search(r"^\s*References", t[a.end():], re.M | re.I) if a else None
    return (a.start(), a.end() + b.start()) if a and b else (0, 0)


def main():
    bad = []
    for pdf in ("main.pdf", "supplement.pdf"):
        t = text(pdf)
        lo, hi = repro_span(t) if pdf == "main.pdf" else (0, 0)
        if pdf == "main.pdf" and hi == 0:
            bad.append(f"{pdf}: Reproducibility section not found")
        for i, line in enumerate(t.splitlines(), 1):
            if META.search(line):
                bad.append(f"{pdf}:{i}: meta text: {line.strip()}")
        inside = 0
        for mt in VERSION.finditer(t):
            if lo <= mt.start() < hi:
                inside += 1
            else:
                bad.append(f"{pdf}: version string outside Reproducibility: {t[max(0, mt.start() - 40):mt.end() + 20]!r}")
        if inside > 1:
            bad.append(f"{pdf}: the Reproducibility section names a software version {inside} times")
        md = info(pdf)
        for k in ("Title", "Author"):
            if not md.get(k, "").strip():
                bad.append(f"{pdf}: PDF {k} is empty")
        for k, v in md.items():
            if META.search(v):
                bad.append(f"{pdf}: PDF metadata {k}: {v.strip()}")
    if "--submission" in sys.argv:
        for tex in ("main.tex", "supplement.tex"):
            for i, line in enumerate(open(tex, encoding="utf-8"), 1):
                if "\\todo" in line:
                    bad.append(f"{tex}:{i}: \\todo left: {line.strip()}")
    print("\n".join(bad) if bad else "lint: main.pdf and supplement.pdf are clean")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
