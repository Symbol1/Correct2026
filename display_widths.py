#!/usr/bin/env python3
"""Measure how wide each display formula $$ ... $$ is when typeset.

Every display is typeset by pdflatex (amsmath, amssymb, plus the \\def macros found in
the notebook) in a box, and its natural width is reported in em of the 10pt text font.
KaTeX and MathJax use TeX's metrics, so the numbers match the rendered notebook up to
a constant factor. An aligned display is as wide as its widest row.
Displays wider than the limit are candidates for splitting with aligned (see AGENTS.md).
The same script is kept in Correct2026, Complex2026, and Differ2026.

Usage:
    python3 display_widths.py                  # every .ipynb under this folder
    python3 display_widths.py 1104.ipynb       # only the given notebooks
    python3 display_widths.py --max 40         # change the limit (default below)
    python3 display_widths.py --all            # list every display, widest first
Exit status is 1 when some display is wider than the limit.
"""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# Calibrated on displays the user called too long (1104.ipynb, 2026-11-03): 25.5em, 27.3em,
# 29.2em, 33.4em; a 28.1em display was borderline, and the user then confirmed 25em.
# When the user calls a narrower display too long, lower this limit to just
# below its width and add the width here, in all three copies (see AGENTS.md).
DEFAULT_MAX_EM = 25
DISPLAY = re.compile(r'\$\$(.+?)\$\$', re.S)
DEF = re.compile(r'\\def\\[A-Za-z]+(?:#\d)*\{(?:[^{}]|\{[^{}]*\})*\}')


def displays(path):
    """Yield (cell id, heading, LaTeX source) for every display, and the \\def macros."""
    nb = json.loads(path.read_text(encoding='utf-8'))
    found, macros, heading = [], [], ''
    for cell in nb['cells']:
        if cell['cell_type'] != 'markdown':
            continue
        text = ''.join(cell['source'])
        if text.startswith('#'):
            heading = text.lstrip('#').strip()
        macros += DEF.findall(text)
        for m in DISPLAY.finditer(text):
            found.append((cell.get('id', '?'), heading, m.group(1).strip()))
    return found, macros


CJK = re.compile('[　-鿿＀-￯]')


def measure(items, macros):
    """Return (width in em, TeX error or None) for each LaTeX snippet.

    After an error TeX still finishes the box, so the width is only approximate then.
    Chinese characters are measured as 1em blanks, since pdflatex has no CJK font.
    """
    body = []
    for i, tex in enumerate(items):
        tex = CJK.sub(r'\\kern1em{}', tex)
        body.append(f'\\typeout{{BEGIN {i}}}\\setbox0\\hbox{{$\\displaystyle {tex}$}}'
                    f'\\typeout{{WIDTH {i} \\the\\wd0}}')
    document = '\n'.join([r'\documentclass{article}', r'\usepackage{amsmath,amssymb,xcolor}',
                          *dict.fromkeys(macros), r'\begin{document}', *body, r'\end{document}'])
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, 'widths.tex').write_text(document, encoding='utf-8')
        subprocess.run(['pdflatex', '-interaction=nonstopmode', '-draftmode', 'widths.tex'],
                       cwd=tmp, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        log = Path(tmp, 'widths.log').read_text(encoding='utf-8', errors='replace').splitlines()
    results, error = {}, None
    for k, line in enumerate(log):
        if line.startswith('BEGIN '):
            error = None
        elif line.startswith('!') and error is None:
            error = line[1:].strip()
            if error.startswith('Undefined control sequence') and k + 1 < len(log):
                names = re.findall(r'\\[A-Za-z]+', log[k + 1])
                error += f' {names[-1]}' if names else ''
        elif line.startswith('WIDTH '):
            _, i, pt = line.split()
            results[int(i)] = (float(pt[:-2]) / 10, error)
    return [results.get(i, (None, 'not typeset')) for i in range(len(items))]


def main(args):
    limit, show_all, paths = DEFAULT_MAX_EM, False, []
    it = iter(args)
    for a in it:
        if a == '--max':
            limit = float(next(it))
        elif a == '--all':
            show_all = True
        else:
            paths.append(Path(a))
    if not paths:
        here = Path(__file__).resolve().parent
        paths = sorted(p for p in here.rglob('*.ipynb') if '.ipynb_checkpoints' not in p.parts)
    too_wide = 0
    for path in paths:
        found, macros = displays(path)
        if not found:
            continue
        results = measure([tex for _, _, tex in found], macros)
        rows = sorted(zip(results, found), key=lambda r: -(r[0][0] or 0))
        shown = [r for r in rows if show_all or r[0][0] is None or r[0][0] > limit]
        too_wide += sum(1 for (w, _), _ in rows if w is not None and w > limit)
        if shown:
            print(f'{path}:')
        for (width, error), (cell_id, heading, tex) in shown:
            size = '     ?  ' if width is None else f"{'~' if error else ' '}{width:5.1f}em"
            print(f"  {size}  {cell_id:12} {heading[:28]:28}  {' '.join(tex.split())[:70]}")
            if error:
                print(f'            approximate; TeX error: {error[:80]}')
    if not too_wide:
        print(f'No display is wider than {limit:g}em.')
    return 1 if too_wide else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
