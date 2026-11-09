#!/usr/bin/env python3
"""Measure how wide each rendered line of Markdown text is, outside display formulas.

A rendered line ends at a hard break (two trailing spaces), a blank line, a display
$$ ... $$, a list item, or the end of the cell; plain newlines only join source lines.
The text of each line is typeset by pdflatex (10pt font, amsmath, amssymb, plus the \\def
macros found in the notebook) and its width is reported in em. Inline math counts 1.21
times its TeX width, because KaTeX and MathJax scale math to 1.21em of the text font.
Headings, tables, and images are skipped. Chinese characters count as 1em.
Lines wider than the limit are candidates for breaking (see AGENTS.md).

With --cells, the script instead estimates how many text lines tall each Markdown cell
renders, since that is how far j and k scroll: each rendered line counts 1, a blank line
between paragraphs 0.5, and a display 1 for its margins plus 1 per row, or 2 for a row
with a fraction, an integral, a sum, a limit, a matrix, or big delimiters.
Headings are skipped; images are not counted. Taller cells are candidates for splitting.
The same script is kept in Correct2026, Complex2026, and Differ2026.

Usage:
    python3 line_widths.py                  # every .ipynb under this folder
    python3 line_widths.py 1106.ipynb       # only the given notebooks
    python3 line_widths.py --max 35         # change the limit (default below)
    python3 line_widths.py --all            # list every line, widest first
    python3 line_widths.py --cells          # list Markdown cells taller than the limit
Exit status is 1 when some line (or cell) exceeds the limit.
"""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# Calibrated on the lines the user broke by hand in Complex2026/1106.ipynb (2026-11-05):
# every line wider than 34em was broken and every line up to 27em was kept;
# between 30em and 33em only lines with a natural break point were broken.
# When the user calls a narrower line too long, lower this limit to just below its width
# and add the width here, in all three copies (see AGENTS.md).
DEFAULT_MAX_EM = 30
# Markdown cells should be about five lines tall (AGENTS.md). In Differ2026/1104.ipynb,
# which the user cut to that rule, the median cell is 5.5 lines and most are 3 to 7.
DEFAULT_MAX_LINES = 7
MATH_SCALE = 1.21
TALL = re.compile(r'\\(?:d?frac|int|iint|oint|sum|prod|lim|binom|Bigg?[lr]|bigg[lr])')
INNER = re.compile(r'\\begin\{([pbvBV]?matrix|cases)\}(.*?)\\end\{\1\}', re.S)
DISPLAY = re.compile(r'\$\$(.+?)\$\$', re.S)
INLINE = re.compile(r'(?<!\\)\$(.+?)(?<!\\)\$')
DEF = re.compile(r'\\def\\[A-Za-z]+(?:#\d)*\{(?:[^{}]|\{[^{}]*\})*\}')
LIST = re.compile(r'\s*(?:[-*+]|\d+[.)])\s+')
CJK = re.compile('[　-鿿＀-￯]')
PUNCT = {'–': '--', '—': '---', '‘': '`', '’': "'", '“': '``', '”': "''", '…': '...',
         '×': r'$\times$', '°': r'$^\circ$', '→': r'$\to$', '⇒': r'$\Rightarrow$'}


def rendered_lines(text):
    """Split the text of a Markdown cell into the lines it renders as."""
    if text.lstrip().startswith('#'):
        return []
    text = DISPLAY.sub('\n\x00\n', text)
    lines, current = [], []
    for raw in text.split('\n'):
        stripped = raw.strip()
        if raw == '\x00' or not stripped or stripped.startswith(('|', '<img', '![')):
            if current:
                lines.append(' '.join(current))
            current = []
            continue
        if LIST.match(raw) and current:
            lines.append(' '.join(current))
            current = []
        current.append(LIST.sub('', raw, count=1).strip() if LIST.match(raw) else stripped)
        if raw.endswith('  ') or raw.endswith('<br>'):
            lines.append(' '.join(current))
            current = []
    if current:
        lines.append(' '.join(current))
    return [line for line in lines if line]


def text_to_tex(text):
    """Turn Markdown text without math into TeX that has about the same width."""
    text = re.sub(r'!?\[([^\]]*)\]\([^)]*\)', r'\1', text)
    text = re.sub(r'<[^>]+>', '', text)
    out = []
    for part in re.split(r'(\*\*[^*]+\*\*|`[^`]+`)', text):
        bold, code = part.startswith('**') and len(part) > 4, part.startswith('`') and len(part) > 2
        if bold:
            part = part[2:-2]
        elif code:
            part = part[1:-1]
        part = part.replace('\\', '\x01')
        for a, b in [('{', r'\{'), ('}', r'\}'), ('&', r'\&'), ('%', r'\%'), ('#', r'\#'),
                     ('_', r'\_'), ('^', r'\^{}'), ('~', r'\~{}'), ('*', ''), ('\x01', r'\textbackslash{}')]:
            part = part.replace(a, b)
        for a, b in PUNCT.items():
            part = part.replace(a, b)
        part = CJK.sub(r'\\kern1em{}', part)
        part = re.sub('[^\x00-\x7f]', 'o', part)
        out.append(r'\textbf{%s}' % part if bold else r'\texttt{%s}' % part if code else part)
    return ''.join(out)


def lines_of(path):
    """Yield (cell id, line) for every rendered Markdown line, and the \\def macros."""
    nb = json.loads(path.read_text(encoding='utf-8'))
    found, macros = [], []
    for cell in nb['cells']:
        if cell['cell_type'] != 'markdown':
            continue
        text = ''.join(cell['source'])
        macros += DEF.findall(text)
        found += [(cell.get('id', '?'), line) for line in rendered_lines(text)]
    return found, macros


def cell_height(text):
    """Estimate how many text lines tall a Markdown cell renders."""
    if text.lstrip().startswith('#'):
        return 0
    height = len(rendered_lines(text)) + 0.5 * len(re.findall(r'\n[ \t]*\n', text.strip()))
    for m in DISPLAY.finditer(text):
        body = re.sub(r'\\(?:begin|end)\{(?:aligned|gathered)\}', '', m.group(1))
        # A matrix or cases block is as tall as its number of rows; it is not a row break.
        body = INNER.sub(lambda e: f'\\INNER{{{e.group(2).count(chr(92) * 2) + 1}}}', body)
        for row in body.split('\\\\'):
            inner = [int(n) for n in re.findall(r'\\INNER\{(\d+)\}', row)]
            height += max([2 if TALL.search(row) else 1] + inner)
        height += 1
    return height


def report_cells(paths, limit, show_all):
    """Print Markdown cells taller than the limit; return how many there are."""
    too_tall = 0
    for path in paths:
        nb = json.loads(path.read_text(encoding='utf-8'))
        rows = []
        for cell in nb['cells']:
            text = ''.join(cell['source'])
            if cell['cell_type'] == 'markdown' and not text.lstrip().startswith('#'):
                rows.append((cell_height(text), cell.get('id', '?'), text.split('\n')[0]))
        shown = sorted((r for r in rows if show_all or r[0] > limit), key=lambda r: -r[0])
        too_tall += sum(1 for r in rows if r[0] > limit)
        if shown:
            print(f'{path}:')
        for height, cell_id, first in shown:
            print(f'  {height:4.1f} lines  {cell_id[:24]:24}  {first[:80]}')
    if not too_tall:
        print(f'No Markdown cell is taller than {limit:g} lines.')
    return too_tall


def measure(lines, macros):
    """Return (width in em, TeX error or None) for each Markdown line."""
    body = []
    for i, line in enumerate(lines):
        maths = INLINE.findall(line)
        text = text_to_tex(INLINE.sub('', line))
        math = ''.join('$' + CJK.sub(r'\\kern1em{}', m) + '$' for m in maths)
        body.append(f'\\typeout{{BEGIN {i}}}\\setbox0\\hbox{{{text}}}\\setbox2\\hbox{{{math}}}'
                    f'\\typeout{{WIDTH {i} \\the\\wd0 \\space\\the\\wd2}}')
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
            _, i, text_pt, math_pt = line.split()
            width = float(text_pt[:-2]) / 10 + MATH_SCALE * float(math_pt[:-2]) / 10
            results[int(i)] = (width, error)
    return [results.get(i, (None, 'not typeset')) for i in range(len(lines))]


def main(args):
    limit, show_all, cells, paths = None, False, False, []
    it = iter(args)
    for a in it:
        if a == '--max':
            limit = float(next(it))
        elif a == '--all':
            show_all = True
        elif a == '--cells':
            cells = True
        else:
            paths.append(Path(a))
    if not paths:
        here = Path(__file__).resolve().parent
        paths = sorted(p for p in here.rglob('*.ipynb') if '.ipynb_checkpoints' not in p.parts)
    if cells:
        return 1 if report_cells(paths, DEFAULT_MAX_LINES if limit is None else limit, show_all) else 0
    limit = DEFAULT_MAX_EM if limit is None else limit
    too_wide = 0
    for path in paths:
        found, macros = lines_of(path)
        if not found:
            continue
        results = measure([line for _, line in found], macros)
        rows = sorted(zip(results, found), key=lambda r: -(r[0][0] or 0))
        shown = [r for r in rows if show_all or r[0][0] is None or r[0][0] > limit]
        too_wide += sum(1 for (w, _), _ in rows if w is not None and w > limit)
        if shown:
            print(f'{path}:')
        for (width, error), (cell_id, line) in shown:
            size = '     ?  ' if width is None else f"{'~' if error else ' '}{width:5.1f}em"
            print(f'  {size}  {cell_id[:24]:24}  {line[:90]}')
            if error:
                print(f'            approximate; TeX error: {error[:80]}')
    if not too_wide:
        print(f'No line is wider than {limit:g}em.')
    return 1 if too_wide else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
