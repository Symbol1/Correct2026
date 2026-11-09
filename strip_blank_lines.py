#!/usr/bin/env python3
"""Remove leading and trailing empty lines from every cell of the lecture notebooks.

AGENTS.md requires that no cell starts or ends with an empty line; blank lines that
separate paragraphs stay only in the middle of a cell. VS Code leaves such lines
behind when a cell is split. The last line of a cell also loses its trailing spaces,
since a Markdown line break (two trailing spaces) means nothing at the end of a cell.
Outputs and metadata are not touched.
The same script is kept in Correct2026, Complex2026, and Differ2026.

Usage:
    python3 strip_blank_lines.py               # fix every .ipynb under this folder
    python3 strip_blank_lines.py 1104.ipynb    # fix the given notebooks
    python3 strip_blank_lines.py --check       # report only; exit 1 if anything needs fixing
"""
import json
import sys
from pathlib import Path


def strip_cell(text):
    lines = text.split('\n')
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if lines:
        lines[-1] = lines[-1].rstrip()
    return '\n'.join(lines)


def fix(nb):
    """Return a list of the changes needed; apply them to nb in place."""
    changes = []
    for index, cell in enumerate(nb['cells']):
        text = ''.join(cell['source'])
        stripped = strip_cell(text)
        if stripped == text:
            continue
        lines = stripped.split('\n')
        cell['source'] = [line + '\n' for line in lines[:-1]] + [lines[-1]] if stripped else []
        first = stripped.split('\n')[0][:50] if stripped else '(empty cell)'
        changes.append(f"cell {index} ({cell.get('id', '?')}): {first}")
    return changes


def main(args):
    check = '--check' in args
    paths = [Path(a) for a in args if a != '--check']
    if not paths:
        here = Path(__file__).resolve().parent
        paths = sorted(p for p in here.rglob('*.ipynb') if '.ipynb_checkpoints' not in p.parts)
    needs_fix = False
    for path in paths:
        nb = json.loads(path.read_text(encoding='utf-8'))
        changes = fix(nb)
        if not changes:
            continue
        needs_fix = True
        print(f"{path}: {len(changes)} cell{'s' if len(changes) > 1 else ''}{' to fix' if check else ' fixed'}")
        for change in changes:
            print('  ' + change)
        if not check:
            # same layout as Jupyter and VS Code: one-space indent, UTF-8, trailing newline
            path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    if not needs_fix:
        print('No cell starts or ends with an empty line.')
    return 1 if check and needs_fix else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
