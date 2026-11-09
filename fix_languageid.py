#!/usr/bin/env python3
"""Keep Python syntax highlighting in the lecture notebooks.

Every code cell gets metadata.vscode.languageId = "python", and the notebook's
metadata.language_info.name is kept at "python". The kernelspec is left alone,
so a notebook on the SageMath kernel ("language": "sage") still runs in Sage.
VS Code drops languageId when it saves, and then falls back to the kernelspec's "sage",
which has no highlighting; run this after every edit.
The same script is kept in Correct2026, Complex2026, and Differ2026.

Usage:
    python3 fix_languageid.py               # fix every .ipynb under this folder
    python3 fix_languageid.py 1104.ipynb    # fix the given notebooks
    python3 fix_languageid.py --check       # report only; exit 1 if anything needs fixing
"""
import json
import sys
from pathlib import Path


def fix(nb):
    """Return a list of the changes needed; apply them to nb in place."""
    changes = []
    for index, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        vscode = cell.setdefault('metadata', {}).setdefault('vscode', {})
        if vscode.get('languageId') != 'python':
            changes.append(f"cell {index} ({cell.get('id', '?')}): languageId {vscode.get('languageId')!r} -> 'python'")
            vscode['languageId'] = 'python'
    language_info = nb.setdefault('metadata', {}).setdefault('language_info', {})
    if language_info.get('name') != 'python':
        changes.append(f"language_info.name {language_info.get('name')!r} -> 'python'")
        language_info['name'] = 'python'
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
        print(f"{path}: {len(changes)} change{'s' if len(changes) > 1 else ''}{' needed' if check else ''}")
        for change in changes:
            print('  ' + change)
        if not check:
            # same layout as Jupyter and VS Code: one-space indent, UTF-8, trailing newline
            path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    if not needs_fix:
        print('All code cells have languageId "python".')
    return 1 if check and needs_fix else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
