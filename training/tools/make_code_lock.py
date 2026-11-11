"""Regenerate audit/code_lock.json after an INTENDED code change (then rerun verify.py and all tests).

Run from the package root:  python -B tools/make_code_lock.py
Any later edit to a locked file makes run.py/verify.py refuse to start.
"""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda:f.read(8*1024*1024),b''):h.update(c)
    return h.hexdigest()
files=sorted(p for p in ROOT.rglob('*.py') if 'results' not in p.relative_to(ROOT).parts and 'original_code' not in p.relative_to(ROOT).parts and '__pycache__' not in p.parts)
files+=[ROOT/'config.json']+sorted(ROOT.glob('*.sbatch'))+[ROOT/'README.md']
lock={p.relative_to(ROOT).as_posix():sha(p) for p in files if p.is_file()}
(ROOT/'audit/code_lock.json').write_text(json.dumps(lock,indent=2),encoding='utf-8')
print('locked %d files'%len(lock))
