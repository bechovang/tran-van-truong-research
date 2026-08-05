# -*- coding: utf-8 -*-
"""Generic builder cho cac paper self-contained (py:percent -> clean ipynb + metadata).

Dung cho cac paper ma code/*.py da KAGGLE-READY (install, auto-discover data, image
index, ... deu nam trong .py). Builder chi: parse -> ipynb (giu newline) + bat/tat
SMOKE + viet kernel-metadata.json. KHONG regex-patch (tranh fragility cua #12).

Cach dung:  python code/_build.py 11        # build paper #11
Them paper: bo sung vao CONFIGS.
"""
import io, os, re, uuid, json, ast, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root

def _uncomment_md(ln):
    s = ln; lead = len(s) - len(s.lstrip())
    if s.lstrip().startswith('#'):
        s = s[lead + 1:]
        if s.startswith(' '):
            s = s[1:]
        s = (' ' * lead) + s
    return s

def parse_percent(text):
    cells, cur = [], None
    for ln in text.split('\n'):
        if ln.strip().startswith('# %%'):
            if cur is not None:
                cells.append(cur)
            cur = {'cell_type': 'markdown' if '[markdown]' in ln else 'code', 'lines': []}
        else:
            if cur is None:
                cur = {'cell_type': 'code', 'lines': []}
            cur['lines'].append(ln)
    if cur is not None:
        cells.append(cur)
    out = []
    for c in cells:
        body = c['lines']
        if c['cell_type'] == 'markdown':
            body = [_uncomment_md(x) for x in body]
        while body and body[0].strip() == '':
            body.pop(0)
        while body and body[-1].strip() == '':
            body.pop()
        if not body and c['cell_type'] == 'markdown':
            continue
        out.append({'cell_type': c['cell_type'], 'source': '\n'.join(body)})
    return out

def _to_src_list(text):
    lines = text.split('\n')
    return [l + '\n' for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])

def set_smoke(cells, force_smoke):
    """Thay 'SMOKE = True'/'SMOKE = False'/'SMOKE = os.environ...' thanh bool phu hop."""
    pat = re.compile(r'^SMOKE\s*=\s*.+$', re.MULTILINE)
    for c in cells:
        if c['cell_type'] == 'code' and pat.search(c['source']):
            c['source'] = pat.sub(f'SMOKE = {bool(force_smoke)}  # set by _build.py', c['source'], count=1)
            return True
    return False

def compile_check(cells):
    errs = 0
    for i, c in enumerate(cells):
        if c['cell_type'] != 'code':
            continue
        try:
            ast.parse(c['source'])
        except SyntaxError as e:
            errs += 1
            print(f"  !! SYNTAX cell[{i}]: {e.msg} @ line {e.lineno}")
            print("    ", c['source'][:160].replace('\n', ' | '))
    return errs

def write_notebook(path, cells):
    nbcells = []
    for c in cells:
        cell = {'cell_type': c['cell_type'], 'id': 'cell-' + uuid.uuid4().hex[:8],
                'metadata': {}, 'source': _to_src_list(c['source'])}
        if c['cell_type'] == 'code':
            cell['outputs'] = []; cell['execution_count'] = None
        nbcells.append(cell)
    nb = {'cells': nbcells,
          'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                       'language_info': {'name': 'python'}},
          'nbformat': 4, 'nbformat_minor': 5}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with io.open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    return len(nbcells)

# ---------------- per-paper config ----------------
CONFIGS = {
    "11": {
        "src": "code/11_COCO-Tree_kaggle.py",
        "kernel_dir": "kernels/coco_tree_11",
        "kaggle_id": "bechovang/11-coco-tree",
        "title": "11 COCO-Tree",
        "dataset_sources": ["lyte69/gqa-images"],
        "kernel_sources": ["khoangoo/test-dataset-visual-cot"],
    },
    # "13": {...}, "14": {...}, "15": {...}  -- them sau
}

def build(num, force_smoke=True):
    cfg = CONFIGS[num]
    src = os.path.join(ROOT, cfg["src"])
    kdir = os.path.join(ROOT, cfg["kernel_dir"])
    main = os.path.join(kdir, os.path.basename(src).replace('.py', '.ipynb'))
    push_dir = os.path.join(kdir, "push")
    push = os.path.join(push_dir, os.path.basename(main))

    base = parse_percent(io.open(src, encoding='utf-8').read())
    print(f"=== #{num} parsed {len(base)} cells ===")
    errs = compile_check(base)
    if errs:
        print(f"!! {errs} syntax error(s) -> KHONG build. Sua code/*.py truoc.")
        return errs

    # main = clean source (SMOKE nhu trong .py)
    n_main = write_notebook(main, [dict(c) for c in base])
    # push = SMOKE theo force_smoke
    push_cells = [dict(c) for c in base]
    if not set_smoke(push_cells, force_smoke):
        print("  (canh bao: khong tim thay dong 'SMOKE = ...' de toggle)")
    n_push = write_notebook(push, push_cells)

    meta = {"id": cfg["kaggle_id"], "title": cfg["title"],
            "code_file": os.path.basename(main), "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
            "enable_internet": True,
            "dataset_sources": cfg["dataset_sources"],
            "kernel_sources": cfg["kernel_sources"],
            "machine_shape": "Gpu",
            "docker_image_pinning_type": "latest"}
    os.makedirs(push_dir, exist_ok=True)
    with io.open(os.path.join(push_dir, 'kernel-metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print(f"main -> {main} ({n_main} cells)")
    print(f"push -> {push} ({n_push} cells, FORCE_SMOKE={force_smoke})")
    print("metadata:", json.dumps(meta))
    return 0

if __name__ == '__main__':
    num = sys.argv[1] if len(sys.argv) > 1 else "11"
    force = os.environ.get("FORCE_SMOKE", "1") != "0"   # mac dinh smoke
    if num not in CONFIGS:
        print("paper chua co config:", num); sys.exit(2)
    sys.exit(0 if build(num, force_smoke=force) == 0 else 1)
