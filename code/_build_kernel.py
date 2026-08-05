# -*- coding: utf-8 -*-
"""Build clean (main + push) #12 LLaVA-CoT notebooks from the py:percent source.

Source of truth : code/12_LLaVA-CoT_kaggle.py   (cells = '# %% [markdown]' / '# %%')
Output:
  kernels/llava_cot_12/12_LLaVA-CoT_kaggle.ipynb        <- ban goc (clean, chay duoc)
  kernels/llava_cot_12/push/12_LLaVA-CoT_kaggle.ipynb   <- + hardening cho Kaggle
  kernels/llava_cot_12/push/kernel-metadata.json

Root-cause fix: cac lan convert .py -> .ipynb truoc day BI MAT NEWLINE
("randomimport torch" -> SyntaxError). Parse truc tiep tu .py (giu nguyen
newline) -> khong con loi do nua.
"""
import io, os, re, uuid, json

SRC     = 'code/12_LLaVA-CoT_kaggle.py'
MAIN    = 'kernels/llava_cot_12/12_LLaVA-CoT_kaggle.ipynb'
PUSH_DIR = 'kernels/llava_cot_12/push'
PUSH    = os.path.join(PUSH_DIR, '12_LLaVA-CoT_kaggle.ipynb')

FORCE_SMOKE = True   # True: push chay SMOKE (n=4, 2 steps) de bat bug truoc. False: full run.

KAGGLE_ID = "bechovang/12-llava-cot"

# Dataset slugs (owner/name) de dua vao dataset_sources cua kernel-metadata.
# NOTE: bat buoc phai CO de `kaggle kernels push` attach dung input.
#   - VISUAL_COT_DATASET : dataset chua gqa_cot_*.jsonl (data reasoning)
#   - GQA_IMAGES_DATASET : dataset chua images/<id>.jpg (private)
# Notebook van chay duoc nho auto-discover (rglob /kaggle/input) bat ke slug,
# nhung day len Kaggle can slug dung o day.
# Inputs (verified tu server metadata cua kernel bechovang/12-llava-cot, 2026-08-06):
#   - visual-cot data = OUTPUT kernel khoangoo/test-dataset-visual-cot  -> kernel_sources
#     (mount /kaggle/input/test-dataset-visual-cot/visual-cot/cot_with_detailed_reasoning_steps/...)
#   - GQA images     = dataset lyte69/gqa-images                        -> dataset_sources
#     (mount /kaggle/input/gqa-images/images/<id>.jpg)
VISUAL_COT_KERNEL = "khoangoo/test-dataset-visual-cot"   # kernel_sources
GQA_IMAGES_DATASET = "lyte69/gqa-images"                  # dataset_sources

# ---------------- percent parser ----------------
def _uncomment_md(ln):
    """# ## heading  ->  ## heading   (chi boc 1 dau '#')."""
    s = ln
    lead = len(s) - len(s.lstrip())
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
            if cur is None:                      # khoi header truoc marker dau
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

# ---------------- Kaggle hardening (chi cho ban push) ----------------
HARDEN_DOWNLOAD = (
    'def _download_jsonl(filename):\n'
    '    """Tai 1 file jsonl tu HF deepcs233/Visual-CoT ve DATA_OUT_DIR (retry + mirror fallback)."""\n'
    '    import time as _t, os as _os, socket as _s\n'
    '    last = None\n'
    '    for attempt in range(6):\n'
    '        try:\n'
    '            return hf_hub_download(repo_id=DATASET_HF, repo_type="dataset",\n'
    '                                   filename=filename, local_dir=DATA_OUT_DIR)\n'
    '        except Exception as e:\n'
    '            last = e\n'
    '            print(f"[retry {attempt+1}/6] {filename}: {repr(e)[:100]}")\n'
    '            try:\n'
    '                print(f"  dns hf: {_s.gethostbyname(\'huggingface.co\')}")\n'
    '            except Exception as e2:\n'
    '                print(f"  dns failed: {e2!r}")\n'
    '            _t.sleep(20 * (attempt + 1))\n'
    '    # Mirror fallback (neu huggingface.co bi chan / loi DNS he thong)\n'
    '    _os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"\n'
    '    try:\n'
    '        import huggingface_hub.constants as _hc\n'
    '        _hc.ENDPOINT = "https://hf-mirror.com"\n'
    '        _hc.HF_HUB_ENDPOINT = "https://hf-mirror.com"\n'
    '    except Exception:\n'
    '        pass\n'
    '    for attempt in range(3):\n'
    '        try:\n'
    '            return hf_hub_download(repo_id=DATASET_HF, repo_type="dataset",\n'
    '                                   filename=filename, local_dir=DATA_OUT_DIR)\n'
    '        except Exception as e:\n'
    '            last = e\n'
    '            print(f"[mirror {attempt+1}/3] {filename}: {repr(e)[:100]}")\n'
    '            _t.sleep(15)\n'
    '    raise RuntimeError(f"HF download failed: {filename}: {last!r}")\n'
    '    return path'
)

IMG_DL_CELL_SRC = (
    "# ---- GQA images: UU TIEN input da attach, fallback kagglehub, luon xay GQA_IMAGE_INDEX ----\n"
    "from pathlib import Path\n"
    "\n"
    "# (a) Script download nhe tu HF neu co san trong input (khoangoo/test-data)\n"
    "dvc = list(Path('/kaggle/input').rglob('download_visual_cot.py'))\n"
    "if dvc:\n"
    "    import subprocess\n"
    "    print('Found download_visual_cot.py:', dvc[0])\n"
    "    subprocess.run(['python', str(dvc[0]), '--mode', 'light'], check=False)\n"
    "else:\n"
    "    print('download_visual_cot.py not found -> bo qua (lay anh tu input/kagglehub)')\n"
    "\n"
    "# (b) Fallback: kagglehub lyte69/gqa-images (can Internet; bo qua neu da co input anh)\n"
    "try:\n"
    "    import kagglehub\n"
    "    gqa_dir = kagglehub.dataset_download('lyte69/gqa-images')\n"
    "    print('kagglehub gqa-images ->', gqa_dir)\n"
    "except Exception as e:\n"
    "    print('kagglehub gqa-images skipped/failed:', repr(e)[:150])\n"
    "\n"
    "# (c) Xay index filename -> path (scan toan bo /kaggle/input + working + cache)\n"
    "#     Day la nguon chinh: bat dataset GQA-images (private) bat ke slug.\n"
    "GQA_IMAGE_INDEX = {}\n"
    "for root in ('/kaggle/input', '/kaggle/working/visual-cot',\n"
    "             '/kaggle/working', '/root/.cache/kagglehub'):\n"
    "    rp = Path(root)\n"
    "    if not rp.exists():\n"
    "        continue\n"
    "    for ext in ('*.jpg', '*.jpeg', '*.png'):\n"
    "        for p in rp.rglob(ext):\n"
    "            GQA_IMAGE_INDEX.setdefault(p.name, str(p))\n"
    "print('GQA images indexed:', len(GQA_IMAGE_INDEX))\n"
)

def harden(cells, force_smoke=FORCE_SMOKE):
    cells = [dict(c) for c in cells]
    # 1) Smoke toggle
    if force_smoke:
        for c in cells:
            if c['cell_type'] == 'code' and 'SMOKE = os.environ.get("SMOKE", "0") == "1"' in c['source']:
                c['source'] = c['source'].replace(
                    'SMOKE = os.environ.get("SMOKE", "0") == "1"',
                    'SMOKE = os.environ.get("SMOKE", "0") == "1" or True  # TODO(full): set False')
                break
    # 2) Harden _download_jsonl (retry + mirror) — chi dung khi input khong duoc attach
    for c in cells:
        if c['cell_type'] == 'code' and 'def _download_jsonl' in c['source']:
            pat = re.compile(r'def _download_jsonl\(filename\):.*?\n    return path', re.DOTALL)
            assert pat.search(c['source']), "_download_jsonl block not found"
            c['source'] = pat.sub(HARDEN_DOWNLOAD, c['source'])
            break
    # 3) Chen image-download cell NGAY SAU cell LOAD
    img_cell = {'cell_type': 'code', 'source': IMG_DL_CELL_SRC}
    new, inserted = [], False
    for c in cells:
        new.append(c)
        if (not inserted) and c['cell_type'] == 'code' and 'def load_visual_cot_gqa' in c['source']:
            new.append(img_cell); inserted = True
    assert inserted, "LOAD cell (load_visual_cot_gqa) not found"
    cells = new
    # 4) resolve_image dung GQA_IMAGE_INDEX truoc
    for c in cells:
        if c['cell_type'] == 'code' and 'def resolve_image' in c['source']:
            pat = re.compile(
                r'(name = str\(name\)\.split\("###", 1\)\[0\][^\n]*\n)'
                r'(    for tpl in GQA_IMG_CANDIDATES \+ \[name\]:)')
            m = pat.search(c['source'])
            assert m, "resolve_image block not found as expected"
            inject = (m.group(1) +
                      '    if name in globals().get("GQA_IMAGE_INDEX", {}):\n'
                      '        return Image.open(GQA_IMAGE_INDEX[name]).convert("RGB")\n'
                      + m.group(2))
            c['source'] = c['source'][:m.start()] + inject + c['source'][m.end():]
            break
    return cells

# ---------------- notebook writer ----------------
def _to_source_list(text):
    lines = text.split('\n')
    return [l + '\n' for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])

def write_notebook(path, cells):
    nbcells = []
    for c in cells:
        cell = {
            'cell_type': c['cell_type'],
            'id': 'cell-' + uuid.uuid4().hex[:8],
            'metadata': {},
            'source': _to_source_list(c['source']),
        }
        if c['cell_type'] == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
        nbcells.append(cell)
    nb = {
        'cells': nbcells,
        'metadata': {
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python'},
        },
        'nbformat': 4, 'nbformat_minor': 5,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with io.open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    return len(nbcells)

def main():
    text = io.open(SRC, encoding='utf-8').read()
    base = parse_percent(text)
    n_main = write_notebook(MAIN, base)

    push_cells = harden(base)
    n_push = write_notebook(PUSH, push_cells)

    meta = {
        "id": KAGGLE_ID, "title": "12 LLaVA-CoT",
        "code_file": "12_LLaVA-CoT_kaggle.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [GQA_IMAGES_DATASET],
        "kernel_sources": [VISUAL_COT_KERNEL],
        "machine_shape": "Gpu",
        # FIX (2026-08-06): image "original" (pin cu) -> Kaggle cap P100 (sm_60),
        # PyTorch/bitsandbytes (sm_70+) crash. "latest" -> T4 x2 + image moi.
        "docker_image_pinning_type": "latest",
    }
    os.makedirs(PUSH_DIR, exist_ok=True)
    with io.open(os.path.join(PUSH_DIR, 'kernel-metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print('main cells:', n_main, '->', MAIN)
    print('push cells:', n_push, '->', PUSH)
    print('FORCE_SMOKE =', FORCE_SMOKE)
    print('dataset_sources:', meta['dataset_sources'])
    print('kernel_sources :', meta['kernel_sources'])

if __name__ == '__main__':
    main()
