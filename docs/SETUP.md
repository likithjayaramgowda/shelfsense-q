# Setup — both machines

## 0. Prerequisites

- Python **3.10 <= v < 3.14**  (`python --version`)
- git
- A Qualcomm ID with AI Hub Workbench access

## 1. Clone

```bash
git clone https://github.com/<you>/shelfsense-q.git
cd shelfsense-q
```

## 2. Secrets

```bash
cp .env.example .env
```

Open `.env` and fill it in. Set `MACHINE=laptop` or `MACHINE=workstation`.

`.env` is gitignored. Verify before your first commit:

```bash
git check-ignore -v .env
```

That must print a matching rule. If it prints nothing, **stop** — the file is
not ignored and you are one `git add -A` away from publishing your token.

## 3. Environments

### `.venv-qai` — both machines, no GPU needed

```bash
python -m venv .venv-qai
source .venv-qai/bin/activate          # Windows: .venv-qai\Scripts\activate
pip install -r requirements-qai.txt
qai-hub configure --api_token <YOUR_TOKEN>
```

Verify:

```bash
python src/qualcomm/list_devices.py
```

### `.venv-train` — WORKSTATION ONLY

```bash
python -m venv .venv-train
source .venv-train/bin/activate        # Windows: .venv-train\Scripts\activate
pip install -r requirements-train.txt
```

Verify Blackwell / sm_120:

```bash
python scripts/check_cuda.py
```

Must report `available=True` and `capability=(12, 0)`.

> **Never** install a `qai-*` package into `.venv-train`, and never install
> `torch` into `.venv-qai` by hand. If you do, delete the env and rebuild it.

## 4. Data — workstation only

SKU-110K is ~13.6 GB: 11,743 images, ~1.73M boxes, ~147 objects/image,
official split 8,219 train / 588 val / 2,936 test.

License: **academic / non-commercial**. This must be stated in the README.

Download into `data/raw/` (gitignored). Then generate the split files:

```bash
python src/data/make_splits.py
```

That writes `data/splits/{train,val,test}.txt`, which ARE committed. Those
files are what make every downstream number reproducible — and what stops
calibration images leaking in from the evaluation splits.
