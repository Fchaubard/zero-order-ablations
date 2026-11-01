"""Datasets for the survey, fully resident on the training device.

``Data(task, device, root)`` loads (and on first use downloads + caches) the dataset
for a task and keeps every split on ``device`` as tensors, so sampling a training
batch is a single gather and there is zero dataloader overhead.

    data = Data("mlp_mnist", "cuda")
    data.kind          # "cls" or "lm"
    data.n_classes     # 10 for images, the PTB character vocabulary size for text
    data.batch_size    # 128 (cls) / 64 (lm)
    data.seq_len       # None (cls) / 64 (lm)
    data.n_train       # training images, or training characters
    x, y = data.train_batch()                 # uniformly random batch
    for x, y in data.eval_batches("test"):    # full split, large sequential batches
        ...

Datasets (see PROTOCOL.md):
* MNIST      55k train / 5k val / 10k test, standardised with mean 0.1307 std 0.3081, (N,1,28,28)
* CIFAR-10   45k / 5k / 10k, per-channel standardisation, (N,3,32,32)
* PTB (char) Mikolov's ptb.{train,valid,test}.txt; lines joined with "\n" (spaces kept);
             character vocabulary built from the training text; LongTensor of ids per split.
             Shared by lstm_ptb and transformer_ptb.

Processed tensors are cached under ``root/processed`` so a second run is instant.
``python -m zoo.data --root ./data --download-all`` prefetches everything.
``Data(..., _synthetic=True)`` makes random data of the right shapes (tests / smoke runs),
and ``Data("lstm_ptb", ..., text={"train": str, "valid": str, "test": str})`` injects a
corpus directly (tests).
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
import zipfile
from typing import Dict, Iterator, Optional, Tuple

import torch

# ----------------------------------------------------------------------------- constants
MNIST_MEAN, MNIST_STD = (0.1307,), (0.3081,)
CIFAR_MEAN, CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)

TASK_DATASET = {
    "mlp_mnist": "mnist",
    "conv_cifar10": "cifar10",
    "lstm_ptb": "ptb",
    "transformer_ptb": "ptb",
}
DATASET_KIND = {"mnist": "cls", "cifar10": "cls", "ptb": "lm"}

CLS_BATCH, CLS_EVAL_BATCH = 128, 2048
LM_BATCH, LM_EVAL_BATCH, SEQ_LEN = 64, 1024, 64
N_VAL = 5000  # last 5k of the MNIST / CIFAR train sets

PTB_FILES = ("ptb.train.txt", "ptb.valid.txt", "ptb.test.txt")
PTB_GITHUB = "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/"
PTB_DEEPAI_ZIP = "https://data.deepai.org/ptbdataset.zip"
HF_DATASET = "ptb_text_only"

_SPLIT_ALIASES = {"val": "valid", "valid": "valid", "validation": "valid", "test": "test", "train": "train"}


# ----------------------------------------------------------------------------- downloads
def _log(msg: str) -> None:
    print(f"[zoo.data] {msg}", flush=True)


def fetch_bytes(url: str, timeout: float = 60.0, retries: int = 4) -> bytes:
    """GET ``url`` with retries and exponential back-off (requests, falling back to urllib)."""
    last: Optional[BaseException] = None
    for attempt in range(retries):
        try:
            try:
                import requests  # type: ignore

                r = requests.get(url, timeout=timeout, headers={"User-Agent": "zoo-survey/1.0"})
                r.raise_for_status()
                return r.content
            except ImportError:
                from urllib.request import Request, urlopen

                with urlopen(Request(url, headers={"User-Agent": "zoo-survey/1.0"}), timeout=timeout) as resp:
                    return resp.read()
        except Exception as e:  # noqa: BLE001 - we really want to retry on anything
            last = e
            wait = 2.0 ** attempt
            _log(f"download failed ({attempt + 1}/{retries}) {url}: {e!r}; retrying in {wait:.0f}s")
            time.sleep(wait)
    raise RuntimeError(f"could not download {url}: {last!r}")


def _md5(path: str, chunk: int = 1 << 20) -> str:
    import hashlib

    h = hashlib.md5()
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(chunk), b""):
            h.update(buf)
    return h.hexdigest()


def download_file(url: str, dest: str, md5: Optional[str] = None, timeout: float = 60.0,
                  retries: int = 8, chunk: int = 1 << 20) -> str:
    """Stream ``url`` to ``dest`` with HTTP-Range resume, retries and an optional md5 check.
    A partial ``dest`` (or ``dest.part``) is resumed rather than restarted."""
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    part = dest + ".part"
    if os.path.isfile(dest):
        if md5 is None or _md5(dest) == md5:
            return dest
        _log(f"{dest} exists but fails the md5 check; resuming it")
        os.replace(dest, part)
    last: Optional[BaseException] = None
    resume_failures = 0
    for attempt in range(retries):
        have = os.path.getsize(part) if os.path.isfile(part) else 0
        try:
            # identity encoding: byte ranges only make sense on the raw bytes (GitHub's CDN
            # otherwise gzip-encodes text files and a resumed fragment cannot be decoded)
            headers = {"User-Agent": "zoo-survey/1.0", "Accept-Encoding": "identity"}
            if have:
                headers["Range"] = f"bytes={have}-"
            try:
                import requests  # type: ignore

                r = requests.get(url, stream=True, timeout=timeout, headers=headers)
                status, length = r.status_code, r.headers.get("Content-Length")
                body = r.iter_content(chunk)
                closer = r.close
            except ImportError:
                from urllib.request import Request, urlopen

                r = urlopen(Request(url, headers=headers), timeout=timeout)
                status, length = r.status, r.headers.get("Content-Length")
                body = iter(lambda: r.read(chunk), b"")
                closer = r.close
            try:
                if status == 416:          # range not satisfiable: the .part is already complete
                    pass
                elif status not in (200, 206):
                    raise RuntimeError(f"HTTP {status}")
                else:
                    if have and status != 206:
                        _log("server ignored the Range header; restarting the download")
                        have = 0
                    total = (int(length) + have) if length else None
                    done, next_pct = have, 10
                    t0 = time.time()
                    with open(part, "ab" if have else "wb") as f:
                        for buf in body:
                            f.write(buf)
                            done += len(buf)
                            if total and 100 * done / total >= next_pct:
                                rate = (done - have) / max(time.time() - t0, 1e-6) / 1e6
                                _log(f"{os.path.basename(dest)}: {100 * done / total:3.0f}% of {total / 1e6:.1f} MB ({rate:.1f} MB/s)")
                                next_pct += 10
            finally:
                closer()
            if md5 is not None and _md5(part) != md5:
                os.remove(part)
                raise RuntimeError("md5 mismatch after download (corrupt file removed)")
            os.replace(part, dest)
            return dest
        except Exception as e:  # noqa: BLE001
            last = e
            if have:
                resume_failures += 1
                if resume_failures >= 2 and os.path.isfile(part):
                    _log("resuming failed twice; restarting the download from scratch")
                    os.remove(part)
            wait = min(60.0, 2.0 ** attempt)
            _log(f"download of {url} failed ({attempt + 1}/{retries}): {e!r}; retrying in {wait:.0f}s")
            time.sleep(wait)
    raise RuntimeError(f"could not download {url}: {last!r}")


def _with_retries(fn, what: str, retries: int = 3):
    last: Optional[BaseException] = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            wait = 3.0 * (attempt + 1)
            _log(f"{what} failed ({attempt + 1}/{retries}): {e!r}; retrying in {wait:.0f}s")
            time.sleep(wait)
    raise RuntimeError(f"{what} failed after {retries} attempts: {last!r}")


# ----------------------------------------------------------------------------- PTB
def _ptb_raw_dir(root: str) -> str:
    return os.path.join(root, "ptb")


def _read_ptb_files(raw_dir: str) -> Optional[Dict[str, str]]:
    paths = [os.path.join(raw_dir, f) for f in PTB_FILES]
    if not all(os.path.isfile(p) and os.path.getsize(p) > 0 for p in paths):
        return None
    out = {}
    for split, p in zip(("train", "valid", "test"), paths):
        with open(p, encoding="utf-8") as f:
            out[split] = f.read()
    return out


def _write_ptb_files(raw_dir: str, texts: Dict[str, str]) -> None:
    os.makedirs(raw_dir, exist_ok=True)
    for split, fname in zip(("train", "valid", "test"), PTB_FILES):
        with open(os.path.join(raw_dir, fname), "w", encoding="utf-8") as f:
            f.write(texts[split])


def _ptb_from_github() -> Dict[str, str]:
    return {split: fetch_bytes(PTB_GITHUB + fname).decode("utf-8")
            for split, fname in zip(("train", "valid", "test"), PTB_FILES)}


def _ptb_from_deepai_zip() -> Dict[str, str]:
    blob = fetch_bytes(PTB_DEEPAI_ZIP, timeout=120)
    zf = zipfile.ZipFile(io.BytesIO(blob))
    names = zf.namelist()
    out = {}
    for split, fname in zip(("train", "valid", "test"), PTB_FILES):
        cands = [n for n in names if n.endswith(fname)]
        if not cands:
            raise RuntimeError(f"{fname} not in {PTB_DEEPAI_ZIP} (members: {names[:10]}...)")
        out[split] = zf.read(cands[0]).decode("utf-8")
    return out


def _ptb_from_hf() -> Dict[str, str]:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as e:
        raise RuntimeError("the `datasets` package is not installed") from e
    try:
        ds = load_dataset(HF_DATASET, "penn_treebank")
    except Exception:  # noqa: BLE001 - config name differs between versions
        ds = load_dataset(HF_DATASET)
    out = {}
    for split, hf_split in (("train", "train"), ("valid", "validation"), ("test", "test")):
        col = "sentence" if "sentence" in ds[hf_split].column_names else ds[hf_split].column_names[0]
        # Mikolov's files have a leading and trailing space on every line
        out[split] = "".join(f" {s.strip()} \n" for s in ds[hf_split][col])
    return out


PTB_SOURCES = (
    ("github:wojzaremba/lstm", _ptb_from_github),
    ("deepai.org/ptbdataset.zip", _ptb_from_deepai_zip),
    (f"huggingface:{HF_DATASET}", _ptb_from_hf),
)


def fetch_ptb_texts(root: str) -> Tuple[Dict[str, str], str]:
    """Return ({train, valid, test}: str, source).  Uses root/ptb/*.txt if present."""
    raw_dir = _ptb_raw_dir(root)
    texts = _read_ptb_files(raw_dir)
    if texts is not None:
        return texts, f"local:{raw_dir}"
    errors = []
    for name, fn in PTB_SOURCES:
        try:
            _log(f"downloading PTB from {name} ...")
            texts = fn()
            for s in ("train", "valid", "test"):
                if len(texts[s]) < 1000:
                    raise RuntimeError(f"{s} split suspiciously small ({len(texts[s])} chars)")
            _write_ptb_files(raw_dir, texts)
            return texts, name
        except Exception as e:  # noqa: BLE001
            errors.append(f"  {name}: {e!r}")
            _log(f"PTB source {name} failed: {e!r}")
    raise RuntimeError(
        "Could not obtain the Penn Treebank files from any source:\n" + "\n".join(errors)
        + f"\nPlace ptb.train.txt / ptb.valid.txt / ptb.test.txt manually under {raw_dir}/ and retry."
    )


def build_char_corpus(texts: Dict[str, str]) -> dict:
    """Join lines with '\\n' (spaces kept), build the vocabulary from the training text and
    map every split to a LongTensor of character ids.  Characters that do not occur in the
    training text (there are none in PTB) are mapped to the space id."""
    joined = {s: "\n".join(texts[s].splitlines()) for s in ("train", "valid", "test")}
    vocab = "".join(sorted(set(joined["train"])))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    unk = stoi.get(" ", 0)
    out: dict = {"vocab": vocab}
    for s, t in joined.items():
        ids = [stoi.get(ch, unk) for ch in t]
        out[s] = torch.tensor(ids, dtype=torch.long)
        out[f"n_unk_{s}"] = sum(1 for ch in t if ch not in stoi)
    return out


def load_ptb(root: str, force: bool = False) -> dict:
    cache = os.path.join(root, "processed", "ptb_char.pt")
    if os.path.isfile(cache) and not force:
        return torch.load(cache, map_location="cpu", weights_only=True)
    texts, source = fetch_ptb_texts(root)
    d = build_char_corpus(texts)
    d["source"] = source
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    torch.save(d, cache)
    _log(f"PTB char corpus cached at {cache} (vocab={len(d['vocab'])}, "
         f"train={d['train'].numel()}, valid={d['valid'].numel()}, test={d['test'].numel()}, source={source})")
    return d


# ----------------------------------------------------------------------------- images
def _load_torchvision(name: str, root: str):
    from torchvision import datasets  # imported lazily: not needed for synthetic / text runs

    cls = {"mnist": datasets.MNIST, "cifar10": datasets.CIFAR10}[name]
    os.makedirs(root, exist_ok=True)
    if name == "cifar10":
        # the archive is ~170 MB from a slow server: fetch it ourselves (resumable, md5-verified);
        # torchvision then only verifies + extracts it.
        download_file(cls.url, os.path.join(root, cls.filename), md5=cls.tgz_md5, timeout=120)
    tr = _with_retries(lambda: cls(root, train=True, download=True), f"{name} train download")
    te = _with_retries(lambda: cls(root, train=False, download=True), f"{name} test download")
    return tr, te


def _as_uint8_nchw(x) -> torch.Tensor:
    x = torch.as_tensor(x)
    if x.dim() == 3:          # MNIST (N,28,28) -> (N,1,28,28)
        x = x.unsqueeze(1)
    elif x.shape[-1] in (1, 3):  # CIFAR numpy (N,32,32,3) -> (N,3,32,32)
        x = x.permute(0, 3, 1, 2)
    return x.contiguous().to(torch.uint8)


def load_images(name: str, root: str, force: bool = False) -> dict:
    """uint8 images + int64 labels per split, cached at root/processed/{name}.pt."""
    cache = os.path.join(root, "processed", f"{name}.pt")
    if os.path.isfile(cache) and not force:
        return torch.load(cache, map_location="cpu", weights_only=True)
    tr, te = _load_torchvision(name, root)
    x = _as_uint8_nchw(tr.data)
    y = torch.as_tensor(tr.targets, dtype=torch.long)
    xt = _as_uint8_nchw(te.data)
    yt = torch.as_tensor(te.targets, dtype=torch.long)
    n_tr = x.shape[0] - N_VAL
    d = {
        "x_train": x[:n_tr], "y_train": y[:n_tr],
        "x_valid": x[n_tr:], "y_valid": y[n_tr:],
        "x_test": xt, "y_test": yt,
    }
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    torch.save(d, cache)
    _log(f"{name} cached at {cache} (train={n_tr}, valid={N_VAL}, test={xt.shape[0]})")
    return d


def standardise(x_uint8: torch.Tensor, mean, std) -> torch.Tensor:
    x = x_uint8.float().div_(255.0)
    m = torch.tensor(mean, dtype=torch.float32, device=x.device).view(1, -1, 1, 1)
    s = torch.tensor(std, dtype=torch.float32, device=x.device).view(1, -1, 1, 1)
    return x.sub_(m).div_(s)


# ----------------------------------------------------------------------------- synthetic
SYNTH_VOCAB = "\n $&'*,-./0123456789<>N\\abcdefghijklmnopqrstuvwxyz"   # 50 symbols, PTB-like


def synthetic_images(shape, n_train=2048, n_valid=512, n_test=512, n_classes=10, seed=0) -> dict:
    """Random Gaussian images with a class-dependent mean so that something is learnable."""
    g = torch.Generator().manual_seed(seed)
    proto = torch.randn((n_classes, *shape), generator=g) * 0.6
    d = {}
    for split, n in (("train", n_train), ("valid", n_valid), ("test", n_test)):
        y = torch.randint(0, n_classes, (n,), generator=g)
        x = torch.randn((n, *shape), generator=g) + proto[y]
        d[f"x_{split}"], d[f"y_{split}"] = x, y
    return d


def synthetic_text(n_train=50_000, n_valid=8_000, n_test=8_000, seed=0) -> dict:
    """Random character ids with a simple Markov structure (next = prev + small step)."""
    g = torch.Generator().manual_seed(seed)
    V = len(SYNTH_VOCAB)
    d = {"vocab": SYNTH_VOCAB, "source": "synthetic"}
    for split, n in (("train", n_train), ("valid", n_valid), ("test", n_test)):
        steps = torch.randint(0, 3, (n,), generator=g)
        ids = torch.cumsum(steps, 0) % V
        d[split] = ids.long()
        d[f"n_unk_{split}"] = 0
    return d


# ----------------------------------------------------------------------------- Data
class Data:
    def __init__(self, task: str, device="cpu", root: str = "./data", _synthetic: bool = False,
                 text: Optional[Dict[str, str]] = None):
        if task not in TASK_DATASET:
            raise KeyError(f"unknown task {task!r}; choose from {list(TASK_DATASET)}")
        self.task = task
        self.dataset = TASK_DATASET[task]
        self.kind = DATASET_KIND[self.dataset]
        self.device = torch.device(device)
        self.root = root
        self.synthetic = _synthetic
        self.source = "synthetic" if _synthetic else None

        if self.kind == "cls":
            self.batch_size, self.eval_batch_size, self.seq_len = CLS_BATCH, CLS_EVAL_BATCH, None
            self.n_classes = 10
            shape = (1, 28, 28) if self.dataset == "mnist" else (3, 32, 32)
            mean, std = (MNIST_MEAN, MNIST_STD) if self.dataset == "mnist" else (CIFAR_MEAN, CIFAR_STD)
            if _synthetic:
                raw = synthetic_images(shape)
                self._xy = {s: (raw[f"x_{s}"].to(self.device), raw[f"y_{s}"].to(self.device))
                            for s in ("train", "valid", "test")}
            else:
                raw = load_images(self.dataset, root)
                self.source = self.dataset
                self._xy = {s: (standardise(raw[f"x_{s}"].to(self.device), mean, std),
                                raw[f"y_{s}"].to(self.device))
                            for s in ("train", "valid", "test")}
            self.shape = tuple(self._xy["train"][0].shape[1:])
            self.n_train = self._xy["train"][0].shape[0]
        else:
            self.batch_size, self.eval_batch_size, self.seq_len = LM_BATCH, LM_EVAL_BATCH, SEQ_LEN
            if _synthetic:
                raw = synthetic_text()
            elif text is not None:
                raw = build_char_corpus(text)
                raw["source"] = "text-override"
            else:
                raw = load_ptb(root)
            self.vocab: str = raw["vocab"]
            self.source = raw.get("source")
            self.n_classes = len(self.vocab)
            self._ids = {s: raw[s].to(self.device) for s in ("train", "valid", "test")}
            self.n_train = self._ids["train"].numel()
            if self.n_train < self.seq_len + 2:
                raise ValueError(f"training text too short ({self.n_train} chars) for seq_len={self.seq_len}")
            self._ar = torch.arange(self.seq_len + 1, device=self.device)
            self._windows: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
            self.shape = (self.seq_len,)

    # ---- sampling ---------------------------------------------------------------------
    def train_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Uniformly random training batch (images+labels, or 64-char windows + next chars)."""
        if self.kind == "cls":
            x, y = self._xy["train"]
            idx = torch.randint(0, self.n_train, (self.batch_size,), device=self.device)
            return x[idx], y[idx]
        ids = self._ids["train"]
        starts = torch.randint(0, self.n_train - self.seq_len, (self.batch_size,), device=self.device)
        chunk = ids[starts[:, None] + self._ar[None, :]]          # (B, T+1)
        return chunk[:, :-1].contiguous(), chunk[:, 1:].contiguous()

    def split_size(self, split: str) -> int:
        s = _SPLIT_ALIASES[split]
        if self.kind == "cls":
            return self._xy[s][0].shape[0]
        return self._ids[s].numel()

    def _lm_windows(self, split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if split not in self._windows:
            ids = self._ids[split]
            T = self.seq_len
            n_win = (ids.numel() - 1) // T
            x = ids[: n_win * T].view(n_win, T)
            y = ids[1: n_win * T + 1].view(n_win, T)
            self._windows[split] = (x, y)
        return self._windows[split]

    def eval_batches(self, split: str = "test") -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Iterate over the FULL split in large sequential batches (2048 images, or 1024
        non-overlapping 64-character windows)."""
        s = _SPLIT_ALIASES[split]
        if self.kind == "cls":
            x, y = self._xy[s]
        else:
            x, y = self._lm_windows(s)
        n = x.shape[0]
        for i in range(0, n, self.eval_batch_size):
            yield x[i: i + self.eval_batch_size], y[i: i + self.eval_batch_size]

    def n_eval(self, split: str) -> int:
        """Number of eval examples (images, or 64-char windows) in a split."""
        s = _SPLIT_ALIASES[split]
        return self._xy[s][0].shape[0] if self.kind == "cls" else self._lm_windows(s)[0].shape[0]

    # ---- text helpers ------------------------------------------------------------------
    def encode(self, text: str) -> torch.Tensor:
        stoi = {ch: i for i, ch in enumerate(self.vocab)}
        return torch.tensor([stoi.get(ch, stoi.get(" ", 0)) for ch in text], dtype=torch.long, device=self.device)

    def decode(self, ids) -> str:
        return "".join(self.vocab[int(i)] for i in ids)

    def summary(self) -> str:
        if self.kind == "cls":
            return (f"{self.task}: {self.dataset} shape={self.shape} "
                    f"train={self.split_size('train')} valid={self.split_size('valid')} test={self.split_size('test')} "
                    f"batch={self.batch_size} eval_batch={self.eval_batch_size}")
        return (f"{self.task}: ptb-char vocab={self.n_classes} "
                f"train={self.n_train} valid={self.split_size('valid')} test={self.split_size('test')} chars "
                f"({self.n_eval('valid')}/{self.n_eval('test')} eval windows) "
                f"batch={self.batch_size}x{self.seq_len} source={self.source}")

    def __repr__(self) -> str:
        return f"Data({self.summary()})"


# ----------------------------------------------------------------------------- CLI
def download_all(root: str, force: bool = False) -> None:
    for name in ("mnist", "cifar10"):
        d = load_images(name, root, force=force)
        print(f"{name:8s} train={tuple(d['x_train'].shape)} valid={tuple(d['x_valid'].shape)} "
              f"test={tuple(d['x_test'].shape)}")
    d = load_ptb(root, force=force)
    print(f"ptb      vocab={len(d['vocab'])} train={d['train'].numel()} valid={d['valid'].numel()} "
          f"test={d['test'].numel()} chars, source={d.get('source')}, vocab={d['vocab']!r}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Download / cache / inspect the survey datasets.")
    ap.add_argument("--root", default="./data")
    ap.add_argument("--download-all", action="store_true", help="prefetch MNIST, CIFAR-10 and PTB")
    ap.add_argument("--force", action="store_true", help="rebuild the processed caches")
    ap.add_argument("--task", choices=list(TASK_DATASET), help="load one task's data and print a summary")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    if args.download_all:
        download_all(args.root, force=args.force)
    if args.task:
        d = Data(args.task, args.device, root=args.root)
        print(d.summary())
        x, y = d.train_batch()
        print(f"train_batch: x{tuple(x.shape)} {x.dtype}  y{tuple(y.shape)} {y.dtype}")
        if d.kind == "lm":
            print("sample:", repr(d.decode(x[0][:80])))
    if not args.download_all and not args.task:
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
