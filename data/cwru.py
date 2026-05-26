import os
import re
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.cluster import KMeans
from typing import List, Tuple, Dict, Optional


CLASS_MAP: Dict[str, int] = {
    "NM": 0,
    "B007": 1,
    "B014": 2,
    "B021": 3,
    "IR007": 4,
    "IR014": 5,
    "IR021": 6,
    "OR007": 7,
    "OR014": 8,
    "OR021": 9,
}
NUM_CLASSES = len(CLASS_MAP)

LOAD_MAP: Dict[int, int] = {0: 0, 1: 1, 2: 2, 3: 3}

RPM_MAP: Dict[int, int] = {0: 1797, 1: 1772, 2: 1750, 3: 1730}


def _list_mat(dir_path: str) -> List[str]:
    if not os.path.isdir(dir_path):
        return []
    return sorted([f for f in os.listdir(dir_path) if f.endswith(".mat")])


def _add_or_class(name: str, severity: str) -> Optional[str]:
    sev = severity.lstrip("0") or "0"
    cls = f"{name}{int(sev):03d}"
    return cls if cls in CLASS_MAP else None


def parse_cwru_files(
    root: str = "CRWU",
    section: str = "12k Drive End Bearing Fault Data",
    or_position: str = "Centered",
) -> List[Tuple[str, str, int]]:
    samples: List[Tuple[str, str, int]] = []

    nm_dir = os.path.join(root, "Normal Baseline")
    for fn in _list_mat(nm_dir):
        m = re.match(r"normal_(\d+)\.mat", fn)
        if m:
            samples.append((os.path.join(nm_dir, fn), "NM", int(m.group(1))))

    sec = os.path.join(root, section)

    ball_dir = os.path.join(sec, "Ball")
    for sev in sorted(os.listdir(ball_dir) if os.path.isdir(ball_dir) else []):
        sev_dir = os.path.join(ball_dir, sev)
        if not os.path.isdir(sev_dir):
            continue
        cls = _add_or_class("B", sev)
        if cls is None:
            continue
        for fn in _list_mat(sev_dir):
            m = re.match(rf"B{int(sev):03d}_(\d+)\.mat", fn)
            if m:
                samples.append((os.path.join(sev_dir, fn), cls, int(m.group(1))))

    ir_dir = os.path.join(sec, "Inner Race")
    for sev in sorted(os.listdir(ir_dir) if os.path.isdir(ir_dir) else []):
        sev_dir = os.path.join(ir_dir, sev)
        if not os.path.isdir(sev_dir):
            continue
        cls = _add_or_class("IR", sev)
        if cls is None:
            continue
        for fn in _list_mat(sev_dir):
            m = re.match(rf"IR{int(sev):03d}_(\d+)\.mat", fn)
            if m:
                samples.append((os.path.join(sev_dir, fn), cls, int(m.group(1))))

    or_root = os.path.join(sec, "Outer Race", or_position)
    for sev in sorted(os.listdir(or_root) if os.path.isdir(or_root) else []):
        sev_dir = os.path.join(or_root, sev)
        if not os.path.isdir(sev_dir):
            continue
        cls = _add_or_class("OR", sev)
        if cls is None:
            continue
        for fn in _list_mat(sev_dir):
            m = re.match(rf"OR{int(sev):03d}@\d+_(\d+)\.mat", fn)
            if m:
                samples.append((os.path.join(sev_dir, fn), cls, int(m.group(1))))

    return samples


def _load_de_signal(fp: str) -> Optional[np.ndarray]:
    d = sio.loadmat(fp)
    de_keys = [k for k in d.keys() if k.endswith("_DE_time")]
    if not de_keys:
        return None
    sigs = [d[k].flatten().astype(np.float32) for k in de_keys]
    sig = max(sigs, key=len)
    return sig


def _zscore(x: np.ndarray) -> np.ndarray:
    mu = x.mean()
    sd = x.std() + 1e-8
    return (x - mu) / sd


def _segment(signal: np.ndarray, win: int, stride: int) -> np.ndarray:
    if len(signal) < win:
        return np.empty((0, win), dtype=np.float32)
    n = (len(signal) - win) // stride + 1
    idx = np.arange(win)[None, :] + stride * np.arange(n)[:, None]
    return signal[idx].astype(np.float32)


def _spectral_features(segments: np.ndarray, fs: int = 12000) -> np.ndarray:
    n_fft = min(256, segments.shape[1])
    win = np.hanning(n_fft).astype(np.float32)
    feats = []
    for s in segments:
        x = s[:n_fft] * win
        spec = np.abs(np.fft.rfft(x))
        spec = spec / (spec.sum() + 1e-8)
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
        centroid = (freqs * spec).sum()
        bandwidth = np.sqrt(((freqs - centroid) ** 2 * spec).sum())
        flatness = np.exp(np.mean(np.log(spec + 1e-8))) / (spec.mean() + 1e-8)
        entropy = -(spec * np.log(spec + 1e-8)).sum()
        feats.append([centroid, bandwidth, flatness, entropy, s.std(), np.abs(s).max()])
    return np.asarray(feats, dtype=np.float32)


class CWRUDataset(Dataset):
    def __init__(
        self,
        file_records: List[Tuple[str, str, int]],
        window_size: int = 1024,
        stride: int = 256,
        normalize: str = "zscore",
        fs: int = 12000,
        pseudo_condition: bool = False,
        n_clusters: int = 4,
        cluster_method: str = "spectral_kmeans",
        precomputed_kmeans: Optional[KMeans] = None,
        keep_meta: bool = True,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        self.fs = fs
        self.normalize = normalize
        self.pseudo_condition = pseudo_condition
        self.n_clusters = n_clusters
        self.cluster_method = cluster_method

        self.segments: List[np.ndarray] = []
        self.labels: List[int] = []
        self.loads: List[int] = []
        self.spectra: List[np.ndarray] = []
        self.file_ids: List[int] = []

        for fid, (fp, cls_name, load) in enumerate(file_records):
            sig = _load_de_signal(fp)
            if sig is None or len(sig) < window_size:
                continue
            if normalize == "zscore":
                sig = _zscore(sig)
            segs = _segment(sig, window_size, stride)
            if len(segs) == 0:
                continue
            self.segments.append(segs)
            self.labels.extend([CLASS_MAP[cls_name]] * len(segs))
            self.loads.extend([load] * len(segs))
            self.file_ids.extend([fid] * len(segs))

        if not self.segments:
            raise RuntimeError("Empty CWRU dataset; check root path and section.")

        self.segments = np.concatenate(self.segments, axis=0)
        self.labels = np.asarray(self.labels, dtype=np.int64)
        self.loads = np.asarray(self.loads, dtype=np.int64)
        self.file_ids = np.asarray(self.file_ids, dtype=np.int64)

        self.pseudo_labels: Optional[np.ndarray] = None
        self.kmeans = precomputed_kmeans
        if pseudo_condition:
            self._build_pseudo_labels(precomputed_kmeans)

        self._meta = keep_meta

    def _build_pseudo_labels(self, precomputed: Optional[KMeans]) -> None:
        feats = _spectral_features(self.segments, fs=self.fs)
        if precomputed is None:
            km = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=0)
            km.fit(feats)
            self.kmeans = km
        else:
            self.kmeans = precomputed
        self.pseudo_labels = self.kmeans.predict(feats).astype(np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.segments[idx]).unsqueeze(0)
        y = int(self.labels[idx])
        load = int(self.loads[idx])
        pseudo = int(self.pseudo_labels[idx]) if self.pseudo_labels is not None else -1
        return {
            "x": x,
            "y": y,
            "load": load,
            "pseudo": pseudo,
        }


def _random_split_indices(n: int, val_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    cut = int(n * (1.0 - val_ratio))
    return idx[:cut], idx[cut:]


def build_dataloaders(
    cfg: dict,
    source_load: int,
    seed: int = 2026,
) -> Dict[str, DataLoader]:
    root = cfg["data"]["root"]
    section = cfg["data"]["section"]
    win = cfg["data"]["window_size"]
    stride = cfg["data"]["stride"]
    fs = cfg["data"]["fs"]
    normalize = cfg["data"]["normalize"]
    val_ratio = cfg["data"]["val_ratio"]
    num_workers = cfg["data"]["num_workers"]
    bsz = cfg["train"]["batch_size"]
    pseudo_cfg = cfg["data"].get("pseudo_condition", {})

    records = parse_cwru_files(root=root, section=section)
    src_records = [r for r in records if r[2] == source_load]
    tgt_records = [r for r in records if r[2] != source_load]

    src_full = CWRUDataset(
        file_records=src_records,
        window_size=win,
        stride=stride,
        normalize=normalize,
        fs=fs,
        pseudo_condition=pseudo_cfg.get("enable", False),
        n_clusters=pseudo_cfg.get("n_clusters", 4),
        cluster_method=pseudo_cfg.get("method", "spectral_kmeans"),
    )

    train_idx, val_idx = _random_split_indices(len(src_full), val_ratio, seed)

    train_ds = _SubsetDataset(src_full, train_idx)
    val_ds = _SubsetDataset(src_full, val_idx)

    tgt_by_load: Dict[int, DataLoader] = {}
    for ld in sorted(set(r[2] for r in tgt_records)):
        ld_records = [r for r in tgt_records if r[2] == ld]
        ld_ds = CWRUDataset(
            file_records=ld_records,
            window_size=win,
            stride=stride,
            normalize=normalize,
            fs=fs,
            pseudo_condition=False,
            keep_meta=True,
        )
        tgt_by_load[ld] = DataLoader(
            ld_ds,
            batch_size=bsz,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return {
        "train": DataLoader(
            train_ds,
            batch_size=bsz,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=bsz,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "targets": tgt_by_load,
        "source_dataset": src_full,
    }


class _SubsetDataset(Dataset):
    def __init__(self, base: CWRUDataset, indices: np.ndarray) -> None:
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        return self.base[int(self.indices[idx])]
