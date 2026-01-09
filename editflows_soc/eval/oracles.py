'''
from dataclasses import dataclass
from typing import List, Optional, Dict

import numpy as np
import torch
import torch.nn.functional as F

from grelu.lightning import LightningModel

DNA_ALPHABET = set(["A", "C", "G", "T"])
_DNA2ID = {"A": 0, "C": 1, "G": 2, "T": 3}

# eval/oracles.py
import io
import torch
from grelu.lightning import LightningModel

def load_from_checkpoint_safe(ckpt_path: str, device: str = "cuda") -> LightningModel:
    ckpt = torch.load(ckpt_path, map_location="cpu")

    hp = ckpt.get("hyper_parameters", {}) or {}
    ckpt.setdefault("data_params", hp.get("data_params", {}))
    ckpt.setdefault("performance", {})
    if not ckpt["performance"]:
        ckpt["performance"] = {
            "best_step": ckpt.get("global_step", 0),
            "best_metric": None,
        }

    buf = io.BytesIO()
    torch.save(ckpt, buf)
    buf.seek(0)

    model = LightningModel.load_from_checkpoint(buf, map_location="cpu")
    model.to(torch.device(device))
    model.eval()
    return model


def _sanitize_seq(seq: str) -> str:
    seq = str(seq).strip().upper()
    out = []
    for ch in seq:
        if ch in DNA_ALPHABET:
            out.append(ch)
        else:
            out.append(np.random.choice(["A", "C", "G", "T"]))
    return "".join(out)


def normalize_len(seq: str, L: int = 200, mode: str = "center_crop") -> str:
    """
    Normalize sequence length to L for oracle input.
    - if longer: crop (center/left/right)
    - if shorter: pad with random bases
    """
    seq = _sanitize_seq(seq)
    n = len(seq)
    if n == L:
        return seq

    if n > L:
        if mode == "center_crop":
            s = (n - L) // 2
            return seq[s : s + L]
        elif mode == "left_crop":
            return seq[:L]
        elif mode == "right_crop":
            return seq[-L:]
        else:
            raise ValueError(f"Unknown crop mode: {mode}")

    # n < L
    pad = "".join(np.random.choice(["A", "C", "G", "T"], size=(L - n)))
    return seq + pad


def batch_tokenize_acgt(seqs: List[str], L: int = 200) -> torch.Tensor:
    """
    Convert list[str] of A/C/G/T into LongTensor (N, L) with ids 0..3.
    Assumes sequences already normalized to length L.
    """
    N = len(seqs)
    arr = np.empty((N, L), dtype=np.int64)
    for i, s in enumerate(seqs):
        # s must be length L
        arr[i] = np.fromiter((_DNA2ID[c] for c in s), dtype=np.int64, count=L)
    return torch.tensor(arr, dtype=torch.long)


def seqs_to_oracle_input(seqs: List[str], device: torch.device, L: int = 200, crop_mode: str = "center_crop") -> torch.Tensor:
    """
    Oracle expects one-hot inputs shaped (N, 4, L).
    """
    seqs = [normalize_len(s, L=L, mode=crop_mode) for s in seqs]
    tokens = batch_tokenize_acgt(seqs, L=L).to(device)           # (N,L)
    onehot = F.one_hot(tokens, num_classes=4).float()            # (N,L,4)
    x = onehot.transpose(1, 2).contiguous()                      # (N,4,L)
    return x


@dataclass
class OracleBundle:
    gosai_eval: Optional[object]
    gosai_ft: Optional[object]
    atac: Optional[object]


class GosaiActivityOracle:
    """
    Predicts enhancer activity (3 tasks: hepg2, k562, sknsh).
    """
    def __init__(self, ckpt_path: str, device: str = "cuda"):
        self.ckpt_path = ckpt_path
        self.device = torch.device(device)
        self.model = load_from_checkpoint_safe(ckpt_path, device=str(self.device))
        self.model.eval()

    @torch.no_grad()
    def predict(self, seqs: List[str], L: int = 200, crop_mode: str = "center_crop") -> np.ndarray:
        x = seqs_to_oracle_input(seqs, device=self.device, L=L, crop_mode=crop_mode)
        preds = self.model(x).detach().cpu().numpy()   # (N, 3)
        return preds


class ATACBinaryOracle:
    """
    Binary chromatin accessibility classifier.
    Output is often (N, 7) for different cell lines; you can pick HepG2 index in metrics config.
    """
    def __init__(self, ckpt_path: str, device: str = "cuda"):
        self.ckpt_path = ckpt_path
        self.device = torch.device(device)
        self.model = load_from_checkpoint_safe(ckpt_path, device=str(self.device))
        self.model.eval()

    @torch.no_grad()
    def predict_logits(self, seqs: List[str], L: int = 200, crop_mode: str = "center_crop") -> np.ndarray:
        x = seqs_to_oracle_input(seqs, device=self.device, L=L, crop_mode=crop_mode)
        logits = self.model(x).detach().cpu().numpy()
        return logits

    @torch.no_grad()
    def predict_proba(self, seqs: List[str], L: int = 200, crop_mode: str = "center_crop") -> np.ndarray:
        logits = self.predict_logits(seqs, L=L, crop_mode=crop_mode)
        return 1.0 / (1.0 + np.exp(-logits))


def load_oracles(
    gosai_eval_ckpt: str,
    gosai_ft_ckpt: Optional[str],
    atac_ckpt: str,
    device: str = "cuda",
) -> Dict[str, object]:
    out = {
        "gosai_eval": GosaiActivityOracle(gosai_eval_ckpt, device=device) if gosai_eval_ckpt else None,
        "gosai_ft": GosaiActivityOracle(gosai_ft_ckpt, device=device) if gosai_ft_ckpt else None,
        "atac": ATACBinaryOracle(atac_ckpt, device=device) if atac_ckpt else None,
    }
    return out
'''

# eval/oracle.py (Updated: data_params + performance injection)
import torch
import torch.nn.functional as F
import numpy as np
import io
import random
from typing import List
from grelu.lightning import LightningModel

DNA_ALPHABET = ["A", "C", "G", "T"]
_DNA2ID = {c: i for i, c in enumerate(DNA_ALPHABET)}

class OracleWrapper:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.models = {}

    def load_model(self, name: str, ckpt_path: str):
        print(f"Loading {name} from {ckpt_path}...")
        try:
            # 1. Raw Checkpoint 로드 (CPU에서)
            ckpt = torch.load(ckpt_path, map_location="cpu")

            # 2. [주입 1] 누락된 'data_params' 강제 주입
            if "data_params" not in ckpt:
                print(f"  -> Injecting missing 'data_params' for {name}...")
                ckpt["data_params"] = {}
            
            if "hyper_parameters" in ckpt:
                if "data_params" not in ckpt["hyper_parameters"]:
                    ckpt["hyper_parameters"]["data_params"] = {}

            # 3. [주입 2] 누락된 'performance' 강제 주입 (이번 에러 해결)
            if "performance" not in ckpt or ckpt["performance"] is None:
                print(f"  -> Injecting missing 'performance' for {name}...")
                # Grelu가 기대하는 최소한의 구조
                ckpt["performance"] = {
                    "best_step": ckpt.get("global_step", 0),
                    "best_metric": None,
                }

            # 4. 수정된 딕셔너리를 메모리 버퍼에 다시 저장
            buf = io.BytesIO()
            torch.save(ckpt, buf)
            buf.seek(0)

            # 5. 버퍼로부터 모델 로드
            model = LightningModel.load_from_checkpoint(buf, map_location="cpu")
            model.to(self.device)
            model.eval()
            self.models[name] = model
            print(f"✅ Successfully loaded {name}")
            
        except Exception as e:
            print(f"🚨 Failed to load {name}: {e}")
            # import traceback
            # traceback.print_exc()

    def _normalize_seq(self, seq: str, L: int = 200) -> str:
        seq = seq.upper()
        curr_len = len(seq)

        if curr_len == L:
            return seq
        
        if curr_len > L:
            start = (curr_len - L) // 2
            return seq[start : start + L]
        else:
            pad_len = L - curr_len
            left_pad = pad_len // 2
            right_pad = pad_len - left_pad
            bases = ["A", "C", "G", "T"]
            prefix = "".join(random.choices(bases, k=left_pad))
            suffix = "".join(random.choices(bases, k=right_pad))
            return prefix + seq + suffix

    def _preprocess(self, seqs: List[str], L: int = 200) -> torch.Tensor:
        normalized_seqs = [self._normalize_seq(s, L) for s in seqs]
        
        tokens = np.zeros((len(normalized_seqs), L), dtype=np.int64)
        for i, s in enumerate(normalized_seqs):
            tokens[i] = [_DNA2ID.get(c, 0) for c in s]
            
        tokens_t = torch.tensor(tokens, dtype=torch.long, device=self.device)
        
        # 기본 시도: (N, 4, L)
        onehot = F.one_hot(tokens_t, num_classes=4).float().transpose(1, 2)
        return onehot

    @torch.no_grad()
    def predict_activity(self, seqs: List[str], model_name: str = "eval") -> np.ndarray:
        if model_name not in self.models:
            print(f"⚠️ Model {model_name} not loaded.")
            return np.zeros((len(seqs), 3))
            
        x = self._preprocess(seqs, L=200)
        
        try:
            out = self.models[model_name](x)
        except RuntimeError as e:
            if "size mismatch" in str(e) or "mat1 and mat2" in str(e):
                x = x.transpose(1, 2)
                out = self.models[model_name](x)
            else:
                raise e
                
        return out.detach().cpu().numpy()

    @torch.no_grad()
    def predict_atac(self, seqs: List[str]) -> np.ndarray:
        if "atac" not in self.models:
            print("⚠️ ATAC model not loaded.")
            return np.zeros((len(seqs), 1))

        x = self._preprocess(seqs, L=200)
        
        try:
            raw_out = self.models["atac"](x)
        except RuntimeError as e:
            # Shape Mismatch 시 Transpose 시도
            x = x.transpose(1, 2)
            raw_out = self.models["atac"](x)
        
        if raw_out.min() >= 0 and raw_out.max() <= 1.0:
            probs = raw_out
        else:
            # Logit -> Sigmoid
            probs = torch.sigmoid(raw_out)

        return probs.detach().cpu().numpy()