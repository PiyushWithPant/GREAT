"""
Universal Embedding & Clustering Script
---------------------------------------

Generates sentence embeddings from any Hugging Face transformer model,
performs K-Means clustering, and selects medoid examples (cluster representatives).

Features:
- Supports any encoder or causal model (auto-detects)
- Optional PCA dimensionality reduction
- Optional DataParallel for multi-GPU
- Clean Hugging Face Hub authentication
- Saves medoids and cluster plots

Usage:
    export HF_TOKEN="your_hf_token_here"

    python cluster_embeddings.py \
        --model_id meta-llama/Llama-3.2-1B \
        --data_path data/triggers/triggers_train.jsonl \
        --output_dir results/clustering \
        --k 10 \
        --use_pca \
        --n_pcs 50 \
        --batch_size 128 \
        --max_length 128
"""

import os
import json
import time
import torch
import random
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    AutoConfig,
)
from datasets import load_dataset
from huggingface_hub import login


# ========================== UTILITIES ========================== #

def set_seed(seed: int = 42):
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _select_torch_dtype() -> torch.dtype:
    """Use float16 on CUDA to avoid bf16 issues, else float32."""
    return torch.float16 if torch.cuda.is_available() else torch.float32


def login_to_huggingface():
    """Authenticate with Hugging Face Hub via environment token."""
    token = os.getenv("HF_TOKEN")
    if token:
        try:
            login(token=token)
            print("[INFO] Hugging Face login successful.")
        except Exception as e:
            print(f"[WARN] Hugging Face login failed: {e}")
    else:
        print("[WARN] No HF_TOKEN found in environment. Some models may require authentication.")


# ========================== MODEL LOADING ========================== #

class GenericEncoderWrapper(torch.nn.Module):
    """Wraps any Hugging Face model to extract token embeddings."""
    def __init__(self, model, causal: bool = False):
        super().__init__()
        self.model = model
        self.causal = causal

    def forward(self, input_ids, attention_mask):
        if self.causal:
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = out.hidden_states[-1]
        else:
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            last_hidden = out.last_hidden_state
        return last_hidden


def load_encoder(model_id: str, use_dataparallel=True, device_ids=None):
    """Load any Hugging Face model and wrap it for embedding extraction."""
    dtype = _select_torch_dtype()
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    cfg = AutoConfig.from_pretrained(model_id)
    cfg.attn_implementation = "eager"

    causal = False
    try:
        model = AutoModel.from_pretrained(model_id, torch_dtype=dtype, config=cfg)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, config=cfg)
        causal = True

    encoder = GenericEncoderWrapper(model, causal=causal)

    if torch.cuda.is_available():
        if use_dataparallel and torch.cuda.device_count() >= 2:
            if device_ids is None:
                device_ids = list(range(torch.cuda.device_count()))
            encoder = torch.nn.DataParallel(encoder, device_ids=device_ids).cuda(device_ids[0])
        else:
            encoder = encoder.cuda()

    encoder.eval()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    return tokenizer, encoder


# ========================== EMBEDDINGS ========================== #

def mean_pool(last_hidden_state, attention_mask):
    """Masked mean pooling over tokens."""
    mask = attention_mask.unsqueeze(-1)
    masked = last_hidden_state * mask
    lengths = mask.sum(dim=1).clamp(min=1)
    return masked.sum(dim=1) / lengths


def embed_texts(texts, tokenizer, encoder, batch_size=64, max_length=256) -> np.ndarray:
    """Compute embeddings for input texts."""
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)

    all_embeddings = []
    device = next(encoder.parameters()).device if not isinstance(encoder, torch.nn.DataParallel) else encoder.device_ids[0]

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        last_hidden = encoder(input_ids=input_ids, attention_mask=attention_mask)
        emb = mean_pool(last_hidden, attention_mask)
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        all_embeddings.append(emb.detach().to("cpu", dtype=torch.float32).numpy())

    return np.vstack(all_embeddings).astype(np.float32)


# ========================== CLUSTERING ========================== #

def cluster_and_find_medoids(
    ids: List[Any],
    sentences: List[str],
    k: int,
    model_id: str,
    random_state: int = 42,
    use_dataparallel=True,
    use_pca=True,
    n_pcs=50,
    batch_size=64,
    max_length=256
) -> Tuple[List[Dict[str, Any]], np.ndarray, KMeans, np.ndarray]:
    """Cluster embeddings and find medoids."""
    if not sentences:
        raise ValueError("`sentences` is empty.")
    if len(ids) != len(sentences):
        raise ValueError("`ids` and `sentences` must have the same length.")

    keep = [i for i, s in enumerate(sentences) if isinstance(s, str) and s.strip()]
    ids = [ids[i] for i in keep]
    sentences = [sentences[i] for i in keep]
    n = len(sentences)
    k = max(1, min(k, n))

    tokenizer, encoder = load_encoder(model_id, use_dataparallel=use_dataparallel)
    emb = embed_texts(sentences, tokenizer, encoder, batch_size=batch_size, max_length=max_length)

    if use_pca:
        n_pcs_eff = min(n_pcs, emb.shape[1], emb.shape[0])
        pca = PCA(n_components=n_pcs_eff, random_state=random_state)
        km_input = pca.fit_transform(emb)
    else:
        km_input = emb

    print(f"[INFO] Running KMeans (k={k}) on {km_input.shape[0]} samples, dim={km_input.shape[1]}")
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = km.fit_predict(km_input)

    medoids = []
    for c in range(k):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        E = emb[idx]
        dist = 1.0 - (E @ E.T)
        medoid_local = int(np.argmin(dist.sum(axis=1)))
        medoid_idx = int(idx[medoid_local])
        medoids.append({
            "cluster": c,
            "medoid_id": ids[medoid_idx],
            "medoid_sentence": sentences[medoid_idx],
            "cluster_size": int(len(idx))
        })

    medoids.sort(key=lambda x: x["cluster"])
    return medoids, labels, km, km_input


# ========================== PLOTTING ========================== #

def plot_clusters_2d(km_input, kmeans, labels, medoid_indices, save_path, title="Cluster Visualization"):
    """Plot 2D PCA projections with medoids."""
    plt.figure(figsize=(7, 6))
    emb_2d = km_input[:, :2]
    for c in range(kmeans.n_clusters):
        mask = (labels == c)
        plt.scatter(emb_2d[mask, 0], emb_2d[mask, 1], s=8, alpha=0.15, label=f"Cluster {c}")
    if medoid_indices:
        medoids_2d = emb_2d[medoid_indices]
        plt.scatter(medoids_2d[:, 0], medoids_2d[:, 1], s=140, facecolors="none", edgecolor="black", linewidths=1.5)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"[INFO] Saved plot to {save_path}")


# ========================== MAIN ========================== #

def main():
    parser = argparse.ArgumentParser(description="Universal model-based clustering for sentence embeddings.")
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--use_pca", action="store_true")
    parser.add_argument("--n_pcs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    login_to_huggingface()

    os.makedirs(args.output_dir, exist_ok=True)
    ds = load_dataset("json", data_files=args.data_path)["train"]

    ids = ds["id"]
    texts = ds["text"]

    medoids, labels, km, km_input = cluster_and_find_medoids(
        ids, texts, k=args.k, model_id=args.model_id,
        use_pca=args.use_pca, n_pcs=args.n_pcs,
        batch_size=args.batch_size, max_length=args.max_length
    )

    for m in medoids:
        print(f"[Cluster {m['cluster']}] size={m['cluster_size']} → {m['medoid_sentence']}")

    out_path = os.path.join(args.output_dir, f"medoids_k{args.k}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for m in medoids:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    medoid_indices = [m["cluster"] for m in medoids]
    plot_path = os.path.join(args.output_dir, "clusters_2d.png")
    plot_clusters_2d(km_input, km, labels, medoid_indices, plot_path, title=f"{args.model_id} clusters")

    print(f"[DONE] Medoids and plot saved under {args.output_dir}")


if __name__ == "__main__":
    main()
