#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

from layers.utils import Config_80M, SCrna, Prepare, build_dataset
from model import Finetune_Cell_FM


# def prepare_adata(h5ad_path, require_labels=False):
#     adata = sc.read_h5ad(h5ad_path)

#     if "gene_ids" not in adata.var.columns:
#         raise ValueError(f"{h5ad_path} is missing adata.var['gene_ids']")

#     if not sp.issparse(adata.X):
#         adata.X = sp.csr_matrix(adata.X)

#     if require_labels:
#         if "cell_type" not in adata.obs.columns:
#             raise ValueError(f"{h5ad_path} is missing adata.obs['cell_type']")
#         if adata.obs["cell_type"].isna().all():
#             raise ValueError(f"{h5ad_path} has cell_type column but all values are NA")
#     else:
#         if "cell_type" not in adata.obs.columns:
#             adata.obs["cell_type"] = "unknown"

#     # CellFM-torch data pipeline fields
#     adata.obs["celltype"] = adata.obs["cell_type"].astype("category")
#     adata.obs["batch_id"] = 0
#     adata.obs["feat"] = 0
#     adata.obs["train"] = 2  # mark all as test/inference

#     return adata

def prepare_adata(h5ad_path, require_labels=False):
    adata = sc.read_h5ad(h5ad_path)

    if "gene_ids" not in adata.var.columns:
        raise ValueError(f"{h5ad_path} is missing adata.var['gene_ids']")

    # --- Prefer raw counts if available ---
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()
        print(f"[{os.path.basename(h5ad_path)}] Using adata.layers['counts'] as X")
    elif "raw_counts" in adata.layers:
        adata.X = adata.layers["raw_counts"].copy()
        print(f"[{os.path.basename(h5ad_path)}] Using adata.layers['raw_counts'] as X")
    elif adata.raw is not None:
        adata.X = adata.raw.X.copy()
        print(f"[{os.path.basename(h5ad_path)}] Using adata.raw.X as X")
    else:
        print(f"[{os.path.basename(h5ad_path)}] No counts layer/raw found; using adata.X as-is")

    # Make sparse CSR
    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)
    else:
        adata.X = adata.X.tocsr()

    # --- Guardrails: X must be nonnegative counts ---
    if adata.X.data.size > 0:
        neg = (adata.X.data < 0).sum()
        if neg > 0:
            print(f"[{os.path.basename(h5ad_path)}] WARNING: found {neg} negative entries in X; clipping to 0")
            adata.X.data[adata.X.data < 0] = 0
            adata.X.eliminate_zeros()

        # Optional: force integer-ish counts (binomial n really wants counts)
        adata.X.data = np.rint(adata.X.data).astype(np.float32)

    # --- Drop all-zero observations (avoid read=0 -> NaNs in normalization/log1p) ---
    libsize = np.asarray(adata.X.sum(axis=1)).ravel()
    zero_rows = (libsize <= 0).sum()
    if zero_rows > 0:
        print(f"[{os.path.basename(h5ad_path)}] WARNING: dropping {zero_rows} all-zero samples (no counts in gene set)")
        adata = adata[libsize > 0].copy()

    # Labels
    if require_labels:
        if "cell_type" not in adata.obs.columns:
            raise ValueError(f"{h5ad_path} is missing adata.obs['cell_type']")
        if adata.obs["cell_type"].isna().all():
            raise ValueError(f"{h5ad_path} has cell_type column but all values are NA")
    else:
        if "cell_type" not in adata.obs.columns:
            adata.obs["cell_type"] = "unknown"

    # CellFM-torch expected fields
    adata.obs["celltype"] = adata.obs["cell_type"].astype("category")
    adata.obs["batch_id"] = 0
    adata.obs["feat"] = 0
    adata.obs["train"] = 2

    return adata


def check_feature_alignment(ref_adata, test_adata):
    same_var_names = np.array_equal(ref_adata.var_names.astype(str), test_adata.var_names.astype(str))
    same_gene_ids = np.array_equal(
        ref_adata.var["gene_ids"].astype(str).to_numpy(),
        test_adata.var["gene_ids"].astype(str).to_numpy()
    )

    if not same_var_names:
        raise ValueError("Reference and test var_names do not match exactly.")
    if not same_gene_ids:
        raise ValueError("Reference and test var['gene_ids'] do not match exactly.")


def make_loader(adata, batch_size=8, mask_ratio=0.5):
    dataset = SCrna(adata, mode="test")
    prep = Prepare(2048, pad=0, mask_ratio=mask_ratio)

    try:
        loader = build_dataset(
            dataset,
            prep=prep,
            batch_size=batch_size,
            pad_zero=True,
            drop=False,
            shuffle=False,
            num_workers=0,
        )
    except TypeError:
        loader = build_dataset(
            dataset,
            prep=prep,
            batch_size=batch_size,
            pad_zero=True,
            drop=False,
            shuffle=False,
        )

    return loader


def extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for key in ["state_dict", "model_state_dict", "model", "net"]:
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return ckpt_obj[key]
    return ckpt_obj


def strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    out = {}
    for k, v in state_dict.items():
        out[k[7:] if k.startswith("module.") else k] = v
    return out


def infer_num_classes_for_wrapper(state_dict):
    # Finetune_Cell_FM requires num_cls even if we only use embeddings.
    # We just need a compatible wrapper to load the checkpoint.
    candidate_keys = ["cls.weight", "classifier.weight", "head.weight", "fc.weight"]
    for key in candidate_keys:
        if key in state_dict and hasattr(state_dict[key], "shape") and len(state_dict[key].shape) == 2:
            return int(state_dict[key].shape[0])

    # fallback: harmless default for unsupervised-style checkpoints
    return 2


def load_cellfm_model(checkpoint_path, device="cuda:0"):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = strip_module_prefix(extract_state_dict(ckpt))

    drop_keys = [k for k in state_dict.keys() if k.startswith("cls.")]
    for k in drop_keys:
        del state_dict[k]
    print(f"Dropped {len(drop_keys)} cls.* keys from checkpoint (embedding-only inference).")

    cfg = Config_80M()

    # Required runtime/config fields
    cfg.device = device
    cfg.ckpt_path = checkpoint_path
    cfg.num_cls = infer_num_classes_for_wrapper(state_dict)

    # Data / masking settings commonly expected by CellFM-torch
    cfg.pad_zero = True
    cfg.add_zero = True
    cfg.mask_ratio = 0.5

    # Attributes referenced by model/layers code
    cfg.ecs = True
    cfg.ecs_threshold = 0.8
    cfg.recompute = False
    cfg.sim = False

    # Safe defaults for wrapper/inference
    cfg.use_bs = 8
    cfg.epoch = 1

    model = Finetune_Cell_FM(cfg).to(device)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Inferred num_classes: {cfg.num_cls}")
    print(f"Missing keys: {missing}")
    print(f"Unexpected keys: {unexpected}")

    model.eval()
    return model

@torch.no_grad()
def extract_embeddings(model, loader, device="cuda:0"):
    embs = []

    for batch in loader:
        raw_nzdata = batch["raw_nzdata"].to(device)
        dw_nzdata = batch["dw_nzdata"].to(device)
        ST_feat = batch["ST_feat"].to(device)
        nonz_gene = batch["nonz_gene"].to(device)
        mask_gene = batch["mask_gene"].to(device)
        zero_idx = batch["zero_idx"].to(device)

        outputs = model(
            raw_nzdata=raw_nzdata,
            dw_nzdata=dw_nzdata,
            ST_feat=ST_feat,
            nonz_gene=nonz_gene,
            mask_gene=mask_gene,
            zero_idx=zero_idx,
        )

        # Finetune_Cell_FM typically returns (cls_logits, mask_loss, cls_token)
        if isinstance(outputs, (tuple, list)) and len(outputs) >= 3:
            cls_token = outputs[2]
        else:
            raise ValueError("Unexpected model output format; could not find cls_token.")

        embs.append(cls_token.detach().cpu().numpy())

    return np.vstack(embs)


def run_label_transfer(
    ref_h5ad,
    test_h5ad,
    checkpoint,
    out_pred_csv,
    out_ref_emb_csv,
    out_test_emb_csv,
    k=5,
    batch_size=8,
    device="cuda:0"
):
    ref_adata = prepare_adata(ref_h5ad, require_labels=True)
    test_adata = prepare_adata(test_h5ad, require_labels=False)

    check_feature_alignment(ref_adata, test_adata)

    model = load_cellfm_model(checkpoint, device=device)

    ref_loader = make_loader(ref_adata, batch_size=batch_size)
    test_loader = make_loader(test_adata, batch_size=batch_size)

    print("Extracting reference embeddings...")
    ref_emb = extract_embeddings(model, ref_loader, device=device)

    print("Extracting test embeddings...")
    test_emb = extract_embeddings(model, test_loader, device=device)

    # Encode labels from reference
    ref_labels = ref_adata.obs["cell_type"].astype(str).to_numpy()
    le = LabelEncoder()
    y_ref = le.fit_transform(ref_labels)

    # kNN in embedding space
    knn = KNeighborsClassifier(n_neighbors=k, weights="distance", metric="euclidean")
    knn.fit(ref_emb, y_ref)

    pred_idx = knn.predict(test_emb)
    pred_prob = knn.predict_proba(test_emb)
    pred_label = le.inverse_transform(pred_idx)

    # nearest-neighbor metadata
    distances, neighbor_idx = knn.kneighbors(test_emb, n_neighbors=k, return_distance=True)

    out_df = pd.DataFrame({
        "sample_id": test_adata.obs_names.to_numpy(),
        "pred_cell_type": pred_label,
        "pred_class_idx": pred_idx,
    })

    for i, cls in enumerate(le.classes_):
        safe_cls = str(cls).replace(" ", "_")
        out_df[f"vote_frac_{safe_cls}"] = pred_prob[:, i]

    # include nearest neighbors from reference
    for j in range(k):
        out_df[f"nn{j+1}_ref_sample"] = ref_adata.obs_names.to_numpy()[neighbor_idx[:, j]]
        out_df[f"nn{j+1}_ref_cell_type"] = ref_labels[neighbor_idx[:, j]]
        out_df[f"nn{j+1}_distance"] = distances[:, j]

    out_df.to_csv(out_pred_csv, index=False)

    ref_emb_df = pd.DataFrame(ref_emb, index=ref_adata.obs_names)
    ref_emb_df.index.name = "sample_id"
    ref_emb_df["cell_type"] = ref_labels
    ref_emb_df.to_csv(out_ref_emb_csv)

    test_emb_df = pd.DataFrame(test_emb, index=test_adata.obs_names)
    test_emb_df.index.name = "sample_id"
    test_emb_df.to_csv(out_test_emb_csv)

    print(f"Saved predictions to: {out_pred_csv}")
    print(f"Saved reference embeddings to: {out_ref_emb_csv}")
    print(f"Saved test embeddings to: {out_test_emb_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_h5ad", required=True, help="Labeled reference h5ad with obs['cell_type']")
    parser.add_argument("--test_h5ad", required=True, help="Unlabeled test h5ad")
    parser.add_argument("--checkpoint", required=True, help="Unsupervised CellFM checkpoint (.pth)")
    parser.add_argument("--out_pred_csv", default="cellfm_knn_predictions.csv")
    parser.add_argument("--out_ref_emb_csv", default="cellfm_reference_embeddings.csv")
    parser.add_argument("--out_test_emb_csv", default="cellfm_test_embeddings.csv")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    run_label_transfer(
        ref_h5ad=args.ref_h5ad,
        test_h5ad=args.test_h5ad,
        checkpoint=args.checkpoint,
        out_pred_csv=args.out_pred_csv,
        out_ref_emb_csv=args.out_ref_emb_csv,
        out_test_emb_csv=args.out_test_emb_csv,
        k=args.k,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()

