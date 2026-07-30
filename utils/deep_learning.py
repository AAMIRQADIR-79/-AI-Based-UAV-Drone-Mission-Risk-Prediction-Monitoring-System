"""
deep_learning.py - Four PyTorch Deep Learning Architectures
1. Deep Residual MLP    - single point risk classification
2. Bidirectional LSTM   - sequential route risk + attention
3. Waypoint Transformer - parallel per-waypoint risk
4. Telemetry Autoencoder- unsupervised anomaly detection
"""

import os
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from utils.config import FEATURE_NAMES, SEQ_LEN
from utils.data_generator import generate_tabular_data, generate_sequential_data

MODEL_DIR = "models"
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _seed(s=42):
    torch.manual_seed(s)
    np.random.seed(s)


# ═══════════════════════════════════════════════════════════
# 1. DEEP RESIDUAL MLP
# ═══════════════════════════════════════════════════════════
class ResidualBlock(nn.Module):
    """Skip connection block - same idea as ResNet for images."""
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.fc1  = nn.Linear(dim, dim)
        self.bn1  = nn.BatchNorm1d(dim)
        self.fc2  = nn.Linear(dim, dim)
        self.bn2  = nn.BatchNorm1d(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.drop(out)
        out = self.bn2(self.fc2(out))
        return F.relu(out + x)   # skip connection


class DeepRiskMLP(nn.Module):
    """
    10 features → 256 → [4 Residual Blocks] → 64 → 1
    Skip connections prevent vanishing gradients in deep network.
    """
    def __init__(self, n_features=10):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList([
            ResidualBlock(256, dropout=0.25) for _ in range(4)
        ])
        self.head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def train_deep_mlp(force_retrain=False, epochs=50):
    path_model  = os.path.join(MODEL_DIR, "deep_mlp.pt")
    path_scaler = os.path.join(MODEL_DIR, "deep_mlp_scaler.joblib")
    path_meta   = os.path.join(MODEL_DIR, "deep_mlp_meta.joblib")

    if not force_retrain and all(os.path.exists(p) for p in [path_model, path_scaler, path_meta]):
        scaler = joblib.load(path_scaler)
        meta   = joblib.load(path_meta)
        model  = DeepRiskMLP(len(FEATURE_NAMES)).to(DEVICE)
        model.load_state_dict(torch.load(path_model, map_location=DEVICE))
        model.eval()
        return model, scaler, meta

    _seed()
    df = generate_tabular_data(n_samples=8000)
    X  = df[FEATURE_NAMES].values
    y  = df["risk"].values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    train_dl = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=64, shuffle=True)

    model = DeepRiskMLP(len(FEATURE_NAMES)).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit  = nn.BCELoss()

    history = {"loss": [], "val_acc": []}
    for epoch in range(epochs):
        model.train()
        ep_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb).squeeze(), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item() * xb.size(0)
        sched.step()
        ep_loss /= len(train_dl.dataset)
        history["loss"].append(round(ep_loss, 4))

        model.eval()
        with torch.no_grad():
            vp = model(torch.tensor(X_test).to(DEVICE)).squeeze().cpu().numpy()
        history["val_acc"].append(round(accuracy_score(y_test, vp >= 0.5), 4))

    acc = round(accuracy_score(y_test, vp >= 0.5) * 100, 2)
    auc = round(roc_auc_score(y_test, vp) * 100, 2)
    f1  = round(f1_score(y_test, vp >= 0.5) * 100, 2)

    meta = {
        "status":       "trained",
        "accuracy":     acc,
        "auc_roc":      auc,
        "f1_score":     f1,
        "history":      history,
        "architecture": "10→256→[ResBlock×4]→64→1",
        "n_params":     sum(p.numel() for p in model.parameters()),
    }

    torch.save(model.state_dict(), path_model)
    joblib.dump(scaler, path_scaler)
    joblib.dump(meta,   path_meta)
    return model, scaler, meta


def predict_deep_mlp(model, scaler, features: dict) -> dict:
    model.eval()
    X = np.array([[features[f] for f in FEATURE_NAMES]], dtype=np.float32)
    with torch.no_grad():
        p = model(torch.tensor(scaler.transform(X))).item()
    return _format_pred(p)


# ═══════════════════════════════════════════════════════════
# 2. BIDIRECTIONAL LSTM + ATTENTION
# ═══════════════════════════════════════════════════════════
class AttentionPool(nn.Module):
    """Learns which waypoint matters most."""
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x):
        w = F.softmax(self.attn(x).squeeze(-1), dim=1)
        return torch.bmm(w.unsqueeze(1), x).squeeze(1), w


class FlightLSTM(nn.Module):
    """
    Bidirectional 2-layer LSTM over waypoint sequence.
    Attention pooling shows WHICH waypoint drove the decision.
    Input:  (batch, seq_len, n_features)
    Output: route risk probability + attention weights
    """
    def __init__(self, n_features=10, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features, hidden, num_layers=2,
            batch_first=True, bidirectional=True, dropout=0.3)
        self.attn = AttentionPool(hidden * 2)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, 32), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, x):
        out, _ = self.lstm(x)
        pooled, weights = self.attn(out)
        return self.head(pooled).squeeze(-1), weights


def train_lstm(force_retrain=False, epochs=40):
    path_model  = os.path.join(MODEL_DIR, "lstm.pt")
    path_scaler = os.path.join(MODEL_DIR, "lstm_scaler.joblib")
    path_meta   = os.path.join(MODEL_DIR, "lstm_meta.joblib")

    if not force_retrain and all(os.path.exists(p) for p in [path_model, path_scaler, path_meta]):
        scaler = joblib.load(path_scaler)
        meta   = joblib.load(path_meta)
        model  = FlightLSTM(len(FEATURE_NAMES)).to(DEVICE)
        model.load_state_dict(torch.load(path_model, map_location=DEVICE))
        model.eval()
        return model, scaler, meta

    _seed()
    X, _, y_route = generate_sequential_data(n_routes=3000)
    n, t, f = X.shape

    scaler  = StandardScaler()
    X_flat  = scaler.fit_transform(X.reshape(-1, f)).astype(np.float32)
    X       = X_flat.reshape(n, t, f)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_route, test_size=0.2, random_state=42, stratify=y_route)

    train_dl = DataLoader(
        TensorDataset(torch.tensor(X_train),
                      torch.tensor(y_train.astype(np.float32))),
        batch_size=64, shuffle=True)

    model = FlightLSTM(f).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit  = nn.BCELoss()

    history = {"loss": [], "val_acc": []}
    for epoch in range(epochs):
        model.train()
        ep_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred, _ = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * xb.size(0)
        ep_loss /= len(train_dl.dataset)
        history["loss"].append(round(ep_loss, 4))

        model.eval()
        with torch.no_grad():
            vp, _ = model(torch.tensor(X_test).to(DEVICE))
            vp = vp.cpu().numpy()
        history["val_acc"].append(round(accuracy_score(y_test, vp >= 0.5), 4))

    acc = round(accuracy_score(y_test, vp >= 0.5) * 100, 2)
    auc = round(roc_auc_score(y_test, vp) * 100, 2)
    f1  = round(f1_score(y_test, vp >= 0.5) * 100, 2)

    meta = {
        "status":       "trained",
        "accuracy":     acc,
        "auc_roc":      auc,
        "f1_score":     f1,
        "history":      history,
        "architecture": "BiLSTM(2L,h=64)+AttentionPool→32→1",
        "n_params":     sum(p.numel() for p in model.parameters()),
    }

    torch.save(model.state_dict(), path_model)
    joblib.dump(scaler, path_scaler)
    joblib.dump(meta,   path_meta)
    return model, scaler, meta


def predict_lstm(model, scaler, waypoint_features: list) -> dict:
    """
    waypoint_features: list of feature dicts, one per waypoint.
    Returns route risk + attention weights per waypoint.
    """
    model.eval()
    X  = np.array([[wf[f] for f in FEATURE_NAMES]
                   for wf in waypoint_features], dtype=np.float32)
    Xs = scaler.transform(X).astype(np.float32)[np.newaxis]
    with torch.no_grad():
        pred, attn = model(torch.tensor(Xs).to(DEVICE))
    p = pred.item()
    result = _format_pred(p)
    result["attention_weights"]    = attn.squeeze(0).cpu().numpy().tolist()
    result["most_critical_waypoint"] = int(np.argmax(result["attention_weights"]))
    return result


# ═══════════════════════════════════════════════════════════
# 3. WAYPOINT TRANSFORMER
# ═══════════════════════════════════════════════════════════
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super().__init__()
        import math
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class WaypointTransformer(nn.Module):
    """
    Each waypoint = one token. Self-attention lets every
    waypoint attend to every other simultaneously.
    Outputs per-waypoint risk scores in one parallel pass.
    """
    def __init__(self, n_features=10, d_model=64, n_heads=4, n_layers=3):
        super().__init__()
        self.proj    = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        enc_layer    = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.2, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head    = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, x):
        h = self.pos_enc(self.proj(x))
        h = self.encoder(h)
        return self.head(h).squeeze(-1)


def train_transformer(force_retrain=False, epochs=40):
    path_model  = os.path.join(MODEL_DIR, "transformer.pt")
    path_scaler = os.path.join(MODEL_DIR, "transformer_scaler.joblib")
    path_meta   = os.path.join(MODEL_DIR, "transformer_meta.joblib")

    if not force_retrain and all(os.path.exists(p) for p in [path_model, path_scaler, path_meta]):
        scaler = joblib.load(path_scaler)
        meta   = joblib.load(path_meta)
        model  = WaypointTransformer(len(FEATURE_NAMES)).to(DEVICE)
        model.load_state_dict(torch.load(path_model, map_location=DEVICE))
        model.eval()
        return model, scaler, meta

    _seed()
    X, y_step, _ = generate_sequential_data(n_routes=3000)
    n, t, f = X.shape

    scaler  = StandardScaler()
    X_flat  = scaler.fit_transform(X.reshape(-1, f)).astype(np.float32)
    X       = X_flat.reshape(n, t, f)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_step, test_size=0.2, random_state=42)

    train_dl = DataLoader(
        TensorDataset(torch.tensor(X_train),
                      torch.tensor(y_train.astype(np.float32))),
        batch_size=48, shuffle=True)

    model = WaypointTransformer(f).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    crit  = nn.BCELoss()

    history = {"loss": [], "val_acc": []}
    for epoch in range(epochs):
        model.train()
        ep_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * xb.size(0)
        ep_loss /= len(train_dl.dataset)
        history["loss"].append(round(ep_loss, 4))

        model.eval()
        with torch.no_grad():
            vp = model(torch.tensor(X_test).to(DEVICE)).cpu().numpy()
        history["val_acc"].append(
            round(accuracy_score(y_test.flatten(), vp.flatten() >= 0.5), 4))

    acc = round(accuracy_score(y_test.flatten(), vp.flatten() >= 0.5) * 100, 2)
    auc = round(roc_auc_score(y_test.flatten(), vp.flatten()) * 100, 2)
    f1  = round(f1_score(y_test.flatten(), vp.flatten() >= 0.5) * 100, 2)

    meta = {
        "status":       "trained",
        "accuracy":     acc,
        "auc_roc":      auc,
        "f1_score":     f1,
        "history":      history,
        "architecture": "Transformer(3L,4heads,d=64)→32→1 per waypoint",
        "n_params":     sum(p.numel() for p in model.parameters()),
    }

    torch.save(model.state_dict(), path_model)
    joblib.dump(scaler, path_scaler)
    joblib.dump(meta,   path_meta)
    return model, scaler, meta


def predict_transformer(model, scaler, waypoint_features: list) -> dict:
    model.eval()
    X  = np.array([[wf[f] for f in FEATURE_NAMES]
                   for wf in waypoint_features], dtype=np.float32)
    Xs = scaler.transform(X).astype(np.float32)[np.newaxis]
    with torch.no_grad():
        per_wp = model(torch.tensor(Xs).to(DEVICE)).squeeze(0).cpu().numpy()
    route_risk = float(per_wp.max())
    result = _format_pred(route_risk)
    result["per_waypoint_risk"] = (per_wp * 100).round(1).tolist()
    result["worst_waypoint"]    = int(np.argmax(per_wp))
    return result


# ═══════════════════════════════════════════════════════════
# 4. TELEMETRY AUTOENCODER (Anomaly Detection)
# ═══════════════════════════════════════════════════════════
class TelemetryAutoencoder(nn.Module):
    """
    Trained ONLY on safe missions.
    High reconstruction error = anomalous mission pattern.
    10 → 16 → 8 → 4(latent) → 8 → 16 → 10
    """
    def __init__(self, n_features=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 16), nn.ReLU(),
            nn.Linear(16, 8),          nn.ReLU(),
            nn.Linear(8, 4),
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),           nn.ReLU(),
            nn.Linear(8, 16),          nn.ReLU(),
            nn.Linear(16, n_features),
        )

    def forward(self, x):
        z    = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


def train_autoencoder(force_retrain=False, epochs=80):
    path_model  = os.path.join(MODEL_DIR, "autoencoder.pt")
    path_scaler = os.path.join(MODEL_DIR, "autoencoder_scaler.joblib")
    path_thresh = os.path.join(MODEL_DIR, "autoencoder_thresh.joblib")
    path_meta   = os.path.join(MODEL_DIR, "autoencoder_meta.joblib")

    if not force_retrain and all(os.path.exists(p) for p in
                                  [path_model, path_scaler, path_thresh, path_meta]):
        scaler = joblib.load(path_scaler)
        thresh = joblib.load(path_thresh)
        meta   = joblib.load(path_meta)
        model  = TelemetryAutoencoder(len(FEATURE_NAMES)).to(DEVICE)
        model.load_state_dict(torch.load(path_model, map_location=DEVICE))
        model.eval()
        return model, scaler, thresh, meta

    _seed()
    df      = generate_tabular_data(n_samples=8000)
    df_safe = df[df["risk"] == 0]
    X       = df_safe[FEATURE_NAMES].values.astype(np.float32)

    scaler  = StandardScaler()
    Xs      = scaler.fit_transform(X).astype(np.float32)
    X_train, X_val = train_test_split(Xs, test_size=0.15, random_state=42)

    train_dl = DataLoader(
        TensorDataset(torch.tensor(X_train)),
        batch_size=64, shuffle=True)

    model = TelemetryAutoencoder(len(FEATURE_NAMES)).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit  = nn.MSELoss()

    history = {"loss": []}
    for epoch in range(epochs):
        model.train()
        ep_loss = 0.0
        for (xb,) in train_dl:
            xb = xb.to(DEVICE)
            opt.zero_grad()
            recon, _ = model(xb)
            loss = crit(recon, xb)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * xb.size(0)
        ep_loss /= len(train_dl.dataset)
        history["loss"].append(round(ep_loss, 5))

    # Set anomaly threshold at 95th percentile of validation errors
    model.eval()
    with torch.no_grad():
        recon_val, _ = model(torch.tensor(X_val).to(DEVICE))
        errors = ((recon_val.cpu().numpy() - X_val) ** 2).mean(axis=1)
    threshold = float(np.percentile(errors, 95))

    meta = {
        "status":       "trained",
        "final_loss":   round(history["loss"][-1], 5),
        "threshold":    round(threshold, 5),
        "history":      history,
        "architecture": "10→16→8→4(latent)→8→16→10",
        "trained_on":   f"{len(X)} safe missions only",
        "n_params":     sum(p.numel() for p in model.parameters()),
    }

    torch.save(model.state_dict(), path_model)
    joblib.dump(scaler,    path_scaler)
    joblib.dump(threshold, path_thresh)
    joblib.dump(meta,      path_meta)
    return model, scaler, threshold, meta


def detect_anomaly(model, scaler, threshold, features: dict) -> dict:
    model.eval()
    X  = np.array([[features[f] for f in FEATURE_NAMES]], dtype=np.float32)
    Xs = scaler.transform(X).astype(np.float32)
    with torch.no_grad():
        recon, latent = model(torch.tensor(Xs).to(DEVICE))
    error = float(((recon.cpu().numpy() - Xs) ** 2).mean())
    score = min(10.0, (error / threshold) * 5) if threshold > 0 else 0

    return {
        "anomaly_score":        round(score, 2),
        "reconstruction_error": round(error, 5),
        "threshold":            round(threshold, 5),
        "is_anomaly":           error > threshold,
        "latent_vector":        latent.cpu().numpy().tolist()[0],
        "message": (
            "⚠️ Anomalous mission profile detected!"
            if error > threshold else
            "✅ Mission profile matches normal patterns."
        ),
    }


# ═══════════════════════════════════════════════════════════
# TRAIN ALL MODELS
# ═══════════════════════════════════════════════════════════
def train_all(force_retrain=False, progress_fn=None):
    """Train or load all 4 deep learning models."""
    results = {}
    steps   = [
        ("deep_mlp",     "Deep Residual MLP",    10),
        ("lstm",         "BiLSTM + Attention",    35),
        ("transformer",  "Waypoint Transformer",  65),
        ("autoencoder",  "Telemetry Autoencoder", 90),
    ]

    for key, name, pct in steps:
        if progress_fn:
            progress_fn(f"Training {name}...", pct)

        if key == "deep_mlp":
            m, s, meta = train_deep_mlp(force_retrain)
            results[key] = {"model": m, "scaler": s, "meta": meta}
        elif key == "lstm":
            m, s, meta = train_lstm(force_retrain)
            results[key] = {"model": m, "scaler": s, "meta": meta}
        elif key == "transformer":
            m, s, meta = train_transformer(force_retrain)
            results[key] = {"model": m, "scaler": s, "meta": meta}
        elif key == "autoencoder":
            m, s, t, meta = train_autoencoder(force_retrain)
            results[key] = {"model": m, "scaler": s, "threshold": t, "meta": meta}

    if progress_fn:
        progress_fn("All models ready!", 100)
    return results


# ═══════════════════════════════════════════════════════════
# SHARED HELPER
# ═══════════════════════════════════════════════════════════
def _format_pred(p: float) -> dict:
    score = p * 100
    if score >= 80:
        level, emoji, color = "Critical",  "🚨", "#FF0000"
    elif score >= 60:
        level, emoji, color = "High Risk", "🔴", "#FF6600"
    elif score >= 35:
        level, emoji, color = "Moderate",  "🟡", "#FFAA00"
    else:
        level, emoji, color = "Safe",      "🟢", "#00CC00"
    return {
        "risk_score": round(score, 1),
        "safe_score": round(100 - score, 1),
        "label":      f"{emoji} {level}",
        "level":      level,
        "emoji":      emoji,
        "color":      color,
        "confidence": round(max(p, 1 - p) * 100, 1),
    }