"""
=============================================================
 Gold ML Model — XGBoost + LSTM + Ensemble
=============================================================
 ใช้ต่อจาก gold_features.py
 ติดตั้ง:
   pip install xgboost lightgbm scikit-learn torch pandas numpy joblib

 วิธีใช้:
   python gold_model.py            # train + save model
   python gold_model.py --predict  # โหลด model + predict ราคาปัจจุบัน
=============================================================
"""

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

# ─── torch (optional — ถ้าไม่มีก็ยังใช้ XGBoost ได้) ────
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[WARN] torch ไม่ได้ติดตั้ง — จะข้าม LSTM model")

import xgboost as xgb
import lightgbm as lgb

# ─── Config ───────────────────────────────────────────────
MODEL_DIR   = Path("./gold_models")
DATA_DIR    = Path("./gold_data")
MODEL_DIR.mkdir(exist_ok=True)

SEQ_LEN     = 30        # LSTM ดูย้อนหลัง 30 candle
EPOCHS      = 50
BATCH_SIZE  = 64
LR          = 1e-3
DEVICE      = "cuda" if HAS_TORCH and torch.cuda.is_available() else "cpu"

# ════════════════════════════════════════════════════════════
#  FEATURE GROUPS (นำมาจาก gold_features.py)
# ════════════════════════════════════════════════════════════

FEATURE_GROUPS = {
    "trend"     : ["dist_EMA9","dist_EMA21","dist_EMA50","dist_EMA200",
                   "slope_EMA9","slope_EMA21","ema_bull_stack","ema_bear_stack",
                   "MACD","MACD_signal","MACD_hist","MACD_cross",
                   "ADX","ADX_diff","lr_slope20","above_cloud","below_cloud"],
    "momentum"  : ["RSI7","RSI14","RSI21","RSI14_slope",
                   "RSI_oversold","RSI_overbought",
                   "Stoch_K","Stoch_D","Stoch_diff",
                   "stoch_cross_up","stoch_cross_down",
                   "CCI20","WilliamsR","ROC5","ROC10","MOM10","TSI"],
    "volatility": ["ATR7","ATR14","ATR21","ATR_ratio",
                   "BB_width","BB_pct","BB_squeeze",
                   "price_above_BB_upper","price_below_BB_lower",
                   "HV20","candle_range_pct"],
    "volume"    : ["OBV_slope","VOL_ratio","VOL_spike",
                   "dist_VWAP","MFI14","CMF20"],
    "candle"    : ["body_ratio","doji","hammer","shooting_star",
                   "bull_engulf","bear_engulf","inside_bar",
                   "3_bull","3_bear","is_bull"],
    "structure" : ["dist_high_10","dist_low_10","dist_high_20","dist_low_20",
                   "dist_pivot","breakout_up","breakout_down"],
    "time"      : ["hour_sin","hour_cos","dow_sin","dow_cos",
                   "is_london","is_ny","is_overlap","is_asian"],
    "lag"       : ["ret_1","ret_2","ret_3","ret_5",
                   "lag_RSI14_1","lag_RSI14_2",
                   "lag_MACD_hist_1","lag_MACD_hist_2",
                   "rolling_std_5","rolling_std_10","rolling_std_20"],
}

def get_feature_cols(df):
    cols = []
    for g in FEATURE_GROUPS.values():
        cols += [c for c in g if c in df.columns]
    return list(dict.fromkeys(cols))   # dedup รักษาลำดับ


# ════════════════════════════════════════════════════════════
#  DATA LOADER
# ════════════════════════════════════════════════════════════

def load_data(tf: str = "M15") -> tuple[pd.DataFrame, list, str]:
    path = DATA_DIR / f"XAUUSD_{tf}_features.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"ไม่พบ {path}\n"
            "รัน gold_features.py ก่อน เพื่อสร้างไฟล์ features"
        )

    df = pd.read_csv(path, index_col=0, parse_dates=True)

    target_col = "target"
    if target_col not in df.columns:
        raise ValueError("ไม่พบ column 'target' — รัน build_features(df, target_pips=3.0) ก่อน")

    feat_cols = get_feature_cols(df)
    df = df[feat_cols + [target_col]].dropna()

    # แปลง target: -1,0,1 → 0,1,2 สำหรับ classifier
    df["label"] = df[target_col].map({-1: 0, 0: 1, 1: 2})

    print(f"[DATA] โหลด {tf}: {len(df)} แถว | {len(feat_cols)} features")
    print(f"       DOWN(0)={( df['label']==0).sum()} "
          f"NEUTRAL(1)={(df['label']==1).sum()} "
          f"UP(2)={(df['label']==2).sum()}")
    return df, feat_cols, target_col


def train_test_split_ts(df, test_ratio=0.2):
    n = len(df)
    split = int(n * (1 - test_ratio))
    return df.iloc[:split], df.iloc[split:]


# ════════════════════════════════════════════════════════════
#  MODEL 1 — XGBoost
# ════════════════════════════════════════════════════════════

class XGBModel:
    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators     = 500,
            max_depth        = 6,
            learning_rate    = 0.05,
            subsample        = 0.8,
            colsample_bytree = 0.8,
            min_child_weight = 5,
            gamma            = 0.1,
            reg_alpha        = 0.1,
            reg_lambda       = 1.0,
            use_label_encoder= False,
            eval_metric      = "mlogloss",
            random_state     = 42,
            n_jobs           = -1,
            verbosity        = 0,
        )
        self.scaler = RobustScaler()
        self.feat_cols = None

    def fit(self, df_train, df_val, feat_cols):
        self.feat_cols = feat_cols
        X_tr = self.scaler.fit_transform(df_train[feat_cols])
        y_tr = df_train["label"].values
        X_va = self.scaler.transform(df_val[feat_cols])
        y_va = df_val["label"].values

        # ── ตรวจสอบว่ามีครบ 3 class (0,1,2) ──────────────────
        # ถ้าข้อมูลไม่มี NEUTRAL (1) ให้เพิ่ม synthetic rows เข้าไป
        unique = np.unique(y_tr)
        if len(unique) < 3:
            missing = [c for c in [0, 1, 2] if c not in unique]
            print(f"[XGB] พบ class ไม่ครบ — เพิ่ม synthetic rows สำหรับ class {missing}")
            mean_X = X_tr.mean(axis=0, keepdims=True)
            for cls in missing:
                X_tr = np.vstack([X_tr, mean_X])
                y_tr = np.append(y_tr, cls)

        self.model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
        )
        return self

    def predict_proba(self, df):
        X = self.scaler.transform(df[self.feat_cols])
        return self.model.predict_proba(X)

    def predict(self, df):
        return np.argmax(self.predict_proba(df), axis=1)

    def feature_importance(self, top_n=20):
        imp = pd.Series(
            self.model.feature_importances_,
            index=self.feat_cols
        ).sort_values(ascending=False)
        return imp.head(top_n)

    def save(self, path):
        joblib.dump({"model": self.model, "scaler": self.scaler,
                     "feat_cols": self.feat_cols}, path)

    @classmethod
    def load(cls, path):
        obj = cls()
        d = joblib.load(path)
        obj.model = d["model"]
        obj.scaler = d["scaler"]
        obj.feat_cols = d["feat_cols"]
        return obj


# ════════════════════════════════════════════════════════════
#  MODEL 2 — LightGBM
# ════════════════════════════════════════════════════════════

class LGBModel:
    def __init__(self):
        self.model = lgb.LGBMClassifier(
            n_estimators    = 500,
            max_depth       = 6,
            learning_rate   = 0.05,
            num_leaves      = 63,
            min_child_samples=20,
            subsample       = 0.8,
            colsample_bytree= 0.8,
            reg_alpha       = 0.1,
            reg_lambda      = 1.0,
            random_state    = 42,
            n_jobs          = -1,
            verbose         = -1,
        )
        self.scaler    = RobustScaler()
        self.feat_cols = None

    def fit(self, df_train, df_val, feat_cols):
        self.feat_cols = feat_cols
        X_tr = self.scaler.fit_transform(df_train[feat_cols])
        y_tr = df_train["label"].values
        X_va = self.scaler.transform(df_val[feat_cols])
        y_va = df_val["label"].values

        # ── ตรวจสอบว่ามีครบ 3 class ──────────────────────────
        unique_lgb = np.unique(y_tr)
        if len(unique_lgb) < 3:
            missing_lgb = [c for c in [0, 1, 2] if c not in unique_lgb]
            print(f"[LGB] พบ class ไม่ครบ — เพิ่ม synthetic rows สำหรับ class {missing_lgb}")
            mean_X_lgb = X_tr.mean(axis=0, keepdims=True)
            for cls in missing_lgb:
                X_tr = np.vstack([X_tr, mean_X_lgb])
                y_tr = np.append(y_tr, cls)

        self.model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(period=-1)],
        )
        return self

    def predict_proba(self, df):
        X = self.scaler.transform(df[self.feat_cols])
        return self.model.predict_proba(X)

    def predict(self, df):
        return np.argmax(self.predict_proba(df), axis=1)

    def save(self, path):
        joblib.dump({"model": self.model, "scaler": self.scaler,
                     "feat_cols": self.feat_cols}, path)

    @classmethod
    def load(cls, path):
        obj = cls()
        d = joblib.load(path)
        obj.model = d["model"]
        obj.scaler = d["scaler"]
        obj.feat_cols = d["feat_cols"]
        return obj


# ════════════════════════════════════════════════════════════
#  MODEL 3 — LSTM (PyTorch)
# ════════════════════════════════════════════════════════════

if HAS_TORCH:
    class LSTMNet(nn.Module):
        def __init__(self, input_size, hidden=128, num_layers=2,
                     dropout=0.3, n_classes=3):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size, hidden,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )
            self.attn = nn.Linear(hidden, 1)   # self-attention
            self.head = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, 64),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(64, n_classes),
            )

        def forward(self, x):
            out, _ = self.lstm(x)               # (B, T, H)
            w = torch.softmax(self.attn(out), dim=1)  # (B, T, 1)
            ctx = (out * w).sum(dim=1)          # (B, H) weighted sum
            return self.head(ctx)


class LSTMModel:
    def __init__(self):
        self.net       = None
        self.scaler    = RobustScaler()
        self.feat_cols = None

    def _make_sequences(self, X_arr, y_arr=None):
        Xs, ys = [], []
        for i in range(SEQ_LEN, len(X_arr)):
            Xs.append(X_arr[i-SEQ_LEN:i])
            if y_arr is not None:
                ys.append(y_arr[i])
        X_out = np.array(Xs, dtype=np.float32)
        y_out = np.array(ys, dtype=np.int64) if y_arr is not None else None
        return X_out, y_out

    def fit(self, df_train, df_val, feat_cols):
        if not HAS_TORCH:
            print("[SKIP] torch ไม่มี — ข้าม LSTM")
            return self

        self.feat_cols = feat_cols
        X_tr_raw = self.scaler.fit_transform(df_train[feat_cols])
        X_va_raw = self.scaler.transform(df_val[feat_cols])
        y_tr = df_train["label"].values
        y_va = df_val["label"].values

        X_tr, y_tr = self._make_sequences(X_tr_raw, y_tr)
        X_va, y_va = self._make_sequences(X_va_raw, y_va)

        tr_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
        va_ds = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va))
        tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
        va_dl = DataLoader(va_ds, batch_size=BATCH_SIZE)

        self.net = LSTMNet(len(feat_cols)).to(DEVICE)
        opt      = torch.optim.AdamW(self.net.parameters(), lr=LR, weight_decay=1e-4)
        sched    = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
        loss_fn  = nn.CrossEntropyLoss(
            weight=torch.tensor([1.5, 0.5, 1.5]).to(DEVICE)  # ให้น้ำหนัก UP/DOWN มากกว่า
        )

        best_acc = 0
        best_state = None

        for ep in range(1, EPOCHS + 1):
            # Train
            self.net.train()
            for xb, yb in tr_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                loss_fn(self.net(xb), yb).backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
            sched.step()

            # Validate
            self.net.eval()
            preds, trues = [], []
            with torch.no_grad():
                for xb, yb in va_dl:
                    p = self.net(xb.to(DEVICE)).argmax(1).cpu().numpy()
                    preds.extend(p); trues.extend(yb.numpy())
            acc = accuracy_score(trues, preds)
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}

            if ep % 10 == 0:
                print(f"   LSTM ep {ep:3d}/{EPOCHS} | val_acc={acc:.4f} (best={best_acc:.4f})")

        self.net.load_state_dict(best_state)
        print(f"   LSTM best val_acc = {best_acc:.4f}")
        return self

    def predict_proba(self, df):
        if not HAS_TORCH or self.net is None:
            n = len(df)
            return np.full((max(0, n - SEQ_LEN), 3), 1/3)

        X_raw = self.scaler.transform(df[self.feat_cols])
        X, _  = self._make_sequences(X_raw)
        self.net.eval()
        with torch.no_grad():
            logits = self.net(torch.from_numpy(X).to(DEVICE))
            proba  = torch.softmax(logits, dim=1).cpu().numpy()
        return proba

    def predict(self, df):
        return np.argmax(self.predict_proba(df), axis=1)

    def save(self, path):
        data = {"scaler": self.scaler, "feat_cols": self.feat_cols}
        if HAS_TORCH and self.net is not None:
            data["state_dict"]  = self.net.state_dict()
            data["input_size"]  = len(self.feat_cols)
        joblib.dump(data, path)

    @classmethod
    def load(cls, path):
        obj = cls()
        d = joblib.load(path)
        obj.scaler    = d["scaler"]
        obj.feat_cols = d["feat_cols"]
        if HAS_TORCH and "state_dict" in d:
            obj.net = LSTMNet(d["input_size"]).to(DEVICE)
            obj.net.load_state_dict(d["state_dict"])
            obj.net.eval()
        return obj


# ════════════════════════════════════════════════════════════
#  ENSEMBLE
# ════════════════════════════════════════════════════════════

class EnsembleModel:
    """
    รวม XGBoost + LightGBM + LSTM ด้วย weighted average
    weights ปรับตาม validation accuracy ของแต่ละโมเดล
    """

    LABEL_MAP = {0: "DOWN ↓", 1: "NEUTRAL —", 2: "UP ↑"}

    def __init__(self, weights=(0.35, 0.35, 0.30)):
        self.xgb_m  = XGBModel()
        self.lgb_m  = LGBModel()
        self.lstm_m = LSTMModel()
        self.weights = np.array(weights)
        self.weights /= self.weights.sum()

    def fit(self, df, feat_cols, tf="M15"):
        df_train, df_val = train_test_split_ts(df)
        print(f"\n{'─'*50}")
        print(f"  Train: {len(df_train)} | Val: {len(df_val)}")
        print(f"{'─'*50}")

        print("\n[1/3] Training XGBoost...")
        self.xgb_m.fit(df_train, df_val, feat_cols)
        xgb_acc = accuracy_score(df_val["label"], self.xgb_m.predict(df_val))
        print(f"      XGBoost val_acc = {xgb_acc:.4f}")

        print("\n[2/3] Training LightGBM...")
        self.lgb_m.fit(df_train, df_val, feat_cols)
        lgb_acc = accuracy_score(df_val["label"], self.lgb_m.predict(df_val))
        print(f"      LightGBM val_acc = {lgb_acc:.4f}")

        print("\n[3/3] Training LSTM...")
        self.lstm_m.fit(df_train, df_val, feat_cols)

        # ปรับ weight ตาม accuracy
        accs = np.array([xgb_acc, lgb_acc, xgb_acc])  # fallback ถ้า LSTM skip
        if HAS_TORCH and self.lstm_m.net is not None:
            lstm_preds = self.lstm_m.predict(df_val)
            # align กับ df_val (LSTM ตัด SEQ_LEN แถวแรก)
            lstm_acc = accuracy_score(
                df_val["label"].values[SEQ_LEN:], lstm_preds
            )
            accs[2] = lstm_acc
            print(f"      LSTM val_acc = {lstm_acc:.4f}")

        self.weights = accs / accs.sum()
        print(f"\n  Ensemble weights: "
              f"XGB={self.weights[0]:.2f} "
              f"LGB={self.weights[1]:.2f} "
              f"LSTM={self.weights[2]:.2f}")

        # Full report บน val set
        self._eval_report(df_val, "Validation")
        self._save_all(tf)
        return self

    def predict_proba_ensemble(self, df):
        p_xgb  = self.xgb_m.predict_proba(df)                    # (N, 3)
        p_lgb  = self.lgb_m.predict_proba(df)                    # (N, 3)
        p_lstm = self.lstm_m.predict_proba(df)                   # (N-SEQ_LEN, 3)

        n = min(len(p_xgb), len(p_lgb), len(p_lstm))
        if n == 0:
            return np.full((1, 3), 1/3)

        p = (self.weights[0] * p_xgb[-n:]
           + self.weights[1] * p_lgb[-n:]
           + self.weights[2] * p_lstm[-n:])
        return p

    def predict_signal(self, df, last_close: float, atr: float) -> dict:
        """
        คืน signal พร้อม Entry / SL / TP สำหรับ 1 แถวล่าสุด
        """
        proba = self.predict_proba_ensemble(df)
        if len(proba) == 0:
            return {"error": "ข้อมูลไม่พอ"}

        last_proba = proba[-1]
        label      = int(np.argmax(last_proba))
        confidence = float(last_proba[label]) * 100

        direction  = self.LABEL_MAP[label]
        atr_sl     = round(atr * 1.5, 2)
        atr_tp     = round(atr * 2.5, 2)

        if label == 2:   # UP
            entry = round(last_close, 2)
            sl    = round(last_close - atr_sl, 2)
            tp    = round(last_close + atr_tp, 2)
            rr    = round(atr_tp / atr_sl, 2)
        elif label == 0: # DOWN
            entry = round(last_close, 2)
            sl    = round(last_close + atr_sl, 2)
            tp    = round(last_close - atr_tp, 2)
            rr    = round(atr_tp / atr_sl, 2)
        else:
            entry = sl = tp = rr = None

        return {
            "direction"  : direction,
            "confidence" : f"{confidence:.1f}%",
            "entry"      : entry,
            "SL"         : sl,
            "TP"         : tp,
            "RR"         : rr,
            "pips_target": round(atr_tp, 1) if tp else None,
            "proba"      : {
                "DOWN"   : f"{last_proba[0]*100:.1f}%",
                "NEUTRAL": f"{last_proba[1]*100:.1f}%",
                "UP"     : f"{last_proba[2]*100:.1f}%",
            },
        }

    def _eval_report(self, df_val, name="Test"):
        print(f"\n  ── {name} Report ──")
        label_map = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}

        preds_xgb = self.xgb_m.predict(df_val)
        present   = sorted(np.unique(np.concatenate([df_val["label"].values, preds_xgb])))
        tnames    = [label_map[i] for i in present]
        print(f"\n  XGBoost:\n{classification_report(df_val['label'], preds_xgb, labels=present, target_names=tnames, zero_division=0)}")

        proba  = self.predict_proba_ensemble(df_val)
        n      = len(proba)
        y_true = df_val["label"].values[-n:]
        y_pred = np.argmax(proba, axis=1)
        present2 = sorted(np.unique(np.concatenate([y_true, y_pred])))
        tnames2  = [label_map[i] for i in present2]
        print(f"  Ensemble accuracy: {accuracy_score(y_true, y_pred):.4f}")
        print(f"\n{classification_report(y_true, y_pred, labels=present2, target_names=tnames2, zero_division=0)}")

    def _save_all(self, tf):
        self.xgb_m.save(MODEL_DIR / f"xgb_{tf}.pkl")
        self.lgb_m.save(MODEL_DIR / f"lgb_{tf}.pkl")
        self.lstm_m.save(MODEL_DIR / f"lstm_{tf}.pkl")
        meta = {
            "weights"  : self.weights.tolist(),
            "tf"       : tf,
            "seq_len"  : SEQ_LEN,
            "feat_cols": self.xgb_m.feat_cols,
        }
        with open(MODEL_DIR / f"meta_{tf}.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"\n  💾 บันทึก model ที่ {MODEL_DIR}/")

    @classmethod
    def load_all(cls, tf="M15"):
        obj = cls()
        obj.xgb_m  = XGBModel.load(MODEL_DIR / f"xgb_{tf}.pkl")
        obj.lgb_m  = LGBModel.load(MODEL_DIR / f"lgb_{tf}.pkl")
        obj.lstm_m = LSTMModel.load(MODEL_DIR / f"lstm_{tf}.pkl")
        with open(MODEL_DIR / f"meta_{tf}.json") as f:
            meta = json.load(f)
        obj.weights = np.array(meta["weights"])
        return obj


# ════════════════════════════════════════════════════════════
#  TRAIN ALL TIMEFRAMES
# ════════════════════════════════════════════════════════════

def train_all():
    ensembles = {}
    for tf in ["M15", "M30", "H1"]:
        feat_path = DATA_DIR / f"XAUUSD_{tf}_features.csv"
        if not feat_path.exists():
            print(f"\n[SKIP] ไม่พบ {feat_path}")
            continue

        print(f"\n{'═'*55}")
        print(f"  Training TF = {tf}")
        print(f"{'═'*55}")

        df, feat_cols, _ = load_data(tf)
        ens = EnsembleModel()
        ens.fit(df, feat_cols, tf)
        ensembles[tf] = ens

    print("\n✅ Train เสร็จทุก Timeframe!")
    return ensembles


# ════════════════════════════════════════════════════════════
#  PREDICT (ใช้ตอน realtime)
# ════════════════════════════════════════════════════════════

def predict_realtime():
    try:
        import MetaTrader5 as mt5
        from gold_mt5_pipeline import connect_mt5, fetch_candles, TIMEFRAMES
        from gold_features import build_features
        has_mt5 = connect_mt5()
    except ImportError:
        has_mt5 = False

    results = {}

    for tf in ["M15", "M30", "H1"]:
        meta_path = MODEL_DIR / f"meta_{tf}.json"
        if not meta_path.exists():
            print(f"[SKIP] ยังไม่มี model {tf} — รัน train ก่อน")
            continue

        print(f"\n── Predicting {tf} ──")

        try:
            ens = EnsembleModel.load_all(tf)
        except Exception as e:
            print(f"[ERROR] โหลด model {tf}: {e}")
            continue

        # ดึงข้อมูล
        if has_mt5:
            tf_map = {"M15": mt5.TIMEFRAME_M15,
                      "M30": mt5.TIMEFRAME_M30,
                      "H1" : mt5.TIMEFRAME_H1}
            df_raw = fetch_candles(tf, tf_map[tf], n=600)
            df_feat = build_features(df_raw)
        else:
            feat_path = DATA_DIR / f"XAUUSD_{tf}_features.csv"
            if not feat_path.exists():
                print(f"[SKIP] ไม่พบ {feat_path}")
                continue
            df_feat = pd.read_csv(feat_path, index_col=0, parse_dates=True)
            df_feat = df_feat.dropna()

        last_close = float(df_feat["Close"].iloc[-1]) if "Close" in df_feat.columns else 0
        atr = float(df_feat["ATR14"].iloc[-1]) if "ATR14" in df_feat.columns else 5.0

        sig = ens.predict_signal(df_feat, last_close, atr)
        results[tf] = sig

        print(f"  ทิศทาง   : {sig.get('direction', 'N/A')}")
        print(f"  Confidence: {sig.get('confidence', 'N/A')}")
        print(f"  Entry    : {sig.get('entry', '-')}")
        print(f"  SL       : {sig.get('SL', '-')}  TP: {sig.get('TP', '-')}")
        print(f"  R:R      : 1:{sig.get('RR', '-')}")
        print(f"  Proba    : {sig.get('proba', {})}")

    if has_mt5:
        mt5.shutdown()

    # บันทึก
    with open(DATA_DIR / "realtime_signals.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 บันทึก signal ที่ {DATA_DIR}/realtime_signals.json")
    return results


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict", action="store_true",
                        help="โหลด model ที่ train แล้ว + predict realtime")
    args = parser.parse_args()

    if args.predict:
        predict_realtime()
    else:
        train_all()