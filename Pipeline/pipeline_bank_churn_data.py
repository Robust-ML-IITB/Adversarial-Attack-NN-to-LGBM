# #!/usr/bin/env python3
# # pipeline_website_arff_enhanced.py
# """
# Adversarial pipeline for ARFF dataset with target 'WebsiteType'.
# ENHANCED VERSION:
# - Whitebox Attacks (FGSM, PGD) + Blackbox
# - Weight Saving (Autoencoder & Surrogate)
# - Categorical Flip (Hamming Distance) Analysis
# - Results saved to CSV

"""
CORRECTED ADVERSARIAL PIPELINE FOR DISCRETE DATASETS (Phishing/Website)
- Forces integer/discrete columns to be treated as Categorical (One-Hot).
- Solves the "Continuous vs Discrete" attack disconnect via Argmax decoding.
- Prints exact count of Categorical vs Numeric features.
- Saves results and weights.
"""

import argparse
import warnings
import time
import os
from io import BytesIO
from itertools import product
from collections import Counter

import numpy as np
import pandas as pd
import arff

warnings.filterwarnings("ignore")

# PyTorch, sklearn, lightgbm, ART
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import (
    HopSkipJump, 
    FastGradientMethod, 
    ProjectedGradientDescent
)

# -------------------------
# CONFIGURATION
# -------------------------
class Config:
    TEST_SIZE = 0.2
    VAL_SIZE = 0.15
    RANDOM_STATE = 42

    # Output Files
    AE_WEIGHTS_FILE = "best_autoencoder_website.pth"
    SURROGATE_WEIGHTS_FILE = "surrogate_model_website.pth"
    RESULTS_FILE = "attack_summary_website.csv"

    # Hyperparameters
    LATENT_DIMS = [16, 32, 48]
    LEARNING_RATES = [1e-3, 2e-3]
    BATCH_SIZES = [256]
    DROPOUTS = [0.1, 0.2]

    DEFAULT_LATENT_DIM = 32
    DEFAULT_LR = 1e-3
    DEFAULT_BATCH_SIZE = 256
    DEFAULT_DROPOUT = 0.2

    # Training
    GRID_SEARCH_EPOCHS = 15
    FINAL_TRAINING_EPOCHS = 60
    EARLY_STOPPING_PATIENCE = 8

    N_ATTACK_SAMPLES = 10  # Increased slightly for better stats
    
    # Attack Config (Whitebox + Blackbox)
    ATTACKS_CONFIG = {
        "FGSM_Whitebox": {"method": "FGSM", "eps": 0.5},
        # PGD is iterative, giving it more power to find the 'flip' boundary
        "PGD_Whitebox": {"method": "PGD", "eps": 1.0, "eps_step": 0.1, "max_iter": 40},
        "HopSkipJump": {"method": "HopSkipJump", "max_iter": 50, "max_eval": 2000, "init_eval": 100}
    }

# -------------------------
# DATA PROCESSING HELPERS
# -------------------------
def guess_target_column(df: pd.DataFrame):
    # Common target names in Phishing datasets
    candidates = ['Result', 'class', 'WebsiteType', 'target', 'Target']
    for c in candidates:
        if c in df.columns:
            return c
    # Fallback: usually the last column
    return df.columns[-1]

def detect_cat_cont(X: pd.DataFrame, categorical_threshold: int = 50):
    """
    Aggressive detection: Treat integer columns with few unique values as Categorical.
    This is critical for datasets like Phishing where features are {-1, 0, 1}.
    """
    cat_cols = []
    cont_cols = []
    
    for c in X.columns:
        # If explicitly object/string
        if X[c].dtype == object:
            cat_cols.append(c)
        else:
            # Check unique values
            nunique = X[c].nunique(dropna=True)
            # If it's an integer type (or float acting like int) AND has few unique values
            if nunique <= categorical_threshold:
                cat_cols.append(c)
            else:
                cont_cols.append(c)
                
    return cont_cols, cat_cols

def _onehot_kwargs_compat():
    import inspect
    sig = inspect.signature(OneHotEncoder.__init__)
    if "sparse_output" in sig.parameters:
        return dict(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    else:
        return dict(handle_unknown="ignore", sparse=False, dtype=np.float32)

def load_arff_dataset(dataset_path: str):
    print(f"→ Loading ARFF dataset from: {dataset_path}")
    with open(dataset_path, "r", encoding="utf8", errors="ignore") as f:
        arff_data = arff.load(f)
    df = pd.DataFrame(arff_data["data"])
    df.columns = [attr[0] for attr in arff_data["attributes"]]
    
    # Numeric Coercion
    for col in df.columns:
        if df[col].dtype == object:
            # Try converting to numeric
            coerced = pd.to_numeric(df[col], errors="coerce")
            # If mostly numeric, keep it.
            if coerced.notna().sum() / len(coerced) > 0.9:
                df[col] = coerced
    
    print(f"  Raw ARFF shape: {df.shape}")
    return df

# -------------------------
# PREPROCESSOR
# -------------------------
def define_preprocessor(X: pd.DataFrame):
    # Force detection with threshold 50 to catch {-1, 0, 1} features
    cont_features, cat_features = detect_cat_cont(X, categorical_threshold=50)
    
    print("\n🔍 FEATURE ANALYSIS:")
    print(f"  Total Features: {X.shape[1]}")
    print(f"  Numeric (Continuous): {len(cont_features)}")
    print(f"  Categorical (Discrete): {len(cat_features)}")
    if len(cont_features) > 0:
        print(f"  Example Numeric: {cont_features[:3]}")
    if len(cat_features) > 0:
        print(f"  Example Categorical: {cat_features[:3]}")
        
    ohe_args = _onehot_kwargs_compat()
    
    # Critical: Categorical features must be strings for OneHotEncoder if they were int
    # We do this transformation inside the main loop, but here we set up the transformer
    
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), cont_features),
        ("cat", OneHotEncoder(**ohe_args), cat_features)
    ], remainder='drop')
    
    # We need to fit to know the output dimension
    # (Transformation happens later after casting types)
    return {
        "preprocessor": preprocessor, 
        "cont_features": cont_features, 
        "cat_features": cat_features
    }

# -------------------------
# AUTOENCODER
# -------------------------
class SimpleAE(nn.Module):
    def __init__(self, input_dim, latent_dim=16, dropout=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, input_dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

def train_autoencoder(X_train_np, X_val_np, latent_dim=16, learning_rate=1e-3, 
                      batch_size=256, dropout=0.2, epochs=50, patience=6, verbose=False):
    device = torch.device("cpu")
    train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
    val_tensor = torch.tensor(X_val_np, dtype=torch.float32)
    
    train_loader = DataLoader(TensorDataset(train_tensor, train_tensor), batch_size=batch_size, shuffle=True)
    
    model = SimpleAE(X_train_np.shape[1], latent_dim=latent_dim, dropout=dropout)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    best_val = float('inf')
    best_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        for bx, _ in train_loader:
            recon, _ = model(bx)
            loss = criterion(recon, bx)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            v_recon, _ = model(val_tensor)
            v_loss = criterion(v_recon, val_tensor).item()
            
        if verbose and (epoch+1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Val Loss={v_loss:.6f}")
            
        if v_loss < best_val:
            best_val = v_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose: print("  Early stopping triggered.")
                break
                
    model.load_state_dict(best_state)
    return model, best_val

# -------------------------
# TRANSFORM & RECONSTRUCT
# -------------------------
def transform_to_Z(X_np, autoencoder):
    """Maps preprocessed X to Latent Z"""
    with torch.no_grad():
        X_tensor = torch.tensor(X_np, dtype=torch.float32)
        _, Z = autoencoder(X_tensor) # forward returns (recon, z)
    return Z.numpy()

def inverse_transform_from_Z(Z, preprocessor, autoencoder, cont_features, cat_features, X_template):
    """
    Decodes Z -> X_reconstructed with strict Categorical Argmax.
    """
    with torch.no_grad():
        Zt = torch.tensor(Z, dtype=torch.float32)
        X_prep_recon = autoencoder.decoder(Zt).numpy()
        
    cont_t = preprocessor.named_transformers_["num"]
    cat_t = preprocessor.named_transformers_["cat"]
    
    n_cont = len(cont_features)
    
    # 1. Handle Numerical
    if n_cont > 0:
        X_cont = X_prep_recon[:, :n_cont]
        X_num_orig = cont_t.inverse_transform(X_cont)
    else:
        X_num_orig = np.empty((Z.shape[0], 0))
        
    # 2. Handle Categorical (THE FIX)
    parts = []
    idx_start = n_cont
    
    if hasattr(cat_t, "categories_"):
        for cats in cat_t.categories_:
            n_c = len(cats)
            # Slice the probability distribution for this feature
            probs_block = X_prep_recon[:, idx_start : idx_start + n_c]
            
            # ARGMAX: Force the highest probability to be the chosen category
            # This is what allows "flipping" from one discrete state to another
            choice_indices = np.argmax(probs_block, axis=1)
            
            # Map index back to value (e.g., 0 -> "-1", 1 -> "0", 2 -> "1")
            chosen_vals = np.array(cats)[choice_indices]
            parts.append(chosen_vals.reshape(-1, 1))
            
            idx_start += n_c
            
    if parts:
        X_cat_orig = np.hstack(parts)
        # Combine
        X_final = np.concatenate([X_num_orig, X_cat_orig], axis=1)
        cols = cont_features + cat_features
    else:
        X_final = X_num_orig
        cols = cont_features
        
    # Create DataFrame
    X_df = pd.DataFrame(X_final, columns=cols)
    
    # Reorder to match original
    X_df = X_df[X_template.columns]
    
    # Ensure types match original for cleanliness
    # (LightGBM handles strings, but we prefer consistency)
    for c in X_df.columns:
        try:
            X_df[c] = pd.to_numeric(X_df[c], errors='ignore')
        except:
            pass
            
    return X_df

# -------------------------
# MODELS (Surrogate & Target)
# -------------------------
def train_target_model(X_train, y_train):
    print("→ Training Target Model (LightGBM)...")
    # LightGBM handles categoricals natively if dtype is 'category', 
    # but here we pass raw values. It's robust.
    clf = lgb.LGBMClassifier(random_state=42, verbose=-1)
    clf.fit(X_train, y_train)
    return clf

class SurrogateMLP(nn.Module):
    def __init__(self, input_dim, classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, classes)
        )
    def forward(self, x):
        return self.net(x)

def train_surrogate(Z_train, y_train, epochs=30):
    print("→ Training Surrogate Model (MLP on Latent Space)...")
    device = torch.device("cpu")
    X_t = torch.tensor(Z_train, dtype=torch.float32)
    y_t = torch.tensor(y_train.values, dtype=torch.long)
    
    ds = TensorDataset(X_t, y_t)
    dl = DataLoader(ds, batch_size=256, shuffle=True)
    
    model = SurrogateMLP(Z_train.shape[1], classes=len(np.unique(y_train)))
    opt = optim.Adam(model.parameters(), lr=0.001)
    crit = nn.CrossEntropyLoss()
    
    model.train()
    for _ in range(epochs):
        for bx, by in dl:
            opt.zero_grad()
            out = model(bx)
            loss = crit(out, by)
            loss.backward()
            opt.step()
    model.eval()
    return model

# -------------------------
# MAIN PIPELINE
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=False, default="dataset_ (2)", help="Path to ARFF")
    parser.add_argument("--sample-size", type=int, default=20000)
    args = parser.parse_args()

    # 1. LOAD
    df = load_arff_dataset(args.dataset_path)
    target_col = guess_target_column(df)
    print(f"Target Column: {target_col}")
    
    # Downsample
    if len(df) > args.sample_size:
        df = df.sample(n=args.sample_size, random_state=42).reset_index(drop=True)

    # 2. SPLIT X/Y
    X = df.drop(columns=[target_col])
    y_raw = df[target_col]
    
    # Encode Target
    if y_raw.dtype == object or not pd.api.types.is_numeric_dtype(y_raw):
        y_codes, _ = pd.factorize(y_raw)
        y = pd.Series(y_codes, name="target")
    else:
        y = y_raw

    # 3. PREPROCESSOR SETUP
    # CAST CATEGORICALS TO STRING TO ENSURE ONEHOT WORKS CORRECTLY
    # (Otherwise sklearn might treat integers as continuous)
    prep_info = define_preprocessor(X)
    for c in prep_info["cat_features"]:
        X[c] = X[c].astype(str)
        
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=Config.TEST_SIZE, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=Config.VAL_SIZE, random_state=42, stratify=y_train_full)
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Fit Preprocessor
    preprocessor = prep_info["preprocessor"]
    preprocessor.fit(X_train)
    
    X_train_enc = preprocessor.transform(X_train).astype(np.float32)
    X_val_enc = preprocessor.transform(X_val).astype(np.float32)
    X_test_enc = preprocessor.transform(X_test).astype(np.float32)
    
    input_dim = X_train_enc.shape[1]
    print(f"Encoded Input Dimension: {input_dim}")

    # 4. TRAIN AUTOENCODER (Grid Search or Direct)
    print("\n🏗️ Autoencoder Phase...")
    # Simple direct training for brevity, insert grid search here if needed
    ae, _ = train_autoencoder(X_train_enc, X_val_enc, 
                              latent_dim=Config.DEFAULT_LATENT_DIM, 
                              epochs=Config.FINAL_TRAINING_EPOCHS,
                              verbose=True)
    
    torch.save(ae.state_dict(), Config.AE_WEIGHTS_FILE)

    # 5. TRAIN MODELS
    # Target
    # LightGBM needs 'category' dtype for best performance on strings, or we rely on it parsing
    # For simplicity/robustness in this script, we pass the raw X (with strings) - LGBM handles it.
    for c in X_train.columns:
        if X_train[c].dtype == object:
            X_train[c] = X_train[c].astype('category')
            X_test[c] = X_test[c].astype('category')
            
    target_model = train_target_model(X_train, y_train)
    print(f"Target Accuracy: {accuracy_score(y_test, target_model.predict(X_test))*100:.2f}%")
    
    # Surrogate
    Z_train = transform_to_Z(X_train_enc, ae)
    Z_test = transform_to_Z(X_test_enc, ae)
    
    surrogate_model = train_surrogate(Z_train, y_train)
    torch.save(surrogate_model.state_dict(), Config.SURROGATE_WEIGHTS_FILE)

    # 6. ATTACK & EVALUATE
    print("\n⚡ Running Attacks...")
    
    # Wrap Surrogate
    z_min, z_max = float(Z_train.min()), float(Z_train.max())
    art_wrapper = PyTorchClassifier(
        model=surrogate_model,
        loss=nn.CrossEntropyLoss(),
        optimizer=optim.Adam(surrogate_model.parameters()),
        input_shape=(Z_train.shape[1],),
        nb_classes=len(np.unique(y)),
        clip_values=(z_min, z_max),
        device_type="cpu"
    )
    
    # Pick Victims (Correctly classified by both)
    surr_preds = surrogate_model(torch.tensor(Z_test)).detach().numpy().argmax(axis=1)
    targ_preds = target_model.predict(X_test)
    
    valid_mask = (surr_preds == y_test) & (targ_preds == y_test)
    valid_idxs = np.where(valid_mask)[0]
    
    if len(valid_idxs) > Config.N_ATTACK_SAMPLES:
        victim_idxs = np.random.choice(valid_idxs, Config.N_ATTACK_SAMPLES, replace=False)
    else:
        victim_idxs = valid_idxs

    results_data = []

    for idx in victim_idxs:
        Z_orig = Z_test[idx:idx+1] # Shape (1, latent)
        y_true = int(y_test.iloc[idx])
        X_orig_row = X_test.iloc[idx]
        
        for name, cfg in Config.ATTACKS_CONFIG.items():
            t0 = time.time()
            if cfg['method'] == "FGSM":
                atk = FastGradientMethod(estimator=art_wrapper, eps=cfg['eps'])
                Z_adv = atk.generate(Z_orig)
            elif cfg['method'] == "PGD":
                atk = ProjectedGradientDescent(estimator=art_wrapper, eps=cfg['eps'], eps_step=cfg['eps_step'], max_iter=cfg['max_iter'], verbose=False)
                Z_adv = atk.generate(Z_orig)
            elif cfg['method'] == "HopSkipJump":
                atk = HopSkipJump(classifier=art_wrapper, max_iter=cfg['max_iter'], verbose=False)
                Z_adv = atk.generate(Z_orig)
                
            elapsed = time.time() - t0
            
            # DECODE & MEASURE
            X_adv_df = inverse_transform_from_Z(Z_adv, preprocessor, ae, 
                                                prep_info["cont_features"], 
                                                prep_info["cat_features"], 
                                                X_train)
            
            # Ensure categories match for prediction
            for c in X_adv_df.columns:
                if X_adv_df[c].dtype == object:
                    X_adv_df[c] = X_adv_df[c].astype('category')
            
            # Metrics
            s_pred = art_wrapper.predict(Z_adv).argmax(axis=1)[0]
            t_pred = target_model.predict(X_adv_df)[0]
            
            # FLIP ANALYSIS
            flips = 0
            for c in prep_info["cat_features"]:
                # Comparison: Convert to string to be safe
                if str(X_orig_row[c]) != str(X_adv_df.iloc[0][c]):
                    flips += 1
            
            results_data.append({
                "Attack": name,
                "S-ASR": int(s_pred != y_true),
                "T-ASR": int(t_pred != y_true),
                "L2": np.linalg.norm(Z_adv - Z_orig),
                "CatFlips": flips,
                "Time": elapsed
            })
            
            print(f"  [{name}] Sample {idx}: Flips={flips}, T-ASR={int(t_pred != y_true)}")

    # 7. SUMMARY
    if results_data:
        df_res = pd.DataFrame(results_data)
        print("\n" + "="*50)
        print("FINAL SUMMARY (Averaged)")
        print("="*50)
        summary = df_res.groupby("Attack").mean()
        print(summary)
        summary.to_csv(Config.RESULTS_FILE)
        print(f"\nSaved to {Config.RESULTS_FILE}")

if __name__ == "__main__":

    main()
