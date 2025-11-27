# #!/usr/bin/env python3
# # pipeline_website_arff_enhanced.py
# """
# Adversarial pipeline for ARFF dataset with target 'WebsiteType'.
# ENHANCED VERSION:
# - Whitebox Attacks (FGSM, PGD) + Blackbox
# - Weight Saving (Autoencoder & Surrogate)
# - Categorical Flip (Hamming Distance) Analysis
# - Results saved to CSV

# Usage example:
#     python pipeline_website_arff_enhanced.py --dataset-path "dataset_ (2)" --sample-size 20000
# """

# import argparse
# import warnings
# import time
# import os
# from io import BytesIO
# from itertools import product
# from collections import Counter

# import numpy as np
# import pandas as pd
# import arff

# warnings.filterwarnings("ignore")

# # PyTorch, sklearn, lightgbm, ART
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# import lightgbm as lgb

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.metrics import accuracy_score

# from art.estimators.classification import PyTorchClassifier
# from art.attacks.evasion import (
#     HopSkipJump, 
#     FastGradientMethod, 
#     ProjectedGradientDescent
# )

# # -------------------------
# # CONFIG
# # -------------------------
# class Config:
#     TEST_SIZE = 0.2
#     VAL_SIZE = 0.15
#     RANDOM_STATE = 42

#     # Paths for saving
#     AE_WEIGHTS_FILE = "best_autoencoder_website.pth"
#     SURROGATE_WEIGHTS_FILE = "surrogate_model_website.pth"
#     RESULTS_FILE = "attack_summary_website.csv"

#     # Grid search config
#     LATENT_DIMS = [4, 8, 16, 24, 32, 48, 56]
#     LEARNING_RATES = [5e-4, 1e-3, 2e-3]
#     BATCH_SIZES = [128, 256]
#     DROPOUTS = [0.1, 0.2, 0.3]

#     DEFAULT_LATENT_DIM = 16
#     DEFAULT_LR = 1e-3
#     DEFAULT_BATCH_SIZE = 256
#     DEFAULT_DROPOUT = 0.2

#     GRID_SEARCH_EPOCHS = 15
#     FINAL_TRAINING_EPOCHS = 60
#     EARLY_STOPPING_PATIENCE = 6

#     N_ATTACK_SAMPLES = 5
    
#     # Enhanced Attack Config
#     ATTACKS_CONFIG = {
#         "FGSM_Whitebox": {"method": "FGSM", "eps": 0.5},
#         "PGD_Whitebox": {"method": "PGD", "eps": 0.5, "eps_step": 0.1, "max_iter": 20},
#         "RandomWalk": {"method": "RandomWalk", "max_iter": 500, "step_size": 0.02},
#         "SquareAttack": {"method": "SquareAttack", "max_iter": 500, "eps": 0.3, "p": 0.05},
#         "QEBA": {"method": "QEBA", "max_iter": 300},
#         "HopSkipJump": {"method": "HopSkipJump", "max_iter": 50, "max_eval": 2000, "init_eval": 100}
#     }

# # -------------------------
# # Helpers
# # -------------------------
# def guess_target_column(df: pd.DataFrame):
#     # Primary: WebsiteType
#     if "WebsiteType" in df.columns:
#         return "WebsiteType"
#     # fallback common names
#     for name in df.columns:
#         if str(name).strip().lower() in ('target', 'class', 'label', 'y', 'outcome'):
#             return name
#     return df.columns[-1]

# def detect_cat_cont(X: pd.DataFrame, categorical_threshold: int = 20):
#     cat_cols = []
#     cont_cols = []
#     for c in X.columns:
#         if X[c].dtype == object:
#             cat_cols.append(c)
#         else:
#             nunique = X[c].nunique(dropna=True)
#             if nunique <= categorical_threshold and np.issubdtype(X[c].dtype, np.integer):
#                 cat_cols.append(c)
#             else:
#                 cont_cols.append(c)
#     return cont_cols, cat_cols

# def _onehot_kwargs_compat():
#     import inspect
#     sig = inspect.signature(OneHotEncoder.__init__)
#     if "sparse_output" in sig.parameters:
#         return dict(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
#     else:
#         return dict(handle_unknown="ignore", sparse=False, dtype=np.float32)

# # -------------------------
# # ARFF loader + encoding
# # -------------------------
# def load_arff_dataset(dataset_path: str):
#     print(f"→ Loading ARFF dataset from: {dataset_path}")
#     with open(dataset_path, "r", encoding="utf8", errors="ignore") as f:
#         arff_data = arff.load(f)
#     df = pd.DataFrame(arff_data["data"])
#     df.columns = [attr[0] for attr in arff_data["attributes"]]
#     print(f"  Raw ARFF shape: {df.shape}")

#     # Try to coerce object columns to numeric where appropriate
#     for col in df.columns:
#         if df[col].dtype == object:
#             coerced = pd.to_numeric(df[col], errors="coerce")
#             # if most values convert to numeric, keep numeric
#             if coerced.notna().sum() / max(1, len(coerced)) > 0.9:
#                 df[col] = coerced

#     return df

# # -------------------------
# # Preprocessor creation
# # -------------------------
# def define_preprocessor(X: pd.DataFrame):
#     cont_features, cat_features = detect_cat_cont(X, categorical_threshold=20)
#     ohe_args = _onehot_kwargs_compat()
#     preprocessor = ColumnTransformer([
#         ("num", StandardScaler(), cont_features),
#         ("cat", OneHotEncoder(**ohe_args), cat_features)
#     ], remainder='drop')
#     preprocessor.fit(X)
#     # compute output dim
#     try:
#         dim = len(preprocessor.get_feature_names_out())
#     except Exception:
#         dim = preprocessor.transform(X.iloc[:1]).shape[1]
#     return {
#         "preprocessor": preprocessor, 
#         "cont_features": cont_features, 
#         "cat_features": cat_features, 
#         "dim_preprocessed": dim
#     }

# # -------------------------
# # Autoencoder (fallback)
# # -------------------------
# class SimpleAE(nn.Module):
#     def __init__(self, input_dim, latent_dim=16, dropout=0.2):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(dropout),
#             nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
#             nn.Linear(128, latent_dim)
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(latent_dim, 128), nn.ReLU(), nn.Dropout(dropout),
#             nn.Linear(128, 256), nn.ReLU(), nn.Dropout(dropout),
#             nn.Linear(256, input_dim)
#         )
#     def encode(self, x): return self.encoder(x)
#     def decode(self, z): return self.decoder(z)
#     def forward(self, x):
#         z = self.encode(x)
#         return self.decode(z), z

# def train_autoencoder_fallback(X_train_np, X_val_np, latent_dim=16, learning_rate=1e-3, batch_size=256, dropout=0.2, epochs=50, patience=6, verbose=False):
#     device = torch.device("cpu")
#     train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
#     val_tensor = torch.tensor(X_val_np, dtype=torch.float32)
#     train_loader = DataLoader(TensorDataset(train_tensor, train_tensor), batch_size=batch_size, shuffle=True)
#     model = SimpleAE(X_train_np.shape[1], latent_dim=latent_dim, dropout=dropout)
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#     criterion = nn.MSELoss()
#     best_val = float('inf'); best_state = None; no_imp = 0
#     for epoch in range(epochs):
#         model.train(); tr_loss = 0.0
#         for bx, _ in train_loader:
#             recon, _ = model(bx)
#             loss = criterion(recon, bx)
#             optimizer.zero_grad(); loss.backward(); optimizer.step()
#             tr_loss += loss.item()
#         tr_loss /= len(train_loader)
#         model.eval()
#         with torch.no_grad():
#             v_recon, _ = model(val_tensor)
#             v_loss = criterion(v_recon, val_tensor).item()
#         if verbose and (epoch+1) % 10 == 0:
#             print(f"  AE epoch {epoch+1}/{epochs}: train={tr_loss:.6f}, val={v_loss:.6f}")
#         if v_loss < best_val:
#             best_val = v_loss
#             best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
#             no_imp = 0
#         else:
#             no_imp += 1
#             if no_imp >= patience:
#                 if verbose:
#                     print("  Early stopping AE")
#                 break
#     model.load_state_dict(best_state)
#     model.eval()
#     return model, best_val

# # -------------------------
# # Transform helpers
# # -------------------------
# def transform_to_Z(X, preprocessor, autoencoder):
#     X_prep = preprocessor.transform(X).astype(np.float32)
#     with torch.no_grad():
#         X_tensor = torch.tensor(X_prep, dtype=torch.float32)
#         Z = autoencoder.encode(X_tensor).numpy()
#     return Z

# def inverse_transform_from_Z(Z, preprocessor, autoencoder, cont_features, cat_features, X_template, original_dtypes):
#     # decode to preprocessed space and approximate inverse
#     with torch.no_grad():
#         Zt = torch.tensor(Z, dtype=torch.float32)
#         X_prep = autoencoder.decode(Zt).numpy()
#     cont_t = preprocessor.named_transformers_["num"]
#     cat_t = preprocessor.named_transformers_["cat"]
#     n_cont = len(cont_features)
#     X_cont = X_prep[:, :n_cont] if n_cont > 0 else np.zeros((Z.shape[0], 0))
#     X_cat = X_prep[:, n_cont:] if X_prep.shape[1] > n_cont else np.zeros((Z.shape[0], 0))
#     # inverse numeric
#     if n_cont > 0:
#         X_num = cont_t.inverse_transform(X_cont)
#     else:
#         X_num = np.zeros((Z.shape[0], 0))
#     # categorical: choose argmax per category block
#     parts = []
#     i = 0
#     if hasattr(cat_t, "categories_"):
#         for cats in cat_t.categories_:
#             k = len(cats)
#             if X_cat.shape[1] >= i + k:
#                 block = X_cat[:, i:i+k]
#                 idxs = np.nanargmax(block, axis=1)
#                 cat_vals = np.array(cats)[idxs]
#             else:
#                 # fallback: choose first category
#                 cat_vals = np.array(cats)[0]
#                 cat_vals = np.repeat(cat_vals, Z.shape[0])
#             parts.append(cat_vals.reshape(-1, 1))
#             i += k
#     if parts:
#         X_cat_inv = np.hstack(parts)
#         X_final = np.concatenate([X_num, X_cat_inv.astype(object)], axis=1)
#         cols = cont_features + cat_features
#         X_df = pd.DataFrame(X_final, columns=cols)
#     else:
#         X_df = pd.DataFrame(X_num, columns=cont_features)
#     # reorder columns to match original template (if possible)
#     try:
#         X_df = X_df[X_template.columns]
#     except Exception:
#         # If mismatch, try to add missing cols filled with NaN
#         for c in X_template.columns:
#             if c not in X_df.columns:
#                 X_df[c] = np.nan
#         X_df = X_df[X_template.columns]
#     # Enforce numeric dtypes for originally numeric columns, else string for categorical
#     for col in X_df.columns:
#         if col in original_dtypes:
#             orig = original_dtypes[col]
#             if pd.api.types.is_numeric_dtype(orig):
#                 X_df[col] = pd.to_numeric(X_df[col], errors='coerce')
#             else:
#                 X_df[col] = X_df[col].astype(str)
#         else:
#             # best effort
#             try:
#                 X_df[col] = pd.to_numeric(X_df[col], errors='coerce')
#             except Exception:
#                 X_df[col] = X_df[col].astype(str)
#     return X_df

# # -------------------------
# # Models
# # -------------------------
# def get_target_model(X_train, y_train, X_test, y_test):
#     print("→ Training target model (LightGBM)...")
#     g = lgb.LGBMClassifier(random_state=Config.RANDOM_STATE, verbose=-1)
#     g.fit(X_train, y_train)
#     acc = accuracy_score(y_test, g.predict(X_test))
#     print(f"  Target accuracy: {acc*100:.2f}%")
#     return g

# class SurrogateMLP(nn.Module):
#     def __init__(self, input_dim, nclasses=2):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.3),
#             nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
#             nn.Linear(64, nclasses)
#         )
#     def forward(self, x):
#         return self.net(x)

# def train_surrogate(Z_train, y_train, epochs=30, batch_size=256):
#     print("→ Training surrogate MLP on Z-space...")
#     device = torch.device("cpu")
#     X_tensor = torch.tensor(Z_train, dtype=torch.float32).to(device)
#     y_tensor = torch.tensor(y_train.values, dtype=torch.long).to(device)
#     dataset = TensorDataset(X_tensor, y_tensor)
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#     model = SurrogateMLP(Z_train.shape[1], nclasses=len(np.unique(y_train)))
#     optimizer = optim.Adam(model.parameters(), lr=1e-3)
#     criterion = nn.CrossEntropyLoss()
#     model.train()
#     for epoch in range(epochs):
#         correct = 0; total = 0
#         for batch_x, batch_y in loader:
#             out = model(batch_x)
#             loss = criterion(out, batch_y)
#             optimizer.zero_grad(); loss.backward(); optimizer.step()
#             correct += (out.argmax(dim=1) == batch_y).sum().item()
#             total += batch_y.size(0)
#         if (epoch + 1) % 10 == 0:
#             print(f"  Epoch {epoch+1}/{epochs}, Accuracy: {correct/total*100:.2f}%")
#     model.eval()
#     return model

# def wrap_for_art(model, Z_train):
#     z_min, z_max = float(Z_train.min()), float(Z_train.max())
#     optimizer = optim.Adam(model.parameters(), lr=1e-3)
#     wrapped = PyTorchClassifier(
#         model=model,
#         loss=nn.CrossEntropyLoss(),
#         optimizer=optimizer,
#         input_shape=(Z_train.shape[1],),
#         nb_classes=model.net[-1].out_features,
#         clip_values=(z_min, z_max),
#         device_type="cpu"
#     )
#     return wrapped

# # -------------------------
# # Attacks (Whitebox + Blackbox)
# # -------------------------
# class RandomWalkAttack:
#     def __init__(self, classifier, max_iter=500, step_size=0.02):
#         self.classifier = classifier
#         self.max_iter = max_iter
#         self.step_size = step_size
#     def generate(self, x):
#         x = np.atleast_2d(x).astype(np.float32)
#         x_adv = x.copy()
#         y_orig = self.classifier.predict(x).argmax(axis=1)[0]
#         target_class = 1 - y_orig
#         for i in range(self.max_iter):
#             delta = np.random.randn(*x.shape).astype(np.float32) * self.step_size
#             x_candidate = x_adv + delta
#             if self.classifier.predict(x_candidate).argmax(axis=1)[0] == target_class:
#                 return x_candidate
#             if np.random.rand() < 0.05:
#                 x_adv = x_candidate
#         return x_adv

# class SquareAttack:
#     def __init__(self, classifier, max_iter=500, eps=0.3, p=0.05):
#         self.classifier = classifier
#         self.max_iter = max_iter
#         self.eps = eps
#         self.p = p
#     def generate(self, x):
#         x = np.atleast_2d(x).astype(np.float32)
#         d = x.shape[1]
#         n_features = max(1, int(self.p * d))
#         x_adv = x.copy()
#         y_orig = self.classifier.predict(x).argmax(axis=1)[0]
#         target_class = 1 - y_orig
#         for i in range(self.max_iter):
#             indices = np.random.choice(d, n_features, replace=False)
#             delta = np.zeros_like(x_adv, dtype=np.float32)
#             delta[0, indices] = np.random.uniform(-self.eps, self.eps, n_features).astype(np.float32)
#             x_candidate = np.clip(x_adv + delta, x - self.eps, x + self.eps).astype(np.float32)
#             if self.classifier.predict(x_candidate).argmax(axis=1)[0] == target_class:
#                 return x_candidate
#         return x_adv

# class QEBAAttack:
#     def __init__(self, classifier, max_iter=300):
#         self.classifier = classifier
#         self.max_iter = max_iter
#     def generate(self, x):
#         x = np.atleast_2d(x).astype(np.float32)
#         y_orig = self.classifier.predict(x).argmax(axis=1)[0]
#         target_class = 1 - y_orig
#         x_adv = None
#         for _ in range(200):
#             candidate = (x + np.random.randn(*x.shape) * 0.5).astype(np.float32)
#             if self.classifier.predict(candidate).argmax(axis=1)[0] == target_class:
#                 x_adv = candidate
#                 break
#         if x_adv is None:
#             return x
#         for i in range(self.max_iter):
#             alpha = np.random.uniform(0.8, 1.0)
#             x_candidate = (alpha * x_adv + (1 - alpha) * x).astype(np.float32)
#             if self.classifier.predict(x_candidate).argmax(axis=1)[0] == target_class:
#                 x_adv = x_candidate
#         return x_adv

# def run_attacks(f_wrapper, Z_sample, attacks_config):
#     results = {}
#     Z_sample = Z_sample.astype(np.float32)
    
#     for attack_name, params in attacks_config.items():
#         t0 = time.time()
#         method = params.get("method", attack_name)
        
#         # --- Whitebox Attacks ---
#         if method == "FGSM":
#             attack = FastGradientMethod(estimator=f_wrapper, eps=params.get("eps", 0.5))
#             Z_adv = attack.generate(x=Z_sample)
            
#         elif method == "PGD":
#             attack = ProjectedGradientDescent(
#                 estimator=f_wrapper, 
#                 eps=params.get("eps", 0.5),
#                 eps_step=params.get("eps_step", 0.1),
#                 max_iter=params.get("max_iter", 20),
#                 verbose=False
#             )
#             Z_adv = attack.generate(x=Z_sample)
        
#         # --- Blackbox Attacks ---
#         elif method == "HopSkipJump":
#             attack = HopSkipJump(
#                 classifier=f_wrapper,
#                 max_iter=params.get("max_iter", 50),
#                 max_eval=params.get("max_eval", 2000),
#                 init_eval=params.get("init_eval", 100),
#                 verbose=False
#             )
#             try:
#                 Z_adv = attack.generate(x=Z_sample)
#             except Exception as e:
#                 print(f"  HopSkipJump failed: {e}")
#                 Z_adv = Z_sample
#         elif method == "RandomWalk":
#             attack = RandomWalkAttack(
#                 classifier=f_wrapper,
#                 max_iter=params.get("max_iter", 500),
#                 step_size=params.get("step_size", 0.02)
#             )
#             Z_adv = attack.generate(Z_sample)
#         elif method == "SquareAttack":
#             attack = SquareAttack(
#                 classifier=f_wrapper,
#                 max_iter=params.get("max_iter", 500),
#                 eps=params.get("eps", 0.3),
#                 p=params.get("p", 0.05)
#             )
#             Z_adv = attack.generate(Z_sample)
#         elif method == "QEBA":
#             attack = QEBAAttack(
#                 classifier=f_wrapper,
#                 max_iter=params.get("max_iter", 300)
#             )
#             Z_adv = attack.generate(Z_sample)
#         else:
#             print(f"  Unknown attack: {attack_name}")
#             continue
            
#         t1 = time.time()
#         results[attack_name] = {"Z_adv": Z_adv, "time": t1 - t0}
#         print(f"  {attack_name} done in {t1 - t0:.2f}s")
        
#     return results

# # -------------------------
# # Evaluation (Enhanced with Flip Analysis)
# # -------------------------
# def evaluate_adversarial(Z_orig, Z_adv, y_orig_label, f_wrapper, g_model,
#                         preprocessor, autoencoder, cont_features,
#                         cat_features, X_template, dtypes, X_orig_row=None):
    
#     # 1. Predictions
#     f_pred_orig = f_wrapper.predict(Z_orig).argmax(axis=1)[0]
#     f_pred_adv = f_wrapper.predict(Z_adv).argmax(axis=1)[0]
#     surrogate_fooled = (f_pred_adv != y_orig_label)
    
#     # 2. Reconstruct X_adv
#     X_adv = inverse_transform_from_Z(Z_adv, preprocessor, autoencoder,
#                                      cont_features, cat_features,
#                                      X_template, dtypes)
    
#     # Ensure dtypes match training frame before predict
#     for c in X_adv.columns:
#         if c in dtypes and pd.api.types.is_numeric_dtype(dtypes[c]):
#             X_adv[c] = pd.to_numeric(X_adv[c], errors='coerce')
#         else:
#             X_adv[c] = X_adv[c].astype(str)
            
#     g_pred_adv = g_model.predict(X_adv)[0]
#     target_fooled = (g_pred_adv != y_orig_label)
#     l2_dist = float(np.linalg.norm(Z_adv - Z_orig))
    
#     # 3. Categorical Flip Analysis (Hamming Distance)
#     cat_flips = 0
#     flipped_features = []
    
#     if X_orig_row is not None and len(cat_features) > 0:
#         for feat in cat_features:
#             val_orig = str(X_orig_row[feat])
#             val_adv = str(X_adv.iloc[0][feat])
            
#             if val_orig != val_adv:
#                 cat_flips += 1
#                 flipped_features.append(feat)

#     return {
#         "surrogate_fooled": surrogate_fooled,
#         "target_fooled": target_fooled,
#         "l2_distance": l2_dist,
#         "cat_flips": cat_flips,
#         "flipped_features": flipped_features,
#         "f_pred_adv": int(f_pred_adv),
#         "g_pred_adv": int(g_pred_adv)
#     }

# # -------------------------
# # Main pipeline
# # -------------------------
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--dataset-path", type=str, required=False, default="dataset_ (2)", help="Path to local ARFF dataset")
#     parser.add_argument("--sample-size", type=int, default=20000, help="Downsample size (default 20000). If smaller, uses full dataset.")
#     parser.add_argument("--no-tune", action="store_true", help="Disable AE grid search")
#     parser.add_argument("--n-attack-samples", type=int, default=Config.N_ATTACK_SAMPLES, help="Number of test samples to attack")
#     args = parser.parse_args()

#     Config.TUNE_HYPERPARAMETERS = not args.no_tune
#     Config.N_ATTACK_SAMPLES = args.n_attack_samples

#     # 1) Load ARFF dataset
#     df = load_arff_dataset(args.dataset_path)

#     # 2) Detect target column (use WebsiteType preferentially)
#     target_col = guess_target_column(df)
#     print(f"Detected target column: '{target_col}'")
#     if target_col not in df.columns:
#         raise RuntimeError(f"Target column '{target_col}' not present in dataset.")

#     # 3) Downsample
#     def downsample_local(df_, target_col_, sample_size_, rs=Config.RANDOM_STATE):
#         n = len(df_)
#         if sample_size_ is None or sample_size_ >= n:
#             return df_.reset_index(drop=True)
#         y = df_[target_col_]
#         try:
#             vc = y.value_counts()
#             if len(vc) > 1 and vc.min() >= 2:
#                 df_s = df_.groupby(target_col_, group_keys=False).apply(
#                     lambda x: x.sample(n=max(1, int(sample_size_ * len(x) / n)), random_state=rs)
#                 ).reset_index(drop=True)
#                 if len(df_s) > sample_size_:
#                     df_s = df_s.sample(n=sample_size_, random_state=rs).reset_index(drop=True)
#                 elif len(df_s) < sample_size_:
#                     remaining = df_.drop(df_s.index)
#                     need = sample_size_ - len(df_s)
#                     add = remaining.sample(n=need, random_state=rs)
#                     df_s = pd.concat([df_s, add], ignore_index=True)
#                 return df_s.reset_index(drop=True)
#         except Exception:
#             pass
#         return df_.sample(n=sample_size_, random_state=rs).reset_index(drop=True)

#     df_sampled = downsample_local(df, target_col, args.sample_size)

#     # 4) Prepare X, y
#     y_raw = df_sampled[target_col]
#     if y_raw.dtype == object or not pd.api.types.is_numeric_dtype(y_raw.dtype):
#         y_arr, uniques = pd.factorize(y_raw)
#         y = pd.Series(y_arr, name="target")
#     else:
#         y = pd.Series(y_raw.values, name="target")
#     X = df_sampled.drop(columns=[target_col]).reset_index(drop=True)
#     y = y.reset_index(drop=True)
#     print(f"Dataset after sampling: X={X.shape}, y={y.shape}")
    
#     # --- FIX FOR LIGHTGBM: Convert all object columns into integer category codes ---
#     for col in X.columns:
#         if X[col].dtype == object:
#             X[col] = X[col].astype("category").cat.codes.astype("float32")

#     # 5) Split
#     try:
#         X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE, stratify=y)
#     except Exception:
#         X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE)
#     X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=Config.VAL_SIZE, random_state=Config.RANDOM_STATE, stratify=y_train_full if len(np.unique(y_train_full))>1 else None)
#     print(f"Split shapes -> Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

#     # 6) Preprocessor
#     T_info = define_preprocessor(X_train)
#     preprocessor = T_info["preprocessor"]
#     cont_features = T_info["cont_features"]
#     cat_features = T_info["cat_features"]
#     dim_pre = T_info["dim_preprocessed"]
#     print(f"Preprocessed dims -> D = {dim_pre}")

#     # Save original dtypes for inverse transforms
#     dtypes = X.dtypes.to_dict()

#     X_train_prep = preprocessor.transform(X_train).astype(np.float32)
#     X_val_prep = preprocessor.transform(X_val).astype(np.float32)
#     X_test_prep = preprocessor.transform(X_test).astype(np.float32)

#     # 7) Autoencoder training / tuning
#     if Config.TUNE_HYPERPARAMETERS:
#         print("Running AE grid search (may take time)...")
#         best_val = float('inf'); best_model = None; best_params = None
#         combos = list(product(Config.LATENT_DIMS, Config.LEARNING_RATES, Config.BATCH_SIZES, Config.DROPOUTS))
#         for latent_dim, lr, bs, drop in combos:
#             m, val = train_autoencoder_fallback(X_train_prep, X_val_prep, latent_dim=latent_dim, learning_rate=lr, batch_size=bs, dropout=drop, epochs=Config.GRID_SEARCH_EPOCHS, patience=3, verbose=False)
#             print(f"  AE config lat={latent_dim}, lr={lr}, bs={bs}, drop={drop} -> val={val:.6f}")
#             if val < best_val:
#                 best_val = val; best_model = m; best_params = dict(latent_dim=latent_dim, learning_rate=lr, batch_size=bs, dropout=drop)
#         print("Best AE params:", best_params)
#         # retrain best
#         ae_model, _ = train_autoencoder_fallback(X_train_prep, X_val_prep, latent_dim=best_params['latent_dim'], learning_rate=best_params['learning_rate'], batch_size=best_params['batch_size'], dropout=best_params['dropout'], epochs=Config.FINAL_TRAINING_EPOCHS, patience=Config.EARLY_STOPPING_PATIENCE, verbose=True)
#         latent_dim = best_params['latent_dim']
#     else:
#         ae_model, _ = train_autoencoder_fallback(X_train_prep, X_val_prep, latent_dim=Config.DEFAULT_LATENT_DIM, learning_rate=Config.DEFAULT_LR, batch_size=Config.DEFAULT_BATCH_SIZE, dropout=Config.DEFAULT_DROPOUT, epochs=Config.FINAL_TRAINING_EPOCHS, patience=Config.EARLY_STOPPING_PATIENCE, verbose=True)
#         latent_dim = Config.DEFAULT_LATENT_DIM

#     print(f"Autoencoder ready (latent_dim={latent_dim})")
#     print(f"Compression: original D = {dim_pre}, latent d = {latent_dim}, ratio D/d = {dim_pre/latent_dim:.4f}")
    
#     # --- TASK: Save Autoencoder Weights ---
#     print(f"💾 Saving Best Autoencoder to '{Config.AE_WEIGHTS_FILE}'...")
#     torch.save(ae_model.state_dict(), Config.AE_WEIGHTS_FILE)

#     # 8) Train target model g(X)
#     g = get_target_model(X_train, y_train, X_test, y_test)

#     # 9) Transform to Z
#     Z_train = transform_to_Z(X_train, preprocessor, ae_model)
#     Z_test = transform_to_Z(X_test, preprocessor, ae_model)
#     print(f"Z dims: {Z_train.shape}")

#     # 10) Train surrogate f(Z)
#     f_model = train_surrogate(Z_train, y_train)
#     f_wrapper = wrap_for_art(f_model, Z_train)
    
#     # --- TASK: Save Surrogate Weights ---
#     print(f"💾 Saving Surrogate Model to '{Config.SURROGATE_WEIGHTS_FILE}'...")
#     torch.save(f_model.state_dict(), Config.SURROGATE_WEIGHTS_FILE)

#     # 11) Choose samples to attack
#     g_preds = g.predict(X_test)
#     f_preds = f_wrapper.predict(Z_test).argmax(axis=1)
#     correct_mask = (g_preds == y_test) & (f_preds == y_test)
#     valid_indices = np.where(correct_mask)[0]
#     if len(valid_indices) > 0:
#         chosen = valid_indices[:Config.N_ATTACK_SAMPLES]
#     else:
#         chosen = np.arange(min(Config.N_ATTACK_SAMPLES, len(X_test)))
#     print(f"Selected {len(chosen)} samples for attacks.")

#     all_results = {k: [] for k in Config.ATTACKS_CONFIG.keys()}

#     for i, idx in enumerate(chosen):
#         print(f"\n--- Sample {i+1}/{len(chosen)} (index={idx}) ---")
#         Z_orig = Z_test[idx:idx+1]
#         y_orig = int(y_test.iloc[idx])
        
#         # Get original row for Flip Analysis
#         X_orig_row = X_test.iloc[idx]
        
#         attack_results = run_attacks(f_wrapper, Z_orig, Config.ATTACKS_CONFIG)
        
#         for attack_name, attack_data in attack_results.items():
#             Z_adv = attack_data["Z_adv"]
#             attack_time = attack_data["time"]
            
#             # Enhanced Evaluation
#             eval_result = evaluate_adversarial(
#                 Z_orig, Z_adv, y_orig, f_wrapper, g,
#                 preprocessor, ae_model,
#                 cont_features, cat_features,
#                 X_train, dtypes,
#                 X_orig_row=X_orig_row
#             )
            
#             eval_result["time"] = attack_time
#             all_results[attack_name].append(eval_result)
            
#             print(f"  {attack_name}: S-Fool={eval_result['surrogate_fooled']}, "
#                   f"T-Fool={eval_result['target_fooled']}, L2={eval_result['l2_distance']:.4f}, "
#                   f"CatFlips={eval_result['cat_flips']}")

#     # 12) Summary
#     summary_rows = []
#     for attack_name, results in all_results.items():
#         if not results:
#             continue
#         s_asr = np.mean([r["surrogate_fooled"] for r in results]) * 100
#         t_asr = np.mean([r["target_fooled"] for r in results]) * 100
#         avg_l2 = np.mean([r["l2_distance"] for r in results])
#         avg_time = np.mean([r["time"] for r in results])
#         avg_flips = np.mean([r["cat_flips"] for r in results])
        
#         summary_rows.append({
#             "Attack": attack_name,
#             "S-ASR (%)": f"{s_asr:.1f}",
#             "T-ASR (%)": f"{t_asr:.1f}",
#             "Avg L2": f"{avg_l2:.4f}",
#             "Avg Cat Flips": f"{avg_flips:.2f}",
#             "Avg Time (s)": f"{avg_time:.2f}"
#         })
        
#     summary_df = pd.DataFrame(summary_rows)
#     if not summary_df.empty:
#         print("\nFINAL SUMMARY")
#         print(summary_df.to_string(index=False))
#         print("\nNote: 'Avg Cat Flips' represents the average Hamming Distance.")
        
#         # --- TASK: Save Results to CSV ---
#         print(f"💾 Saving results to '{Config.RESULTS_FILE}'...")
#         summary_df.to_csv(Config.RESULTS_FILE, index=False)
#     else:
#         print("\nNo attack results to summarize.")

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
# pipeline_website_phishy_corrected.py
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