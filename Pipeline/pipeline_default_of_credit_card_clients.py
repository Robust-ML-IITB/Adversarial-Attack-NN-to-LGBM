# """
# COMPLETE ENHANCED ADVERSARIAL ATTACK PIPELINE
# With hyperparameter tuning, weight saving, whitebox/blackbox attacks,
# and categorical perturbation analysis (Hamming Distance).
# """

# import warnings
# warnings.filterwarnings('ignore')

# import numpy as np
# import pandas as pd
# import requests
# from io import BytesIO
# import time
# import inspect
# import os
# from typing import Dict, Tuple, List, Any
# from itertools import product

# # ML libs
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# import lightgbm as lgb

# # sklearn
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.metrics import accuracy_score

# # ART
# from art.estimators.classification import PyTorchClassifier
# from art.attacks.evasion import (
#     HopSkipJump, 
#     FastGradientMethod, 
#     ProjectedGradientDescent
# )


# # ══════════════════════════════════════════════════════════════
# # CONFIGURATION
# # ══════════════════════════════════════════════════════════════
# class Config:
#     """Global configuration"""
#     # Data splits
#     TEST_SIZE = 0.2
#     VAL_SIZE = 0.15
#     RANDOM_STATE = 42
    
#     # Hyperparameter tuning
#     TUNE_HYPERPARAMETERS = True  # Set to False to skip grid search
    
#     # Grid search space (only used if TUNE_HYPERPARAMETERS=True)
#     LATENT_DIMS = [16, 32]  # Reduced for demo speed
#     LEARNING_RATES = [1e-3, 2e-3]
#     BATCH_SIZES = [256]
#     DROPOUTS = [0.2]
    
#     # Default hyperparameters (used if no tuning)
#     DEFAULT_LATENT_DIM = 32
#     DEFAULT_LR = 1e-3
#     DEFAULT_BATCH_SIZE = 256
#     DEFAULT_DROPOUT = 0.2
    
#     # Training
#     GRID_SEARCH_EPOCHS = 10     # Reduced for demo speed
#     FINAL_TRAINING_EPOCHS = 50 
#     EARLY_STOPPING_PATIENCE = 5
    
#     # Attack configuration
#     N_ATTACK_SAMPLES = 5
#     ATTACKS_CONFIG = {
#         # --- Whitebox Attacks (Gradient-based on Surrogate) ---
#         "FGSM": {"eps": 0.5},
#         "PGD": {"eps": 0.5, "eps_step": 0.1, "max_iter": 20},
        
#         # --- Blackbox Attacks (Query-based / Decision-based) ---
#         "HopSkipJump": {"max_iter": 20, "max_eval": 1000, "init_eval": 100},
#         "RandomWalk": {"max_iter": 200, "step_size": 0.05},
#         "SquareAttack": {"max_iter": 200, "eps": 0.5, "p": 0.1},
#     }
    
#     # Paths for saving weights
#     AE_WEIGHTS_PATH = "best_autoencoder_weights.pth"
#     SURROGATE_WEIGHTS_PATH = "surrogate_weights.pth"


# # ══════════════════════════════════════════════════════════════
# # 1. DATA LOADING
# # ══════════════════════════════════════════════════════════════
# def generate_synthetic_credit_data(n_samples=10000):
#     """Generate synthetic credit card default data for testing"""
#     print("  📝 Generating synthetic data (for testing)...")
    
#     np.random.seed(42)
    
#     data = {
#         'LIMIT_BAL': np.random.randint(10000, 500000, n_samples),
#         'SEX': np.random.choice([1, 2], n_samples),
#         'EDUCATION': np.random.choice([1, 2, 3, 4], n_samples, p=[0.3, 0.4, 0.2, 0.1]),
#         'MARRIAGE': np.random.choice([1, 2, 3], n_samples, p=[0.45, 0.45, 0.1]),
#         'AGE': np.random.randint(21, 75, n_samples),
#     }
    
#     for i in range(1, 7):
#         data[f'PAY_{i}'] = np.random.choice([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], 
#                                            n_samples, 
#                                            p=[0.05, 0.1, 0.4, 0.15, 0.1, 0.08, 0.05, 0.03, 0.02, 0.01, 0.01])
    
#     for i in range(1, 7):
#         data[f'BILL_AMT{i}'] = np.random.randint(-10000, 200000, n_samples)
    
#     for i in range(1, 7):
#         data[f'PAY_AMT{i}'] = np.random.randint(0, 100000, n_samples)
    
#     df = pd.DataFrame(data)
#     df['ID'] = range(1, n_samples + 1)
    
#     # Generate target
#     default_prob = 0.2 + 0.1 * (df['PAY_1'] > 0).astype(int) + 0.1 * (df['PAY_2'] > 0).astype(int)
#     df['TARGET'] = (np.random.random(n_samples) < default_prob).astype(int)
    
#     print(f"  ✓ Generated {n_samples} synthetic samples")
#     return df


# def load_and_clean_data():
#     """Load dataset with fallbacks"""
#     print("📊 Loading dataset...")
#     # For this demonstration, defaulting directly to synthetic if local file missing
#     # to ensure code runs immediately in notebook environments.
#     try:
#         if os.path.exists("default_credit_card.xls"):
#             df = pd.read_excel("default_credit_card.xls", header=1)
#             print("  ✓ Loaded local file")
#         else:
#             raise FileNotFoundError("Local file not found")
#     except Exception:
#         df = generate_synthetic_credit_data()

#     if "default payment next month" in df.columns:
#         df = df.rename(columns={"default payment next month": "TARGET"})
#     if "PAY_0" in df.columns:
#         df = df.rename(columns={"PAY_0": "PAY_1"})
    
#     # Standardization of categories
#     if "EDUCATION" in df.columns:
#         df["EDUCATION"] = df["EDUCATION"].replace([0, 5, 6], 4)
#     if "MARRIAGE" in df.columns:
#         df["MARRIAGE"] = df["MARRIAGE"].replace(0, 3)

#     X = df.drop(columns=["ID", "TARGET"], errors='ignore')
#     y = df["TARGET"].astype(int)
#     print(f"✓ Loaded X: {X.shape}, y: {y.shape}")
#     return X, y, X.dtypes


# # ══════════════════════════════════════════════════════════════
# # 2. IMPROVED AUTOENCODER ARCHITECTURE
# # ══════════════════════════════════════════════════════════════
# def _onehot_kwargs_compat():
#     sig = inspect.signature(OneHotEncoder.__init__)
#     if "sparse_output" in sig.parameters:
#         return dict(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
#     else:
#         return dict(handle_unknown="ignore", sparse=False, dtype=np.float32)


# class ImprovedAutoencoder(nn.Module):
#     def __init__(self, input_dim, latent_dim=32, dropout=0.2, use_batchnorm=True):
#         super().__init__()
        
#         # Encoder
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, 256),
#             nn.BatchNorm1d(256) if use_batchnorm else nn.Identity(),
#             nn.LeakyReLU(0.2),
#             nn.Dropout(dropout),
            
#             nn.Linear(256, 128),
#             nn.BatchNorm1d(128) if use_batchnorm else nn.Identity(),
#             nn.LeakyReLU(0.2),
#             nn.Dropout(dropout),
            
#             nn.Linear(128, 64),
#             nn.BatchNorm1d(64) if use_batchnorm else nn.Identity(),
#             nn.LeakyReLU(0.2),
#             nn.Dropout(dropout),
            
#             nn.Linear(64, latent_dim)
#         )
        
#         # Decoder
#         self.decoder = nn.Sequential(
#             nn.Linear(latent_dim, 64),
#             nn.BatchNorm1d(64) if use_batchnorm else nn.Identity(),
#             nn.LeakyReLU(0.2),
#             nn.Dropout(dropout),
            
#             nn.Linear(64, 128),
#             nn.BatchNorm1d(128) if use_batchnorm else nn.Identity(),
#             nn.LeakyReLU(0.2),
#             nn.Dropout(dropout),
            
#             nn.Linear(128, 256),
#             nn.BatchNorm1d(256) if use_batchnorm else nn.Identity(),
#             nn.LeakyReLU(0.2),
#             nn.Dropout(dropout),
            
#             nn.Linear(256, input_dim)
#         )
    
#     def encode(self, x):
#         return self.encoder(x)
    
#     def decode(self, z):
#         return self.decoder(z)
    
#     def forward(self, x):
#         z = self.encode(x)
#         x_recon = self.decode(z)
#         return x_recon, z


# # ══════════════════════════════════════════════════════════════
# # 3. TRAINING WITH VALIDATION & SAVING
# # ══════════════════════════════════════════════════════════════
# def train_autoencoder_single_config(X_train, X_val, latent_dim=32, learning_rate=1e-3,
#                                    batch_size=256, dropout=0.2, epochs=30,
#                                    early_stopping=True, patience=10, verbose=True):
#     train_tensor = torch.tensor(X_train, dtype=torch.float32)
#     val_tensor = torch.tensor(X_val, dtype=torch.float32)
    
#     train_loader = DataLoader(
#         TensorDataset(train_tensor, train_tensor),
#         batch_size=batch_size,
#         shuffle=True
#     )
    
#     input_dim = X_train.shape[1]
#     model = ImprovedAutoencoder(input_dim, latent_dim, dropout, use_batchnorm=True)
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#     criterion = nn.MSELoss()
    
#     best_val_loss = float('inf')
#     epochs_no_improve = 0
#     best_model_state = None
    
#     for epoch in range(epochs):
#         model.train()
#         train_loss = 0.0
#         for batch_x, _ in train_loader:
#             x_recon, _ = model(batch_x)
#             loss = criterion(x_recon, batch_x)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item()
        
#         train_loss /= len(train_loader)
        
#         model.eval()
#         with torch.no_grad():
#             val_recon, _ = model(val_tensor)
#             val_loss = criterion(val_recon, val_tensor).item()
        
#         if verbose and (epoch + 1) % 10 == 0:
#             print(f"    Epoch {epoch+1:3d}/{epochs}: Train={train_loss:.6f}, Val={val_loss:.6f}")
        
#         if early_stopping:
#             if val_loss < best_val_loss:
#                 best_val_loss = val_loss
#                 epochs_no_improve = 0
#                 best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
#             else:
#                 epochs_no_improve += 1
#             if epochs_no_improve >= patience:
#                 if verbose: print(f"    Early stopping at epoch {epoch+1}")
#                 break
#         else:
#             best_val_loss = val_loss
#             best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
#     if best_model_state:
#         model.load_state_dict(best_model_state)
    
#     return model, best_val_loss


# def grid_search_hyperparameters(X_train, X_val, config: Config):
#     print("\n🔍 HYPERPARAMETER GRID SEARCH")
    
#     best_val_loss = float('inf')
#     best_params = None
#     best_model = None
#     results = []
    
#     total = len(config.LATENT_DIMS) * len(config.LEARNING_RATES) * len(config.BATCH_SIZES) * len(config.DROPOUTS)
#     count = 0
    
#     for latent_dim, lr, batch_size, dropout in product(config.LATENT_DIMS, config.LEARNING_RATES, 
#                                                        config.BATCH_SIZES, config.DROPOUTS):
#         count += 1
#         print(f"[{count}/{total}] Testing: dim={latent_dim}, lr={lr}, batch={batch_size}, drop={dropout}")
        
#         model, val_loss = train_autoencoder_single_config(
#             X_train, X_val, latent_dim, lr, batch_size, dropout,
#             epochs=config.GRID_SEARCH_EPOCHS, early_stopping=True, patience=3, verbose=False
#         )
        
#         results.append({'dim': latent_dim, 'lr': lr, 'val_loss': val_loss})
        
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             best_params = {'latent_dim': latent_dim, 'learning_rate': lr, 
#                            'batch_size': batch_size, 'dropout': dropout}
#             best_model = model
#             print(f"  ✓ New best val_loss: {val_loss:.6f}")
    
#     return best_model, best_params, pd.DataFrame(results)


# # ══════════════════════════════════════════════════════════════
# # 4. PREPROCESSING & INVERSION (Required for Flip Calculation)
# # ══════════════════════════════════════════════════════════════
# def define_preprocessor(X):
#     print("\n🔄 Defining preprocessor...")
#     categorical = ["SEX", "EDUCATION", "MARRIAGE"] + [f"PAY_{i}" for i in range(1, 7)]
#     continuous = [c for c in X.columns if c not in categorical]
    
#     preprocessor = ColumnTransformer([
#         ("num", StandardScaler(), continuous),
#         ("cat", OneHotEncoder(**_onehot_kwargs_compat()), categorical)
#     ])
    
#     preprocessor.fit(X)
#     return {
#         "preprocessor": preprocessor,
#         "cont_features": continuous,
#         "cat_features": categorical,
#         "dim_preprocessed": preprocessor.transform(X[:1]).shape[1]
#     }

# def transform_to_Z(X, preprocessor, autoencoder):
#     X_prep = preprocessor.transform(X).astype(np.float32)
#     with torch.no_grad():
#         Z = autoencoder.encode(torch.tensor(X_prep)).numpy()
#     return Z

# def inverse_transform_from_Z(Z, preprocessor, autoencoder, cont_features, 
#                             cat_features, X_template, original_dtypes):
#     """
#     Decodes Z-space vectors back to original DataFrame format.
#     Crucial for categorical flip analysis.
#     """
#     Z = np.atleast_2d(Z).astype(np.float32)
#     with torch.no_grad():
#         X_prep = autoencoder.decode(torch.tensor(Z)).numpy()
    
#     cont_t = preprocessor.named_transformers_["num"]
#     cat_t = preprocessor.named_transformers_["cat"]
    
#     n_cont = len(cont_features)
#     X_cont = X_prep[:, :n_cont]
#     X_cat = X_prep[:, n_cont:]

#     # Snap categorical to one-hot
#     parts = []
#     i = 0
#     for cats in cat_t.categories_:
#         k = len(cats)
#         block = X_cat[:, i:i+k]
#         onehot = np.zeros_like(block)
#         onehot[np.arange(block.shape[0]), np.argmax(block, axis=1)] = 1
#         parts.append(onehot)
#         i += k
#     X_cat_clean = np.concatenate(parts, axis=1)

#     X_num = cont_t.inverse_transform(X_cont)

#     cat_cols = []
#     start = 0
#     for cats in cat_t.categories_:
#         k = len(cats)
#         block = X_cat_clean[:, start:start+k]
#         idxs = np.argmax(block, axis=1)
#         cat_vals = np.array(cats)[idxs]
#         cat_cols.append(cat_vals)
#         start += k

#     X_cat_inv = np.column_stack(cat_cols)
#     X_final = np.concatenate([X_num, X_cat_inv], axis=1)
#     X_df = pd.DataFrame(X_final, columns=cont_features + cat_features)

#     # Restore types
#     for c in X_df.columns:
#         dt = original_dtypes[c]
#         if "int" in str(dt):
#             X_df[c] = pd.to_numeric(X_df[c], errors="coerce").round().astype(dt)
#         elif "float" in str(dt):
#             X_df[c] = pd.to_numeric(X_df[c], errors="coerce").astype(float)
            
#     return X_df


# # ══════════════════════════════════════════════════════════════
# # 5. SURROGATE MODEL
# # ══════════════════════════════════════════════════════════════
# class SurrogateMLP(nn.Module):
#     def __init__(self, input_dim):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.3),
#             nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
#             nn.Linear(64, 2)
#         )
#     def forward(self, x): return self.net(x)

# def train_surrogate(Z_train, y_train, epochs=30):
#     print("\n🤖 Training surrogate model f(Z)...")
#     dataset = TensorDataset(torch.tensor(Z_train, dtype=torch.float32), 
#                           torch.tensor(y_train.values, dtype=torch.long))
#     loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
#     model = SurrogateMLP(Z_train.shape[1])
#     optimizer = optim.Adam(model.parameters(), lr=1e-3)
#     criterion = nn.CrossEntropyLoss()
    
#     model.train()
#     for _ in range(epochs):
#         for bx, by in loader:
#             optimizer.zero_grad()
#             loss = criterion(model(bx), by)
#             loss.backward()
#             optimizer.step()
    
#     model.eval()
#     return model

# def wrap_for_art(model, Z_train):
#     z_min, z_max = float(Z_train.min()), float(Z_train.max())
#     return PyTorchClassifier(
#         model=model, loss=nn.CrossEntropyLoss(), optimizer=optim.Adam(model.parameters()),
#         input_shape=(Z_train.shape[1],), nb_classes=2, clip_values=(z_min, z_max)
#     )


# # ══════════════════════════════════════════════════════════════
# # 6. ATTACK IMPLEMENTATIONS (WHITEBOX & BLACKBOX)
# # ══════════════════════════════════════════════════════════════
# class RandomWalkAttack:
#     def __init__(self, classifier, max_iter=200, step_size=0.01):
#         self.classifier = classifier
#         self.max_iter = max_iter
#         self.step_size = step_size
    
#     def generate(self, x):
#         x = np.atleast_2d(x).astype(np.float32)
#         x_adv = x.copy()
#         y_orig = self.classifier.predict(x).argmax(axis=1)[0]
#         target = 1 - y_orig
        
#         for _ in range(self.max_iter):
#             cand = x_adv + np.random.randn(*x_adv.shape).astype(np.float32) * self.step_size
#             if self.classifier.predict(cand).argmax(axis=1)[0] == target:
#                 return cand
#         return x_adv

# class SquareAttack:
#     def __init__(self, classifier, max_iter=200, eps=0.5, p=0.1):
#         self.classifier = classifier
#         self.max_iter = max_iter
#         self.eps = eps
#         self.p = p
    
#     def generate(self, x):
#         x = np.atleast_2d(x).astype(np.float32)
#         x_adv = x.copy()
#         y_orig = self.classifier.predict(x).argmax(axis=1)[0]
#         target = 1 - y_orig
#         d = x.shape[1]
#         n_feat = max(1, int(self.p * d))
        
#         for _ in range(self.max_iter):
#             idx = np.random.choice(d, n_feat, replace=False)
#             delta = np.zeros_like(x_adv); delta[0, idx] = np.random.uniform(-self.eps, self.eps, n_feat)
#             cand = np.clip(x_adv + delta, x - self.eps, x + self.eps).astype(np.float32)
#             if self.classifier.predict(cand).argmax(axis=1)[0] == target:
#                 return cand
#         return x_adv

# def run_attacks(f_wrapper, Z_sample, attacks_config):
#     results = {}
#     Z_sample = Z_sample.astype(np.float32)
    
#     for name, params in attacks_config.items():
#         t0 = time.time()
        
#         # --- Whitebox Attacks ---
#         if name == "FGSM":
#             # Fast Gradient Sign Method
#             attack = FastGradientMethod(estimator=f_wrapper, eps=params["eps"])
#             Z_adv = attack.generate(x=Z_sample)
            
#         elif name == "PGD":
#             # Projected Gradient Descent
#             attack = ProjectedGradientDescent(
#                 estimator=f_wrapper, 
#                 eps=params["eps"], 
#                 eps_step=params["eps_step"], 
#                 max_iter=params["max_iter"],
#                 verbose=False
#             )
#             Z_adv = attack.generate(x=Z_sample)
            
#         # --- Blackbox Attacks ---
#         elif name == "HopSkipJump":
#             attack = HopSkipJump(
#                 classifier=f_wrapper,
#                 max_iter=params["max_iter"],
#                 max_eval=params["max_eval"],
#                 init_eval=params["init_eval"],
#                 verbose=False
#             )
#             Z_adv = attack.generate(x=Z_sample)
            
#         elif name == "RandomWalk":
#             attack = RandomWalkAttack(f_wrapper, params["max_iter"], params["step_size"])
#             Z_adv = attack.generate(Z_sample)
            
#         elif name == "SquareAttack":
#             attack = SquareAttack(f_wrapper, params["max_iter"], params["eps"], params["p"])
#             Z_adv = attack.generate(Z_sample)
            
#         else:
#             continue
            
#         results[name] = {"Z_adv": Z_adv, "time": time.time() - t0}
        
#     return results


# # ══════════════════════════════════════════════════════════════
# # 7. EVALUATION WITH CATEGORICAL FLIP METRICS
# # ══════════════════════════════════════════════════════════════
# def evaluate_adversarial(Z_orig, Z_adv, y_orig, f_wrapper, g_model, 
#                         preprocessor, autoencoder, cont_features, 
#                         cat_features, X_template, dtypes):
#     # 1. Prediction checks
#     f_pred_adv = f_wrapper.predict(Z_adv).argmax(axis=1)[0]
    
#     # 2. Decode BOTH Original and Adversarial to feature space
#     X_orig_df = inverse_transform_from_Z(Z_orig, preprocessor, autoencoder, 
#                                         cont_features, cat_features, X_template, dtypes)
#     X_adv_df = inverse_transform_from_Z(Z_adv, preprocessor, autoencoder, 
#                                        cont_features, cat_features, X_template, dtypes)
    
#     # 3. Target Model Check
#     g_pred_adv = g_model.predict(X_adv_df)[0]
    
#     # 4. Latent Distance
#     l2_dist = float(np.linalg.norm(Z_adv - Z_orig))
    
#     # 5. CATEGORICAL PERTURBATION ANALYSIS (Flip Metrics)
#     # Compare categorical columns row by row (though here it's 1 row)
#     cat_flips = 0
#     flipped_features = []
    
#     if len(cat_features) > 0:
#         # Comparison logic: compare values in the DataFrames
#         orig_cats = X_orig_df[cat_features].values[0]
#         adv_cats = X_adv_df[cat_features].values[0]
        
#         # Hamming distance calculation (L0 for categorical)
#         mismatches = (orig_cats != adv_cats)
#         cat_flips = np.sum(mismatches)
        
#         # Identify which features flipped
#         flipped_features = [cat_features[i] for i, m in enumerate(mismatches) if m]
    
#     return {
#         "surrogate_fooled": (f_pred_adv != y_orig),
#         "target_fooled": (g_pred_adv != y_orig),
#         "l2_distance": l2_dist,
#         "cat_flips": cat_flips,          # L0 Norm equivalent
#         "flipped_features": flipped_features
#     }


# # ══════════════════════════════════════════════════════════════
# # 8. MAIN PIPELINE
# # ══════════════════════════════════════════════════════════════
# def main():
#     print("="*60 + "\n  ADVANCED ADVERSARIAL PIPELINE\n" + "="*60)
    
#     # 1. Load & Split
#     X, y, dtypes = load_and_clean_data()
#     X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=Config.TEST_SIZE, stratify=y)
#     X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=Config.VAL_SIZE, stratify=y_train_full)
    
#     # 2. Preprocessing
#     T_info = define_preprocessor(X_train)
#     preprocessor = T_info["preprocessor"]
#     X_train_prep = preprocessor.transform(X_train).astype(np.float32)
#     X_val_prep = preprocessor.transform(X_val).astype(np.float32)
    
#     # 3. Autoencoder Training & Saving
#     if Config.TUNE_HYPERPARAMETERS:
#         best_ae, best_params, _ = grid_search_hyperparameters(X_train_prep, X_val_prep, Config)
#         # Retrain with more epochs
#         autoencoder, _ = train_autoencoder_single_config(
#             X_train_prep, X_val_prep, **best_params, 
#             epochs=Config.FINAL_TRAINING_EPOCHS, early_stopping=True
#         )
#     else:
#         autoencoder, _ = train_autoencoder_single_config(
#             X_train_prep, X_val_prep, latent_dim=Config.DEFAULT_LATENT_DIM, 
#             epochs=Config.FINAL_TRAINING_EPOCHS
#         )
    
#     # [Task 2: Save AE Weights]
#     print(f"\n💾 Saving Autoencoder weights to '{Config.AE_WEIGHTS_PATH}'...")
#     torch.save(autoencoder.state_dict(), Config.AE_WEIGHTS_PATH)
    
#     # 4. Train Target & Surrogate
#     g_model = lgb.LGBMClassifier(random_state=42, verbose=-1).fit(X_train, y_train)
    
#     Z_train = transform_to_Z(X_train, preprocessor, autoencoder)
#     Z_test = transform_to_Z(X_test, preprocessor, autoencoder)
    
#     f_model = train_surrogate(Z_train, y_train)
    
#     # [Task 3: Save Surrogate Weights]
#     print(f"💾 Saving Surrogate weights to '{Config.SURROGATE_WEIGHTS_PATH}'...")
#     torch.save(f_model.state_dict(), Config.SURROGATE_WEIGHTS_PATH)
    
#     f_wrapper = wrap_for_art(f_model, Z_train)
    
#     # 5. Attack Generation & Evaluation
#     valid_indices = np.where((g_model.predict(X_test) == y_test) & (y_test == 0))[0][:Config.N_ATTACK_SAMPLES]
#     print(f"\n⚡ Running Attacks on {len(valid_indices)} samples...")
    
#     results_list = []
    
#     for idx in valid_indices:
#         Z_orig = Z_test[idx:idx+1]
#         y_orig = int(y_test.iloc[idx])
        
#         attack_results = run_attacks(f_wrapper, Z_orig, Config.ATTACKS_CONFIG)
        
#         for name, data in attack_results.items():
#             res = evaluate_adversarial(
#                 Z_orig, data["Z_adv"], y_orig, f_wrapper, g_model,
#                 preprocessor, autoencoder, T_info["cont_features"], 
#                 T_info["cat_features"], X_train, dtypes
#             )
            
#             # Print specific flip info for verification
#             if res["target_fooled"]:
#                 print(f"  > {name} (Success): L2={res['l2_distance']:.2f}, "
#                       f"CatFlips={res['cat_flips']} {res['flipped_features']}")
            
#             results_list.append({
#                 "Attack": name,
#                 "Sample_ID": idx,
#                 "Success": res["target_fooled"],
#                 "L2_Latent": res["l2_distance"],
#                 "Cat_Flips": res["cat_flips"],  # [Task 4: Metric]
#                 "Time": data["time"]
#             })
            
#     # 6. Summary
#     df_res = pd.DataFrame(results_list)
#     if not df_res.empty:
#         print("\n" + "="*60 + "\n📊 ATTACK PERFORMANCE SUMMARY\n" + "="*60)
#         summary = df_res.groupby("Attack").agg({
#             "Success": lambda x: f"{np.mean(x)*100:.1f}%",
#             "L2_Latent": "mean",
#             "Cat_Flips": "mean",
#             "Time": "mean"
#         }).reset_index()
#         print(summary.to_string(index=False))
#         print("\nNOTE: 'Cat_Flips' represents the average Hamming Distance (L0 norm)")
#         print("      calculated on categorical features between Original and Adversarial.")

# if __name__ == "__main__":
#     main()

"""
UCI CREDIT CARD ADVERSARIAL PIPELINE
- Real UCI Dataset
- SASR & TASR Metrics
- Detailed Categorical Flip/Perturbation Analysis
- Whitebox (Surrogate) & Blackbox Attacks
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import requests
import os
import io
import zipfile
from itertools import product
from collections import Counter

# ML libs
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import lightgbm as lgb

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score

# ART
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import (
    HopSkipJump, 
    FastGradientMethod, 
    ProjectedGradientDescent
)

# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════
class Config:
    # URL for the specific UCI dataset requested
    DATA_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
    DATA_FILENAME = "default of credit card clients.xls"
    
    # Experiment Settings
    TEST_SIZE = 0.2
    VAL_SIZE = 0.15
    RANDOM_STATE = 42
    
    # Autoencoder Params
    LATENT_DIM = 32  # Can be tuned, fixed here for consistency
    EPOCHS = 40
    BATCH_SIZE = 256
    LR = 1e-3
    
    # Attack Settings
    N_ATTACK_SAMPLES = 10  # Number of samples to generate attacks for
    
    # Defined Attack Methods
    ATTACKS = {
        "FGSM (Whitebox)": {"type": "whitebox", "eps": 0.5},
        "PGD (Whitebox)": {"type": "whitebox", "eps": 0.5, "steps": 10},
        "HopSkipJump (Blackbox)": {"type": "blackbox", "iter": 15}
    }

# ══════════════════════════════════════════════════════════════
# 1. DATA LOADING (REAL UCI DATA)
# ══════════════════════════════════════════════════════════════
def load_uci_data():
    """Downloads and loads the specific UCI Credit Card dataset."""
    print("📊 Data Pipeline initiated...")
    
    if not os.path.exists(Config.DATA_FILENAME):
        print(f"  Downloading data from {Config.DATA_URL}...")
        r = requests.get(Config.DATA_URL)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall()
        
    print("  Loading Excel file...")
    # The UCI file usually has a header in the second row (index 1)
    df = pd.read_excel(Config.DATA_FILENAME, header=1)
    
    # Rename for consistency
    rename_map = {"default payment next month": "TARGET", "PAY_0": "PAY_1"}
    df = df.rename(columns=rename_map)
    
    # Clean ID
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])
        
    # Validation of schema
    print(f"  ✓ Loaded Dataset: {df.shape}")
    return df

# ══════════════════════════════════════════════════════════════
# 2. PREPROCESSING & REVERSE MAPPING ARCHITECTURE
# ══════════════════════════════════════════════════════════════
class TabularPreprocessor:
    """
    Handles the complexity of OneHot encoding and, crucially,
    the INVERSE transformation required to measure feature flips.
    """
    def __init__(self, df, target_col='TARGET'):
        self.target_col = target_col
        self.cat_cols = ["SEX", "EDUCATION", "MARRIAGE"] + [f"PAY_{i}" for i in range(1, 7)]
        self.cont_cols = [c for c in df.columns if c not in self.cat_cols and c != target_col]
        
        # We need to ensure we know the exact feature order
        self.feature_names = self.cont_cols + self.cat_cols
        
        self.preprocessor = ColumnTransformer([
            ("num", StandardScaler(), self.cont_cols),
            ("cat", OneHotEncoder(sparse_output=False, handle_unknown='ignore'), self.cat_cols)
        ])
        
    def fit_transform(self, X):
        return self.preprocessor.fit_transform(X).astype(np.float32)
    
    def transform(self, X):
        return self.preprocessor.transform(X).astype(np.float32)
        
    def inverse_transform(self, X_processed):
        """
        Architecture for decoding:
        Continuous -> Inverse Scale
        OneHot Blocks -> Argmax -> Map to Category Label
        """
        # 1. Split Continuous and Categorical parts of the array
        n_cont = len(self.cont_cols)
        X_cont_scaled = X_processed[:, :n_cont]
        X_cat_onehot = X_processed[:, n_cont:]
        
        # 2. Inverse Scale Continuous
        scaler = self.preprocessor.named_transformers_['num']
        X_cont = scaler.inverse_transform(X_cont_scaled)
        
        # 3. Decode Categorical (The "Flip" Logic)
        encoder = self.preprocessor.named_transformers_['cat']
        
        # We need to manually reconstruct because standard inverse_transform 
        # expects exact binary one-hot, but our adversarial examples are soft probabilities.
        # We use Argmax to force the decision.
        
        decoded_cats = []
        start_idx = 0
        for i, cats in enumerate(encoder.categories_):
            n_cats = len(cats)
            # Extract the block for this feature
            block = X_cat_onehot[:, start_idx : start_idx + n_cats]
            # ARGMAX: Find the category with highest probability/value
            chosen_indices = np.argmax(block, axis=1)
            # Map index back to original label (e.g., 0 -> 'Married')
            original_labels = cats[chosen_indices]
            decoded_cats.append(original_labels)
            start_idx += n_cats
            
        X_cat = np.column_stack(decoded_cats)
        
        # Combine
        df_rec = pd.DataFrame(X_cont, columns=self.cont_cols)
        df_cat = pd.DataFrame(X_cat, columns=self.cat_cols)
        return pd.concat([df_rec, df_cat], axis=1)

# ══════════════════════════════════════════════════════════════
# 3. MODELS (Autoencoder, Surrogate, Target)
# ══════════════════════════════════════════════════════════════
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

class SurrogateModel(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, x): return self.net(x)

# ══════════════════════════════════════════════════════════════
# 4. ANALYSIS HELPER (Label Flipping Logic)
# ══════════════════════════════════════════════════════════════
def analyze_feature_flips(X_orig_df, X_adv_df, categorical_cols):
    """
    Computes detailed statistics on which categorical features were flipped.
    """
    flip_report = {}
    total_samples = len(X_orig_df)
    total_flips_per_sample = []
    
    # 1. Per-Feature Analysis
    for col in categorical_cols:
        # Check inequality between Original and Adversarial
        # Ensure types match
        orig_vals = X_orig_df[col].astype(str)
        adv_vals = X_adv_df[col].astype(str)
        
        is_flipped = (orig_vals != adv_vals)
        num_flips = is_flipped.sum()
        
        if num_flips > 0:
            flip_report[col] = (num_flips / total_samples) * 100
            
    # 2. Per-Sample Analysis (Hamming Distance)
    # Get just the categorical portion
    orig_cat = X_orig_df[categorical_cols].astype(str).values
    adv_cat = X_adv_df[categorical_cols].astype(str).values
    
    # Sum mismatches across columns for each row
    hamming_distances = np.sum(orig_cat != adv_cat, axis=1)
    
    return {
        "feature_breakdown": flip_report,
        "avg_hamming": np.mean(hamming_distances),
        "max_hamming": np.max(hamming_distances),
        "flipped_indices": np.where(hamming_distances > 0)[0]
    }

# ══════════════════════════════════════════════════════════════
# 5. MAIN PIPELINE
# ══════════════════════════════════════════════════════════════
def main():
    # --- Step 1: Data ---
    raw_df = load_uci_data()
    X = raw_df.drop(columns=['TARGET'])
    y = raw_df['TARGET']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Config.TEST_SIZE, stratify=y, random_state=Config.RANDOM_STATE)
    
    # Preprocess
    processor = TabularPreprocessor(raw_df)
    X_train_proc = processor.fit_transform(X_train)
    X_test_proc = processor.transform(X_test)
    
    input_dim = X_train_proc.shape[1]
    print(f"  Processed Input Dimension: {input_dim}")

    # --- Step 2: Autoencoder Training ---
    print("\n🏗️ Training Autoencoder...")
    ae = Autoencoder(input_dim, Config.LATENT_DIM)
    optimizer_ae = optim.Adam(ae.parameters(), lr=Config.LR)
    criterion_ae = nn.MSELoss()
    
    train_tensor = torch.tensor(X_train_proc)
    loader = DataLoader(TensorDataset(train_tensor), batch_size=Config.BATCH_SIZE, shuffle=True)
    
    for epoch in range(10): # Short epochs for demo
        for batch in loader:
            recon, _ = ae(batch[0])
            loss = criterion_ae(recon, batch[0])
            optimizer_ae.zero_grad()
            loss.backward()
            optimizer_ae.step()
            
    # Generate Latent Representations
    with torch.no_grad():
        Z_train = ae.encoder(torch.tensor(X_train_proc)).numpy()
        Z_test = ae.encoder(torch.tensor(X_test_proc)).numpy()

    # --- Step 3: Models Training ---
    print("🤖 Training Surrogate (MLP on Latent Space)...")
    surrogate = SurrogateModel(Config.LATENT_DIM)
    opt_surr = optim.Adam(surrogate.parameters(), lr=0.005)
    crit_surr = nn.CrossEntropyLoss()
    
    # Train Surrogate
    z_tensor = torch.tensor(Z_train)
    y_tensor = torch.tensor(y_train.values).long()
    surr_loader = DataLoader(TensorDataset(z_tensor, y_tensor), batch_size=256, shuffle=True)
    
    for epoch in range(10):
        for bz, by in surr_loader:
            preds = surrogate(bz)
            loss = crit_surr(preds, by)
            opt_surr.zero_grad()
            loss.backward()
            opt_surr.step()
            
    # Wrap Surrogate for ART
    surrogate_art = PyTorchClassifier(
        model=surrogate,
        loss=crit_surr,
        optimizer=opt_surr,
        input_shape=(Config.LATENT_DIM,),
        nb_classes=2
    )

    print("🎯 Training Target Model (LightGBM on Feature Space)...")
    target_model = lgb.LGBMClassifier(verbose=-1).fit(X_train, y_train)
    
    # --- Step 4: Attack Loop & Analysis ---
    print(f"\n⚡ Generating Attacks on {Config.N_ATTACK_SAMPLES} samples...")
    
    # Select samples that the Target classifies correctly (Class 0)
    # We want to force them to Class 1
    base_preds = target_model.predict(X_test)
    valid_idxs = np.where((base_preds == 0) & (y_test == 0))[0][:Config.N_ATTACK_SAMPLES]
    
    if len(valid_idxs) == 0:
        print("No valid samples found (target accuracy might be too low/high).")
        return

    results_table = []

    for attack_name, cfg in Config.ATTACKS.items():
        print(f"  > Running {attack_name}...")
        
        # 1. Generate Attack in Latent Space
        if cfg["type"] == "whitebox":
            if "FGSM" in attack_name:
                attacker = FastGradientMethod(estimator=surrogate_art, eps=cfg["eps"])
            else:
                attacker = ProjectedGradientDescent(estimator=surrogate_art, eps=cfg["eps"], max_iter=cfg.get("steps", 10))
            Z_adv_batch = attacker.generate(Z_test[valid_idxs])
            
        else: # Blackbox
            # HSJ requires querying the surrogate (which mimics the target in a blackbox setting)
            attacker = HopSkipJump(classifier=surrogate_art, max_iter=cfg["iter"], verbose=False)
            Z_adv_batch = attacker.generate(Z_test[valid_idxs])

        # 2. Decode Adversarial Examples to Feature Space
        with torch.no_grad():
            X_adv_raw_proc = ae.decoder(torch.tensor(Z_adv_batch)).numpy()
            
        # 3. Inverse Transform (The Architecture of Discretization)
        # This converts continuous outputs -> discrete categories (Label Flipping step)
        X_orig_df = X_test.iloc[valid_idxs].reset_index(drop=True)
        X_adv_df = processor.inverse_transform(X_adv_raw_proc)

        # 4. Calculate Metrics
        # SASR: Does Surrogate predict Class 1?
        surr_preds = np.argmax(surrogate_art.predict(Z_adv_batch), axis=1)
        sasr = np.mean(surr_preds != 0) * 100
        
        # TASR: Does Target (LGBM) predict Class 1?
        target_preds = target_model.predict(X_adv_df)
        tasr = np.mean(target_preds != 0) * 100
        
        # Categorical Analysis
        flip_stats = analyze_feature_flips(X_orig_df, X_adv_df, processor.cat_cols)
        
        results_table.append({
            "Attack Method": attack_name,
            "S-ASR (%)": sasr,
            "T-ASR (%)": tasr,
            "Avg L0 (Cat Flips)": f"{flip_stats['avg_hamming']:.2f}",
            "Most Flipped Feature": max(flip_stats['feature_breakdown'], key=flip_stats['feature_breakdown'].get) if flip_stats['feature_breakdown'] else "None"
        })
        
        # Detailed print for the first attack
        print(f"    S-ASR: {sasr:.1f}%, T-ASR: {tasr:.1f}%")
        print(f"    Avg Categorical Features Flipped: {flip_stats['avg_hamming']:.2f}")

    # --- Step 5: Final Report ---
    print("\n" + "="*60)
    print("🔬 FINAL ADVERSARIAL ANALYSIS REPORT")
    print("="*60)
    df_res = pd.DataFrame(results_table)
    print(df_res.to_string(index=False))
    print("\nLegend:")
    print("S-ASR: Surrogate Attack Success Rate (Whitebox/Blackbox success on the proxy)")
    print("T-ASR: Target Attack Success Rate (Transfer success on the real LightGBM)")
    print("Avg L0: Average Hamming Distance (Count of categorical variables flipped per sample)")

if __name__ == "__main__":
    main()
