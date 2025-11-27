"""
================================================================================
ADVERSARIAL ATTACKS ON TABULAR DATA: PIPELINE V2.1 (FIXED)
================================================================================
Fixes:
- Resolved KeyError by aligning column renaming logic with target definition.
- Added robust column checking.
- Includes complete Label Flipping Architecture & Artifact Saving.
================================================================================
"""

import os
import sys
import time
import logging
import random
import requests
import zipfile
import io
import warnings
from typing import Dict, List, Tuple, Any, Optional
from itertools import product
from collections import Counter

# Scientific Computing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning - PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Machine Learning - Scikit-Learn / LightGBM
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Adversarial Robustness Toolbox (ART)
try:
    from art.estimators.classification import PyTorchClassifier
    from art.attacks.evasion import (
        FastGradientMethod,
        ProjectedGradientDescent,
        HopSkipJump
    )
except ImportError:
    print("❌ Error: 'adversarial-robustness-toolbox' not installed.")
    print("   Run: pip install adversarial-robustness-toolbox")
    sys.exit(1)

# Suppress warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

class Config:
    """Global Experiment Configuration"""
    
    # --- Paths & URLs ---
    DATA_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
    DATA_FILE = "default of credit card clients.xls"
    OUTPUT_DIR = "adversarial_results"
    AE_WEIGHTS_FILE = "best_autoencoder.pth"
    SURROGATE_WEIGHTS_FILE = "surrogate_model.pth"
    
    # --- Data Definition ---
    # The name as it appears in the raw Excel file
    RAW_TARGET_NAME = "default payment next month" 
    # The name we will use internally after renaming
    TARGET_COL = "TARGET" 
    
    # Renaming map to standardize columns
    RENAME_MAP = {
        RAW_TARGET_NAME: TARGET_COL, 
        "PAY_0": "PAY_1"
    }
    
    # --- Experiment Settings ---
    TEST_SIZE = 0.2
    VAL_SIZE = 0.15
    RANDOM_STATE = 42
    
    # --- Grid Search Space (Autoencoder) ---
    GRID_SEARCH = {
        'latent_dim': [8,16,32,48, 64,80],
        'lr': [5e-4,1e-3,2e-3],
        'batch_size': [64,128,256],
        'dropout': [0.1,0.2,0.3]
    }
    
    # --- Training Params ---
    EPOCHS_AE = 40
    EPOCHS_SURROGATE = 30
    PATIENCE = 5
    
    # --- Attack Configuration ---
    ATTACK_SCENARIOS = {
        "FGSM_Whitebox": {
            "method": "FGSM",
            "type": "whitebox",
            "eps": 0.5,
            "description": "Fast Gradient Sign Method on Surrogate"
        },
        "PGD_Whitebox": {
            "method": "PGD",
            "type": "whitebox",
            "eps": 0.5,
            "eps_step": 0.1,
            "max_iter": 20,
            "description": "Projected Gradient Descent on Surrogate"
        },
        "HopSkipJump_Blackbox": {
            "method": "HSJ",
            "type": "blackbox",
            "max_iter": 15,
            "max_eval": 1000,
            "init_eval": 100,
            "description": "Query-based Blackbox attack"
        }
    }
    
    N_ATTACK_SAMPLES = 50  # Samples to evaluate
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(Config.OUTPUT_DIR, exist_ok=True)


# ==============================================================================
# 1. DATA PIPELINE
# ==============================================================================

class DataManager:
    """Handles downloading, loading, and cleaning the UCI dataset."""
    
    @staticmethod
    def get_data():
        """Downloads (if needed) and loads the dataset."""
        if not os.path.exists(Config.DATA_FILE):
            logger.info(f"Downloading dataset from {Config.DATA_URL}...")
            try:
                r = requests.get(Config.DATA_URL)
                with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                    z.extractall()
                logger.info("Download and extraction complete.")
            except Exception as e:
                logger.error(f"Failed to download data: {e}")
                sys.exit(1)
        
        logger.info("Loading Excel file into Pandas...")
        try:
            # Header is usually row 1 in this dataset
            df = pd.read_excel(Config.DATA_FILE, header=1)
            
            # Apply renaming immediately
            df = df.rename(columns=Config.RENAME_MAP)
            
            # Verify Target Exists
            if Config.TARGET_COL not in df.columns:
                # Fallback: Check if the raw name still exists (in case renaming failed or header diff)
                available = list(df.columns)
                raise KeyError(f"Target column '{Config.TARGET_COL}' not found. Available: {available[:5]}...")
                
            # Drop ID if exists
            if 'ID' in df.columns:
                df = df.drop('ID', axis=1)
                
            # Cleaning Categories
            if 'EDUCATION' in df.columns:
                df['EDUCATION'] = df['EDUCATION'].apply(lambda x: 4 if x in [0, 5, 6] else x)
            if 'MARRIAGE' in df.columns:
                df['MARRIAGE'] = df['MARRIAGE'].apply(lambda x: 3 if x == 0 else x)
                
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error reading Excel file: {e}")
            sys.exit(1)

class FeatureEngineer:
    """
    Manages Preprocessing and INVERSE Preprocessing.
    Crucial for calculating Categorical Flips.
    """
    
    def __init__(self, df: pd.DataFrame, target_col: str):
        self.target_col = target_col
        self.feature_cols = [c for c in df.columns if c != target_col]
        
        # Define types explicitly for this dataset
        self.cat_features = ["SEX", "EDUCATION", "MARRIAGE"] + [f"PAY_{i}" for i in range(1, 7)]
        # Robust check to ensure these columns actually exist
        self.cat_features = [c for c in self.cat_features if c in df.columns]
        
        self.num_features = [c for c in self.feature_cols if c not in self.cat_features]
        
        logger.info(f"Categorical Features: {self.cat_features}")
        logger.info(f"Numerical Features: {self.num_features}")
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.num_features),
                ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), self.cat_features)
            ],
            verbose_feature_names_out=False
        )
        
    def fit(self, X):
        self.preprocessor.fit(X)
        self.input_dim = self.preprocessor.transform(X[:1]).shape[1]
        
        self.ohe = self.preprocessor.named_transformers_['cat']
        self.scaler = self.preprocessor.named_transformers_['num']
        self.cat_indices_start = len(self.num_features)
        
        # Map features to their categories for decoding
        self.categories_map = {}
        if hasattr(self.ohe, 'categories_'):
            for i, feature in enumerate(self.cat_features):
                self.categories_map[feature] = self.ohe.categories_[i]
            
        logger.info(f"Preprocessing fitted. Input Dim: {self.input_dim}")
        return self

    def transform(self, X):
        return self.preprocessor.transform(X).astype(np.float32)
    
    def inverse_transform_categorical(self, X_processed_batch):
        """
        Reconstructs DataFrame from continuous (soft) Autoencoder output.
        Uses Argmax for categorical decision (Label Flipping Architecture).
        """
        # 1. Split Num/Cat
        X_num_scaled = X_processed_batch[:, :self.cat_indices_start]
        X_cat_encoded = X_processed_batch[:, self.cat_indices_start:]
        
        # 2. Inverse Scale Numerical
        X_num = self.scaler.inverse_transform(X_num_scaled)
        
        # 3. Decode Categorical
        decoded_cats = {}
        curr_idx = 0
        
        for feature in self.cat_features:
            cats = self.categories_map[feature]
            n_cats = len(cats)
            
            # Slice probabilities
            probs = X_cat_encoded[:, curr_idx : curr_idx + n_cats]
            
            # Argmax to determine flipped category
            chosen_indices = np.argmax(probs, axis=1)
            decoded_values = cats[chosen_indices]
            
            decoded_cats[feature] = decoded_values
            curr_idx += n_cats
            
        df_num = pd.DataFrame(X_num, columns=self.num_features)
        df_cat = pd.DataFrame(decoded_cats)
        
        return pd.concat([df_num, df_cat], axis=1)


# ==============================================================================
# 2. MODELS
# ==============================================================================

class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, dropout: float = 0.2):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(128, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(256, input_dim)
        )
        
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

class SurrogateModel(nn.Module):
    """Whitebox Proxy trained on Latent Space"""
    def __init__(self, latent_dim: int):
        super(SurrogateModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        
    def forward(self, x):
        return self.net(x)


# ==============================================================================
# 3. TRAINING & HELPERS
# ==============================================================================

class ModelTrainer:
    @staticmethod
    def train_ae(model, train_loader, val_loader, lr, epochs, patience):
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        model.to(Config.DEVICE)
        
        best_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for X_batch, in train_loader:
                X_batch = X_batch.to(Config.DEVICE)
                optimizer.zero_grad()
                recon, _ = model(X_batch)
                loss = criterion(recon, X_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, in val_loader:
                    X_batch = X_batch.to(Config.DEVICE)
                    recon, _ = model(X_batch)
                    val_loss += criterion(recon, X_batch).item()
            
            avg_val_loss = val_loss / len(val_loader)
            
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                break
                
        if best_state:
            model.load_state_dict(best_state)
        return model, best_loss

    @staticmethod
    def grid_search(X_train, X_val, input_dim):
        logger.info("Starting Autoencoder Grid Search...")
        best_params = None
        best_val_loss = float('inf')
        
        # Prepare DataLoaders once to save time? 
        # Actually batch size is a param, so we create inside loop
        X_train_t = torch.FloatTensor(X_train)
        X_val_t = torch.FloatTensor(X_val)
        
        keys, values = zip(*Config.GRID_SEARCH.items())
        combinations = [dict(zip(keys, v)) for v in product(*values)]
        
        for i, params in enumerate(combinations):
            train_loader = DataLoader(TensorDataset(X_train_t), batch_size=params['batch_size'], shuffle=True)
            val_loader = DataLoader(TensorDataset(X_val_t), batch_size=params['batch_size'])
            
            model = Autoencoder(input_dim, params['latent_dim'], params['dropout'])
            
            # Quick training for GS
            _, val_loss = ModelTrainer.train_ae(
                model, train_loader, val_loader, 
                lr=params['lr'], epochs=5, patience=2
            )
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = params
                print(f"  [{i+1}/{len(combinations)}] New Best: {params} | Loss: {val_loss:.5f}")
                
        return best_params

    @staticmethod
    def train_surrogate(model, Z_train, y_train, epochs):
        model.to(Config.DEVICE)
        dataset = TensorDataset(torch.FloatTensor(Z_train), torch.LongTensor(y_train))
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(epochs):
            for zb, yb in loader:
                zb, yb = zb.to(Config.DEVICE), yb.to(Config.DEVICE)
                optimizer.zero_grad()
                outputs = model(zb)
                loss = criterion(outputs, yb)
                loss.backward()
                optimizer.step()
        return model

# ==============================================================================
# 4. ANALYSIS LOGIC (Flip Detection)
# ==============================================================================

class AttackAnalyzer:
    def __init__(self, feature_engine, target_model):
        self.fe = feature_engine
        self.target = target_model
        
    def analyze(self, X_orig_df, X_adv_df, y_true, surr_preds):
        """Calculates SASR, TASR, and Flip Stats"""
        
        # 1. Flip Stats (L0 Norm / Hamming)
        # We compare categorical columns row by row
        total_flips = 0
        flipped_features_count = Counter()
        n_samples = len(X_orig_df)
        
        hamming_dists = []
        
        for i in range(n_samples):
            row_orig = X_orig_df.iloc[i]
            row_adv = X_adv_df.iloc[i]
            sample_flips = 0
            
            for feat in self.fe.cat_features:
                # String comparison is safest for categories
                if str(row_orig[feat]) != str(row_adv[feat]):
                    sample_flips += 1
                    flipped_features_count[feat] += 1
            
            hamming_dists.append(sample_flips)
            total_flips += sample_flips
            
        avg_hamming = np.mean(hamming_dists)
        
        # 2. Attack Success Rates
        # Target Model Success (TASR)
        target_preds = self.target.predict(X_adv_df)
        # Success = Prediction is NOT the true label (assuming untargeted evasion)
        tasr = np.mean(target_preds != y_true) * 100
        
        # Surrogate Success (SASR)
        sasr = np.mean(surr_preds != y_true) * 100
        
        return {
            "S-ASR": sasr,
            "T-ASR": tasr,
            "Avg_Cat_Flips": avg_hamming,
            "Most_Flipped": flipped_features_count.most_common(1)
        }


# ==============================================================================
# 5. MAIN
# ==============================================================================

def main():
    logger.info("Initializing Pipeline...")
    
    # 1. Load Data
    df = DataManager.get_data()
    
    # 2. Split X/y
    # Use Config.TARGET_COL which is now "TARGET" (the renamed one)
    X = df.drop(Config.TARGET_COL, axis=1)
    y = df[Config.TARGET_COL]
    
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=Config.TEST_SIZE, stratify=y, random_state=Config.RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=Config.VAL_SIZE, stratify=y_train_full, random_state=Config.RANDOM_STATE
    )
    
    # 3. Feature Engineering
    logger.info("Fitting Feature Engineer...")
    fe = FeatureEngineer(X_train, Config.TARGET_COL)
    fe.fit(X_train)
    
    X_train_enc = fe.transform(X_train)
    X_val_enc = fe.transform(X_val)
    X_test_enc = fe.transform(X_test)
    input_dim = X_train_enc.shape[1]
    
    # 4. Autoencoder (Grid Search + Final Train)
    best_params = ModelTrainer.grid_search(X_train_enc, X_val_enc, input_dim)
    logger.info(f"Training Final AE with: {best_params}")
    
    ae = Autoencoder(input_dim, best_params['latent_dim'], best_params['dropout'])
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train_enc)), batch_size=best_params['batch_size'], shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val_enc)), batch_size=best_params['batch_size'])
    
    ae, _ = ModelTrainer.train_ae(
        ae, train_loader, val_loader, 
        lr=best_params['lr'], epochs=Config.EPOCHS_AE, patience=Config.PATIENCE
    )
    
    # SAVE WEIGHTS (Task 3)
    torch.save(ae.state_dict(), os.path.join(Config.OUTPUT_DIR, Config.AE_WEIGHTS_FILE))
    
    # Generate Latent Representations
    ae.to(Config.DEVICE)
    ae.eval()
    with torch.no_grad():
        Z_train = ae.encoder(torch.FloatTensor(X_train_enc).to(Config.DEVICE)).cpu().numpy()
        Z_test = ae.encoder(torch.FloatTensor(X_test_enc).to(Config.DEVICE)).cpu().numpy()
        
    # 5. Train Models
    # Target (LightGBM)
    logger.info("Training Target (LightGBM)...")
    target = lgb.LGBMClassifier(random_state=42, verbose=-1)
    target.fit(X_train, y_train)
    logger.info(f"Target Accuracy: {accuracy_score(y_test, target.predict(X_test))*100:.2f}%")
    
    # Surrogate (MLP)
    logger.info("Training Surrogate...")
    surrogate = SurrogateModel(best_params['latent_dim'])
    surrogate = ModelTrainer.train_surrogate(
        surrogate, Z_train, y_train.values, Config.EPOCHS_SURROGATE
    )
    torch.save(surrogate.state_dict(), os.path.join(Config.OUTPUT_DIR, Config.SURROGATE_WEIGHTS_FILE))
    
    # 6. Attack Phase
    # Select victims (True Negatives - we want to flip them to 1)
    preds = target.predict(X_test)
    victim_idxs = np.where((preds == 0) & (y_test == 0))[0]
    if len(victim_idxs) > Config.N_ATTACK_SAMPLES:
        victim_idxs = np.random.choice(victim_idxs, Config.N_ATTACK_SAMPLES, replace=False)
        
    Z_vic = Z_test[victim_idxs]
    X_vic_df = X_test.iloc[victim_idxs].reset_index(drop=True)
    y_vic = y_test.iloc[victim_idxs].values
    
    # Prepare ART
    z_min, z_max = Z_train.min(), Z_train.max()
    art_classifier = PyTorchClassifier(
        model=surrogate,
        loss=nn.CrossEntropyLoss(),
        optimizer=optim.Adam(surrogate.parameters()),
        input_shape=(best_params['latent_dim'],),
        nb_classes=2,
        clip_values=(z_min, z_max),
        device_type='gpu' if torch.cuda.is_available() else 'cpu'
    )
    
    analyzer = AttackAnalyzer(fe, target)
    results = []
    
    for name, cfg in Config.ATTACK_SCENARIOS.items():
        logger.info(f"Running {name}...")
        start = time.time()
        
        # Generate
        if cfg['method'] == "FGSM":
            attack = FastGradientMethod(estimator=art_classifier, eps=cfg['eps'])
            Z_adv = attack.generate(Z_vic)
        elif cfg['method'] == "PGD":
            attack = ProjectedGradientDescent(estimator=art_classifier, eps=cfg['eps'], eps_step=cfg['eps_step'], max_iter=cfg['max_iter'], verbose=False)
            Z_adv = attack.generate(Z_vic)
        elif cfg['method'] == "HSJ":
            attack = HopSkipJump(classifier=art_classifier, max_iter=cfg['max_iter'], verbose=False)
            Z_adv = attack.generate(Z_vic)
            
        # Reconstruct
        with torch.no_grad():
            X_adv_recon = ae.decoder(torch.FloatTensor(Z_adv).to(Config.DEVICE)).cpu().numpy()
        
        # Label Flipping Architecture (Argmax decoding)
        X_adv_df = fe.inverse_transform_categorical(X_adv_recon)
        
        # Metrics
        surr_preds = np.argmax(art_classifier.predict(Z_adv), axis=1)
        metrics = analyzer.analyze(X_vic_df, X_adv_df, y_vic, surr_preds)
        
        metrics['Attack'] = name
        metrics['Time'] = time.time() - start
        results.append(metrics)
        logger.info(f"Done. T-ASR: {metrics['T-ASR']:.2f}%, AvgFlips: {metrics['Avg_Cat_Flips']:.2f}")

    # 7. Summary
    res_df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(res_df.to_string(index=False))
    res_df.to_csv(os.path.join(Config.OUTPUT_DIR, "final_results.csv"), index=False)
    print("\nPipeline Complete.")

if __name__ == "__main__":
    main()