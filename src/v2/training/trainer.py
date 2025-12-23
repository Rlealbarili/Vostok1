"""
VOSTOK V2 :: LightGBM Training Pipeline (The Dojo)
===================================================
Pipeline completo de treinamento com:
- Triple Barrier Labeling (dinâmico com ATR)
- Purged K-Fold Cross-Validation (evita data leakage)
- LightGBM com early stopping

Arquiteto: Petrovich | Operador: Vostok
"""

import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    lgb = None

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Import do FeatureGenerator (path relativo)
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.v2.features.engineering import FeatureGenerator, FEATURE_COLUMNS

logger = logging.getLogger("vostok_trainer")


# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_MODEL_PATH = "models/v2/lgbm_model.txt"
DEFAULT_METRICS_PATH = "models/v2/model_metrics.json"
DEFAULT_DATA_PATH = "data/training/dataset.jsonl"

# Triple Barrier defaults
TP_MULTIPLIER = 2.0   # Take Profit = 2x ATR
SL_MULTIPLIER = 1.0   # Stop Loss = 1x ATR
TIME_LIMIT = 45       # Candles (45 minutes)
ATR_PERIOD = 14

# LightGBM HFT Parameters
LGBM_PARAMS = {
    "objective": "binary",
    "metric": ["binary_logloss", "auc"],
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,          # Sem limite (LightGBM lida bem)
    "min_data_in_leaf": 100,  # Evitar overfitting
    "feature_fraction": 0.8,  # Diversidade
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,         # Regularização
    "lambda_l2": 1.0,
    "verbose": -1,
    "n_jobs": -1,
    "seed": 42,
}

N_ESTIMATORS = 1000
EARLY_STOPPING_ROUNDS = 50
N_FOLDS = 5
EMBARGO_PERIOD = 10  # Gap entre treino e validação


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class FoldResult:
    """Resultado de um fold de validação."""
    fold_idx: int
    precision: float
    recall: float
    f1: float
    accuracy: float
    confusion: np.ndarray


@dataclass
class TrainingResult:
    """Resultado do treinamento completo."""
    success: bool
    model_path: str
    n_samples: int
    n_features: int
    n_positive: int
    n_negative: int
    folds: List[FoldResult]
    avg_precision: float
    avg_recall: float
    avg_f1: float
    feature_importance: Dict[str, float]
    training_time_seconds: float


# ============================================================================
# TRIPLE BARRIER LABELING
# ============================================================================

def apply_triple_barrier(
    df: pd.DataFrame,
    tp_multiplier: float = TP_MULTIPLIER,
    sl_multiplier: float = SL_MULTIPLIER,
    time_limit: int = TIME_LIMIT,
    atr_period: int = ATR_PERIOD,
) -> pd.Series:
    """
    Aplica Triple Barrier Labeling para criar targets.
    
    Para cada candle, olha 'time_limit' candles para frente:
    - Se tocar TP (close + tp_mult * ATR) primeiro → 1 (WIN/LONG)
    - Se tocar SL (close - sl_mult * ATR) primeiro → 0 (LOSS)
    - Se tempo acabar → 0 (NO TRADE)
    
    Args:
        df: DataFrame com colunas 'high', 'low', 'close'
        tp_multiplier: Multiplicador ATR para Take Profit
        sl_multiplier: Multiplicador ATR para Stop Loss
        time_limit: Número de candles para frente
        atr_period: Período do ATR
        
    Returns:
        Series com labels (0 ou 1)
    """
    df = df.copy()
    df.columns = df.columns.str.lower()
    
    n = len(df)
    labels = np.zeros(n, dtype=int)
    
    # Calcular ATR
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - np.roll(close, 1)),
            np.abs(low - np.roll(close, 1))
        )
    )
    tr[0] = high[0] - low[0]
    
    # ATR rolling (usando convolution para eficiência)
    atr = pd.Series(tr).rolling(window=atr_period).mean().values
    
    logger.info(f"Applying Triple Barrier: TP={tp_multiplier}x, SL={sl_multiplier}x, Time={time_limit}")
    
    for i in range(n - time_limit):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue
        
        entry_price = close[i]
        tp_price = entry_price + (tp_multiplier * atr[i])
        sl_price = entry_price - (sl_multiplier * atr[i])
        
        # Olhar candles futuros
        for j in range(1, time_limit + 1):
            future_idx = i + j
            if future_idx >= n:
                break
            
            future_high = high[future_idx]
            future_low = low[future_idx]
            
            # Verificar se tocou TP primeiro
            if future_high >= tp_price:
                labels[i] = 1  # WIN
                break
            
            # Verificar se tocou SL
            if future_low <= sl_price:
                labels[i] = 0  # LOSS
                break
        
        # Se não tocou nenhum = 0 (timeout)
    
    # Últimos 'time_limit' candles não podem ser rotulados
    labels[-time_limit:] = -1  # Marcar como inválido
    
    return pd.Series(labels, index=df.index)


# ============================================================================
# PURGED K-FOLD CROSS-VALIDATION
# ============================================================================

class PurgedKFold:
    """
    K-Fold com Embargo para séries temporais.
    
    Remove observações adjacentes ao split para evitar
    vazamento de dados de olhar para frente.
    """
    
    def __init__(self, n_splits: int = 5, embargo: int = EMBARGO_PERIOD):
        self.n_splits = n_splits
        self.embargo = embargo
    
    def split(self, X: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Gera índices de treino e validação com embargo.
        
        Yields:
            (train_indices, val_indices) para cada fold
        """
        n_samples = len(X)
        fold_size = n_samples // self.n_splits
        
        splits = []
        
        for fold_idx in range(self.n_splits):
            val_start = fold_idx * fold_size
            val_end = (fold_idx + 1) * fold_size if fold_idx < self.n_splits - 1 else n_samples
            
            val_indices = np.arange(val_start, val_end)
            
            # Treino: tudo que não está no val + embargo
            train_indices = []
            
            # Antes do val (com embargo no final)
            train_end_before = max(0, val_start - self.embargo)
            if train_end_before > 0:
                train_indices.extend(range(0, train_end_before))
            
            # Depois do val (com embargo no início)
            train_start_after = min(n_samples, val_end + self.embargo)
            if train_start_after < n_samples:
                train_indices.extend(range(train_start_after, n_samples))
            
            train_indices = np.array(train_indices)
            
            splits.append((train_indices, val_indices))
        
        return splits


# ============================================================================
# VOSTOK TRAINER CLASS
# ============================================================================

class VostokTrainer:
    """
    Pipeline de treinamento do modelo LightGBM para Vostok V2.
    
    Uso:
        trainer = VostokTrainer()
        result = trainer.train("data/training/ohlcv_btc.csv")
        
        if result.success:
            print(f"Model saved to {result.model_path}")
            print(f"Average Precision: {result.avg_precision:.2%}")
    """
    
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        metrics_path: str = DEFAULT_METRICS_PATH,
        tp_multiplier: float = TP_MULTIPLIER,
        sl_multiplier: float = SL_MULTIPLIER,
        time_limit: int = TIME_LIMIT,
        n_folds: int = N_FOLDS,
        embargo: int = EMBARGO_PERIOD,
    ):
        self.model_path = Path(model_path)
        self.metrics_path = Path(metrics_path)
        self.tp_multiplier = tp_multiplier
        self.sl_multiplier = sl_multiplier
        self.time_limit = time_limit
        self.n_folds = n_folds
        self.embargo = embargo
        
        # Components
        self.feature_generator = FeatureGenerator()
        self.purged_kfold = PurgedKFold(n_splits=n_folds, embargo=embargo)
        
        # Ensure directories exist
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"VostokTrainer initialized: "
            f"TP={tp_multiplier}x, SL={sl_multiplier}x, "
            f"Time={time_limit}, Folds={n_folds}"
        )
    
    def train(self, data_path: str) -> TrainingResult:
        """
        Executa o pipeline completo de treinamento.
        
        Args:
            data_path: Caminho para CSV/JSONL com dados OHLCV
            
        Returns:
            TrainingResult com métricas e status
        """
        start_time = datetime.now()
        
        if not HAS_LIGHTGBM:
            logger.error("LightGBM not installed!")
            return self._create_failed_result("LightGBM not installed")
        
        # 1. Carregar dados
        logger.info("=" * 60)
        logger.info("STEP 1: Loading data...")
        df = self._load_data(data_path)
        
        if df is None or len(df) < 1000:
            return self._create_failed_result(f"Insufficient data: {len(df) if df is not None else 0} rows")
        
        logger.info(f"  Loaded {len(df):,} rows")
        
        # 2. Gerar features
        logger.info("=" * 60)
        logger.info("STEP 2: Generating features...")
        df_features = self.feature_generator.generate(df)
        logger.info(f"  Generated {len(df_features):,} rows with features")
        
        # 3. Gerar labels (Triple Barrier)
        logger.info("=" * 60)
        logger.info("STEP 3: Applying Triple Barrier labeling...")
        labels = apply_triple_barrier(
            df_features,
            tp_multiplier=self.tp_multiplier,
            sl_multiplier=self.sl_multiplier,
            time_limit=self.time_limit,
        )
        
        # Adicionar labels ao df
        df_features['label'] = labels
        
        # Remover labels inválidos (-1) e NaNs
        df_features = df_features[df_features['label'] >= 0].dropna()
        
        n_positive = (df_features['label'] == 1).sum()
        n_negative = (df_features['label'] == 0).sum()
        
        logger.info(f"  Labels: {n_positive:,} WINS (1), {n_negative:,} LOSSES (0)")
        logger.info(f"  Win Rate: {n_positive / len(df_features):.1%}")
        
        # 4. Preparar X e y
        logger.info("=" * 60)
        logger.info("STEP 4: Preparing feature matrix...")
        
        feature_cols = [c for c in FEATURE_COLUMNS if c in df_features.columns]
        X = df_features[feature_cols].values
        y = df_features['label'].values
        
        logger.info(f"  X shape: {X.shape}")
        logger.info(f"  Features: {len(feature_cols)}")
        
        # 5. Purged K-Fold Cross-Validation
        logger.info("=" * 60)
        logger.info(f"STEP 5: Purged K-Fold CV ({self.n_folds} folds, embargo={self.embargo})...")
        
        fold_results = []
        splits = self.purged_kfold.split(X)
        
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            logger.info(f"\n  --- Fold {fold_idx + 1}/{self.n_folds} ---")
            logger.info(f"  Train: {len(train_idx):,} | Val: {len(val_idx):,}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Criar datasets LightGBM
            train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
            val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols, reference=train_data)
            
            # Treinar
            model = lgb.train(
                LGBM_PARAMS,
                train_data,
                num_boost_round=N_ESTIMATORS,
                valid_sets=[val_data],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS),
                    lgb.log_evaluation(period=100),
                ],
            )
            
            # Avaliar
            y_pred_proba = model.predict(X_val)
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            accuracy = accuracy_score(y_val, y_pred)
            conf_matrix = confusion_matrix(y_val, y_pred)
            
            fold_result = FoldResult(
                fold_idx=fold_idx,
                precision=precision,
                recall=recall,
                f1=f1,
                accuracy=accuracy,
                confusion=conf_matrix,
            )
            fold_results.append(fold_result)
            
            logger.info(f"  Precision: {precision:.4f}")
            logger.info(f"  Recall: {recall:.4f}")
            logger.info(f"  F1: {f1:.4f}")
            logger.info(f"  Confusion Matrix:\n{conf_matrix}")
        
        # 6. Calcular métricas médias
        avg_precision = np.mean([f.precision for f in fold_results])
        avg_recall = np.mean([f.recall for f in fold_results])
        avg_f1 = np.mean([f.f1 for f in fold_results])
        
        logger.info("=" * 60)
        logger.info("CROSS-VALIDATION SUMMARY:")
        logger.info(f"  Avg Precision: {avg_precision:.4f}")
        logger.info(f"  Avg Recall: {avg_recall:.4f}")
        logger.info(f"  Avg F1: {avg_f1:.4f}")
        
        # 7. Treinar modelo final em todo o dataset
        logger.info("=" * 60)
        logger.info("STEP 6: Training final model on full dataset...")
        
        full_train_data = lgb.Dataset(X, label=y, feature_name=feature_cols)
        
        final_model = lgb.train(
            LGBM_PARAMS,
            full_train_data,
            num_boost_round=N_ESTIMATORS,
            callbacks=[lgb.log_evaluation(period=200)],
        )
        
        # 8. Feature importance
        importance = dict(zip(feature_cols, final_model.feature_importance(importance_type='gain')))
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        logger.info("\nTop 10 Feature Importance:")
        for i, (feat, imp) in enumerate(list(importance.items())[:10]):
            logger.info(f"  {i+1}. {feat}: {imp:.2f}")
        
        # 9. Salvar modelo
        logger.info("=" * 60)
        logger.info(f"STEP 7: Saving model to {self.model_path}...")
        final_model.save_model(str(self.model_path))
        
        # 10. Salvar métricas
        training_time = (datetime.now() - start_time).total_seconds()
        
        metrics = {
            "training_date": datetime.now().isoformat(),
            "data_path": data_path,
            "n_samples": len(df_features),
            "n_features": len(feature_cols),
            "n_positive": int(n_positive),
            "n_negative": int(n_negative),
            "win_rate": float(n_positive / len(df_features)),
            "tp_multiplier": self.tp_multiplier,
            "sl_multiplier": self.sl_multiplier,
            "time_limit": self.time_limit,
            "n_folds": self.n_folds,
            "embargo": self.embargo,
            "avg_precision": float(avg_precision),
            "avg_recall": float(avg_recall),
            "avg_f1": float(avg_f1),
            "training_time_seconds": training_time,
            "feature_importance": {k: float(v) for k, v in importance.items()},
            "lgbm_params": LGBM_PARAMS,
        }
        
        with open(self.metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"  Metrics saved to {self.metrics_path}")
        
        logger.info("=" * 60)
        logger.info(f"✅ TRAINING COMPLETE in {training_time:.1f}s")
        logger.info(f"   Model: {self.model_path}")
        logger.info(f"   Precision: {avg_precision:.2%}")
        logger.info("=" * 60)
        
        return TrainingResult(
            success=True,
            model_path=str(self.model_path),
            n_samples=len(df_features),
            n_features=len(feature_cols),
            n_positive=int(n_positive),
            n_negative=int(n_negative),
            folds=fold_results,
            avg_precision=avg_precision,
            avg_recall=avg_recall,
            avg_f1=avg_f1,
            feature_importance=importance,
            training_time_seconds=training_time,
        )
    
    def _load_data(self, data_path: str) -> Optional[pd.DataFrame]:
        """Carrega dados de CSV ou JSONL."""
        path = Path(data_path)
        
        if not path.exists():
            logger.error(f"Data file not found: {path}")
            return None
        
        try:
            if path.suffix == '.csv':
                df = pd.read_csv(path)
            elif path.suffix == '.jsonl':
                df = pd.read_json(path, lines=True)
            else:
                logger.error(f"Unsupported file format: {path.suffix}")
                return None
            
            # Normalizar colunas
            df.columns = df.columns.str.lower()
            
            # Verificar colunas mínimas
            required = ['high', 'low', 'close', 'volume']
            missing = [c for c in required if c not in df.columns]
            if missing:
                logger.error(f"Missing columns: {missing}")
                return None
            
            return df
            
        except Exception as e:
            logger.exception(f"Failed to load data: {e}")
            return None
    
    def _create_failed_result(self, reason: str) -> TrainingResult:
        """Cria resultado de falha."""
        logger.error(f"Training failed: {reason}")
        return TrainingResult(
            success=False,
            model_path="",
            n_samples=0,
            n_features=0,
            n_positive=0,
            n_negative=0,
            folds=[],
            avg_precision=0.0,
            avg_recall=0.0,
            avg_f1=0.0,
            feature_importance={},
            training_time_seconds=0.0,
        )


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Entry point para treinamento via CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
    )
    
    print("=" * 60)
    print("🥋 VOSTOK V2 :: Training Dojo")
    print("=" * 60)
    
    # Default data path
    data_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_PATH
    
    print(f"\nData source: {data_path}")
    
    trainer = VostokTrainer()
    result = trainer.train(data_path)
    
    if result.success:
        print(f"\n✅ Model saved: {result.model_path}")
        print(f"   Samples: {result.n_samples:,}")
        print(f"   Precision: {result.avg_precision:.2%}")
        print(f"   Recall: {result.avg_recall:.2%}")
        print(f"   F1: {result.avg_f1:.2%}")
    else:
        print("\n❌ Training failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
