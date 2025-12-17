"""
VOSTOK-1 :: Trainer Module (RandomForest Meta-Labeling)
========================================================
Treina modelo de classificação para filtrar sinais do Sniper.
Lê dataset rotulado pelo Decision Engine e salva modelo sklearn.

Arquiteto: Petrovich | Operador: Vostok
"""

import json
import logging
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "module": "%(name)s", "message": "%(message)s"}',
    datefmt='%Y-%m-%dT%H:%M:%S%z',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("trainer")

DATA_DIR = Path(os.getenv("DATA_DIR", "/app/data"))
TRAINING_DIR = DATA_DIR / "training"
MODELS_DIR = Path(os.getenv("MODELS_DIR", "/app/models"))
DATASET_FILE = TRAINING_DIR / "dataset.jsonl"
MODEL_FILE = MODELS_DIR / "sniper_v1.pkl"

MIN_SAMPLES = 50  # Mínimo de amostras para treinar


# ============================================================================
# DATA LOADER
# ============================================================================
def load_dataset() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Carrega dataset do arquivo JSONL.
    
    Returns:
        X: Features array (n_samples, n_features)
        y: Labels array (n_samples,)
        feature_names: Lista de nomes das features
    """
    if not DATASET_FILE.exists():
        logger.error(f"Dataset não encontrado: {DATASET_FILE}")
        return np.array([]), np.array([]), []
    
    records = []
    with open(DATASET_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                if record.get('outcome_label') is not None:
                    records.append(record)
            except json.JSONDecodeError:
                continue
    
    if not records:
        return np.array([]), np.array([]), []
    
    # Extrair features e labels
    feature_names = ['rsi', 'cvd', 'entropy', 'volatility_atr', 
                     'volatility_parkinson', 'funding_rate', 'macd', 'macd_hist']
    
    X_list = []
    y_list = []
    
    for record in records:
        features = record.get('features', {})
        
        # Criar vetor de features
        x = [
            features.get('rsi', 50.0),
            features.get('cvd', 0.0),
            features.get('entropy', 0.0),
            features.get('volatility_atr', 0.0),
            features.get('volatility_parkinson', 0.0),
            features.get('funding_rate', 0.0),
            features.get('macd', 0.0),
            features.get('macd_hist', 0.0),
        ]
        
        X_list.append(x)
        y_list.append(record['outcome_label'])
    
    return np.array(X_list), np.array(y_list), feature_names


# ============================================================================
# TRAINER
# ============================================================================
def train_model(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> dict[str, Any]:
    """
    Treina RandomForest e retorna métricas.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    
    # Split treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )
    
    logger.info(f"Split: train={len(X_train)}, test={len(X_test)}")
    
    # Treinar RandomForest
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Predições
    y_pred = model.predict(X_test)
    
    # Métricas
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'win_rate_train': np.mean(y_train),
        'win_rate_test': np.mean(y_test),
    }
    
    # Feature importance
    importances = dict(zip(feature_names, model.feature_importances_))
    
    # Salvar modelo
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    model_data = {
        'model': model,
        'feature_names': feature_names,
        'metrics': metrics,
        'importances': importances,
        'trained_at': datetime.now().isoformat(),
    }
    
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model_data, f)
    
    return metrics, importances


# ============================================================================
# MAIN
# ============================================================================
def main() -> None:
    logger.info("=" * 60)
    logger.info("VOSTOK-1 :: Trainer Module (RandomForest)")
    logger.info(f"Dataset: {DATASET_FILE}")
    logger.info(f"Model Output: {MODEL_FILE}")
    logger.info("=" * 60)
    
    # Carregar dataset
    X, y, feature_names = load_dataset()
    
    n_samples = len(X)
    logger.info(f"Dataset carregado: {n_samples} amostras")
    
    # Validar tamanho
    if n_samples < MIN_SAMPLES:
        logger.warning(
            f"⚠️  Dataset incipiente ({n_samples} linhas). "
            f"Aguardando mais trades do Sniper. Mínimo: {MIN_SAMPLES}"
        )
        print(f"\n⚠️  WARN: Dataset incipiente ({n_samples} linhas).")
        print(f"    Aguardando mais trades do Sniper.")
        print(f"    Mínimo necessário: {MIN_SAMPLES} amostras.\n")
        sys.exit(0)
    
    # Verificar distribuição de classes
    n_wins = np.sum(y == 1)
    n_losses = np.sum(y == 0)
    win_rate = n_wins / n_samples * 100
    
    logger.info(f"Distribuição: {n_wins} wins, {n_losses} losses ({win_rate:.1f}% win rate)")
    
    if n_wins == 0 or n_losses == 0:
        logger.warning("⚠️  Dataset desbalanceado (apenas uma classe). Aguarde mais dados.")
        print("\n⚠️  WARN: Dataset tem apenas uma classe. Aguarde mais trades.\n")
        sys.exit(0)
    
    # Treinar modelo
    logger.info("Iniciando treinamento...")
    
    try:
        metrics, importances = train_model(X, y, feature_names)
        
        # Resultados
        print("\n" + "=" * 60)
        print("✅ TREINAMENTO CONCLUÍDO")
        print("=" * 60)
        print(f"\n📊 MÉTRICAS:")
        print(f"   Accuracy:  {metrics['accuracy']:.2%}")
        print(f"   Precision: {metrics['precision']:.2%}")
        print(f"   Recall:    {metrics['recall']:.2%}")
        print(f"   F1 Score:  {metrics['f1']:.2%}")
        
        print(f"\n📈 FEATURE IMPORTANCE:")
        sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        for name, imp in sorted_imp[:5]:
            print(f"   {name}: {imp:.3f}")
        
        print(f"\n💾 Modelo salvo em: {MODEL_FILE}")
        print("=" * 60 + "\n")
        
        logger.info(f"Modelo salvo: {MODEL_FILE}")
        logger.info(f"Métricas: Acc={metrics['accuracy']:.2%}, Prec={metrics['precision']:.2%}, Rec={metrics['recall']:.2%}")
        
    except ImportError as e:
        logger.error(f"Erro de import (sklearn não instalado?): {e}")
        print(f"\n❌ Erro: {e}")
        print("   Instale sklearn: pip install scikit-learn\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
