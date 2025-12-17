"""
VOSTOK-1 :: Model Trainer Pipeline
===================================
Pipeline de treinamento batch para Meta-Labeling com Random Forest.
Processa histórico de trades do Sniper e gera modelo preditivo.

Arquiteto: Petrovich | Operador: Vostok
Stack: Python 3.11 + scikit-learn + pandas + joblib
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

# ============================================================================
# CONFIGURAÇÃO DE LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("trainer")

# ============================================================================
# CONFIGURAÇÕES
# ============================================================================
DATA_DIR = Path(os.getenv("DATA_DIR", "/app/data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", "/app/models"))
TRAINING_DIR = DATA_DIR / "training"
DATASET_FILE = TRAINING_DIR / "dataset.jsonl"
MODEL_FILE = MODELS_DIR / "sniper_v1.pkl"
METRICS_FILE = MODELS_DIR / "model_metrics.json"

# Requisitos mínimos
MIN_SAMPLES = 50
MIN_PRECISION = 0.35  # Threshold de qualidade (não aceita spam)

# Threshold de probabilidade para previsão
PROBA_THRESHOLD = 0.70  # Só considera sinal se confiança > 70% (mais seletivo)

# Features a extrair
FEATURE_NAMES = [
    "rsi",
    "cvd",
    "entropy",
    "volatility_atr",
    "funding_rate",
]


# ============================================================================
# STEP 1: INGESTÃO DE DADOS
# ============================================================================
def load_dataset() -> pd.DataFrame | None:
    """
    Carrega o dataset JSONL e retorna como DataFrame.
    Retorna None se dados insuficientes.
    """
    logger.info("=" * 60)
    logger.info("STEP 1: INGESTÃO DE DADOS")
    logger.info("=" * 60)
    
    if not DATASET_FILE.exists():
        logger.error(f"Dataset não encontrado: {DATASET_FILE}")
        return None
    
    # Ler arquivo JSONL
    records = []
    with open(DATASET_FILE, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                logger.warning(f"Linha {line_num} ignorada (JSON inválido): {e}")
    
    n_samples = len(records)
    logger.info(f"📁 Amostras encontradas: {n_samples}")
    
    # Verificação de mínimo
    if n_samples < MIN_SAMPLES:
        logger.warning(
            f"⚠️  WARN: Dataset incipiente ({n_samples} linhas). "
            f"Aguardando mais trades do Sniper. Mínimo necessário: {MIN_SAMPLES}"
        )
        return None
    
    df = pd.DataFrame(records)
    logger.info(f"✅ Dataset carregado: {len(df)} registros")
    
    return df


# ============================================================================
# STEP 2: PREPARAÇÃO DE FEATURES
# ============================================================================
def prepare_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    """
    Prepara Features (X) e Target (y) para treinamento.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 2: PREPARAÇÃO DE FEATURES")
    logger.info("=" * 60)
    
    # Verificar coluna target
    if 'outcome_label' not in df.columns:
        logger.error("Coluna 'outcome_label' não encontrada no dataset")
        return None
    
    # Extrair features do objeto aninhado
    feature_data = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        features = row.get('features', {})
        if not isinstance(features, dict):
            continue
        
        feature_row = {}
        valid = True
        
        for feat_name in FEATURE_NAMES:
            value = features.get(feat_name)
            if value is None:
                # Tentar nome alternativo
                if feat_name == "funding_rate":
                    value = features.get("funding", 0)
                else:
                    value = 0
            
            try:
                feat_value = float(value)
                if not np.isfinite(feat_value):
                    feat_value = 0.0
                feature_row[feat_name] = feat_value
            except (ValueError, TypeError):
                feature_row[feat_name] = 0.0
        
        if valid:
            feature_data.append(feature_row)
            valid_indices.append(idx)
    
    if not feature_data:
        logger.error("Nenhuma feature válida extraída")
        return None
    
    # Criar DataFrame de features
    X_df = pd.DataFrame(feature_data)
    y_series = df.loc[valid_indices, 'outcome_label']
    
    # Limpeza: remover NaN e infinitos de AMBOS X e y
    y_values = pd.to_numeric(y_series, errors='coerce')
    valid_mask = (
        ~X_df.isna().any(axis=1) & 
        ~np.isinf(X_df.values).any(axis=1) &
        y_values.notna().values
    )
    
    X_clean = X_df[valid_mask].values
    y_clean = y_values[valid_mask].values.astype(int)
    
    logger.info(f"📊 Features extraídas: {FEATURE_NAMES}")
    logger.info(f"📊 Amostras válidas após limpeza: {len(X_clean)}")
    logger.info(f"📊 Distribuição target:")
    logger.info(f"   - Class 0 (Loss): {sum(y_clean == 0)}")
    logger.info(f"   - Class 1 (Win):  {sum(y_clean == 1)}")
    
    if len(X_clean) < MIN_SAMPLES:
        logger.warning(f"⚠️  Amostras insuficientes após limpeza: {len(X_clean)}")
        return None
    
    return X_clean, y_clean, FEATURE_NAMES


# ============================================================================
# STEP 3: TREINAMENTO
# ============================================================================
def train_model(
    X: np.ndarray, 
    y: np.ndarray, 
    feature_names: list[str]
) -> tuple[RandomForestClassifier, dict[str, Any]] | None:
    """
    Treina RandomForestClassifier com os dados preparados.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 3: TREINAMENTO (O CÉREBRO)")
    logger.info("=" * 60)
    
    # Train/Test Split (80/20, preservar ordem temporal)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        shuffle=False,  # IMPORTANTE: preservar ordem temporal
        random_state=42
    )
    
    logger.info(f"📊 Split temporal (shuffle=False):")
    logger.info(f"   - Train: {len(X_train)} amostras")
    logger.info(f"   - Test:  {len(X_test)} amostras")
    
    # Instanciar modelo - OTIMIZADO PARA IMBALANCE
    model = RandomForestClassifier(
        n_estimators=200,              # Mais árvores para estabilidade
        max_depth=10,                  # Limita profundidade (evitar decorar ruído)
        min_samples_leaf=50,           # Exige evidência forte
        class_weight='balanced_subsample',  # Penaliza erros na classe minoritária
        random_state=42,
        n_jobs=-1,
    )
    
    logger.info("🧠 Treinando RandomForestClassifier (Otimizado para Imbalance)...")
    logger.info(f"   - n_estimators: 200")
    logger.info(f"   - max_depth: 10")
    logger.info(f"   - min_samples_leaf: 50")
    logger.info(f"   - class_weight: balanced_subsample")
    
    # Treinar
    model.fit(X_train, y_train)
    
    logger.info("✅ Modelo treinado com sucesso!")
    
    # Preparar dados para validação
    validation_data = {
        'model': model,
        'X_test': X_test,
        'y_test': y_test,
        'X_train': X_train,
        'y_train': y_train,
        'feature_names': feature_names,
    }
    
    return model, validation_data


# ============================================================================
# STEP 4: VALIDAÇÃO E MÉTRICAS
# ============================================================================
def validate_model(model: RandomForestClassifier, validation_data: dict) -> dict[str, Any]:
    """
    Valida o modelo e calcula métricas de performance.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 4: VALIDAÇÃO E MÉTRICAS")
    logger.info("=" * 60)
    
    X_test = validation_data['X_test']
    y_test = validation_data['y_test']
    feature_names = validation_data['feature_names']
    
    # Previsões usando THRESHOLD DE PROBABILIDADE (mais seletivo)
    if len(model.classes_) > 1:
        y_proba = model.predict_proba(X_test)[:, 1]
        # Só considera sinal se confiança > PROBA_THRESHOLD
        y_pred = (y_proba >= PROBA_THRESHOLD).astype(int)
        logger.info(f"🎚️  Usando threshold de probabilidade: {PROBA_THRESHOLD}")
    else:
        y_pred = model.predict(X_test)
        y_proba = None
    
    # Métricas principais
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)
    
    logger.info("📈 MÉTRICAS DE PERFORMANCE:")
    logger.info(f"   🎯 Precision: {precision:.4f} {'✅' if precision > MIN_PRECISION else '⚠️'}")
    logger.info(f"   📊 Recall:    {recall:.4f}")
    logger.info(f"   📊 F1-Score:  {f1:.4f}")
    logger.info(f"   📊 Accuracy:  {accuracy:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    logger.info("")
    logger.info("📊 CONFUSION MATRIX:")
    logger.info(f"   [[TN={cm[0][0]:3d}  FP={cm[0][1]:3d}]")
    logger.info(f"    [FN={cm[1][0]:3d}  TP={cm[1][1]:3d}]]")
    
    # Feature Importance
    importances = model.feature_importances_
    importance_pairs = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True
    )
    
    logger.info("")
    logger.info("🔍 FEATURE IMPORTANCE (Top Contributors):")
    for feat_name, importance in importance_pairs:
        bar = "█" * int(importance * 20)
        logger.info(f"   {feat_name:20s} {importance:.4f} {bar}")
    
    # Preparar métricas para exportação
    metrics = {
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
        'accuracy': round(accuracy, 4),
        'train_samples': len(validation_data['X_train']),
        'test_samples': len(X_test),
        'feature_importance': {name: round(imp, 4) for name, imp in importance_pairs},
        'confusion_matrix': cm.tolist(),
        'trained_at': datetime.now().isoformat(),
        'model_version': 'sniper_v1',
    }
    
    return metrics


# ============================================================================
# STEP 5: EXPORTAÇÃO
# ============================================================================
def export_model(
    model: RandomForestClassifier, 
    metrics: dict[str, Any],
    feature_names: list[str]
) -> bool:
    """
    Exporta modelo e métricas se precision > 0.5.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 5: EXPORTAÇÃO")
    logger.info("=" * 60)
    
    precision = metrics['precision']
    
    if precision < MIN_PRECISION:
        logger.warning(
            f"⚠️  Precision ({precision:.4f}) abaixo do mínimo ({MIN_PRECISION}). "
            f"Modelo NÃO será salvo. Mais dados necessários."
        )
        return False
    
    # Garantir diretório existe
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Salvar modelo com metadados
    model_data = {
        'model': model,
        'feature_names': feature_names,
        'metrics': metrics,
        'version': 'sniper_v1',
    }
    
    joblib.dump(model_data, MODEL_FILE)
    logger.info(f"✅ Modelo salvo: {MODEL_FILE}")
    
    # Salvar métricas em JSON
    with open(METRICS_FILE, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"✅ Métricas salvas: {METRICS_FILE}")
    
    return True


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main() -> int:
    """Pipeline principal de treinamento."""
    logger.info("")
    logger.info("╔══════════════════════════════════════════════════════════════╗")
    logger.info("║   VOSTOK-1 :: MODEL TRAINER PIPELINE                        ║")
    logger.info("║   Random Forest Meta-Labeling for Sniper Protocol           ║")
    logger.info("╚══════════════════════════════════════════════════════════════╝")
    logger.info("")
    
    # Step 1: Ingestão
    df = load_dataset()
    if df is None:
        logger.info("")
        logger.info("Pipeline encerrado (dados insuficientes)")
        return 0  # Exit gracefully
    
    # Step 2: Preparação
    result = prepare_features(df)
    if result is None:
        logger.error("Falha na preparação de features")
        return 1
    
    X, y, feature_names = result
    
    # Step 3: Treinamento
    train_result = train_model(X, y, feature_names)
    if train_result is None:
        logger.error("Falha no treinamento")
        return 1
    
    model, validation_data = train_result
    
    # Step 4: Validação
    metrics = validate_model(model, validation_data)
    
    # Step 5: Exportação
    success = export_model(model, metrics, feature_names)
    
    logger.info("")
    logger.info("=" * 60)
    if success:
        logger.info("🎯 PIPELINE CONCLUÍDO COM SUCESSO!")
        logger.info(f"   Modelo: {MODEL_FILE}")
        logger.info(f"   Precision: {metrics['precision']:.4f}")
    else:
        logger.info("⚠️  PIPELINE CONCLUÍDO (modelo não salvo)")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
