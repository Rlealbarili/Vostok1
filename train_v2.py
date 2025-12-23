#!/usr/bin/env python3
"""
VOSTOK V2 :: Training Script
============================
Script para treinar o modelo LightGBM do Vostok V2.

Uso:
    python train_v2.py                         # Usa data/training/dataset.jsonl
    python train_v2.py data/custom/ohlcv.csv   # Usa arquivo específico

Output:
    models/v2/lgbm_model.txt     # Modelo treinado
    models/v2/model_metrics.json # Métricas de validação
"""

import sys
import os

# Adicionar src ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)7s | %(message)s',
    datefmt='%H:%M:%S',
)

logger = logging.getLogger("train_v2")


def main():
    """Entry point do treinamento."""
    from src.v2.training.trainer import VostokTrainer, DEFAULT_DATA_PATH
    
    print()
    print("=" * 70)
    print("  🥋 VOSTOK V2 :: TRAINING DOJO")
    print("=" * 70)
    print()
    
    # Data path
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = DEFAULT_DATA_PATH
    
    print(f"  📂 Data Source: {data_path}")
    print(f"  💾 Model Output: models/v2/lgbm_model.txt")
    print()
    
    # Verificar se arquivo existe
    if not os.path.exists(data_path):
        print(f"  ❌ ERROR: File not found: {data_path}")
        print()
        print("  Para criar dados de treino, rode o backfill primeiro:")
        print("    docker compose run --rm backfill")
        print()
        print("  Ou converta o dataset V1 existente:")
        print("    python scripts/convert_dataset_v1_to_v2.py")
        print()
        sys.exit(1)
    
    # Criar trainer
    trainer = VostokTrainer()
    
    # Treinar
    print("  🚀 Starting training...")
    print()
    
    result = trainer.train(data_path)
    
    # Resultado
    print()
    print("=" * 70)
    
    if result.success:
        print("  ✅ TRAINING SUCCESSFUL!")
        print()
        print(f"  📊 Dataset:")
        print(f"     Samples: {result.n_samples:,}")
        print(f"     Features: {result.n_features}")
        print(f"     Wins: {result.n_positive:,} ({result.n_positive/(result.n_positive+result.n_negative):.1%})")
        print(f"     Losses: {result.n_negative:,}")
        print()
        print(f"  📈 Cross-Validation Metrics:")
        print(f"     Precision: {result.avg_precision:.2%}")
        print(f"     Recall: {result.avg_recall:.2%}")
        print(f"     F1 Score: {result.avg_f1:.2%}")
        print()
        print(f"  🏆 Top 5 Features:")
        for i, (feat, imp) in enumerate(list(result.feature_importance.items())[:5]):
            print(f"     {i+1}. {feat}: {imp:.1f}")
        print()
        print(f"  💾 Model saved: {result.model_path}")
        print(f"  ⏱️ Training time: {result.training_time_seconds:.1f}s")
    else:
        print("  ❌ TRAINING FAILED!")
        print()
        print("  Check the logs above for details.")
        sys.exit(1)
    
    print()
    print("=" * 70)
    print("  Next: The LightGBMStrategy will now use this model automatically.")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
