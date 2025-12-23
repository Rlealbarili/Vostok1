#!/usr/bin/env python3
"""
VOSTOK :: Dataset Converter V1 → V2
===================================
Converte o dataset V1 (features pré-calculadas) para formato V2 (pseudo-OHLCV).

O dataset V1 contém:
- features: {rsi, cvd, entropy, volatility_atr, atr, funding_rate, macd, macd_hist}
- outcome_label: 0 ou 1

Para V2, precisamos criar pseudo-OHLCV a partir das features existentes.
Como não temos OHLCV real, vamos usar as features diretamente no training.
"""

import json
import sys
import pandas as pd
import numpy as np
from pathlib import Path


def convert_v1_to_v2(input_path: str, output_path: str):
    """Converte dataset V1 para formato V2."""
    print(f"Loading V1 dataset from {input_path}...")
    
    # Carregar JSONL
    records = []
    with open(input_path, 'r') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                records.append(record)
            except:
                continue
    
    print(f"Loaded {len(records):,} records")
    
    # Extrair features e labels
    data = []
    for r in records:
        features = r.get('features', {})
        label = r.get('outcome_label', 0)
        entry_price = r.get('entry_price', 0)
        
        row = {
            # Pseudo-OHLCV
            'open': entry_price * 0.9999,
            'high': entry_price * 1.001,
            'low': entry_price * 0.999,
            'close': entry_price,
            'volume': 1000,  # Placeholder
            
            # Features V1 (já calculadas)
            'rsi': features.get('rsi', 50),
            'cvd': features.get('cvd', 0),
            'entropy': features.get('entropy', 0.5),
            'atr': features.get('atr', 100),
            'volatility_atr': features.get('volatility_atr', 0.1),
            'macd': features.get('macd', 0),
            'macd_hist': features.get('macd_hist', 0),
            
            # Label
            'label': label,
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Stats
    print(f"\nDataset stats:")
    print(f"  Total samples: {len(df):,}")
    print(f"  Label 1 (WIN): {(df['label'] == 1).sum():,} ({(df['label'] == 1).mean():.1%})")
    print(f"  Label 0 (LOSS): {(df['label'] == 0).sum():,}")
    print(f"  Price range: ${df['close'].min():.0f} - ${df['close'].max():.0f}")
    
    # Salvar como CSV
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    
    return df


if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv) > 1 else "data/training/dataset.jsonl"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "data/training/dataset_v2.csv"
    
    convert_v1_to_v2(input_path, output_path)
