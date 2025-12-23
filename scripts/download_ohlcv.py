#!/usr/bin/env python3
"""
VOSTOK V2 :: Download OHLCV Data
================================
Baixa dados OHLCV limpos da Binance para treinamento do modelo V2.

Uso:
    python scripts/download_ohlcv.py              # 365 dias default
    python scripts/download_ohlcv.py --days 90    # 90 dias
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import time

try:
    import requests
    import pandas as pd
except ImportError:
    print("Installing required packages...")
    os.system("pip install requests pandas -q")
    import requests
    import pandas as pd


# ============================================================================
# CONSTANTS
# ============================================================================

BINANCE_API_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "1m"
LIMIT = 1000  # Max per request
OUTPUT_DIR = "data/training"
OUTPUT_FILE = "ohlcv_btc_365d.csv"


def download_klines(symbol: str, interval: str, start_time: int, end_time: int) -> list:
    """Baixa klines de um período específico."""
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time,
        "endTime": end_time,
        "limit": LIMIT,
    }
    
    response = requests.get(BINANCE_API_URL, params=params, timeout=30)
    response.raise_for_status()
    
    return response.json()


def download_ohlcv(days: int = 365, output_path: str = None) -> pd.DataFrame:
    """
    Baixa dados OHLCV da Binance.
    
    Args:
        days: Número de dias para baixar
        output_path: Caminho para salvar CSV
        
    Returns:
        DataFrame com OHLCV
    """
    print(f"\n{'='*60}")
    print(f"📥 VOSTOK V2 :: OHLCV Data Downloader")
    print(f"{'='*60}")
    print(f"Symbol: {SYMBOL}")
    print(f"Interval: {INTERVAL}")
    print(f"Days: {days}")
    print(f"{'='*60}\n")
    
    # Calcular timestamps
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    
    # Total de candles esperado
    total_candles = days * 24 * 60  # 1 candle por minuto
    print(f"Expected candles: ~{total_candles:,}")
    
    all_klines = []
    current_start = start_time
    
    print("\nDownloading...", end="", flush=True)
    
    while current_start < end_time:
        try:
            klines = download_klines(SYMBOL, INTERVAL, current_start, end_time)
            
            if not klines:
                break
            
            all_klines.extend(klines)
            
            # Próximo batch
            current_start = klines[-1][0] + 1  # timestamp do último + 1ms
            
            # Progress
            progress = len(all_klines) / total_candles * 100
            print(f"\rDownloading... {len(all_klines):,} candles ({progress:.1f}%)", end="", flush=True)
            
            # Rate limit
            time.sleep(0.1)
            
        except Exception as e:
            print(f"\nError: {e}")
            time.sleep(1)
    
    print(f"\n\nDownloaded {len(all_klines):,} candles")
    
    # Converter para DataFrame
    df = pd.DataFrame(all_klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    # Limpar e converter tipos
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remover duplicatas
    df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
    
    # Stats
    print(f"\n📊 Dataset Statistics:")
    print(f"   Rows: {len(df):,}")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    print(f"   Memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    
    # Salvar
    if output_path is None:
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        output_path = Path(OUTPUT_DIR) / OUTPUT_FILE
    
    df.to_csv(output_path, index=False)
    print(f"\n💾 Saved to: {output_path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Download OHLCV data from Binance")
    parser.add_argument("--days", type=int, default=365, help="Number of days to download")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    args = parser.parse_args()
    
    download_ohlcv(days=args.days, output_path=args.output)
    
    print(f"\n✅ Done! Now run:")
    print(f"   python train_v2.py data/training/ohlcv_btc_365d.csv")
    print()


if __name__ == "__main__":
    main()
