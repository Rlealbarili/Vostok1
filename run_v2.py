#!/usr/bin/env python3
"""
VOSTOK V2 :: Run Engine Script
==============================
Script para executar o VostokV2Engine em modo de teste.

Simula um ciclo de vida do engine:
1. Carrega dados de exemplo
2. Instancia o Engine
3. Passa o DataFrame e imprime a decisão

Uso:
    python run_v2.py                    # Usa dados dummy
    python run_v2.py data/ohlcv.csv     # Usa dados reais

"""

import asyncio
import sys
import os

# Adicionar src ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)7s | %(message)s',
    datefmt='%H:%M:%S',
)

logger = logging.getLogger("run_v2")


def load_data(path: str = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega dados M1 e H1 (ou gera dummy)."""
    if path and os.path.exists(path):
        logger.info(f"Loading data from {path}")
        df = pd.read_csv(path)
        df.columns = df.columns.str.lower()
        
        # Para H1, resample
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df_h1 = df.resample('1H').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
            }).dropna()
            df_m1 = df.reset_index()
            df_h1 = df_h1.reset_index()
        else:
            df_m1 = df
            df_h1 = None
        
        return df_m1, df_h1
    
    # Gerar dados dummy
    logger.info("Generating dummy data...")
    np.random.seed(42)
    n = 500
    
    # Simular preço de BTC
    base_price = 95000
    returns = np.random.randn(n) * 50  # Volatilidade
    prices = base_price + returns.cumsum()
    
    df_m1 = pd.DataFrame({
        'open': prices + np.random.randn(n) * 10,
        'high': prices + np.abs(np.random.randn(n) * 50),
        'low': prices - np.abs(np.random.randn(n) * 50),
        'close': prices,
        'volume': np.random.randint(100, 1000, n) * 1000,
    })
    
    # Garantir OHLC válido
    df_m1['high'] = df_m1[['open', 'high', 'close']].max(axis=1)
    df_m1['low'] = df_m1[['open', 'low', 'close']].min(axis=1)
    
    # H1 (cada 60 candles M1)
    df_h1 = df_m1.groupby(df_m1.index // 60).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }).reset_index(drop=True)
    
    return df_m1, df_h1


async def run_engine(df_m1, df_h1, news_samples):
    """Executa o engine com diferentes cenários."""
    from src.v2.engine import VostokV2Engine, Action
    
    # Criar engine
    engine = VostokV2Engine(ollama_host="localhost")
    
    print("\n" + "=" * 70)
    print("  🧪 RUNNING ENGINE TESTS")
    print("=" * 70)
    
    # Teste 1: Sem notícias
    print("\n  📊 Test 1: Market Analysis (No News)")
    print("  " + "-" * 66)
    
    decision1 = await engine.analyze_market(df_m1, df_h1)
    
    print(f"    Action:     {decision1.action.value}")
    print(f"    Direction:  {decision1.direction or 'N/A'}")
    print(f"    Confidence: {decision1.confidence:.1%}")
    print(f"    Reason:     {decision1.reason}")
    print(f"    Regime:     {decision1.regime_status} ({decision1.regime_reason})")
    print(f"    ML Signal:  {decision1.ml_signal} ({decision1.ml_confidence:.1%})")
    
    # Teste 2 e 3: Com notícias
    for i, news in enumerate(news_samples, start=2):
        print(f"\n  📰 Test {i}: With News Context")
        print("  " + "-" * 66)
        print(f"    News: \"{news}\"")
        
        decision = await engine.analyze_market(df_m1, df_h1, news_context=news)
        
        print(f"    Action:     {decision.action.value}")
        print(f"    Direction:  {decision.direction or 'N/A'}")
        print(f"    Buffett:    {decision.buffett_verdict} ({decision.buffett_permission})")
        print(f"    Reason:     {decision.reason}")
    
    # Estatísticas
    print("\n  📈 ENGINE STATISTICS")
    print("  " + "-" * 66)
    
    stats = engine.get_stats()
    for key, value in stats.items():
        print(f"    {key:<20}: {value}")
    
    return engine


def main():
    """Entry point."""
    print()
    print("=" * 70)
    print("  🚀 VOSTOK V2 ENGINE - Demo Run")
    print("=" * 70)
    
    # Carregar dados
    data_path = sys.argv[1] if len(sys.argv) > 1 else None
    df_m1, df_h1 = load_data(data_path)
    
    print(f"\n  📊 Data Loaded:")
    print(f"     M1 Candles: {len(df_m1)}")
    print(f"     H1 Candles: {len(df_h1) if df_h1 is not None else 'N/A'}")
    print(f"     Price Range: ${df_m1['close'].min():.0f} - ${df_m1['close'].max():.0f}")
    
    # Notícias de exemplo para testar Buffett
    news_samples = [
        "Bitcoin surges 5% on ETF approval rumors",
        "SEC sues major exchange for fraud - regulatory crackdown imminent",
    ]
    
    # Rodar engine
    try:
        asyncio.run(run_engine(df_m1, df_h1, news_samples))
    except KeyboardInterrupt:
        print("\n  ⚠️ Interrupted by user")
    except Exception as e:
        print(f"\n  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print()
    print("=" * 70)
    print("  ✅ Demo complete!")
    print("=" * 70)
    print()
    print("  Next steps:")
    print("    1. Train model: python train_v2.py")
    print("    2. Run again to see ACTIVE mode predictions")
    print()


if __name__ == "__main__":
    main()
