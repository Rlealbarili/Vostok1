"""
VOSTOK-1 :: Historical Backfill Module (Time Machine) v2.0
===========================================================
Triple Barrier Method - Labels Dinâmicos Baseados em Volatilidade (ATR)

MUDANÇA CRÍTICA:
- Antes: TP/SL fixos (0.5%/0.25%) - Ensinava o modelo a prever sorte
- Agora: Barreiras dinâmicas baseadas em ATR - Ensina o modelo a ler o mercado

Arquiteto: Petrovich | Operador: Vostok
Stack: Python 3.11 + ccxt + pandas + ta
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import ccxt
import numpy as np
import pandas as pd
import ta

# ============================================================================
# CONFIGURAÇÃO DE LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("backfill")

# ============================================================================
# CONFIGURAÇÕES
# ============================================================================
SYMBOL = os.getenv("SYMBOL", "BTC/USDT")
TIMEFRAME = os.getenv("TIMEFRAME", "1m")
DAYS_BACK = int(os.getenv("DAYS_BACK", 90))

# Paths
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
TRAINING_DIR = DATA_DIR / "training"
DATASET_FILE = TRAINING_DIR / "dataset.jsonl"

# ============================================================================
# TRIPLE BARRIER PARAMETERS (ATR-BASED)
# ============================================================================
ATR_PERIOD = 14
ATR_MULTIPLIER_TP = 2.0   # Take Profit = Close + (ATR * 2.0)
ATR_MULTIPLIER_SL = 1.0   # Stop Loss = Close - (ATR * 1.0) -> R:R 2:1
LOOKAHEAD_BARS = 45       # 45 minutos de janela

# Feature Engineering
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9


# ============================================================================
# DATA FETCHER (CCXT)
# ============================================================================
class BinanceFetcher:
    """Baixa dados históricos da Binance via CCXT."""

    def __init__(self) -> None:
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })

    def fetch_ohlcv(self, symbol: str, timeframe: str, days_back: int) -> pd.DataFrame:
        """
        Baixa velas históricas da Binance.
        Retorna DataFrame com OHLCV.
        """
        logger.info(f"📊 Baixando {days_back} dias de dados para {symbol}...")
        
        all_ohlcv = []
        since = int((datetime.now(timezone.utc) - timedelta(days=days_back)).timestamp() * 1000)
        
        batch_size = 1000
        
        while True:
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=since,
                    limit=batch_size
                )
                
                if not ohlcv:
                    break
                
                all_ohlcv.extend(ohlcv)
                since = ohlcv[-1][0] + 1
                
                logger.info(f"   → {len(all_ohlcv):,} velas baixadas...")
                
                if ohlcv[-1][0] >= datetime.now(timezone.utc).timestamp() * 1000 - 60000:
                    break
                
            except Exception as e:
                logger.warning(f"⚠️  Erro no fetch, continuando: {e}")
                break
        
        if not all_ohlcv:
            logger.error("❌ Nenhuma vela obtida")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df = df.set_index('timestamp')
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='last')]
        
        logger.info(f"✅ Total: {len(df):,} velas ({df.index.min()} a {df.index.max()})")
        return df


# ============================================================================
# FEATURE ENGINEERING (ATR-FOCUSED)
# ============================================================================
class FeatureEngineer:
    """Calcula features técnicas com foco em ATR para Triple Barrier."""

    @staticmethod
    def calculate_cvd_proxy(df: pd.DataFrame) -> pd.Series:
        """Aproxima CVD a partir de OHLCV."""
        body = df['close'] - df['open']
        range_ = df['high'] - df['low']
        range_ = range_.replace(0, np.nan).fillna(1)
        direction = body / range_
        cvd_delta = df['volume'] * direction
        cvd = cvd_delta.cumsum()
        cvd_normalized = cvd / (cvd.abs().max() + 1e-10)
        return cvd_normalized

    @staticmethod
    def calculate_entropy_proxy(df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Proxy de entropia usando desvio padrão normalizado."""
        returns = df['close'].pct_change()
        volatility = returns.rolling(window=window).std()
        entropy = (volatility - volatility.min()) / (volatility.max() - volatility.min() + 1e-10)
        return entropy.fillna(0.5)

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula todas as features necessárias incluindo ATR."""
        logger.info("🔧 Calculando features (com ATR para Triple Barrier)...")
        
        # ATR - CRITICAL FOR TRIPLE BARRIER
        atr_indicator = ta.volatility.AverageTrueRange(
            df['high'], df['low'], df['close'], window=ATR_PERIOD
        )
        df['atr'] = atr_indicator.average_true_range()
        df['volatility_atr'] = df['atr'] / df['close']  # Normalizado
        
        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=RSI_PERIOD).rsi()
        
        # MACD
        macd_indicator = ta.trend.MACD(
            df['close'], window_slow=MACD_SLOW, window_fast=MACD_FAST, window_sign=MACD_SIGNAL
        )
        df['macd'] = macd_indicator.macd()
        df['macd_hist'] = macd_indicator.macd_diff()
        
        # CVD Proxy
        df['cvd'] = self.calculate_cvd_proxy(df)
        
        # Entropy Proxy
        df['entropy'] = self.calculate_entropy_proxy(df)
        
        # Funding Rate (não disponível no histórico)
        df['funding_rate'] = 0.0
        
        logger.info(f"✅ Features calculadas: {len(df):,} linhas")
        logger.info(f"   ATR médio: {df['atr'].mean():.2f}")
        logger.info(f"   ATR min/max: {df['atr'].min():.2f} / {df['atr'].max():.2f}")
        
        return df


# ============================================================================
# TRIPLE BARRIER LABELER (DYNAMIC ATR-BASED)
# ============================================================================
class TripleBarrierLabeler:
    """
    Aplica rotulagem Triple Barrier DINÂMICA baseada em ATR.
    
    Conceito:
    - Teto: Close + (ATR * k_tp)
    - Chão: Close - (ATR * k_sl)  
    - Parede Vertical: Time limit (candles)
    
    Quem tocar primeiro define o resultado.
    """

    def __init__(
        self,
        atr_mult_tp: float = ATR_MULTIPLIER_TP,
        atr_mult_sl: float = ATR_MULTIPLIER_SL,
        lookahead_bars: int = LOOKAHEAD_BARS
    ) -> None:
        self.atr_mult_tp = atr_mult_tp
        self.atr_mult_sl = atr_mult_sl
        self.lookahead = lookahead_bars
        
        logger.info(f"🎯 Triple Barrier configurado:")
        logger.info(f"   - Take Profit: Close + (ATR × {atr_mult_tp})")
        logger.info(f"   - Stop Loss:   Close - (ATR × {atr_mult_sl})")
        logger.info(f"   - Time Limit:  {lookahead_bars} candles")
        logger.info(f"   - Risk:Reward: 1:{atr_mult_tp/atr_mult_sl:.1f}")

    def apply_triple_barrier(self, df: pd.DataFrame, idx: int) -> tuple[int | None, float, float]:
        """
        Aplica Triple Barrier para um único candle.
        
        Returns:
            (label, tp_price, sl_price)
            label: 1=Win (TP hit), 0=Loss (SL hit or Timeout), None=Skip
        """
        if idx + self.lookahead >= len(df):
            return None, 0, 0
        
        # Preço de entrada
        entry_price = df['close'].iloc[idx]
        atr_value = df['atr'].iloc[idx]
        
        # Skip se ATR é NaN ou zero
        if pd.isna(atr_value) or atr_value <= 0:
            return None, 0, 0
        
        # Barreiras DINÂMICAS baseadas em ATR
        tp_price = entry_price + (atr_value * self.atr_mult_tp)
        sl_price = entry_price - (atr_value * self.atr_mult_sl)
        
        # Olhar para o futuro (próximos N candles)
        for future_idx in range(idx + 1, idx + 1 + self.lookahead):
            if future_idx >= len(df):
                break
            
            high = df['high'].iloc[future_idx]
            low = df['low'].iloc[future_idx]
            
            # Verificar qual barreira foi tocada PRIMEIRO
            # Usamos high/low para ser mais realista (intracandle)
            
            hit_tp = high >= tp_price
            hit_sl = low <= sl_price
            
            if hit_tp and hit_sl:
                # Ambos tocados no mesmo candle - usar close para decidir
                close = df['close'].iloc[future_idx]
                if close >= entry_price:
                    return 1, tp_price, sl_price  # Win
                else:
                    return 0, tp_price, sl_price  # Loss
            elif hit_tp:
                return 1, tp_price, sl_price  # Win - TP hit first
            elif hit_sl:
                return 0, tp_price, sl_price  # Loss - SL hit first
        
        # Timeout - nenhuma barreira tocada
        # Para HFT, ficar preso é ruim -> Label 0
        return 0, tp_price, sl_price

    def label_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica Triple Barrier a todo o DataFrame."""
        logger.info(f"🏷️  Aplicando Triple Barrier Dinâmico (ATR-based)...")
        
        labels = []
        tp_prices = []
        sl_prices = []
        
        for i in range(len(df)):
            label, tp, sl = self.apply_triple_barrier(df, i)
            labels.append(label)
            tp_prices.append(tp)
            sl_prices.append(sl)
            
            if (i + 1) % 10000 == 0:
                logger.info(f"   → {i+1:,} / {len(df):,} rotulados...")
        
        df['outcome_label'] = labels
        df['barrier_tp'] = tp_prices
        df['barrier_sl'] = sl_prices
        
        # Estatísticas
        valid_labels = df['outcome_label'].notna()
        labeled = valid_labels.sum()
        wins = (df.loc[valid_labels, 'outcome_label'] == 1).sum()
        losses = (df.loc[valid_labels, 'outcome_label'] == 0).sum()
        
        win_rate = 100 * wins / labeled if labeled > 0 else 0
        
        logger.info(f"✅ Rotulagem Triple Barrier completa:")
        logger.info(f"   → Total rotulado: {labeled:,}")
        logger.info(f"   → Wins (TP hit): {wins:,} ({win_rate:.1f}%)")
        logger.info(f"   → Losses (SL/Timeout): {losses:,} ({100-win_rate:.1f}%)")
        
        # Análise de ATR
        avg_atr = df.loc[valid_labels, 'atr'].mean()
        avg_tp_dist = (df.loc[valid_labels, 'barrier_tp'] - df.loc[valid_labels, 'close']).mean()
        avg_sl_dist = (df.loc[valid_labels, 'close'] - df.loc[valid_labels, 'barrier_sl']).mean()
        
        logger.info(f"   → ATR médio: ${avg_atr:.2f}")
        logger.info(f"   → Distância TP média: ${avg_tp_dist:.2f}")
        logger.info(f"   → Distância SL média: ${avg_sl_dist:.2f}")
        
        return df


# ============================================================================
# EXPORTER (OVERWRITE MODE)
# ============================================================================
class DatasetExporter:
    """Exporta para formato dataset.jsonl - SOBRESCREVE para limpar labels antigos."""

    @staticmethod
    def export(df: pd.DataFrame, output_path: Path) -> int:
        """
        Exporta DataFrame para JSONL.
        Mode: 'w' = OVERWRITE (labels antigos estão "sujos")
        """
        logger.info(f"💾 Exportando para {output_path} (OVERWRITE mode)...")
        
        df_valid = df[df['outcome_label'].notna()].copy()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        count = 0
        with open(output_path, 'w', encoding='utf-8') as f:  # OVERWRITE
            for idx, row in df_valid.iterrows():
                record = {
                    "timestamp": idx.isoformat(),
                    "timestamp_utc": int(idx.timestamp() * 1000),
                    "action": "LONG",
                    "entry_price": float(row['close']),
                    "barrier_tp": float(row['barrier_tp']),
                    "barrier_sl": float(row['barrier_sl']),
                    "features": {
                        "rsi": float(row['rsi']) if pd.notna(row['rsi']) else 50.0,
                        "cvd": float(row['cvd']) if pd.notna(row['cvd']) else 0.0,
                        "entropy": float(row['entropy']) if pd.notna(row['entropy']) else 0.5,
                        "volatility_atr": float(row['volatility_atr']) if pd.notna(row['volatility_atr']) else 0.0,
                        "atr": float(row['atr']) if pd.notna(row['atr']) else 0.0,
                        "funding_rate": float(row['funding_rate']),
                        "macd": float(row['macd']) if pd.notna(row['macd']) else 0.0,
                        "macd_hist": float(row['macd_hist']) if pd.notna(row['macd_hist']) else 0.0,
                    },
                    "outcome_label": int(row['outcome_label']),
                    "source": "backfill_triple_barrier_v2",
                    "labeling_method": "atr_dynamic",
                    "atr_mult_tp": ATR_MULTIPLIER_TP,
                    "atr_mult_sl": ATR_MULTIPLIER_SL,
                    "lookahead_bars": LOOKAHEAD_BARS,
                }
                f.write(json.dumps(record) + '\n')
                count += 1
        
        logger.info(f"✅ Exportados {count:,} registros para {output_path}")
        return count


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def run_backfill() -> None:
    """Executa o pipeline completo de backfill com Triple Barrier."""
    logger.info("")
    logger.info("╔══════════════════════════════════════════════════════════════╗")
    logger.info("║   VOSTOK-1 :: HISTORICAL BACKFILL v2.0 (Triple Barrier)     ║")
    logger.info("║   Dynamic ATR-Based Labeling for Precision Trading          ║")
    logger.info("╚══════════════════════════════════════════════════════════════╝")
    logger.info("")
    logger.info(f"Symbol: {SYMBOL}")
    logger.info(f"Timeframe: {TIMEFRAME}")
    logger.info(f"Days Back: {DAYS_BACK}")
    logger.info(f"Output: {DATASET_FILE}")
    logger.info("")
    logger.info("Triple Barrier Parameters:")
    logger.info(f"  TP Multiplier: {ATR_MULTIPLIER_TP}x ATR")
    logger.info(f"  SL Multiplier: {ATR_MULTIPLIER_SL}x ATR")
    logger.info(f"  Time Limit: {LOOKAHEAD_BARS} candles")
    logger.info("")
    
    # Step 1: Fetch Data
    logger.info("=" * 60)
    logger.info("STEP 1: FETCHING HISTORICAL DATA")
    logger.info("=" * 60)
    
    fetcher = BinanceFetcher()
    df = fetcher.fetch_ohlcv(SYMBOL, TIMEFRAME, DAYS_BACK)
    
    if df.empty:
        logger.error("❌ Sem dados para processar. Abortando.")
        return
    
    # Step 2: Feature Engineering (includes ATR)
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 2: FEATURE ENGINEERING (ATR-FOCUSED)")
    logger.info("=" * 60)
    
    engineer = FeatureEngineer()
    df = engineer.engineer_features(df)
    
    # Step 3: Triple Barrier Labeling
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 3: TRIPLE BARRIER LABELING (DYNAMIC)")
    logger.info("=" * 60)
    
    labeler = TripleBarrierLabeler()
    df = labeler.label_dataframe(df)
    
    # Step 4: Export (OVERWRITE to clean old labels)
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 4: EXPORTING DATASET (OVERWRITE)")
    logger.info("=" * 60)
    
    exporter = DatasetExporter()
    count = exporter.export(df, DATASET_FILE)
    
    # Summary
    logger.info("")
    logger.info("╔══════════════════════════════════════════════════════════════╗")
    logger.info("║           TRIPLE BARRIER BACKFILL COMPLETE!                 ║")
    logger.info("╚══════════════════════════════════════════════════════════════╝")
    logger.info(f"📊 Records exported: {count:,}")
    logger.info(f"📁 Dataset file: {DATASET_FILE}")
    logger.info("")
    logger.info("🎯 Labels agora são DINÂMICOS baseados em volatilidade!")
    logger.info("   - Alta volatilidade = Barreiras largas")
    logger.info("   - Baixa volatilidade = Barreiras curtas")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Run trainer: docker compose run --rm trainer")
    logger.info("  2. Check metrics (expect better precision!)")
    logger.info("")


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    run_backfill()
