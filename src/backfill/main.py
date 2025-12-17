"""
VOSTOK-1 :: Historical Backfill Module (Time Machine)
======================================================
Baixa dados históricos da Binance e gera dataset de treinamento.
Calcula features idênticas ao Quant e aplica rotulagem Triple Barrier.

Arquiteto: Petrovich | Operador: Vostok
Stack: Python 3.11 + ccxt + pandas + pandas_ta
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
import pandas_ta as ta

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

# Labeling Parameters (Triple Barrier)
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", 0.5))   # 0.5%
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", 0.25))      # 0.25%
LOOKAHEAD_BARS = int(os.getenv("LOOKAHEAD_BARS", 30))        # 30 min

# Feature Engineering
RSI_PERIOD = 14
ATR_PERIOD = 14
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
        
        # Binance limit = 1000 candles per request
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
                
                # Atualizar since para próximo batch
                since = ohlcv[-1][0] + 1
                
                logger.info(f"   → {len(all_ohlcv):,} velas baixadas...")
                
                # Verificar se chegamos ao presente
                if ohlcv[-1][0] >= datetime.now(timezone.utc).timestamp() * 1000 - 60000:
                    break
                
            except Exception as e:
                logger.warning(f"⚠️  Erro no fetch, continuando: {e}")
                break
        
        if not all_ohlcv:
            logger.error("❌ Nenhuma vela obtida")
            return pd.DataFrame()
        
        # Criar DataFrame
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df = df.set_index('timestamp')
        df = df.sort_index()
        
        # Remover duplicatas
        df = df[~df.index.duplicated(keep='last')]
        
        logger.info(f"✅ Total: {len(df):,} velas ({df.index.min()} a {df.index.max()})")
        return df


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
class FeatureEngineer:
    """Calcula features técnicas idênticas ao Quant."""

    @staticmethod
    def calculate_cvd_proxy(df: pd.DataFrame) -> pd.Series:
        """
        Aproxima CVD a partir de OHLCV.
        Lógica: Velas verdes = compra, vermelhas = venda.
        Magnitude proporcional ao corpo da vela.
        """
        body = df['close'] - df['open']
        range_ = df['high'] - df['low']
        
        # Evitar divisão por zero
        range_ = range_.replace(0, np.nan).fillna(1)
        
        # CVD = volume * direção normalizada
        direction = body / range_
        cvd_delta = df['volume'] * direction
        
        # CVD cumulativo (normalizado)
        cvd = cvd_delta.cumsum()
        
        # Normalizar para escala -1 a +1
        cvd_normalized = cvd / (cvd.abs().max() + 1e-10)
        
        return cvd_normalized

    @staticmethod
    def calculate_entropy_proxy(df: pd.DataFrame, window: int = 14) -> pd.Series:
        """
        Proxy de entropia usando desvio padrão normalizado.
        Alta volatilidade = Alto ruído = Alta "entropia".
        """
        returns = df['close'].pct_change()
        volatility = returns.rolling(window=window).std()
        
        # Normalizar para 0-1
        entropy = (volatility - volatility.min()) / (volatility.max() - volatility.min() + 1e-10)
        
        return entropy.fillna(0.5)

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula todas as features necessárias."""
        logger.info("🔧 Calculando features...")
        
        # RSI
        df['rsi'] = ta.rsi(df['close'], length=RSI_PERIOD)
        
        # ATR (Volatility)
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=ATR_PERIOD)
        df['volatility_atr'] = df['atr'] / df['close']  # Normalizar pelo preço
        
        # MACD
        macd = ta.macd(df['close'], fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
        if macd is not None:
            df['macd'] = macd.iloc[:, 0]  # MACD line
            df['macd_hist'] = macd.iloc[:, 2]  # Histogram
        else:
            df['macd'] = 0
            df['macd_hist'] = 0
        
        # CVD Proxy
        df['cvd'] = self.calculate_cvd_proxy(df)
        
        # Entropy Proxy
        df['entropy'] = self.calculate_entropy_proxy(df)
        
        # Funding Rate (não disponível no histórico, usar 0)
        df['funding_rate'] = 0.0
        
        logger.info(f"✅ Features calculadas: {len(df):,} linhas")
        return df


# ============================================================================
# LABELER (Triple Barrier)
# ============================================================================
class TripleBarrierLabeler:
    """Aplica rotulagem Triple Barrier."""

    def __init__(
        self,
        take_profit_pct: float = TAKE_PROFIT_PCT,
        stop_loss_pct: float = STOP_LOSS_PCT,
        lookahead_bars: int = LOOKAHEAD_BARS
    ) -> None:
        self.take_profit = take_profit_pct / 100
        self.stop_loss = stop_loss_pct / 100
        self.lookahead = lookahead_bars

    def label_row(self, df: pd.DataFrame, idx: int) -> int | None:
        """
        Rotula uma única linha olhando para o futuro.
        Returns: 1 (win), 0 (loss), None (lateral/skip)
        """
        if idx + self.lookahead >= len(df):
            return None
        
        entry_price = df['close'].iloc[idx]
        future_prices = df['close'].iloc[idx+1:idx+1+self.lookahead]
        
        tp_price = entry_price * (1 + self.take_profit)
        sl_price = entry_price * (1 - self.stop_loss)
        
        for price in future_prices:
            if price >= tp_price:
                return 1  # Win
            if price <= sl_price:
                return 0  # Loss
        
        return None  # Lateral (skip)

    def label_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica labels a todo o DataFrame."""
        logger.info(f"🏷️  Aplicando Triple Barrier (TP={TAKE_PROFIT_PCT}%, SL={STOP_LOSS_PCT}%)...")
        
        labels = []
        for i in range(len(df)):
            label = self.label_row(df, i)
            labels.append(label)
            
            if (i + 1) % 10000 == 0:
                logger.info(f"   → {i+1:,} / {len(df):,} rotulados...")
        
        df['outcome_label'] = labels
        
        # Estatísticas
        labeled = df['outcome_label'].notna().sum()
        wins = (df['outcome_label'] == 1).sum()
        losses = (df['outcome_label'] == 0).sum()
        
        logger.info(f"✅ Rotulagem completa:")
        logger.info(f"   → Total rotulado: {labeled:,}")
        logger.info(f"   → Wins: {wins:,} ({100*wins/labeled:.1f}%)")
        logger.info(f"   → Losses: {losses:,} ({100*losses/labeled:.1f}%)")
        
        return df


# ============================================================================
# EXPORTER
# ============================================================================
class DatasetExporter:
    """Exporta para formato dataset.jsonl."""

    @staticmethod
    def export(df: pd.DataFrame, output_path: Path, mode: str = 'a') -> int:
        """
        Exporta DataFrame para JSONL no formato do Vostok.
        Mode: 'a' = append, 'w' = overwrite
        """
        logger.info(f"💾 Exportando para {output_path}...")
        
        # Filtrar apenas labels válidos
        df_valid = df[df['outcome_label'].notna()].copy()
        
        # Garantir diretório existe
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        count = 0
        with open(output_path, mode, encoding='utf-8') as f:
            for idx, row in df_valid.iterrows():
                record = {
                    "timestamp": idx.isoformat(),
                    "timestamp_utc": int(idx.timestamp() * 1000),
                    "action": "LONG",
                    "entry_price": float(row['close']),
                    "features": {
                        "rsi": float(row['rsi']) if pd.notna(row['rsi']) else 50.0,
                        "cvd": float(row['cvd']) if pd.notna(row['cvd']) else 0.0,
                        "entropy": float(row['entropy']) if pd.notna(row['entropy']) else 0.5,
                        "volatility_atr": float(row['volatility_atr']) if pd.notna(row['volatility_atr']) else 0.0,
                        "funding_rate": float(row['funding_rate']),
                        "macd": float(row['macd']) if pd.notna(row['macd']) else 0.0,
                        "macd_hist": float(row['macd_hist']) if pd.notna(row['macd_hist']) else 0.0,
                    },
                    "outcome_label": int(row['outcome_label']),
                    "source": "backfill",
                    "pnl_percent": TAKE_PROFIT_PCT if row['outcome_label'] == 1 else -STOP_LOSS_PCT,
                }
                f.write(json.dumps(record) + '\n')
                count += 1
        
        logger.info(f"✅ Exportados {count:,} registros para {output_path}")
        return count


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def run_backfill() -> None:
    """Executa o pipeline completo de backfill."""
    logger.info("")
    logger.info("╔══════════════════════════════════════════════════════════════╗")
    logger.info("║   VOSTOK-1 :: HISTORICAL BACKFILL (Time Machine)            ║")
    logger.info("║   Generating Training Dataset from Historical Data          ║")
    logger.info("╚══════════════════════════════════════════════════════════════╝")
    logger.info("")
    logger.info(f"Symbol: {SYMBOL}")
    logger.info(f"Timeframe: {TIMEFRAME}")
    logger.info(f"Days Back: {DAYS_BACK}")
    logger.info(f"Output: {DATASET_FILE}")
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
    
    # Step 2: Feature Engineering
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 2: FEATURE ENGINEERING")
    logger.info("=" * 60)
    
    engineer = FeatureEngineer()
    df = engineer.engineer_features(df)
    
    # Step 3: Labeling
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 3: TRIPLE BARRIER LABELING")
    logger.info("=" * 60)
    
    labeler = TripleBarrierLabeler()
    df = labeler.label_dataframe(df)
    
    # Step 4: Export
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 4: EXPORTING DATASET")
    logger.info("=" * 60)
    
    exporter = DatasetExporter()
    count = exporter.export(df, DATASET_FILE, mode='a')
    
    # Summary
    logger.info("")
    logger.info("╔══════════════════════════════════════════════════════════════╗")
    logger.info("║                    BACKFILL COMPLETE!                        ║")
    logger.info("╚══════════════════════════════════════════════════════════════╝")
    logger.info(f"📊 Records exported: {count:,}")
    logger.info(f"📁 Dataset file: {DATASET_FILE}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Run trainer: docker compose run --rm trainer")
    logger.info("  2. Check model: ls -la models/")
    logger.info("")


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    run_backfill()
