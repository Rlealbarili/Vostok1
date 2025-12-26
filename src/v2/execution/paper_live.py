"""
VOSTOK V2 :: Paper Trading Bot (Live Simulation)
=================================================
Bot de paper trading que conecta o VostokV2Engine ao mercado real.

Features:
- Loop operacional de 60s sincronizado com candles M1
- Gestão de posição com TP/SL dinâmicos baseados em ATR
- Logging detalhado e bonito
- Persistência de trades em CSV

Arquiteto: Petrovich | Operador: Vostok
"""

import asyncio
import csv
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.v2.engine import VostokV2Engine, Action
from src.v2.features.engineering import FeatureGenerator

logger = logging.getLogger("paper_trading_v2")


# ============================================================================
# CONSTANTS
# ============================================================================

BINANCE_API_URL = "https://api.binance.com/api/v3/klines"
SYMBOL = "BTCUSDT"
INITIAL_BALANCE = 200.0
POSITION_SIZE_PCT = 0.95  # 95% do saldo por trade
TP_MULTIPLIER = 3.0
SL_MULTIPLIER = 1.5
LOOP_INTERVAL = 60  # seconds

LOG_DIR = Path("data/logs")
TRADE_LOG_FILE = LOG_DIR / "v2_paper_trades.csv"

# Redis Configuration for News
REDIS_HOST = os.environ.get("REDIS_HOST", "vostok_redis")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
SENTIMENT_STREAM = "stream:signals:sentiment"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Position:
    """Posição aberta."""
    entry_time: datetime
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    quantity: float
    take_profit: float
    stop_loss: float
    confidence: float
    
    def unrealized_pnl(self, current_price: float) -> float:
        """Calcula PnL não realizado."""
        if self.direction == "LONG":
            return (current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - current_price) * self.quantity
    
    def unrealized_pnl_pct(self, current_price: float) -> float:
        """Calcula PnL % não realizado."""
        if self.direction == "LONG":
            return (current_price - self.entry_price) / self.entry_price * 100
        else:
            return (self.entry_price - current_price) / self.entry_price * 100


@dataclass
class TradeResult:
    """Resultado de um trade fechado."""
    entry_time: datetime
    exit_time: datetime
    direction: str
    entry_price: float
    exit_price: float
    quantity: float
    take_profit: float
    stop_loss: float
    pnl_usd: float
    pnl_pct: float
    result: str  # "TP", "SL", "MANUAL"
    confidence: float
    balance_after: float


@dataclass
class BotState:
    """Estado do bot."""
    balance: float = INITIAL_BALANCE
    position: Optional[Position] = None
    trades: List[TradeResult] = field(default_factory=list)
    cycles: int = 0
    signals: int = 0
    wins: int = 0
    losses: int = 0


# ============================================================================
# NEWS FETCHER (Redis Stream)
# ============================================================================

class NewsFetcher:
    """Busca últimas headlines do Redis Stream de sentimento."""
    
    def __init__(self):
        self.redis_client = None
        self.last_headlines = None
        self._connect()
    
    def _connect(self):
        """Conecta ao Redis."""
        if not HAS_REDIS:
            logger.warning("⚠️ Redis não instalado - Buffett operará sem notícias")
            return
        
        try:
            self.redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                decode_responses=True,
                socket_timeout=5,
            )
            self.redis_client.ping()
            logger.info(f"✅ Redis conectado para News: {REDIS_HOST}:{REDIS_PORT}")
        except Exception as e:
            logger.warning(f"⚠️ Não foi possível conectar ao Redis: {e}")
            self.redis_client = None
    
    def get_latest_headlines(self) -> Optional[str]:
        """
        Busca o último registro do stream de sentimento.
        Retorna o summary das últimas notícias ou None.
        """
        if not self.redis_client:
            return None
        
        try:
            # Buscar último registro do stream
            result = self.redis_client.xrevrange(
                SENTIMENT_STREAM,
                count=1,
            )
            
            if result:
                msg_id, data = result[0]
                summary = data.get("summary", "")
                score = data.get("sentiment_score", "0")
                headlines_count = data.get("headlines_count", "0")
                
                if summary:
                    # Retornar contexto formatado para o Buffett
                    return f"Latest market news ({headlines_count} headlines): {summary} [Sentiment: {score}]"
            
            return None
            
        except Exception as e:
            logger.debug(f"Erro ao buscar headlines: {e}")
            return None


# ============================================================================
# DATA FETCHER
# ============================================================================

class BinanceDataFetcher:
    """Busca dados OHLCV da Binance."""
    
    def __init__(self, symbol: str = SYMBOL):
        self.symbol = symbol
    
    def fetch_klines(self, interval: str, limit: int) -> pd.DataFrame:
        """Busca klines da Binance."""
        if not HAS_REQUESTS:
            raise ImportError("requests not installed")
        
        try:
            response = requests.get(
                BINANCE_API_URL,
                params={
                    "symbol": self.symbol,
                    "interval": interval,
                    "limit": limit,
                },
                timeout=10,
            )
            response.raise_for_status()
            
            data = response.json()
            
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch klines: {e}")
            raise
    
    def get_m1_data(self, limit: int = 500) -> pd.DataFrame:
        """Busca dados M1."""
        return self.fetch_klines("1m", limit)
    
    def get_h1_data(self, limit: int = 200) -> pd.DataFrame:
        """Busca dados H1."""
        return self.fetch_klines("1h", limit)
    
    def get_current_price(self) -> float:
        """Busca preço atual."""
        df = self.fetch_klines("1m", 1)
        return float(df['close'].iloc[-1])


# ============================================================================
# PAPER TRADING BOT
# ============================================================================

class PaperTradingBot:
    """
    Bot de Paper Trading para Vostok V2.
    
    Simula trades em tempo real usando o VostokV2Engine.
    """
    
    def __init__(
        self,
        initial_balance: float = INITIAL_BALANCE,
        position_size_pct: float = POSITION_SIZE_PCT,
        tp_multiplier: float = TP_MULTIPLIER,
        sl_multiplier: float = SL_MULTIPLIER,
        ollama_host: str = "localhost",
    ):
        self.initial_balance = initial_balance
        self.position_size_pct = position_size_pct
        self.tp_multiplier = tp_multiplier
        self.sl_multiplier = sl_multiplier
        
        # Initialize components
        self.engine = VostokV2Engine(ollama_host=ollama_host)
        self.data_fetcher = BinanceDataFetcher()
        self.news_fetcher = NewsFetcher()
        self.feature_gen = FeatureGenerator()
        
        # State
        self.state = BotState(balance=initial_balance)
        
        # Ensure log directory
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self._init_csv_log()
        
        # Running flag
        self._running = False
        
        logger.info("=" * 70)
        logger.info("🚀 VOSTOK V2 PAPER TRADING BOT")
        logger.info("=" * 70)
        logger.info(f"   Initial Balance: ${initial_balance:.2f}")
        logger.info(f"   Position Size: {position_size_pct:.0%}")
        logger.info(f"   TP: {tp_multiplier}x ATR | SL: {sl_multiplier}x ATR")
        logger.info("=" * 70)
    
    def _init_csv_log(self):
        """Inicializa arquivo CSV de log."""
        if not TRADE_LOG_FILE.exists():
            with open(TRADE_LOG_FILE, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'entry_time', 'exit_time', 'duration_min',
                    'direction', 'confidence', 'entry', 'exit', 'tp', 'sl',
                    'pnl_pct', 'pnl_usd', 'balance', 'result'
                ])
    
    def _log_trade(self, trade: TradeResult):
        """Registra trade no CSV."""
        duration = (trade.exit_time - trade.entry_time).total_seconds() / 60
        
        with open(TRADE_LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now(timezone.utc).isoformat(),
                trade.entry_time.isoformat(),
                trade.exit_time.isoformat(),
                f"{duration:.1f}",
                trade.direction,
                f"{trade.confidence:.4f}",
                f"{trade.entry_price:.2f}",
                f"{trade.exit_price:.2f}",
                f"{trade.take_profit:.2f}",
                f"{trade.stop_loss:.2f}",
                f"{trade.pnl_pct:.4f}",
                f"{trade.pnl_usd:.4f}",
                f"{trade.balance_after:.2f}",
                trade.result,
            ])
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calcula ATR atual."""
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
        
        atr = pd.Series(tr).rolling(window=period).mean().iloc[-1]
        return float(atr)
    
    def _check_position(self, current_price: float) -> Optional[TradeResult]:
        """Verifica se posição atingiu TP ou SL."""
        if self.state.position is None:
            return None
        
        pos = self.state.position
        
        # Check TP/SL for LONG
        if pos.direction == "LONG":
            if current_price >= pos.take_profit:
                return self._close_position(current_price, "TP")
            elif current_price <= pos.stop_loss:
                return self._close_position(current_price, "SL")
        
        # Check TP/SL for SHORT
        else:
            if current_price <= pos.take_profit:
                return self._close_position(current_price, "TP")
            elif current_price >= pos.stop_loss:
                return self._close_position(current_price, "SL")
        
        return None
    
    def _close_position(self, exit_price: float, result: str) -> TradeResult:
        """Fecha a posição atual."""
        pos = self.state.position
        
        pnl_usd = pos.unrealized_pnl(exit_price)
        pnl_pct = pos.unrealized_pnl_pct(exit_price)
        
        self.state.balance += pnl_usd
        
        trade = TradeResult(
            entry_time=pos.entry_time,
            exit_time=datetime.now(timezone.utc),
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            quantity=pos.quantity,
            take_profit=pos.take_profit,
            stop_loss=pos.stop_loss,
            pnl_usd=pnl_usd,
            pnl_pct=pnl_pct,
            result=result,
            confidence=pos.confidence,
            balance_after=self.state.balance,
        )
        
        self.state.trades.append(trade)
        self.state.position = None
        
        if result == "TP":
            self.state.wins += 1
        else:
            self.state.losses += 1
        
        # Log
        self._log_trade(trade)
        
        emoji = "✅" if result == "TP" else "❌"
        logger.info(
            f"{emoji} CLOSED | {pos.direction} | {result} | "
            f"${exit_price:,.2f} | PnL: ${pnl_usd:+.2f} ({pnl_pct:+.2f}%) | "
            f"Balance: ${self.state.balance:.2f}"
        )
        
        return trade
    
    def _open_position(
        self,
        direction: str,
        entry_price: float,
        atr: float,
        confidence: float,
    ):
        """Abre uma nova posição."""
        # Calcular tamanho da posição
        position_value = self.state.balance * self.position_size_pct
        quantity = position_value / entry_price
        
        # Calcular TP e SL
        if direction == "LONG":
            take_profit = entry_price + (self.tp_multiplier * atr)
            stop_loss = entry_price - (self.sl_multiplier * atr)
        else:
            take_profit = entry_price - (self.tp_multiplier * atr)
            stop_loss = entry_price + (self.sl_multiplier * atr)
        
        self.state.position = Position(
            entry_time=datetime.now(timezone.utc),
            direction=direction,
            entry_price=entry_price,
            quantity=quantity,
            take_profit=take_profit,
            stop_loss=stop_loss,
            confidence=confidence,
        )
        
        self.state.signals += 1
        
        logger.info(
            f"🎯 OPENED | {direction} | ${entry_price:,.2f} | "
            f"TP: ${take_profit:,.2f} | SL: ${stop_loss:,.2f} | "
            f"Conf: {confidence:.1%}"
        )
    
    def _print_status(
        self,
        price: float,
        decision: Any,
        atr: float,
    ):
        """Imprime status atual no terminal."""
        # Position info
        if self.state.position:
            pos = self.state.position
            pnl = pos.unrealized_pnl(price)
            pnl_pct = pos.unrealized_pnl_pct(price)
            pos_str = f"POS: {pos.direction} (${pnl:+.2f} / {pnl_pct:+.2f}%)"
        else:
            pos_str = "POS: NONE"
        
        # Stats
        total_trades = self.state.wins + self.state.losses
        win_rate = self.state.wins / total_trades * 100 if total_trades > 0 else 0
        
        # Action emoji
        action_emoji = {
            "EXECUTE": "🟢",
            "HOLD": "⚪",
            "WAIT": "🟡",
            "SKIP": "🔴",
        }
        
        # Print
        print(
            f"\r"
            f"BTC: ${price:,.2f} | "
            f"ATR: ${atr:.2f} | "
            f"REGIME: {decision.regime_status[:3]} | "
            f"ML: {decision.ml_signal} ({decision.ml_confidence:.0%}) | "
            f"{action_emoji.get(decision.action.value, '⚪')} {decision.action.value} | "
            f"{pos_str} | "
            f"BAL: ${self.state.balance:.2f} | "
            f"W/L: {self.state.wins}/{self.state.losses} ({win_rate:.0f}%)"
            f"    ",  # Padding para limpar
            end="",
            flush=True,
        )
    
    async def run_cycle(self):
        """Executa um ciclo de análise."""
        self.state.cycles += 1
        
        try:
            # 1. Fetch data
            df_m1 = self.data_fetcher.get_m1_data(500)
            df_h1 = self.data_fetcher.get_h1_data(200)
            
            current_price = float(df_m1['close'].iloc[-1])
            atr = self._calculate_atr(df_m1)
            
            # 2. Check existing position
            if self.state.position:
                closed = self._check_position(current_price)
                if closed:
                    print()  # New line after close
            
            # 2.5 Fetch latest news from Redis for Buffett
            news_context = self.news_fetcher.get_latest_headlines()
            
            # 3. Analyze market (only if no position)
            if self.state.position is None:
                decision = await self.engine.analyze_market(df_m1, df_h1, news_context=news_context)
                
                # 4. Execute if signal
                if decision.action == Action.EXECUTE and decision.direction:
                    self._open_position(
                        direction=decision.direction,
                        entry_price=current_price,
                        atr=atr,
                        confidence=decision.confidence,
                    )
                    print()  # New line after open
            else:
                # Still update decision for display
                decision = await self.engine.analyze_market(df_m1, df_h1, news_context=news_context)
            
            # 5. Print status
            self._print_status(current_price, decision, atr)
            
        except Exception as e:
            logger.error(f"Cycle error: {e}")
    
    async def run_forever(self):
        """Executa o bot em loop infinito."""
        logger.info("\n🚀 Starting live paper trading loop...")
        logger.info(f"   Cycle interval: {LOOP_INTERVAL}s")
        logger.info("   Press Ctrl+C to stop\n")
        
        self._running = True
        
        try:
            while self._running:
                # Wait for next minute (sync with candle close)
                now = datetime.now()
                seconds_until_next = 60 - now.second
                if seconds_until_next > 5:
                    await asyncio.sleep(seconds_until_next)
                
                # Run cycle
                await self.run_cycle()
                
                # Small delay to avoid hammering
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            print("\n")
            logger.info("🛑 Stopping bot...")
        finally:
            self._running = False
            self._print_summary()
    
    def _print_summary(self):
        """Imprime resumo final."""
        print("\n")
        print("=" * 70)
        print("  📊 PAPER TRADING SESSION SUMMARY")
        print("=" * 70)
        
        total_trades = self.state.wins + self.state.losses
        win_rate = self.state.wins / total_trades * 100 if total_trades > 0 else 0
        pnl = self.state.balance - self.initial_balance
        pnl_pct = pnl / self.initial_balance * 100
        
        print(f"  Cycles: {self.state.cycles}")
        print(f"  Signals: {self.state.signals}")
        print(f"  Trades: {total_trades}")
        print(f"  Wins: {self.state.wins} | Losses: {self.state.losses}")
        print(f"  Win Rate: {win_rate:.1f}%")
        print()
        print(f"  Initial Balance: ${self.initial_balance:.2f}")
        print(f"  Final Balance: ${self.state.balance:.2f}")
        print(f"  P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%)")
        print()
        print(f"  Trade Log: {TRADE_LOG_FILE}")
        print("=" * 70)


# ============================================================================
# MAIN
# ============================================================================

async def main():
    """Entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)7s | %(message)s',
        datefmt='%H:%M:%S',
    )
    
    # Reduce noise from other loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("lightgbm").setLevel(logging.WARNING)
    
    bot = PaperTradingBot(
        initial_balance=200.0,
        ollama_host="localhost",  # Use localhost for testing
    )
    
    await bot.run_forever()


if __name__ == "__main__":
    asyncio.run(main())
