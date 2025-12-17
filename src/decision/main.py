"""
VOSTOK-1 :: Módulo Decision Engine (Data Labeling)
====================================================
Motor de Decisão com estratégia base para geração de sinais
e Triple Barrier Labeling para rotulagem de dados de treino.

Arquiteto: Petrovich | Operador: Vostok
Stack: Python 3.11 + asyncio + redis-py
"""

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import redis.asyncio as aioredis

# ============================================================================
# CONFIGURAÇÃO DE LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "module": "%(name)s", "message": "%(message)s"}',
    datefmt='%Y-%m-%dT%H:%M:%S%z',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("decision")

# ============================================================================
# CONFIGURAÇÕES
# ============================================================================
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
SIGNAL_STREAM = os.getenv("SIGNAL_STREAM", "stream:signals:tech")
MARKET_STREAM = os.getenv("MARKET_STREAM", "stream:market:btc_usdt")
CONSUMER_GROUP = os.getenv("CONSUMER_GROUP", "decision_group")
CONSUMER_NAME = os.getenv("CONSUMER_NAME", "decision_worker_1")

# Diretório de dados
DATA_DIR = Path(os.getenv("DATA_DIR", "/app/data"))
TRAINING_DIR = DATA_DIR / "training"
DATASET_FILE = TRAINING_DIR / "dataset.jsonl"

# Parâmetros da estratégia
RSI_OVERSOLD = 35
RSI_OVERBOUGHT = 65

# Parâmetros do Triple Barrier
TP_ATR_MULT = 2.0  # Take Profit = ATR * 2.0
SL_ATR_MULT = 1.0  # Stop Loss = ATR * 1.0
MAX_BARS = 120     # Barreira temporal (2 horas = 120 velas de 1 min)


# ============================================================================
# ENUMS E DATACLASSES
# ============================================================================
class TradeAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class TradeLabel(int, Enum):
    LOSS = 0  # Tocou SL ou tempo expirou
    WIN = 1   # Tocou TP


@dataclass
class VirtualTrade:
    """Representa um trade virtual para rotulagem."""
    id: str
    timestamp: int
    action: TradeAction
    entry_price: float
    tp_price: float  # Take Profit
    sl_price: float  # Stop Loss
    max_bars: int
    bars_elapsed: int = 0
    
    # Features no momento do sinal
    features: dict[str, float] = field(default_factory=dict)
    
    # Resultado (preenchido após fechamento)
    label: TradeLabel | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    pnl_percent: float | None = None

    def check_barriers(self, current_price: float) -> bool:
        """
        Verifica se alguma barreira foi atingida.
        Retorna True se o trade deve ser fechado.
        """
        self.bars_elapsed += 1
        
        if self.action == TradeAction.BUY:
            # Take Profit
            if current_price >= self.tp_price:
                self.label = TradeLabel.WIN
                self.exit_price = current_price
                self.exit_reason = "TP"
                self.pnl_percent = ((current_price - self.entry_price) / self.entry_price) * 100
                return True
            
            # Stop Loss
            if current_price <= self.sl_price:
                self.label = TradeLabel.LOSS
                self.exit_price = current_price
                self.exit_reason = "SL"
                self.pnl_percent = ((current_price - self.entry_price) / self.entry_price) * 100
                return True
                
        elif self.action == TradeAction.SELL:
            # Take Profit (preço caiu)
            if current_price <= self.tp_price:
                self.label = TradeLabel.WIN
                self.exit_price = current_price
                self.exit_reason = "TP"
                self.pnl_percent = ((self.entry_price - current_price) / self.entry_price) * 100
                return True
            
            # Stop Loss (preço subiu)
            if current_price >= self.sl_price:
                self.label = TradeLabel.LOSS
                self.exit_price = current_price
                self.exit_reason = "SL"
                self.pnl_percent = ((self.entry_price - current_price) / self.entry_price) * 100
                return True
        
        # Barreira temporal
        if self.bars_elapsed >= self.max_bars:
            self.label = TradeLabel.LOSS
            self.exit_price = current_price
            self.exit_reason = "TIME"
            if self.action == TradeAction.BUY:
                self.pnl_percent = ((current_price - self.entry_price) / self.entry_price) * 100
            else:
                self.pnl_percent = ((self.entry_price - current_price) / self.entry_price) * 100
            return True
        
        return False


# ============================================================================
# STRATEGY ENGINE - Gerador de Sinais Base
# ============================================================================
class StrategyEngine:
    """
    Motor de estratégia base com alto recall.
    
    Objetivo: Capturar MUITAS oportunidades para posterior filtragem
    pelo modelo de Meta-Labeling.
    
    Regras (intencionalmente simples):
    - BUY: RSI < 35 e CVD > 0 (divergência bullish)
    - SELL: RSI > 65 e CVD < 0 (divergência bearish)
    """

    def __init__(self) -> None:
        self.signals_generated = 0
        self.last_signal_time: int = 0
        self.cooldown_bars = 5  # Evitar sinais consecutivos muito próximos

    def evaluate(self, signal: dict[str, Any]) -> TradeAction | None:
        """
        Avalia o sinal técnico e retorna ação se condições forem atendidas.
        
        Args:
            signal: Dicionário com métricas do stream:signals:tech
            
        Returns:
            TradeAction ou None se nenhuma condição for satisfeita
        """
        try:
            timestamp = int(signal.get('timestamp', 0))
            rsi = float(signal.get('rsi', 50)) if signal.get('rsi') else 50.0
            cvd = float(signal.get('cvd_absolute', 0)) if signal.get('cvd_absolute') else 0.0
            
            # Cooldown: evitar sinais muito próximos
            if timestamp - self.last_signal_time < (self.cooldown_bars * 60000):
                return None
            
            # Regra BUY: RSI oversold + CVD positivo (divergência bullish)
            if rsi < RSI_OVERSOLD and cvd > 0:
                self.last_signal_time = timestamp
                self.signals_generated += 1
                return TradeAction.BUY
            
            # Regra SELL: RSI overbought + CVD negativo (divergência bearish)
            if rsi > RSI_OVERBOUGHT and cvd < 0:
                self.last_signal_time = timestamp
                self.signals_generated += 1
                return TradeAction.SELL
            
            return None
            
        except (ValueError, TypeError) as e:
            logger.warning(f"Erro ao avaliar sinal: {e}")
            return None


# ============================================================================
# TRIPLE BARRIER LABELER - Rotulador de Dados
# ============================================================================
class TripleBarrierLabeler:
    """
    Implementa o método Triple Barrier para rotulagem de dados.
    
    Para cada trade virtual:
    - Define barreiras: Superior (TP), Inferior (SL), Temporal
    - Monitora preço futuro até uma barreira ser atingida
    - Classifica: WIN (TP) ou LOSS (SL/Tempo)
    - Persiste features + label para treinamento
    """

    def __init__(self, dataset_path: Path = DATASET_FILE) -> None:
        self.dataset_path = dataset_path
        self.open_trades: list[VirtualTrade] = []
        self.closed_trades: int = 0
        self.wins: int = 0
        self.losses: int = 0
        
        # Garantir diretório existe
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Inicializar arquivo se não existir
        if not self.dataset_path.exists():
            self.dataset_path.touch()
            logger.info(f"Dataset file initialized at {self.dataset_path}")

    def create_trade(
        self,
        action: TradeAction,
        signal: dict[str, Any]
    ) -> VirtualTrade:
        """
        Cria um novo trade virtual com barreiras definidas.
        """
        timestamp = int(signal.get('timestamp', 0))
        entry_price = float(signal.get('close', 0))
        atr = float(signal.get('volatility_atr', 50)) if signal.get('volatility_atr') else 50.0
        
        # Definir barreiras baseadas em ATR
        if action == TradeAction.BUY:
            tp_price = entry_price + (atr * TP_ATR_MULT)
            sl_price = entry_price - (atr * SL_ATR_MULT)
        else:  # SELL
            tp_price = entry_price - (atr * TP_ATR_MULT)
            sl_price = entry_price + (atr * SL_ATR_MULT)
        
        # Extrair features para o dataset
        features = {
            'rsi': float(signal.get('rsi', 0)) if signal.get('rsi') else 0.0,
            'cvd': float(signal.get('cvd_absolute', 0)) if signal.get('cvd_absolute') else 0.0,
            'entropy': float(signal.get('entropy', 0)) if signal.get('entropy') else 0.0,
            'volatility_atr': atr,
            'volatility_parkinson': float(signal.get('volatility_parkinson', 0)) if signal.get('volatility_parkinson') else 0.0,
            'funding_rate': float(signal.get('funding_rate', 0)) if signal.get('funding_rate') else 0.0,
            'macd': float(signal.get('macd', 0)) if signal.get('macd') else 0.0,
            'macd_hist': float(signal.get('macd_hist', 0)) if signal.get('macd_hist') else 0.0,
        }
        
        trade = VirtualTrade(
            id=f"{action.value}_{timestamp}",
            timestamp=timestamp,
            action=action,
            entry_price=entry_price,
            tp_price=tp_price,
            sl_price=sl_price,
            max_bars=MAX_BARS,
            features=features
        )
        
        self.open_trades.append(trade)
        
        logger.info(
            f"🎯 Trade Virtual #{len(self.open_trades)} | "
            f"{action.value} @ {entry_price:.2f} | "
            f"TP: {tp_price:.2f} | SL: {sl_price:.2f} | "
            f"RSI: {features['rsi']:.1f} | CVD: {features['cvd']:.2f}"
        )
        
        return trade

    def update_trades(self, current_price: float) -> list[VirtualTrade]:
        """
        Atualiza todos os trades abertos e retorna os fechados.
        """
        closed = []
        still_open = []
        
        for trade in self.open_trades:
            if trade.check_barriers(current_price):
                closed.append(trade)
                self.closed_trades += 1
                
                if trade.label == TradeLabel.WIN:
                    self.wins += 1
                else:
                    self.losses += 1
                
                # Persistir trade rotulado
                self._persist_trade(trade)
                
                # Log de fechamento
                label_emoji = "✅" if trade.label == TradeLabel.WIN else "❌"
                logger.info(
                    f"{label_emoji} Trade Finalizado: {trade.exit_reason} | "
                    f"{trade.action.value} @ {trade.entry_price:.2f} → {trade.exit_price:.2f} | "
                    f"PnL: {trade.pnl_percent:+.2f}% | "
                    f"Bars: {trade.bars_elapsed}/{trade.max_bars} | "
                    f"Win Rate: {self.win_rate:.1f}%"
                )
            else:
                still_open.append(trade)
        
        self.open_trades = still_open
        return closed

    def _persist_trade(self, trade: VirtualTrade) -> None:
        """Persiste o trade rotulado no arquivo JSONL."""
        record = {
            "timestamp": trade.timestamp,
            "features": trade.features,
            "action": trade.action.value,
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "exit_reason": trade.exit_reason,
            "outcome_label": trade.label.value if trade.label else None,
            "pnl_percent": round(trade.pnl_percent, 4) if trade.pnl_percent else None,
            "bars_elapsed": trade.bars_elapsed,
        }
        
        try:
            with open(self.dataset_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record) + '\n')
        except IOError as e:
            logger.error(f"Erro ao persistir trade: {e}")

    @property
    def win_rate(self) -> float:
        """Calcula win rate."""
        total = self.wins + self.losses
        return (self.wins / total * 100) if total > 0 else 0.0


# ============================================================================
# DECISION PROCESSOR - Orquestrador Principal
# ============================================================================
class DecisionProcessor:
    """
    Orquestra o fluxo de decisão:
    1. Lê sinais do stream:signals:tech
    2. Avalia via StrategyEngine
    3. Cria trades virtuais
    4. Monitora via TripleBarrierLabeler
    5. Persiste dataset para ML
    """

    def __init__(self) -> None:
        self.redis: aioredis.Redis | None = None
        self.strategy = StrategyEngine()
        self.labeler = TripleBarrierLabeler()
        self.running = False
        self.last_price: float = 0.0

    async def connect_redis(self) -> None:
        """Conecta ao Redis."""
        self.redis = aioredis.Redis(
            host=REDIS_HOST, port=REDIS_PORT, decode_responses=True
        )
        await self.redis.ping()
        logger.info(f"Redis conectado: {REDIS_HOST}:{REDIS_PORT}")

    async def setup_consumer_group(self) -> None:
        """Cria Consumer Group se não existir."""
        try:
            await self.redis.xgroup_create(SIGNAL_STREAM, CONSUMER_GROUP, id='$', mkstream=True)
            logger.info(f"Consumer group '{CONSUMER_GROUP}' criado")
        except aioredis.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.info(f"Consumer group '{CONSUMER_GROUP}' já existe")
            else:
                raise

    async def process_signal(self, signal: dict[str, Any]) -> None:
        """Processa um sinal do stream:signals:tech."""
        # Atualizar último preço conhecido
        close_price = signal.get('close')
        if close_price:
            try:
                self.last_price = float(close_price)
            except (ValueError, TypeError):
                pass
        
        # Atualizar trades abertos
        if self.last_price > 0:
            self.labeler.update_trades(self.last_price)
        
        # Avaliar novo sinal
        action = self.strategy.evaluate(signal)
        
        if action is not None:
            self.labeler.create_trade(action, signal)

    async def consume_stream(self) -> None:
        """Loop principal de consumo."""
        while self.running:
            try:
                messages = await self.redis.xreadgroup(
                    groupname=CONSUMER_GROUP,
                    consumername=CONSUMER_NAME,
                    streams={SIGNAL_STREAM: '>'},
                    count=10,
                    block=1000
                )
                
                if messages:
                    for stream_name, stream_messages in messages:
                        for message_id, data in stream_messages:
                            await self.process_signal(data)
                            await self.redis.xack(SIGNAL_STREAM, CONSUMER_GROUP, message_id)
                            
            except aioredis.ResponseError as e:
                if "NOGROUP" in str(e):
                    await self.setup_consumer_group()
                else:
                    logger.error(f"Erro Redis: {e}")
                    await asyncio.sleep(1)
            except Exception as e:
                logger.exception(f"Erro no consumo: {e}")
                await asyncio.sleep(1)

    async def start(self) -> None:
        """Inicia o processador."""
        logger.info("=" * 60)
        logger.info("VOSTOK-1 :: Decision Engine (Data Labeling Mode)")
        logger.info(f"Signal Stream: {SIGNAL_STREAM}")
        logger.info(f"Dataset: {DATASET_FILE}")
        logger.info(f"Triple Barrier: TP={TP_ATR_MULT}x ATR, SL={SL_ATR_MULT}x ATR, Max={MAX_BARS} bars")
        logger.info("=" * 60)
        
        self.running = True
        await self.connect_redis()
        await self.setup_consumer_group()
        await self.consume_stream()

    async def stop(self) -> None:
        """Para o processador."""
        logger.info("Parando Decision Engine...")
        self.running = False
        
        if self.redis:
            await self.redis.close()
        
        logger.info(
            f"Decision parado. "
            f"Sinais: {self.strategy.signals_generated} | "
            f"Trades: {self.labeler.closed_trades} | "
            f"Win Rate: {self.labeler.win_rate:.1f}%"
        )


# ============================================================================
# ENTRY POINT
# ============================================================================
async def main() -> None:
    processor = DecisionProcessor()
    try:
        await processor.start()
    except KeyboardInterrupt:
        await processor.stop()


if __name__ == "__main__":
    asyncio.run(main())
