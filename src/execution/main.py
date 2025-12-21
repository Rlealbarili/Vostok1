"""
VOSTOK-1 :: Execution Module (Paper Trading)
=============================================
Executa trades virtuais baseado nos sinais do Decision Engine.
Gerencia posição, monitora barreiras e persiste resultados.

Arquiteto: Petrovich | Operador: Vostok
Stack: Python 3.11 + asyncio + redis-py
"""

import asyncio
import csv
import json
import logging
import os
import sys
from dataclasses import dataclass, asdict
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
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("execution")

# ============================================================================
# CONFIGURAÇÕES
# ============================================================================
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# Streams
ORDER_STREAM = os.getenv("ORDER_STREAM", "stream:orders:execute")
PRICE_STREAM = os.getenv("PRICE_STREAM", "stream:signals:live")
STATUS_STREAM = os.getenv("STATUS_STREAM", "stream:execution:status")

# Consumer
CONSUMER_GROUP = os.getenv("CONSUMER_GROUP", "execution_group")
CONSUMER_NAME = os.getenv("CONSUMER_NAME", "executor_1")

# Paper Trading Bankroll Management
INITIAL_BALANCE = float(os.getenv("PAPER_INITIAL_BALANCE", 200.0))
LEVERAGE = int(os.getenv("LEVERAGE", 1))
ORDER_SIZE_PCT = float(os.getenv("ORDER_SIZE_PCT", 0.95))

# Paths
DATA_DIR = Path(os.getenv("DATA_DIR", "/app/data"))
LOGS_DIR = DATA_DIR / "logs"
TRADES_FILE = LOGS_DIR / "paper_trades.csv"

# Time limit (45 minutes in milliseconds)
TIMEOUT_MS = 45 * 60 * 1000


# ============================================================================
# ENUMS E DATACLASSES
# ============================================================================
class TradeStatus(str, Enum):
    IDLE = "IDLE"
    IN_TRADE = "IN_TRADE"


class ExitReason(str, Enum):
    TP = "TP"        # Take Profit
    SL = "SL"        # Stop Loss
    TIME = "TIME"    # Timeout


@dataclass
class PaperTrade:
    """Representa um trade de paper trading."""
    entry_time: int          # Timestamp MS
    entry_price: float
    tp_price: float
    sl_price: float
    confidence: float
    atr: float
    position_size: float     # Tamanho da posição em USD
    
    # Preenchidos no fechamento
    exit_time: int | None = None
    exit_price: float | None = None
    exit_reason: ExitReason | None = None
    pnl_percent: float | None = None
    pnl_usd: float | None = None
    balance_after: float | None = None

    def check_exit(self, current_price: float, current_time: int) -> ExitReason | None:
        """Verifica condições de saída."""
        # Take Profit
        if current_price >= self.tp_price:
            return ExitReason.TP
        
        # Stop Loss
        if current_price <= self.sl_price:
            return ExitReason.SL
        
        # Timeout (45 minutos)
        if current_time - self.entry_time >= TIMEOUT_MS:
            return ExitReason.TIME
        
        return None

    def close(self, exit_price: float, exit_time: int, reason: ExitReason, current_balance: float) -> float:
        """Fecha o trade e retorna o novo saldo."""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_reason = reason
        self.pnl_percent = ((exit_price - self.entry_price) / self.entry_price) * 100
        
        # Calcular PnL em USD baseado no position size com alavancagem
        qty = (self.position_size * LEVERAGE) / self.entry_price
        self.pnl_usd = (exit_price - self.entry_price) * qty
        self.balance_after = current_balance + self.pnl_usd
        
        return self.balance_after


# ============================================================================
# TRADE LOGGER (CSV Persistence)
# ============================================================================
class TradeLogger:
    """Persiste trades em CSV para análise."""

    def __init__(self, filepath: Path = TRADES_FILE) -> None:
        self.filepath = filepath
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_header()

    def _ensure_header(self) -> None:
        """Cria header se arquivo não existe."""
        if not self.filepath.exists():
            with open(self.filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'entry_time', 'exit_time', 'duration_min',
                    'signal_confidence', 'entry', 'exit', 'tp', 'sl',
                    'result_pct', 'result_usd', 'balance_after', 'position_size',
                    'outcome', 'atr'
                ])
            logger.info(f"📁 Trade log initialized: {self.filepath}")

    def log_trade(self, trade: PaperTrade) -> None:
        """Adiciona trade ao CSV."""
        duration_min = (trade.exit_time - trade.entry_time) / 60000 if trade.exit_time else 0
        
        row = [
            datetime.now(timezone.utc).isoformat(),
            trade.entry_time,
            trade.exit_time,
            round(duration_min, 2),
            round(trade.confidence, 4),
            round(trade.entry_price, 2),
            round(trade.exit_price, 2) if trade.exit_price else None,
            round(trade.tp_price, 2),
            round(trade.sl_price, 2),
            round(trade.pnl_percent, 4) if trade.pnl_percent else None,
            round(trade.pnl_usd, 2) if trade.pnl_usd else None,
            round(trade.balance_after, 2) if trade.balance_after else None,
            round(trade.position_size, 2),
            trade.exit_reason.value if trade.exit_reason else None,
            round(trade.atr, 2),
        ]
        
        with open(self.filepath, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        
        logger.info(f"💾 Trade logged to {self.filepath}")


# ============================================================================
# EXECUTION ENGINE (Paper Trading)
# ============================================================================
class ExecutionEngine:
    """
    Motor de execução para Paper Trading.
    
    Estados:
    - IDLE: Sem posição aberta, aguardando sinal
    - IN_TRADE: Posição aberta, monitorando preço
    """

    def __init__(self) -> None:
        self.redis: aioredis.Redis | None = None
        self.status = TradeStatus.IDLE
        self.current_trade: PaperTrade | None = None
        self.current_price: float = 0.0
        self.trade_logger = TradeLogger()
        self.running = False
        
        # Bankroll Management
        self.initial_balance = INITIAL_BALANCE
        self.current_balance = INITIAL_BALANCE
        self.leverage = LEVERAGE
        self.order_size_pct = ORDER_SIZE_PCT
        
        # Estatísticas
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0
        self.total_pnl_usd = 0.0

    async def connect_redis(self) -> None:
        """Conecta ao Redis."""
        self.redis = aioredis.Redis(
            host=REDIS_HOST, port=REDIS_PORT, decode_responses=True
        )
        await self.redis.ping()
        logger.info(f"Redis conectado: {REDIS_HOST}:{REDIS_PORT}")

    async def setup_consumer_groups(self) -> None:
        """Cria Consumer Groups se não existirem."""
        for stream in [ORDER_STREAM, PRICE_STREAM]:
            try:
                await self.redis.xgroup_create(stream, CONSUMER_GROUP, id='$', mkstream=True)
                logger.info(f"Consumer group criado para {stream}")
            except aioredis.ResponseError as e:
                if "BUSYGROUP" not in str(e):
                    raise

    async def open_trade(self, order: dict[str, Any]) -> None:
        """Abre um novo trade de paper trading."""
        if self.status == TradeStatus.IN_TRADE:
            logger.warning("⚠️  Trade já aberto, ignorando novo sinal")
            return
        
        try:
            entry_price = float(order.get('entry_price', 0))
            tp_price = float(order.get('take_profit', 0))
            sl_price = float(order.get('stop_loss', 0))
            confidence = float(order.get('confidence', 0))
            atr = float(order.get('atr', 0))
            timestamp = int(order.get('timestamp', 0))
            
            if entry_price <= 0:
                logger.warning("Preço de entrada inválido")
                return
            
            # Calcular position size baseado na banca atual
            position_size = self.current_balance * self.order_size_pct
            
            self.current_trade = PaperTrade(
                entry_time=timestamp or int(datetime.now(timezone.utc).timestamp() * 1000),
                entry_price=entry_price,
                tp_price=tp_price,
                sl_price=sl_price,
                confidence=confidence,
                atr=atr,
                position_size=position_size,
            )
            
            self.status = TradeStatus.IN_TRADE
            self.current_price = entry_price
            
            logger.info(
                f"🟢 PAPER TRADE OPENED @ ${entry_price:.2f} | "
                f"Size: ${position_size:.2f} (x{self.leverage}) | "
                f"TP: ${tp_price:.2f} | SL: ${sl_price:.2f} | "
                f"Conf: {confidence:.2%}"
            )
            
        except Exception as e:
            logger.error(f"Erro ao abrir trade: {e}")

    async def close_trade(self, reason: ExitReason) -> None:
        """Fecha o trade atual."""
        if not self.current_trade:
            return
        
        exit_time = int(datetime.now(timezone.utc).timestamp() * 1000)
        new_balance = self.current_trade.close(
            self.current_price, exit_time, reason, self.current_balance
        )
        
        # Atualizar saldo
        pnl_usd = self.current_trade.pnl_usd or 0
        self.current_balance = new_balance
        
        # Estatísticas
        self.total_trades += 1
        pnl = self.current_trade.pnl_percent or 0
        self.total_pnl += pnl
        self.total_pnl_usd += pnl_usd
        
        if reason == ExitReason.TP:
            self.wins += 1
            emoji = "🟢"
            msg = f"TAKE PROFIT HIT (+{pnl:.2f}% / +${pnl_usd:.2f})"
        elif reason == ExitReason.SL:
            self.losses += 1
            emoji = "🔴"
            msg = f"STOP LOSS HIT ({pnl:.2f}% / ${pnl_usd:.2f})"
        else:
            self.losses += 1
            emoji = "⚪"
            msg = f"TIMEOUT EXIT ({pnl:.2f}% / ${pnl_usd:.2f})"
        
        logger.info(
            f"{emoji} {msg} | "
            f"Entry: ${self.current_trade.entry_price:.2f} → Exit: ${self.current_price:.2f} | "
            f"Balance: ${self.current_balance:.2f} | "
            f"Win Rate: {self.win_rate:.1f}%"
        )
        
        # Persistir
        self.trade_logger.log_trade(self.current_trade)
        
        # Reset
        self.current_trade = None
        self.status = TradeStatus.IDLE

    @property
    def win_rate(self) -> float:
        """Calcula win rate."""
        if self.total_trades == 0:
            return 0.0
        return (self.wins / self.total_trades) * 100

    @property
    def current_pnl_percent(self) -> float:
        """Calcula PnL atual do trade aberto."""
        if not self.current_trade or self.current_price <= 0:
            return 0.0
        return ((self.current_price - self.current_trade.entry_price) / 
                self.current_trade.entry_price) * 100

    @property
    def current_pnl_usd(self) -> float:
        """Calcula PnL atual em USD do trade aberto."""
        if not self.current_trade or self.current_price <= 0:
            return 0.0
        qty = (self.current_trade.position_size * self.leverage) / self.current_trade.entry_price
        return (self.current_price - self.current_trade.entry_price) * qty

    async def publish_status(self) -> None:
        """Publica status atual no Redis."""
        status_data = {
            "status": self.status.value,
            "pnl_pct": str(round(self.current_pnl_percent, 4)),
            "pnl_usd": str(round(self.current_pnl_usd, 2)),
            "entry": str(round(self.current_trade.entry_price, 2)) if self.current_trade else "0",
            "current": str(round(self.current_price, 2)),
            "tp": str(round(self.current_trade.tp_price, 2)) if self.current_trade else "0",
            "sl": str(round(self.current_trade.sl_price, 2)) if self.current_trade else "0",
            "position_size": str(round(self.current_trade.position_size, 2)) if self.current_trade else "0",
            "initial_balance": str(round(self.initial_balance, 2)),
            "current_balance": str(round(self.current_balance, 2)),
            "total_trades": str(self.total_trades),
            "wins": str(self.wins),
            "losses": str(self.losses),
            "win_rate": str(round(self.win_rate, 2)),
            "total_pnl": str(round(self.total_pnl, 4)),
            "total_pnl_usd": str(round(self.total_pnl_usd, 2)),
            "timestamp": str(int(datetime.now(timezone.utc).timestamp() * 1000)),
        }
        
        try:
            await self.redis.xadd(STATUS_STREAM, status_data, maxlen=100)
        except Exception as e:
            logger.warning(f"Erro ao publicar status: {e}")

    async def process_order(self, order: dict[str, Any]) -> None:
        """Processa ordem do Decision Engine."""
        signal = order.get('signal', '').upper()
        
        if signal == "BUY":
            await self.open_trade(order)

    async def process_price(self, data: dict[str, Any]) -> None:
        """Processa atualização de preço."""
        try:
            price = float(data.get('close') or data.get('price', 0))
            if price > 0:
                self.current_price = price
        except (ValueError, TypeError):
            return
        
        # Verificar condições de saída se em trade
        if self.status == TradeStatus.IN_TRADE and self.current_trade:
            current_time = int(datetime.now(timezone.utc).timestamp() * 1000)
            exit_reason = self.current_trade.check_exit(self.current_price, current_time)
            
            if exit_reason:
                await self.close_trade(exit_reason)

    async def consume_orders(self) -> None:
        """Loop de consumo de ordens."""
        while self.running:
            try:
                messages = await self.redis.xreadgroup(
                    groupname=CONSUMER_GROUP,
                    consumername=CONSUMER_NAME,
                    streams={ORDER_STREAM: '>'},
                    count=1,
                    block=100
                )
                
                if messages:
                    for stream_name, stream_messages in messages:
                        for message_id, data in stream_messages:
                            await self.process_order(data)
                            await self.redis.xack(ORDER_STREAM, CONSUMER_GROUP, message_id)
                            
            except aioredis.ResponseError as e:
                if "NOGROUP" in str(e):
                    await self.setup_consumer_groups()
                else:
                    await asyncio.sleep(0.1)
            except Exception as e:
                logger.exception(f"Erro no consumo de ordens: {e}")
                await asyncio.sleep(0.5)

    async def consume_prices(self) -> None:
        """Loop de consumo de preços."""
        while self.running:
            try:
                messages = await self.redis.xreadgroup(
                    groupname=CONSUMER_GROUP,
                    consumername=CONSUMER_NAME,
                    streams={PRICE_STREAM: '>'},
                    count=10,
                    block=100
                )
                
                if messages:
                    for stream_name, stream_messages in messages:
                        for message_id, data in stream_messages:
                            await self.process_price(data)
                            await self.redis.xack(PRICE_STREAM, CONSUMER_GROUP, message_id)
                            
            except aioredis.ResponseError as e:
                if "NOGROUP" in str(e):
                    await self.setup_consumer_groups()
                else:
                    await asyncio.sleep(0.1)
            except Exception:
                await asyncio.sleep(0.1)

    async def status_publisher(self) -> None:
        """Publica status a cada segundo."""
        while self.running:
            await self.publish_status()
            await asyncio.sleep(1)

    async def start(self) -> None:
        """Inicia o motor de execução."""
        logger.info("=" * 60)
        logger.info("VOSTOK-1 :: Execution Engine (Paper Trading)")
        logger.info("=" * 60)
        logger.info(f"💰 Initial Balance: ${self.initial_balance:.2f}")
        logger.info(f"📊 Leverage: {self.leverage}x")
        logger.info(f"📏 Order Size: {self.order_size_pct * 100:.0f}% of balance")
        logger.info("-" * 60)
        logger.info(f"Order Stream: {ORDER_STREAM}")
        logger.info(f"Price Stream: {PRICE_STREAM}")
        logger.info(f"Status Stream: {STATUS_STREAM}")
        logger.info(f"Trade Log: {TRADES_FILE}")
        logger.info(f"Timeout: {TIMEOUT_MS / 60000:.0f} minutes")
        logger.info("=" * 60)
        
        self.running = True
        await self.connect_redis()
        await self.setup_consumer_groups()
        
        # Executar loops em paralelo
        await asyncio.gather(
            self.consume_orders(),
            self.consume_prices(),
            self.status_publisher(),
        )

    async def stop(self) -> None:
        """Para o motor de execução."""
        logger.info("Parando Execution Engine...")
        self.running = False
        
        if self.redis:
            await self.redis.close()
        
        logger.info(
            f"Execution parado. "
            f"Trades: {self.total_trades} | "
            f"Wins: {self.wins} | Losses: {self.losses} | "
            f"Win Rate: {self.win_rate:.1f}% | "
            f"Total PnL: {self.total_pnl:+.2f}%"
        )


# ============================================================================
# ENTRY POINT
# ============================================================================
async def main() -> None:
    engine = ExecutionEngine()
    try:
        await engine.start()
    except KeyboardInterrupt:
        await engine.stop()


if __name__ == "__main__":
    asyncio.run(main())
