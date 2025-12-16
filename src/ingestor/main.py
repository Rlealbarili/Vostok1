"""
VOSTOK-1 :: Módulo Ingestor
===========================
Captura trades BTC/USDT da Binance via WebSocket e injeta no Redis Streams.

Arquiteto: Petrovich | Operador: Vostok
Stack: Python 3.11 + asyncio + ccxt.pro + redis-py
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any

import ccxt.pro as ccxtpro
import redis.asyncio as aioredis

# ============================================================================
# CONFIGURAÇÃO DE LOGGING (Estruturado JSON-like)
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "module": "%(name)s", "message": "%(message)s"}',
    datefmt='%Y-%m-%dT%H:%M:%S%z',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ingestor")

# ============================================================================
# CONFIGURAÇÕES (via ENV)
# ============================================================================
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_STREAM = os.getenv("REDIS_STREAM", "stream:market:btc_usdt")
SYMBOL = os.getenv("SYMBOL", "BTC/USDT")
EXCHANGE_ID = os.getenv("EXCHANGE_ID", "binance")

# Backoff exponencial
INITIAL_BACKOFF = 1  # segundos
MAX_BACKOFF = 60     # segundos máximos entre reconexões
BACKOFF_MULTIPLIER = 2


# ============================================================================
# CLASSE PRINCIPAL: Ingestor
# ============================================================================
class MarketIngestor:
    """
    Ingestor de trades em tempo real via WebSocket.
    
    Responsabilidades:
    - Conectar à exchange via ccxt.pro (modo assíncrono)
    - Capturar trades do par configurado
    - Injetar no Redis Stream com payload normalizado
    - Reconectar automaticamente com backoff exponencial
    """

    def __init__(self) -> None:
        self.exchange: ccxtpro.Exchange | None = None
        self.redis: aioredis.Redis | None = None
        self.running: bool = False
        self.backoff: float = INITIAL_BACKOFF
        self.trades_count: int = 0
        self.last_trade_time: datetime | None = None

    async def connect_exchange(self) -> None:
        """Inicializa conexão com a exchange."""
        exchange_class = getattr(ccxtpro, EXCHANGE_ID)
        self.exchange = exchange_class({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
            }
        })
        logger.info(f"Exchange conectada: {EXCHANGE_ID.upper()} (Spot)")

    async def connect_redis(self) -> None:
        """Inicializa conexão com Redis."""
        self.redis = aioredis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            decode_responses=True
        )
        # Teste de conexão
        await self.redis.ping()
        logger.info(f"Redis conectado: {REDIS_HOST}:{REDIS_PORT}")

    async def process_trade(self, trade: dict[str, Any]) -> None:
        """
        Processa um trade e injeta no Redis Stream.
        
        Payload normalizado:
        - price: float (preço do trade)
        - amount: float (quantidade)
        - side: str ('buy' ou 'sell')
        - timestamp: int (Unix ms)
        """
        if not self.redis:
            return

        payload = {
            'price': str(trade['price']),
            'amount': str(trade['amount']),
            'side': trade['side'],
            'timestamp': str(trade['timestamp']),
            'symbol': trade['symbol'],
            'trade_id': str(trade.get('id', '')),
        }

        # XADD ao stream (Redis gerencia IDs automaticamente)
        await self.redis.xadd(
            REDIS_STREAM,
            payload,
            maxlen=100000  # Limite para evitar crescimento infinito
        )

        self.trades_count += 1
        self.last_trade_time = datetime.now(timezone.utc)

        # Log a cada 100 trades para não sobrecarregar
        if self.trades_count % 100 == 0:
            logger.info(
                f"Trades processados: {self.trades_count} | "
                f"Último: {payload['price']} @ {payload['side']}"
            )

    async def watch_trades_loop(self) -> None:
        """
        Loop principal de captura de trades.
        Implementa reconexão com backoff exponencial.
        """
        while self.running:
            try:
                if not self.exchange:
                    await self.connect_exchange()

                logger.info(f"Iniciando watch_trades: {SYMBOL}")
                self.backoff = INITIAL_BACKOFF  # Reset backoff on success

                while self.running:
                    trades = await self.exchange.watch_trades(SYMBOL)
                    for trade in trades:
                        await self.process_trade(trade)

            except ccxtpro.NetworkError as e:
                logger.warning(f"Erro de rede: {e}. Reconectando em {self.backoff}s...")
                await asyncio.sleep(self.backoff)
                self.backoff = min(self.backoff * BACKOFF_MULTIPLIER, MAX_BACKOFF)

            except ccxtpro.ExchangeError as e:
                logger.error(f"Erro da exchange: {e}. Reconectando em {self.backoff}s...")
                await asyncio.sleep(self.backoff)
                self.backoff = min(self.backoff * BACKOFF_MULTIPLIER, MAX_BACKOFF)

            except Exception as e:
                logger.exception(f"Erro inesperado: {e}")
                await asyncio.sleep(self.backoff)
                self.backoff = min(self.backoff * BACKOFF_MULTIPLIER, MAX_BACKOFF)

            finally:
                # Cleanup da conexão WebSocket
                if self.exchange:
                    try:
                        await self.exchange.close()
                    except Exception:
                        pass
                    self.exchange = None

    async def start(self) -> None:
        """Inicia o ingestor."""
        logger.info("=" * 60)
        logger.info("VOSTOK-1 :: Módulo Ingestor Iniciando")
        logger.info(f"Symbol: {SYMBOL} | Stream: {REDIS_STREAM}")
        logger.info("=" * 60)

        self.running = True

        # Conectar Redis primeiro
        await self.connect_redis()

        # Iniciar loop de trades
        await self.watch_trades_loop()

    async def stop(self) -> None:
        """Para o ingestor graciosamente."""
        logger.info("Parando ingestor...")
        self.running = False

        if self.exchange:
            await self.exchange.close()
        if self.redis:
            await self.redis.close()

        logger.info(f"Ingestor parado. Total trades: {self.trades_count}")


# ============================================================================
# ENTRY POINT
# ============================================================================
async def main() -> None:
    """Entry point principal."""
    ingestor = MarketIngestor()

    # Graceful shutdown
    loop = asyncio.get_running_loop()

    def shutdown_handler() -> None:
        asyncio.create_task(ingestor.stop())

    # Nota: signal handlers não funcionam no Windows da mesma forma
    # Em produção Linux, adicionar: loop.add_signal_handler(signal.SIGTERM, shutdown_handler)

    try:
        await ingestor.start()
    except KeyboardInterrupt:
        await ingestor.stop()


if __name__ == "__main__":
    asyncio.run(main())
