"""
VOSTOK-1 :: Módulo Ingestor (Sniper Upgrade)
=============================================
Captura trades + funding rates BTC/USDT da Binance via WebSocket
e injeta no Redis Streams em tempo real.

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
INITIAL_BACKOFF = 1
MAX_BACKOFF = 60
BACKOFF_MULTIPLIER = 2


# ============================================================================
# CLASSE PRINCIPAL: Ingestor (Sniper Upgrade)
# ============================================================================
class MarketIngestor:
    """
    Ingestor de trades e funding rates em tempo real via WebSocket.
    
    Sniper Upgrade:
    - watch_trades: Captura trades (payload type='trade')
    - watch_funding_rate: Captura funding rates (payload type='funding')
    - Execução concorrente com asyncio.gather
    """

    def __init__(self) -> None:
        self.exchange: ccxtpro.Exchange | None = None
        self.redis: aioredis.Redis | None = None
        self.running: bool = False
        self.backoff: float = INITIAL_BACKOFF
        
        # Contadores
        self.trades_count: int = 0
        self.funding_count: int = 0
        self.last_trade_time: datetime | None = None
        self.last_funding_rate: float | None = None

    async def connect_exchange(self) -> ccxtpro.Exchange:
        """Inicializa conexão com a exchange."""
        exchange_class = getattr(ccxtpro, EXCHANGE_ID)
        exchange = exchange_class({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',  # Futures para funding rate
            }
        })
        logger.info(f"Exchange conectada: {EXCHANGE_ID.upper()} (Futures)")
        return exchange

    async def connect_redis(self) -> None:
        """Inicializa conexão com Redis."""
        self.redis = aioredis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            decode_responses=True
        )
        await self.redis.ping()
        logger.info(f"Redis conectado: {REDIS_HOST}:{REDIS_PORT}")

    async def process_trade(self, trade: dict[str, Any]) -> None:
        """
        Processa um trade e injeta no Redis Stream.
        
        Payload:
        - type: 'trade'
        - price, amount, side, timestamp, symbol, trade_id
        """
        if not self.redis:
            return

        payload = {
            'type': 'trade',
            'price': str(trade['price']),
            'amount': str(trade['amount']),
            'side': trade['side'],
            'timestamp': str(trade['timestamp']),
            'symbol': trade['symbol'],
            'trade_id': str(trade.get('id', '')),
        }

        await self.redis.xadd(
            REDIS_STREAM,
            payload,
            maxlen=100000
        )

        self.trades_count += 1
        self.last_trade_time = datetime.now(timezone.utc)

        if self.trades_count % 500 == 0:
            logger.info(
                f"Trades: {self.trades_count} | "
                f"Último: {payload['price']} @ {payload['side']} | "
                f"Funding: {self.funding_count}"
            )

    async def process_funding_rate(self, funding: dict[str, Any]) -> None:
        """
        Processa funding rate e injeta no Redis Stream.
        
        Payload:
        - type: 'funding'
        - rate, next_funding_time, timestamp
        """
        if not self.redis:
            return

        rate = funding.get('fundingRate', 0.0)
        next_time = funding.get('fundingTimestamp', 0)
        
        self.last_funding_rate = rate

        payload = {
            'type': 'funding',
            'rate': str(rate) if rate else '0',
            'next_funding_time': str(next_time),
            'timestamp': str(int(datetime.now(timezone.utc).timestamp() * 1000)),
            'symbol': SYMBOL,
        }

        await self.redis.xadd(
            REDIS_STREAM,
            payload,
            maxlen=100000
        )

        self.funding_count += 1
        logger.info(f"Funding Rate capturado: {rate:.6f} | Next: {next_time}")

    async def watch_trades_loop(self, exchange: ccxtpro.Exchange) -> None:
        """Loop de captura de trades (canal rápido)."""
        backoff = INITIAL_BACKOFF
        
        while self.running:
            try:
                logger.info(f"Iniciando watch_trades: {SYMBOL}")
                backoff = INITIAL_BACKOFF

                while self.running:
                    trades = await exchange.watch_trades(SYMBOL)
                    for trade in trades:
                        await self.process_trade(trade)

            except ccxtpro.NetworkError as e:
                logger.warning(f"[Trades] Erro de rede: {e}. Reconectando em {backoff}s...")
                await asyncio.sleep(backoff)
                backoff = min(backoff * BACKOFF_MULTIPLIER, MAX_BACKOFF)

            except ccxtpro.ExchangeError as e:
                logger.error(f"[Trades] Erro da exchange: {e}")
                await asyncio.sleep(backoff)
                backoff = min(backoff * BACKOFF_MULTIPLIER, MAX_BACKOFF)

            except Exception as e:
                logger.exception(f"[Trades] Erro inesperado: {e}")
                await asyncio.sleep(backoff)
                backoff = min(backoff * BACKOFF_MULTIPLIER, MAX_BACKOFF)

    async def watch_funding_loop(self, exchange: ccxtpro.Exchange) -> None:
        """Loop de captura de funding rates (canal lento - não bloqueia trades)."""
        backoff = INITIAL_BACKOFF
        
        while self.running:
            try:
                logger.info(f"Iniciando watch_funding_rate: {SYMBOL}")
                backoff = INITIAL_BACKOFF

                while self.running:
                    # Funding rate é atualizado a cada 8h, mas monitoramos continuamente
                    try:
                        funding = await asyncio.wait_for(
                            exchange.watch_funding_rate(SYMBOL),
                            timeout=300  # 5 min timeout
                        )
                        if funding:
                            await self.process_funding_rate(funding)
                    except asyncio.TimeoutError:
                        # Normal - funding não muda frequentemente
                        logger.debug("Funding rate timeout - aguardando próxima atualização")
                        continue

            except ccxtpro.NetworkError as e:
                logger.warning(f"[Funding] Erro de rede: {e}. Reconectando em {backoff}s...")
                await asyncio.sleep(backoff)
                backoff = min(backoff * BACKOFF_MULTIPLIER, MAX_BACKOFF)

            except ccxtpro.NotSupported:
                logger.warning("[Funding] watch_funding_rate não suportado. Usando fallback polling...")
                await self.funding_polling_fallback(exchange)
                
            except Exception as e:
                logger.exception(f"[Funding] Erro inesperado: {e}")
                await asyncio.sleep(backoff)
                backoff = min(backoff * BACKOFF_MULTIPLIER, MAX_BACKOFF)

    async def funding_polling_fallback(self, exchange: ccxtpro.Exchange) -> None:
        """Fallback: polling do funding rate a cada 5 minutos."""
        while self.running:
            try:
                # Usar API REST para buscar funding rate
                funding = await exchange.fetch_funding_rate(SYMBOL)
                if funding:
                    await self.process_funding_rate(funding)
                await asyncio.sleep(300)  # Poll a cada 5 min
            except Exception as e:
                logger.warning(f"[Funding Polling] Erro: {e}")
                await asyncio.sleep(60)

    async def start(self) -> None:
        """Inicia o ingestor com múltiplos canais concorrentes."""
        logger.info("=" * 60)
        logger.info("VOSTOK-1 :: Módulo Ingestor (Sniper Upgrade)")
        logger.info(f"Symbol: {SYMBOL} | Stream: {REDIS_STREAM}")
        logger.info("Canais: watch_trades + watch_funding_rate")
        logger.info("=" * 60)

        self.running = True
        await self.connect_redis()
        
        # Criar exchange para cada canal (evita conflitos de estado)
        exchange_trades = await self.connect_exchange()
        exchange_funding = await self.connect_exchange()

        try:
            # Executar ambos os loops concorrentemente
            # asyncio.gather garante que trades (rápido) não é bloqueado por funding (lento)
            await asyncio.gather(
                self.watch_trades_loop(exchange_trades),
                self.watch_funding_loop(exchange_funding),
                return_exceptions=True
            )
        finally:
            await exchange_trades.close()
            await exchange_funding.close()

    async def stop(self) -> None:
        """Para o ingestor graciosamente."""
        logger.info("Parando ingestor...")
        self.running = False

        if self.redis:
            await self.redis.close()

        logger.info(
            f"Ingestor parado. Trades: {self.trades_count} | Funding: {self.funding_count}"
        )


# ============================================================================
# ENTRY POINT
# ============================================================================
async def main() -> None:
    """Entry point principal."""
    ingestor = MarketIngestor()

    try:
        await ingestor.start()
    except KeyboardInterrupt:
        await ingestor.stop()


if __name__ == "__main__":
    asyncio.run(main())
