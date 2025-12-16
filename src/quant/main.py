"""
VOSTOK-1 :: Módulo Quant Processor
==================================
Agrega ticks em velas OHLCV, calcula indicadores técnicos (RSI, MACD, BB)
e publica sinais no Redis Streams.

Arquiteto: Petrovich | Operador: Vostok
Stack: Python 3.11 + numpy + TA-Lib + redis-py
"""

import asyncio
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import redis.asyncio as aioredis

# TA-Lib import (pode falhar se não instalado corretamente)
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logging.warning("TA-Lib não disponível. Usando cálculos fallback.")

# ============================================================================
# CONFIGURAÇÃO DE LOGGING (Estruturado JSON-like)
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "module": "%(name)s", "message": "%(message)s"}',
    datefmt='%Y-%m-%dT%H:%M:%S%z',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("quant")

# ============================================================================
# CONFIGURAÇÕES (via ENV)
# ============================================================================
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
INPUT_STREAM = os.getenv("INPUT_STREAM", "stream:market:btc_usdt")
OUTPUT_STREAM = os.getenv("OUTPUT_STREAM", "stream:signals:tech")
CONSUMER_GROUP = os.getenv("CONSUMER_GROUP", "quant_group")
CONSUMER_NAME = os.getenv("CONSUMER_NAME", "quant_worker_1")

# Parâmetros de indicadores
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_STD = 2

# Buffer de velas
MAX_CANDLES = 100  # Manter últimas 100 velas para cálculos


# ============================================================================
# CANDLE MANAGER - Agregação de Ticks em OHLCV
# ============================================================================
class CandleManager:
    """
    Gerencia a agregação de ticks em velas OHLCV de 1 minuto.
    
    Atributos:
        current_candle: Vela atual em construção
        candles: Buffer numpy de velas fechadas (OHLCV)
    """

    def __init__(self, max_candles: int = MAX_CANDLES) -> None:
        self.max_candles = max_candles
        self.current_candle: dict[str, Any] | None = None
        self.current_minute: int | None = None
        
        # Buffer numpy para velas fechadas: [timestamp, open, high, low, close, volume]
        self.candles = np.zeros((0, 6), dtype=np.float64)
        
        self.ticks_processed = 0
        self.candles_closed = 0

    def _get_minute_timestamp(self, ts_ms: int) -> int:
        """Retorna o timestamp truncado para o minuto (em ms)."""
        return (ts_ms // 60000) * 60000

    def process_tick(self, price: float, amount: float, timestamp_ms: int) -> dict[str, Any] | None:
        """
        Processa um tick e retorna a vela fechada se houve fechamento.
        
        Args:
            price: Preço do trade
            amount: Volume do trade
            timestamp_ms: Timestamp em milissegundos
            
        Returns:
            Vela fechada (dict) ou None se não houve fechamento
        """
        minute_ts = self._get_minute_timestamp(timestamp_ms)
        self.ticks_processed += 1
        
        # Primeira vela
        if self.current_candle is None:
            self.current_minute = minute_ts
            self.current_candle = {
                'timestamp': minute_ts,
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': amount
            }
            return None
        
        # Verificar se cruzou o minuto (nova vela)
        if minute_ts > self.current_minute:
            # Verificar tick fora de ordem (timestamp antigo)
            if minute_ts < self.current_minute:
                logger.warning(f"Tick fora de ordem: {timestamp_ms} < {self.current_minute}")
                return None
            
            # Fechar vela atual
            closed_candle = self.current_candle.copy()
            self._add_candle_to_buffer(closed_candle)
            self.candles_closed += 1
            
            # Iniciar nova vela
            self.current_minute = minute_ts
            self.current_candle = {
                'timestamp': minute_ts,
                'open': price,
                'high': price,
                'low': price,
                'close': price,
                'volume': amount
            }
            
            return closed_candle
        
        # Atualizar vela atual
        self.current_candle['high'] = max(self.current_candle['high'], price)
        self.current_candle['low'] = min(self.current_candle['low'], price)
        self.current_candle['close'] = price
        self.current_candle['volume'] += amount
        
        return None

    def _add_candle_to_buffer(self, candle: dict[str, Any]) -> None:
        """Adiciona vela fechada ao buffer numpy."""
        new_row = np.array([[
            candle['timestamp'],
            candle['open'],
            candle['high'],
            candle['low'],
            candle['close'],
            candle['volume']
        ]], dtype=np.float64)
        
        self.candles = np.vstack([self.candles, new_row])
        
        # Manter apenas últimas N velas
        if len(self.candles) > self.max_candles:
            self.candles = self.candles[-self.max_candles:]

    def get_closes(self) -> np.ndarray:
        """Retorna array de preços de fechamento."""
        if len(self.candles) == 0:
            return np.array([])
        return self.candles[:, 4]  # Coluna 4 = close

    def get_highs(self) -> np.ndarray:
        """Retorna array de preços máximos."""
        if len(self.candles) == 0:
            return np.array([])
        return self.candles[:, 2]  # Coluna 2 = high

    def get_lows(self) -> np.ndarray:
        """Retorna array de preços mínimos."""
        if len(self.candles) == 0:
            return np.array([])
        return self.candles[:, 3]  # Coluna 3 = low

    def has_enough_data(self, min_candles: int = 26) -> bool:
        """Verifica se há dados suficientes para cálculos."""
        return len(self.candles) >= min_candles


# ============================================================================
# INDICATOR CALCULATOR - Cálculo de Indicadores Técnicos
# ============================================================================
class IndicatorCalculator:
    """Calcula indicadores técnicos usando TA-Lib ou fallback."""

    @staticmethod
    def calculate_rsi(closes: np.ndarray, period: int = RSI_PERIOD) -> float | None:
        """Calcula RSI (Relative Strength Index)."""
        if len(closes) < period + 1:
            return None
        
        if TALIB_AVAILABLE:
            rsi = talib.RSI(closes, timeperiod=period)
            return float(rsi[-1]) if not np.isnan(rsi[-1]) else None
        
        # Fallback: cálculo manual
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_macd(closes: np.ndarray) -> tuple[float | None, float | None, float | None]:
        """Calcula MACD (line, signal, histogram)."""
        if len(closes) < MACD_SLOW + MACD_SIGNAL:
            return None, None, None
        
        if TALIB_AVAILABLE:
            macd, signal, hist = talib.MACD(
                closes, 
                fastperiod=MACD_FAST, 
                slowperiod=MACD_SLOW, 
                signalperiod=MACD_SIGNAL
            )
            return (
                float(macd[-1]) if not np.isnan(macd[-1]) else None,
                float(signal[-1]) if not np.isnan(signal[-1]) else None,
                float(hist[-1]) if not np.isnan(hist[-1]) else None
            )
        
        # Fallback: EMA manual
        ema_fast = IndicatorCalculator._ema(closes, MACD_FAST)
        ema_slow = IndicatorCalculator._ema(closes, MACD_SLOW)
        macd_line = ema_fast - ema_slow
        signal_line = IndicatorCalculator._ema(
            np.array([macd_line]), MACD_SIGNAL
        ) if len(closes) >= MACD_SLOW + MACD_SIGNAL else 0
        
        return macd_line, signal_line, macd_line - signal_line

    @staticmethod
    def calculate_bollinger(closes: np.ndarray) -> tuple[float | None, float | None, float | None]:
        """Calcula Bollinger Bands (upper, middle, lower)."""
        if len(closes) < BB_PERIOD:
            return None, None, None
        
        if TALIB_AVAILABLE:
            upper, middle, lower = talib.BBANDS(
                closes,
                timeperiod=BB_PERIOD,
                nbdevup=BB_STD,
                nbdevdn=BB_STD
            )
            return (
                float(upper[-1]) if not np.isnan(upper[-1]) else None,
                float(middle[-1]) if not np.isnan(middle[-1]) else None,
                float(lower[-1]) if not np.isnan(lower[-1]) else None
            )
        
        # Fallback: cálculo manual
        sma = np.mean(closes[-BB_PERIOD:])
        std = np.std(closes[-BB_PERIOD:])
        
        return sma + (BB_STD * std), sma, sma - (BB_STD * std)

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> float:
        """Calcula EMA (Exponential Moving Average)."""
        if len(data) < period:
            return float(np.mean(data))
        
        multiplier = 2 / (period + 1)
        ema = data[0]
        for price in data[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        return ema


# ============================================================================
# QUANT PROCESSOR - Classe Principal
# ============================================================================
class QuantProcessor:
    """
    Processador Quantitativo: lê ticks, agrega em velas, calcula indicadores.
    
    Usa Consumer Groups do Redis para garantir:
    - Não processar trades repetidos após restart
    - Possibilidade de escalar horizontalmente
    """

    def __init__(self) -> None:
        self.redis: aioredis.Redis | None = None
        self.candle_manager = CandleManager()
        self.indicator_calc = IndicatorCalculator()
        self.running = False
        self.signals_published = 0

    async def connect_redis(self) -> None:
        """Inicializa conexão com Redis."""
        self.redis = aioredis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            decode_responses=True
        )
        await self.redis.ping()
        logger.info(f"Redis conectado: {REDIS_HOST}:{REDIS_PORT}")

    async def setup_consumer_group(self) -> None:
        """Cria o Consumer Group se não existir."""
        try:
            # Criar grupo começando do ID mais recente ($)
            # Isso significa que só processará mensagens novas
            await self.redis.xgroup_create(
                INPUT_STREAM, 
                CONSUMER_GROUP, 
                id='$',  # Apenas novas mensagens
                mkstream=True
            )
            logger.info(f"Consumer group '{CONSUMER_GROUP}' criado")
        except aioredis.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.info(f"Consumer group '{CONSUMER_GROUP}' já existe")
            else:
                raise

    async def process_message(self, message_id: str, data: dict[str, Any]) -> None:
        """Processa um tick do stream."""
        try:
            price = float(data.get('price', 0))
            amount = float(data.get('amount', 0))
            timestamp = int(data.get('timestamp', 0))
            
            if price <= 0 or timestamp <= 0:
                return
            
            # Processar tick no CandleManager
            closed_candle = self.candle_manager.process_tick(price, amount, timestamp)
            
            # Se uma vela fechou, calcular indicadores
            if closed_candle is not None:
                await self.calculate_and_publish_signals(closed_candle)
                
        except (ValueError, TypeError) as e:
            logger.warning(f"Erro ao processar tick {message_id}: {e}")

    async def calculate_and_publish_signals(self, candle: dict[str, Any]) -> None:
        """Calcula indicadores e publica sinal técnico."""
        if not self.candle_manager.has_enough_data():
            logger.debug(f"Aguardando mais dados... ({len(self.candle_manager.candles)} velas)")
            return
        
        start_time = time.perf_counter()
        
        closes = self.candle_manager.get_closes()
        
        # Calcular indicadores
        rsi = self.indicator_calc.calculate_rsi(closes)
        macd, macd_signal, macd_hist = self.indicator_calc.calculate_macd(closes)
        bb_upper, bb_middle, bb_lower = self.indicator_calc.calculate_bollinger(closes)
        
        calc_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Montar payload do sinal
        signal_payload = {
            'timestamp': str(candle['timestamp']),
            'close': str(candle['close']),
            'rsi': str(round(rsi, 2)) if rsi else '',
            'macd': str(round(macd, 4)) if macd else '',
            'macd_signal': str(round(macd_signal, 4)) if macd_signal else '',
            'macd_hist': str(round(macd_hist, 4)) if macd_hist else '',
            'bb_upper': str(round(bb_upper, 2)) if bb_upper else '',
            'bb_middle': str(round(bb_middle, 2)) if bb_middle else '',
            'bb_lower': str(round(bb_lower, 2)) if bb_lower else '',
            'calc_time_ms': str(round(calc_time_ms, 2))
        }
        
        # Publicar no stream de sinais
        await self.redis.xadd(
            OUTPUT_STREAM,
            signal_payload,
            maxlen=10000  # Manter últimos 10k sinais
        )
        
        self.signals_published += 1
        
        # Log a cada sinal (ou a cada N sinais para produção)
        logger.info(
            f"Sinal #{self.signals_published} | "
            f"RSI: {rsi:.1f if rsi else 'N/A'} | "
            f"MACD: {macd:.2f if macd else 'N/A'} | "
            f"BB: [{bb_lower:.0f if bb_lower else 0}-{bb_upper:.0f if bb_upper else 0}] | "
            f"Calc: {calc_time_ms:.2f}ms"
        )

    async def consume_stream(self) -> None:
        """Loop principal de consumo do stream."""
        while self.running:
            try:
                # Ler do consumer group
                # '>' significa apenas mensagens não entregues a nenhum consumer
                messages = await self.redis.xreadgroup(
                    groupname=CONSUMER_GROUP,
                    consumername=CONSUMER_NAME,
                    streams={INPUT_STREAM: '>'},
                    count=100,  # Processar em lotes
                    block=1000  # Bloquear por 1s se não houver mensagens
                )
                
                if messages:
                    for stream_name, stream_messages in messages:
                        for message_id, data in stream_messages:
                            await self.process_message(message_id, data)
                            # ACK da mensagem processada
                            await self.redis.xack(INPUT_STREAM, CONSUMER_GROUP, message_id)
                            
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
        logger.info("VOSTOK-1 :: Módulo Quant Processor Iniciando")
        logger.info(f"Input: {INPUT_STREAM} | Output: {OUTPUT_STREAM}")
        logger.info(f"TA-Lib: {'Disponível' if TALIB_AVAILABLE else 'Fallback'}")
        logger.info("=" * 60)
        
        self.running = True
        
        await self.connect_redis()
        await self.setup_consumer_group()
        await self.consume_stream()

    async def stop(self) -> None:
        """Para o processador."""
        logger.info("Parando Quant Processor...")
        self.running = False
        
        if self.redis:
            await self.redis.close()
        
        logger.info(
            f"Quant parado. Ticks: {self.candle_manager.ticks_processed} | "
            f"Velas: {self.candle_manager.candles_closed} | "
            f"Sinais: {self.signals_published}"
        )


# ============================================================================
# ENTRY POINT
# ============================================================================
async def main() -> None:
    """Entry point principal."""
    processor = QuantProcessor()
    
    try:
        await processor.start()
    except KeyboardInterrupt:
        await processor.stop()


if __name__ == "__main__":
    asyncio.run(main())
