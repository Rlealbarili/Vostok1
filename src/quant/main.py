"""
VOSTOK-1 :: Módulo Quant Processor (Sniper Upgrade)
====================================================
Agrega ticks em velas OHLCV, calcula indicadores técnicos avançados
(RSI, MACD, BB, ATR, CVD, Entropia) e publica sinais no Redis Streams.

Arquiteto: Petrovich | Operador: Vostok
Stack: Python 3.11 + numpy + TA-Lib + redis-py + scipy
"""

import asyncio
import logging
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import numpy as np
import redis.asyncio as aioredis

# TA-Lib import
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logging.warning("TA-Lib não disponível. Usando cálculos fallback.")

# ============================================================================
# CONFIGURAÇÃO DE LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "module": "%(name)s", "message": "%(message)s"}',
    datefmt='%Y-%m-%dT%H:%M:%S%z',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("quant")

# ============================================================================
# CONFIGURAÇÕES
# ============================================================================
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
INPUT_STREAM = os.getenv("INPUT_STREAM", "stream:market:btc_usdt")
OUTPUT_STREAM = os.getenv("OUTPUT_STREAM", "stream:signals:tech")
LIVE_STREAM = os.getenv("LIVE_STREAM", "stream:signals:live")
CONSUMER_GROUP = os.getenv("CONSUMER_GROUP", "quant_group")
CONSUMER_NAME = os.getenv("CONSUMER_NAME", "quant_worker_1")

# Parâmetros de indicadores
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_STD = 2
ATR_PERIOD = 14

MAX_CANDLES = 100


# ============================================================================
# CANDLE MANAGER (Sniper Upgrade) - OHLCV + CVD + Entropia
# ============================================================================
class CandleManager:
    """
    Gerencia agregação de ticks em velas OHLCV com métricas avançadas.
    
    Sniper Upgrade:
    - CVD (Cumulative Volume Delta): buy_volume - sell_volume
    - Entropia de Shannon: dispersão de preços (detector de ruído)
    """

    def __init__(self, max_candles: int = MAX_CANDLES) -> None:
        self.max_candles = max_candles
        self.current_candle: dict[str, Any] | None = None
        self.current_minute: int | None = None
        
        # Buffer numpy: [timestamp, open, high, low, close, volume, buy_vol, sell_vol, cvd, entropy]
        # Índices: 0=ts, 1=open, 2=high, 3=low, 4=close, 5=vol, 6=buy_vol, 7=sell_vol, 8=cvd, 9=entropy
        self.candles = np.zeros((0, 10), dtype=np.float64)
        
        # Ticks da vela atual (para cálculo de entropia)
        self.current_ticks: list[float] = []
        
        # Último funding rate conhecido
        self.last_funding_rate: float | None = None
        
        self.ticks_processed = 0
        self.candles_closed = 0

    def _get_minute_timestamp(self, ts_ms: int) -> int:
        """Retorna timestamp truncado para o minuto."""
        return (ts_ms // 60000) * 60000

    def _calculate_entropy(self, prices: list[float]) -> float:
        """
        Calcula Entropia de Shannon sobre os preços dos ticks.
        
        Alta entropia = preços dispersos/aleatórios (ruído)
        Baixa entropia = preços direcionais (tendência)
        
        Retorna valor normalizado entre 0 e 1.
        """
        if len(prices) < 2:
            return 0.0
        
        # Discretizar preços em bins relativos
        prices_arr = np.array(prices)
        
        # Calcular retornos (mudanças de preço)
        returns = np.diff(prices_arr) / prices_arr[:-1]
        
        if len(returns) == 0:
            return 0.0
        
        # Categorizar retornos em bins
        # -1 = queda forte, 0 = neutro, 1 = alta forte
        bins = np.digitize(returns, [-0.0005, -0.0001, 0.0001, 0.0005]) - 2
        
        # Contar frequência de cada bin
        unique, counts = np.unique(bins, return_counts=True)
        probabilities = counts / len(bins)
        
        # Calcular entropia de Shannon: H = -Σ p(x) * log2(p(x))
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        
        # Normalizar para 0-1 (max entropia para 5 bins = log2(5) ≈ 2.32)
        max_entropy = np.log2(5)
        normalized_entropy = min(entropy / max_entropy, 1.0)
        
        return round(normalized_entropy, 4)

    def process_tick(self, data: dict[str, Any]) -> dict[str, Any] | None:
        """
        Processa um tick do stream.
        
        Suporta dois tipos:
        - type='trade': atualiza vela OHLCV + CVD
        - type='funding': atualiza último funding rate
        """
        msg_type = data.get('type', 'trade')
        
        # Processar funding rate separadamente
        if msg_type == 'funding':
            rate = data.get('rate', '0')
            try:
                self.last_funding_rate = float(rate)
            except (ValueError, TypeError):
                pass
            return None
        
        # Processar trade
        try:
            price = float(data.get('price', 0))
            amount = float(data.get('amount', 0))
            timestamp = int(data.get('timestamp', 0))
            side = data.get('side', 'buy')
        except (ValueError, TypeError):
            return None
        
        if price <= 0 or timestamp <= 0:
            return None
        
        minute_ts = self._get_minute_timestamp(timestamp)
        self.ticks_processed += 1
        
        # Primeira vela
        if self.current_candle is None:
            self._start_new_candle(minute_ts, price, amount, side)
            return None
        
        # Verificar cruzamento de minuto
        if minute_ts > self.current_minute:
            # Tick fora de ordem (passado)
            if minute_ts < self.current_minute:
                logger.warning(f"Tick fora de ordem: {timestamp}")
                return None
            
            # Fechar vela atual
            closed_candle = self._close_current_candle()
            
            # Iniciar nova vela
            self._start_new_candle(minute_ts, price, amount, side)
            
            return closed_candle
        
        # Atualizar vela atual
        self._update_current_candle(price, amount, side)
        
        return None

    def _start_new_candle(self, minute_ts: int, price: float, amount: float, side: str) -> None:
        """Inicia uma nova vela."""
        self.current_minute = minute_ts
        self.current_ticks = [price]
        
        buy_vol = amount if side == 'buy' else 0.0
        sell_vol = amount if side == 'sell' else 0.0
        
        self.current_candle = {
            'timestamp': minute_ts,
            'open': price,
            'high': price,
            'low': price,
            'close': price,
            'volume': amount,
            'buy_volume': buy_vol,
            'sell_volume': sell_vol,
        }

    def _update_current_candle(self, price: float, amount: float, side: str) -> None:
        """Atualiza a vela atual com novo tick."""
        self.current_candle['high'] = max(self.current_candle['high'], price)
        self.current_candle['low'] = min(self.current_candle['low'], price)
        self.current_candle['close'] = price
        self.current_candle['volume'] += amount
        
        if side == 'buy':
            self.current_candle['buy_volume'] += amount
        else:
            self.current_candle['sell_volume'] += amount
        
        # Armazenar tick para cálculo de entropia
        self.current_ticks.append(price)

    def _close_current_candle(self) -> dict[str, Any]:
        """Fecha a vela atual e adiciona ao buffer."""
        candle = self.current_candle.copy()
        
        # Calcular CVD (Delta)
        cvd = candle['buy_volume'] - candle['sell_volume']
        candle['cvd'] = cvd
        
        # Calcular Entropia
        entropy = self._calculate_entropy(self.current_ticks)
        candle['entropy'] = entropy
        
        # Adicionar funding rate mais recente
        candle['funding_rate'] = self.last_funding_rate
        
        # Adicionar ao buffer numpy
        self._add_candle_to_buffer(candle)
        self.candles_closed += 1
        
        return candle

    def _add_candle_to_buffer(self, candle: dict[str, Any]) -> None:
        """Adiciona vela ao buffer numpy."""
        new_row = np.array([[
            candle['timestamp'],
            candle['open'],
            candle['high'],
            candle['low'],
            candle['close'],
            candle['volume'],
            candle['buy_volume'],
            candle['sell_volume'],
            candle['cvd'],
            candle['entropy']
        ]], dtype=np.float64)
        
        self.candles = np.vstack([self.candles, new_row])
        
        if len(self.candles) > self.max_candles:
            self.candles = self.candles[-self.max_candles:]

    def get_closes(self) -> np.ndarray:
        """Retorna preços de fechamento."""
        return self.candles[:, 4] if len(self.candles) > 0 else np.array([])

    def get_highs(self) -> np.ndarray:
        """Retorna preços máximos."""
        return self.candles[:, 2] if len(self.candles) > 0 else np.array([])

    def get_lows(self) -> np.ndarray:
        """Retorna preços mínimos."""
        return self.candles[:, 3] if len(self.candles) > 0 else np.array([])

    def has_enough_data(self, min_candles: int = 26) -> bool:
        """Verifica se há dados suficientes."""
        return len(self.candles) >= min_candles


# ============================================================================
# INDICATOR CALCULATOR (Sniper Upgrade) - ATR + Parkinson Volatility
# ============================================================================
class IndicatorCalculator:
    """Calcula indicadores técnicos incluindo métricas de regime."""

    @staticmethod
    def calculate_rsi(closes: np.ndarray, period: int = RSI_PERIOD) -> float | None:
        """Calcula RSI."""
        if len(closes) < period + 1:
            return None
        
        if TALIB_AVAILABLE:
            rsi = talib.RSI(closes, timeperiod=period)
            return float(rsi[-1]) if not np.isnan(rsi[-1]) else None
        
        # Fallback
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
                closes, fastperiod=MACD_FAST, slowperiod=MACD_SLOW, signalperiod=MACD_SIGNAL
            )
            return (
                float(macd[-1]) if not np.isnan(macd[-1]) else None,
                float(signal[-1]) if not np.isnan(signal[-1]) else None,
                float(hist[-1]) if not np.isnan(hist[-1]) else None
            )
        return None, None, None

    @staticmethod
    def calculate_bollinger(closes: np.ndarray) -> tuple[float | None, float | None, float | None]:
        """Calcula Bollinger Bands."""
        if len(closes) < BB_PERIOD:
            return None, None, None
        
        if TALIB_AVAILABLE:
            upper, middle, lower = talib.BBANDS(
                closes, timeperiod=BB_PERIOD, nbdevup=BB_STD, nbdevdn=BB_STD
            )
            return (
                float(upper[-1]) if not np.isnan(upper[-1]) else None,
                float(middle[-1]) if not np.isnan(middle[-1]) else None,
                float(lower[-1]) if not np.isnan(lower[-1]) else None
            )
        
        sma = np.mean(closes[-BB_PERIOD:])
        std = np.std(closes[-BB_PERIOD:])
        return sma + (BB_STD * std), sma, sma - (BB_STD * std)

    @staticmethod
    def calculate_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, 
                      period: int = ATR_PERIOD) -> float | None:
        """
        Calcula ATR (Average True Range) - medida de volatilidade.
        """
        if len(closes) < period + 1:
            return None
        
        if TALIB_AVAILABLE:
            atr = talib.ATR(highs, lows, closes, timeperiod=period)
            return float(atr[-1]) if not np.isnan(atr[-1]) else None
        
        # Fallback: cálculo manual
        tr_list = []
        for i in range(1, len(closes)):
            high_low = highs[i] - lows[i]
            high_close = abs(highs[i] - closes[i-1])
            low_close = abs(lows[i] - closes[i-1])
            tr = max(high_low, high_close, low_close)
            tr_list.append(tr)
        
        if len(tr_list) < period:
            return None
        
        return float(np.mean(tr_list[-period:]))

    @staticmethod
    def calculate_parkinson_volatility(highs: np.ndarray, lows: np.ndarray, 
                                        period: int = ATR_PERIOD) -> float | None:
        """
        Calcula Volatilidade de Parkinson (mais eficiente para cripto).
        
        Usa apenas High/Low, não precisa do Close.
        Fórmula: σ² = (1/4ln2) * Σ(ln(H/L))²
        """
        if len(highs) < period:
            return None
        
        # Usar últimos N períodos
        h = highs[-period:]
        l = lows[-period:]
        
        # Evitar divisão por zero e log de zero
        with np.errstate(divide='ignore', invalid='ignore'):
            log_hl = np.log(h / l)
            log_hl = np.where(np.isfinite(log_hl), log_hl, 0)
        
        # Variância de Parkinson
        variance = (1 / (4 * np.log(2))) * np.mean(log_hl ** 2)
        
        # Volatilidade = sqrt(variance) * sqrt(252) para anualizar (ou *100 para %)
        volatility = np.sqrt(variance) * 100  # Em percentual
        
        return round(float(volatility), 4) if np.isfinite(volatility) else None


# ============================================================================
# QUANT PROCESSOR (Sniper Upgrade)
# ============================================================================
class QuantProcessor:
    """Processador Quantitativo com métricas avançadas de regime e order flow."""

    def __init__(self) -> None:
        self.redis: aioredis.Redis | None = None
        self.candle_manager = CandleManager()
        self.indicator_calc = IndicatorCalculator()
        self.running = False
        self.signals_published = 0
        self.live_pulses_published = 0

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
            await self.redis.xgroup_create(INPUT_STREAM, CONSUMER_GROUP, id='$', mkstream=True)
            logger.info(f"Consumer group '{CONSUMER_GROUP}' criado")
        except aioredis.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.info(f"Consumer group '{CONSUMER_GROUP}' já existe")
            else:
                raise

    async def process_message(self, message_id: str, data: dict[str, Any]) -> None:
        """Processa mensagem do stream e publica live_pulse."""
        try:
            # Extrair dados do tick para live_pulse (antes de process_tick)
            tick_price = data.get('price')
            tick_timestamp = data.get('timestamp')
            msg_type = data.get('type', 'trade')
            
            closed_candle = self.candle_manager.process_tick(data)
            
            # Publicar live_pulse para cada trade tick (não funding)
            if msg_type == 'trade' and tick_price and self.candle_manager.current_candle:
                await self.publish_live_pulse(tick_price, tick_timestamp)
            
            # Publicar sinal completo no fechamento da vela
            if closed_candle is not None:
                await self.calculate_and_publish_signals(closed_candle)
                
        except Exception as e:
            logger.warning(f"Erro ao processar {message_id}: {e}")

    async def publish_live_pulse(self, price: str, timestamp: str) -> None:
        """Publica live_pulse com dados parciais da vela atual."""
        try:
            candle = self.candle_manager.current_candle
            if not candle:
                return
            
            # CVD acumulado parcial da vela atual
            cvd_current = candle.get('buy_volume', 0) - candle.get('sell_volume', 0)
            
            live_payload = {
                'type': 'live_pulse',
                'price': str(price),
                'cvd_current': str(round(cvd_current, 6)),
                'timestamp': str(timestamp),
                'candle_high': str(round(candle.get('high', 0), 2)),
                'candle_low': str(round(candle.get('low', 0), 2)),
            }
            
            # Publicar com maxlen pequeno (só precisamos do último)
            await self.redis.xadd(LIVE_STREAM, live_payload, maxlen=100)
            self.live_pulses_published += 1
            
        except Exception:
            pass  # Não logar cada pulse para evitar spam

    async def calculate_and_publish_signals(self, candle: dict[str, Any]) -> None:
        """Calcula indicadores e publica sinal técnico completo."""
        if not self.candle_manager.has_enough_data():
            logger.debug(f"Aguardando dados... ({len(self.candle_manager.candles)} velas)")
            return
        
        start_time = time.perf_counter()
        
        closes = self.candle_manager.get_closes()
        highs = self.candle_manager.get_highs()
        lows = self.candle_manager.get_lows()
        
        # Indicadores clássicos
        rsi = self.indicator_calc.calculate_rsi(closes)
        macd, macd_signal, macd_hist = self.indicator_calc.calculate_macd(closes)
        bb_upper, bb_middle, bb_lower = self.indicator_calc.calculate_bollinger(closes)
        
        # Indicadores de regime (Sniper Upgrade)
        atr = self.indicator_calc.calculate_atr(highs, lows, closes)
        parkinson_vol = self.indicator_calc.calculate_parkinson_volatility(highs, lows)
        
        calc_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Payload completo (Sniper Format)
        signal_payload = {
            # Vela OHLCV
            'timestamp': str(candle['timestamp']),
            'open': str(round(candle['open'], 2)),
            'high': str(round(candle['high'], 2)),
            'low': str(round(candle['low'], 2)),
            'close': str(round(candle['close'], 2)),
            'volume': str(round(candle['volume'], 6)),
            
            # Order Flow (Sniper)
            'cvd_absolute': str(round(candle['cvd'], 6)),
            'buy_volume': str(round(candle['buy_volume'], 6)),
            'sell_volume': str(round(candle['sell_volume'], 6)),
            
            # Regime Detection (Sniper)
            'entropy': str(candle['entropy']),
            'volatility_atr': str(round(atr, 2)) if atr else '',
            'volatility_parkinson': str(parkinson_vol) if parkinson_vol else '',
            'funding_rate': str(candle['funding_rate']) if candle['funding_rate'] else '',
            
            # Indicadores clássicos
            'rsi': str(round(rsi, 2)) if rsi else '',
            'macd': str(round(macd, 4)) if macd else '',
            'macd_signal': str(round(macd_signal, 4)) if macd_signal else '',
            'macd_hist': str(round(macd_hist, 4)) if macd_hist else '',
            'bb_upper': str(round(bb_upper, 2)) if bb_upper else '',
            'bb_middle': str(round(bb_middle, 2)) if bb_middle else '',
            'bb_lower': str(round(bb_lower, 2)) if bb_lower else '',
            
            # Meta
            'calc_time_ms': str(round(calc_time_ms, 2))
        }
        
        await self.redis.xadd(OUTPUT_STREAM, signal_payload, maxlen=10000)
        self.signals_published += 1
        
        # Log conciso
        cvd_sign = '+' if candle['cvd'] >= 0 else ''
        logger.info(
            f"Sinal #{self.signals_published} | "
            f"Close: {candle['close']:.0f} | "
            f"CVD: {cvd_sign}{candle['cvd']:.4f} | "
            f"Entropy: {candle['entropy']:.3f} | "
            f"RSI: {rsi:.1f if rsi else 0} | "
            f"ATR: {atr:.0f if atr else 0} | "
            f"{calc_time_ms:.2f}ms"
        )

    async def consume_stream(self) -> None:
        """Loop principal de consumo."""
        while self.running:
            try:
                messages = await self.redis.xreadgroup(
                    groupname=CONSUMER_GROUP,
                    consumername=CONSUMER_NAME,
                    streams={INPUT_STREAM: '>'},
                    count=100,
                    block=1000
                )
                
                if messages:
                    for stream_name, stream_messages in messages:
                        for message_id, data in stream_messages:
                            await self.process_message(message_id, data)
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
        logger.info("VOSTOK-1 :: Quant Processor (Sniper Upgrade)")
        logger.info(f"Input: {INPUT_STREAM} | Output: {OUTPUT_STREAM}")
        logger.info(f"TA-Lib: {'OK' if TALIB_AVAILABLE else 'Fallback'}")
        logger.info("Métricas: OHLCV + CVD + Entropy + ATR + Parkinson + RSI/MACD/BB")
        logger.info("=" * 60)
        
        self.running = True
        await self.connect_redis()
        await self.setup_consumer_group()
        await self.consume_stream()

    async def stop(self) -> None:
        """Para o processador."""
        logger.info("Parando Quant...")
        self.running = False
        if self.redis:
            await self.redis.close()
        logger.info(f"Quant parado. Sinais: {self.signals_published}")


# ============================================================================
# ENTRY POINT
# ============================================================================
async def main() -> None:
    processor = QuantProcessor()
    try:
        await processor.start()
    except KeyboardInterrupt:
        await processor.stop()


if __name__ == "__main__":
    asyncio.run(main())
