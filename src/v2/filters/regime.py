"""
VOSTOK V2 :: Regime Filter (Multi-Timeframe)
=============================================
Combina múltiplos indicadores para determinar o regime de mercado.

Componentes:
- Choppiness Index (CHOP): Detecta consolidação
- ADX: Força da tendência
- EMA 200 (H1): Direção da tendência primária

Arquiteto: Petrovich
"""

import logging
from dataclasses import dataclass
from enum import Enum

import pandas as pd

from .choppiness import ChoppinessIndex

logger = logging.getLogger("regime_filter")


class MarketRegime(str, Enum):
    """Regimes de mercado detectados."""
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    DEAD_MARKET = "DEAD"
    HIGH_VOLATILITY = "HIGH_VOL"


@dataclass
class RegimeContext:
    """Contexto do regime atual."""
    regime: MarketRegime
    choppiness: float
    adx: float
    ema_200_h1: float
    trend_bias: str  # "LONG_ONLY" | "SHORT_ONLY" | "NEUTRAL"
    confidence: float
    should_trade: bool


class RegimeFilter:
    """
    Filtro de regime multi-indicador.
    
    Regras:
    1. CHOP > 61.8 → RANGING (Skip trades)
    2. ADX < 15 → DEAD_MARKET (Skip trades)
    3. Price > EMA200 H1 → LONG_ONLY
    4. Price < EMA200 H1 → SHORT_ONLY
    """
    
    def __init__(
        self,
        chop_period: int = 14,
        adx_period: int = 14,
        chop_threshold: float = 61.8,
        adx_min_threshold: float = 15.0,
    ):
        self.choppiness = ChoppinessIndex(period=chop_period)
        self.adx_period = adx_period
        self.chop_threshold = chop_threshold
        self.adx_min = adx_min_threshold
        
        # Cache para EMA H1 (atualizado por outro processo)
        self.ema_200_h1: float = 0.0
    
    def calculate_adx(
        self, 
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series
    ) -> float:
        """
        Calcula ADX simplificado.
        
        TODO: Usar pandas_ta.adx() para implementação completa
        """
        # TODO: Implementar ADX real
        # Por ora, retorna placeholder
        return 25.0
    
    def update_ema_h1(self, ema_value: float):
        """Atualiza EMA 200 do timeframe H1."""
        self.ema_200_h1 = ema_value
    
    def analyze(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        current_price: float,
    ) -> RegimeContext:
        """
        Analisa o regime de mercado atual.
        
        Returns:
            RegimeContext com todas as informações do regime
        """
        # 1. Calcular Choppiness
        chop_series = self.choppiness.calculate(high, low, close)
        current_chop = chop_series.iloc[-1] if len(chop_series) > 0 else 50.0
        
        # 2. Calcular ADX
        adx = self.calculate_adx(high, low, close)
        
        # 3. Determinar regime
        if current_chop > self.chop_threshold:
            regime = MarketRegime.RANGING
            should_trade = False
            confidence = 1.0 - (current_chop - self.chop_threshold) / 38.2
        elif adx < self.adx_min:
            regime = MarketRegime.DEAD_MARKET
            should_trade = False
            confidence = adx / self.adx_min
        elif self.ema_200_h1 > 0:
            if current_price > self.ema_200_h1:
                regime = MarketRegime.TRENDING_UP
            else:
                regime = MarketRegime.TRENDING_DOWN
            should_trade = True
            confidence = min(adx / 50.0, 1.0)
        else:
            # Sem EMA H1, usar apenas CHOP/ADX
            regime = MarketRegime.TRENDING_UP if adx > 25 else MarketRegime.RANGING
            should_trade = adx > 25
            confidence = 0.5
        
        # 4. Determinar bias de direção
        if self.ema_200_h1 > 0:
            if current_price > self.ema_200_h1 * 1.001:  # 0.1% buffer
                trend_bias = "LONG_ONLY"
            elif current_price < self.ema_200_h1 * 0.999:
                trend_bias = "SHORT_ONLY"
            else:
                trend_bias = "NEUTRAL"
        else:
            trend_bias = "NEUTRAL"
        
        return RegimeContext(
            regime=regime,
            choppiness=current_chop,
            adx=adx,
            ema_200_h1=self.ema_200_h1,
            trend_bias=trend_bias,
            confidence=max(0.0, min(1.0, confidence)),
            should_trade=should_trade,
        )
