"""
VOSTOK V2 :: Choppiness Index Filter
====================================
Detecta mercados laterais (ranging) para evitar trades em consolidação.

Fórmula:
CHOP = 100 * Log10(SUM(ATR, n) / (Highest(n) - Lowest(n))) / Log10(n)

Interpretação:
- CHOP > 61.8: Mercado lateral (EVITAR TRADES)
- CHOP < 38.2: Mercado em tendência forte (BONS TRADES)
- 38.2 < CHOP < 61.8: Zona neutra

Arquiteto: Petrovich
"""

import numpy as np
import pandas as pd


class ChoppinessIndex:
    """Calcula o Choppiness Index para detecção de regime."""
    
    def __init__(self, period: int = 14):
        self.period = period
        self.threshold_ranging = 61.8
        self.threshold_trending = 38.2
    
    def calculate(
        self, 
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series
    ) -> pd.Series:
        """
        Calcula o Choppiness Index.
        
        Args:
            high: Série de preços máximos
            low: Série de preços mínimos
            close: Série de preços de fechamento
            
        Returns:
            pd.Series: Valores do CHOP (0-100)
        """
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Sum of TR
        atr_sum = tr.rolling(window=self.period).sum()
        
        # Range (Highest - Lowest)
        range_hl = (
            high.rolling(window=self.period).max() - 
            low.rolling(window=self.period).min()
        )
        
        # CHOP = 100 * Log10(SumATR / Range) / Log10(Period)
        chop = 100 * np.log10(atr_sum / range_hl) / np.log10(self.period)
        
        return chop
    
    def is_ranging(self, chop_value: float) -> bool:
        """Verifica se o mercado está lateral."""
        return chop_value > self.threshold_ranging
    
    def is_trending(self, chop_value: float) -> bool:
        """Verifica se o mercado está em tendência."""
        return chop_value < self.threshold_trending
    
    def get_regime(self, chop_value: float) -> str:
        """Retorna o regime baseado no CHOP."""
        if self.is_ranging(chop_value):
            return "RANGING"
        elif self.is_trending(chop_value):
            return "TRENDING"
        else:
            return "NEUTRAL"
