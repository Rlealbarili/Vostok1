"""
VOSTOK V2 :: Choppiness Index Monitor
=====================================
Detecta mercados laterais (ranging) para evitar trades em consolidação.

Fórmula:
CHOP = 100 * Log10(SUM(ATR, n) / (Highest(n) - Lowest(n))) / Log10(n)

Interpretação:
- CHOP > 61.8: Mercado lateral → LOCK TRADING
- CHOP < 38.2: Tendência forte → ALLOW TRADING  
- 38.2 < CHOP < 61.8: Zona de transição → ALLOW WITH CAUTION

Arquiteto: Petrovich | Operador: Vostok
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False

logger = logging.getLogger("choppiness")


# ============================================================================
# CONSTANTS
# ============================================================================

CHOP_THRESHOLD_RANGING = 61.8   # Fibonacci - mercado lateral
CHOP_THRESHOLD_TRENDING = 38.2  # Fibonacci - tendência forte
DEFAULT_PERIOD = 14


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ChoppinessResult:
    """Resultado da análise de Choppiness."""
    value: float
    is_ranging: bool
    is_trending: bool
    status: str  # "RANGING" | "TRENDING" | "TRANSITION"


# ============================================================================
# CHOPPINESS MONITOR
# ============================================================================

class ChoppinessMonitor:
    """
    Monitor de Choppiness Index para detecção de mercado lateral.
    
    Uso:
        monitor = ChoppinessMonitor(period=14)
        result = monitor.calculate(df)
        if result.is_ranging:
            # Skip trading
    """
    
    def __init__(
        self, 
        period: int = DEFAULT_PERIOD,
        threshold_ranging: float = CHOP_THRESHOLD_RANGING,
        threshold_trending: float = CHOP_THRESHOLD_TRENDING,
    ):
        self.period = period
        self.threshold_ranging = threshold_ranging
        self.threshold_trending = threshold_trending
        
        logger.info(
            f"ChoppinessMonitor initialized: period={period}, "
            f"ranging>{threshold_ranging}, trending<{threshold_trending}"
        )
    
    def calculate(self, df: pd.DataFrame) -> ChoppinessResult:
        """
        Calcula o Choppiness Index atual.
        
        Args:
            df: DataFrame com colunas 'high', 'low', 'close'
            
        Returns:
            ChoppinessResult com valor e status
        """
        if len(df) < self.period + 1:
            logger.warning(f"Insufficient data: {len(df)} < {self.period + 1}")
            return ChoppinessResult(
                value=50.0,  # Neutral fallback
                is_ranging=False,
                is_trending=False,
                status="INSUFFICIENT_DATA",
            )
        
        # Usar pandas_ta se disponível (mais preciso)
        if HAS_PANDAS_TA:
            chop_series = self._calculate_with_pandas_ta(df)
        else:
            chop_series = self._calculate_manual(df)
        
        # Pegar último valor válido
        current_chop = chop_series.dropna().iloc[-1] if len(chop_series.dropna()) > 0 else 50.0
        
        # Determinar status
        is_ranging = current_chop > self.threshold_ranging
        is_trending = current_chop < self.threshold_trending
        
        if is_ranging:
            status = "RANGING"
        elif is_trending:
            status = "TRENDING"
        else:
            status = "TRANSITION"
        
        logger.debug(f"CHOP = {current_chop:.2f} → {status}")
        
        return ChoppinessResult(
            value=float(current_chop),
            is_ranging=is_ranging,
            is_trending=is_trending,
            status=status,
        )
    
    def _calculate_with_pandas_ta(self, df: pd.DataFrame) -> pd.Series:
        """Calcula CHOP usando pandas_ta."""
        # Garantir nomes de coluna corretos (case-insensitive)
        df_norm = df.copy()
        df_norm.columns = df_norm.columns.str.lower()
        
        chop = ta.chop(
            high=df_norm['high'],
            low=df_norm['low'],
            close=df_norm['close'],
            length=self.period,
        )
        
        return chop
    
    def _calculate_manual(self, df: pd.DataFrame) -> pd.Series:
        """
        Calcula CHOP manualmente (fallback sem pandas_ta).
        
        Fórmula: 100 * Log10(SUM(ATR, n) / (Highest(n) - Lowest(n))) / Log10(n)
        """
        df_norm = df.copy()
        df_norm.columns = df_norm.columns.str.lower()
        
        high = df_norm['high']
        low = df_norm['low']
        close = df_norm['close']
        
        # True Range components
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        # True Range = max(tr1, tr2, tr3)
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Sum of TR over period
        atr_sum = tr.rolling(window=self.period).sum()
        
        # Range (Highest - Lowest) over period
        highest = high.rolling(window=self.period).max()
        lowest = low.rolling(window=self.period).min()
        range_hl = highest - lowest
        
        # Avoid division by zero
        range_hl = range_hl.replace(0, np.nan)
        
        # CHOP = 100 * Log10(SumATR / Range) / Log10(Period)
        chop = 100 * np.log10(atr_sum / range_hl) / np.log10(self.period)
        
        return chop
    
    def is_market_locked(self, df: pd.DataFrame) -> bool:
        """
        Verifica rapidamente se o mercado está travado (lateral).
        
        Returns:
            True se CHOP > 61.8 (mercado lateral/morto)
        """
        result = self.calculate(df)
        return result.is_ranging


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def quick_chop_check(df: pd.DataFrame, period: int = 14) -> bool:
    """
    Verificação rápida de Choppiness.
    
    Returns:
        True se mercado lateral (CHOP > 61.8)
    """
    monitor = ChoppinessMonitor(period=period)
    return monitor.is_market_locked(df)
