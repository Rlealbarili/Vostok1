"""
VOSTOK V2 :: Regime Filter (Multi-Timeframe)
=============================================
Orquestra múltiplos indicadores para determinar se o mercado está operacional.

Componentes:
- Choppiness Index (CHOP): Detecta lateralização
- ADX: Mede força da tendência
- EMA 200 (H1): Define direção permitida (LONG/SHORT)

Regras de Veto (Hard Rules):
1. ADX(14) < 20 → LOCKED (Volatilidade Morta)
2. CHOP(14) > 61.8 → LOCKED (Mercado Lateral)
3. Preço vs EMA(200) H1 → Define direção permitida

Arquiteto: Petrovich | Operador: Vostok
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import pandas as pd

try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False

from .choppiness import ChoppinessMonitor

logger = logging.getLogger("regime_filter")


# ============================================================================
# CONSTANTS
# ============================================================================

ADX_MIN_THRESHOLD = 20.0       # ADX < 20 = volatilidade morta
CHOP_RANGING_THRESHOLD = 61.8  # CHOP > 61.8 = mercado lateral
EMA_PERIOD_H1 = 200            # EMA para tendência macro
ADX_PERIOD = 14
CHOP_PERIOD = 14


# ============================================================================
# ENUMS & DATA CLASSES
# ============================================================================

class RegimeStatus(str, Enum):
    """Status do regime de mercado."""
    OPERATIONAL = "OPERATIONAL"  # Pode operar
    LOCKED = "LOCKED"            # Não pode operar


class LockReason(str, Enum):
    """Motivo do bloqueio de operações."""
    OK = "OK"                    # Tudo normal
    ADX_LOW = "ADX_LOW"          # ADX muito baixo
    CHOP_HIGH = "CHOP_HIGH"      # Choppiness muito alto
    BOTH = "ADX_LOW_AND_CHOP_HIGH"  # Ambos


class AllowedDirection(str, Enum):
    """Direção de trades permitida."""
    LONG = "LONG"
    SHORT = "SHORT"
    BOTH = "BOTH"
    NONE = "NONE"  # Quando está LOCKED


@dataclass
class RegimeCheckResult:
    """Resultado completo da verificação de regime."""
    status: RegimeStatus
    reason: LockReason
    allowed_direction: AllowedDirection
    
    # Valores dos indicadores
    adx: float
    choppiness: float
    ema_200_h1: float
    current_price: float
    
    # Detalhes
    adx_ok: bool
    chop_ok: bool
    
    def to_dict(self) -> dict:
        """Converte para dict para serialização."""
        return {
            "status": self.status.value,
            "reason": self.reason.value,
            "allowed_direction": self.allowed_direction.value,
            "adx": round(self.adx, 2),
            "choppiness": round(self.choppiness, 2),
            "ema_200_h1": round(self.ema_200_h1, 2),
        }


# ============================================================================
# REGIME FILTER
# ============================================================================

class RegimeFilter:
    """
    Filtro de regime multi-indicador para Vostok V2.
    
    Determina se o mercado está operacional e qual direção é permitida.
    
    Uso:
        regime_filter = RegimeFilter()
        result = regime_filter.check_regime(df_m1, df_h1)
        
        if result.status == RegimeStatus.LOCKED:
            logger.warning(f"Trading LOCKED: {result.reason}")
            return  # Skip trade
        
        if result.allowed_direction == AllowedDirection.LONG:
            # Apenas longs permitidos
    """
    
    def __init__(
        self,
        adx_period: int = ADX_PERIOD,
        adx_min_threshold: float = ADX_MIN_THRESHOLD,
        chop_period: int = CHOP_PERIOD,
        chop_threshold: float = CHOP_RANGING_THRESHOLD,
        ema_period_h1: int = EMA_PERIOD_H1,
    ):
        self.adx_period = adx_period
        self.adx_min_threshold = adx_min_threshold
        self.chop_period = chop_period
        self.chop_threshold = chop_threshold
        self.ema_period_h1 = ema_period_h1
        
        # Sub-monitor
        self.chop_monitor = ChoppinessMonitor(
            period=chop_period,
            threshold_ranging=chop_threshold,
        )
        
        # Cache
        self._last_result: RegimeCheckResult | None = None
        
        logger.info(
            f"RegimeFilter initialized: ADX>{adx_min_threshold}, "
            f"CHOP<{chop_threshold}, EMA_H1={ema_period_h1}"
        )
    
    def check_regime(
        self, 
        df_m1: pd.DataFrame, 
        df_h1: pd.DataFrame | None = None,
    ) -> RegimeCheckResult:
        """
        Verifica o regime de mercado atual.
        
        Args:
            df_m1: DataFrame M1 com OHLCV (execução)
            df_h1: DataFrame H1 com OHLCV (tendência macro, opcional)
            
        Returns:
            RegimeCheckResult com status, reason e allowed_direction
        """
        # Normalizar colunas
        df_m1 = self._normalize_columns(df_m1)
        
        # 1. Calcular ADX
        adx_value = self._calculate_adx(df_m1)
        adx_ok = adx_value >= self.adx_min_threshold
        
        # 2. Calcular Choppiness
        chop_result = self.chop_monitor.calculate(df_m1)
        chop_value = chop_result.value
        chop_ok = not chop_result.is_ranging  # OK se NÃO está lateral
        
        # 3. Calcular EMA 200 H1 e direção
        current_price = float(df_m1['close'].iloc[-1])
        ema_200_h1 = 0.0
        allowed_direction = AllowedDirection.BOTH
        
        if df_h1 is not None and len(df_h1) >= self.ema_period_h1:
            df_h1 = self._normalize_columns(df_h1)
            ema_200_h1 = self._calculate_ema(df_h1, self.ema_period_h1)
            
            if ema_200_h1 > 0:
                if current_price > ema_200_h1:
                    allowed_direction = AllowedDirection.LONG
                else:
                    allowed_direction = AllowedDirection.SHORT
        
        # 4. Determinar status e reason
        if not adx_ok and not chop_ok:
            status = RegimeStatus.LOCKED
            reason = LockReason.BOTH
            allowed_direction = AllowedDirection.NONE
        elif not adx_ok:
            status = RegimeStatus.LOCKED
            reason = LockReason.ADX_LOW
            allowed_direction = AllowedDirection.NONE
        elif not chop_ok:
            status = RegimeStatus.LOCKED
            reason = LockReason.CHOP_HIGH
            allowed_direction = AllowedDirection.NONE
        else:
            status = RegimeStatus.OPERATIONAL
            reason = LockReason.OK
        
        result = RegimeCheckResult(
            status=status,
            reason=reason,
            allowed_direction=allowed_direction,
            adx=adx_value,
            choppiness=chop_value,
            ema_200_h1=ema_200_h1,
            current_price=current_price,
            adx_ok=adx_ok,
            chop_ok=chop_ok,
        )
        
        self._last_result = result
        
        # Log
        emoji = "🟢" if status == RegimeStatus.OPERATIONAL else "🔴"
        logger.info(
            f"{emoji} Regime: {status.value} | "
            f"ADX={adx_value:.1f} ({'OK' if adx_ok else 'LOW'}) | "
            f"CHOP={chop_value:.1f} ({'OK' if chop_ok else 'HIGH'}) | "
            f"Direction: {allowed_direction.value}"
        )
        
        return result
    
    def _calculate_adx(self, df: pd.DataFrame) -> float:
        """Calcula ADX usando pandas_ta ou método simplificado."""
        if len(df) < self.adx_period + 5:
            return 25.0  # Fallback neutro
        
        if HAS_PANDAS_TA:
            try:
                adx_df = ta.adx(
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    length=self.adx_period,
                )
                
                if adx_df is not None and f'ADX_{self.adx_period}' in adx_df.columns:
                    adx_series = adx_df[f'ADX_{self.adx_period}'].dropna()
                    if len(adx_series) > 0:
                        return float(adx_series.iloc[-1])
            except Exception as e:
                logger.warning(f"ADX calculation failed: {e}")
        
        # Fallback: ADX simplificado baseado em range
        return self._calculate_adx_simple(df)
    
    def _calculate_adx_simple(self, df: pd.DataFrame) -> float:
        """Cálculo simplificado de ADX como fallback."""
        # Usar amplitude relativa como proxy
        recent = df.tail(self.adx_period)
        high_range = recent['high'].max() - recent['low'].min()
        avg_price = recent['close'].mean()
        
        if avg_price > 0:
            # Normalizar para escala 0-100
            volatility_pct = (high_range / avg_price) * 100
            # Mapear para ADX-like (mais volatilidade = ADX maior)
            adx_proxy = min(50.0, volatility_pct * 5)  # Cap at 50
            return max(10.0, adx_proxy)  # Min 10
        
        return 25.0  # Fallback
    
    def _calculate_ema(self, df: pd.DataFrame, period: int) -> float:
        """Calcula EMA."""
        if len(df) < period:
            return 0.0
        
        if HAS_PANDAS_TA:
            try:
                ema = ta.ema(df['close'], length=period)
                if ema is not None and len(ema.dropna()) > 0:
                    return float(ema.dropna().iloc[-1])
            except Exception:
                pass
        
        # Fallback: EMA manual
        ema = df['close'].ewm(span=period, adjust=False).mean()
        return float(ema.iloc[-1])
    
    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normaliza nomes de colunas para lowercase."""
        df = df.copy()
        df.columns = df.columns.str.lower()
        return df
    
    def is_operational(self, df_m1: pd.DataFrame, df_h1: pd.DataFrame | None = None) -> bool:
        """Verificação rápida se pode operar."""
        result = self.check_regime(df_m1, df_h1)
        return result.status == RegimeStatus.OPERATIONAL
    
    def get_last_result(self) -> RegimeCheckResult | None:
        """Retorna último resultado (cache)."""
        return self._last_result


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def quick_regime_check(
    df_m1: pd.DataFrame, 
    df_h1: pd.DataFrame | None = None,
) -> dict:
    """
    Verificação rápida de regime.
    
    Returns:
        dict com status, reason e allowed_direction
    """
    regime_filter = RegimeFilter()
    result = regime_filter.check_regime(df_m1, df_h1)
    return result.to_dict()
