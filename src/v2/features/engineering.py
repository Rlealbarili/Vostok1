"""
VOSTOK V2 :: Feature Engineering Pipeline
==========================================
Transforma dados brutos (OHLCV) em features ricas para o modelo LightGBM.

Categorias de Features:
1. Momentum: RSI, MACD, ROC
2. Volatilidade: ATR, Bollinger Bands Width
3. Volume: OBV, CMF (Chaikin Money Flow)
4. Custom Vostok: CVD (Cumulative Volume Delta), Shannon Entropy
5. Lags: Retornos e volume defasados (t-1, t-2, t-3)

Arquiteto: Petrovich | Operador: Vostok
"""

import logging
from typing import List

import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False

logger = logging.getLogger("feature_engineering")


# ============================================================================
# CONSTANTS
# ============================================================================

# Feature list para seleção (deve coincidir com training)
FEATURE_COLUMNS = [
    # Momentum
    "rsi_14", "rsi_7", "macd", "macd_signal", "macd_hist", "roc_10",
    # Volatilidade
    "atr_14", "bb_width", "volatility_pct",
    # Volume
    "obv_norm", "cmf",
    # Custom Vostok
    "cvd", "entropy",
    # Lags
    "return_1", "return_2", "return_3",
    "volume_ratio_1", "volume_ratio_2", "volume_ratio_3",
    # Price Action
    "close_vs_ema_20", "close_vs_ema_50",
]


# ============================================================================
# FEATURE GENERATOR CLASS
# ============================================================================

class FeatureGenerator:
    """
    Gera features para o modelo LightGBM.
    
    Uso:
        generator = FeatureGenerator()
        df_features = generator.generate(df_ohlcv)
    """
    
    def __init__(
        self,
        rsi_period_fast: int = 7,
        rsi_period_slow: int = 14,
        atr_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        bb_period: int = 20,
        bb_std: float = 2.0,
        entropy_period: int = 14,
        lag_periods: List[int] = [1, 2, 3],
    ):
        self.rsi_period_fast = rsi_period_fast
        self.rsi_period_slow = rsi_period_slow
        self.atr_period = atr_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.entropy_period = entropy_period
        self.lag_periods = lag_periods
        
        logger.info("FeatureGenerator initialized")
    
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Gera todas as features a partir de dados OHLCV.
        
        Args:
            df: DataFrame com colunas 'open', 'high', 'low', 'close', 'volume'
            
        Returns:
            DataFrame com features adicionadas
        """
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        # Verificar colunas mínimas
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        if len(df) < 50:
            logger.warning(f"Insufficient data: {len(df)} rows (need 50+)")
            return df
        
        # 1. Momentum Features
        df = self._add_momentum_features(df)
        
        # 2. Volatilidade Features
        df = self._add_volatility_features(df)
        
        # 3. Volume Features
        df = self._add_volume_features(df)
        
        # 4. Custom Vostok Features
        df = self._add_vostok_features(df)
        
        # 5. Lag Features
        df = self._add_lag_features(df)
        
        # 6. Price Action Features
        df = self._add_price_action_features(df)
        
        # Drop NaN rows (primeiras linhas sem dados suficientes)
        initial_len = len(df)
        df = df.dropna()
        dropped = initial_len - len(df)
        if dropped > 0:
            logger.debug(f"Dropped {dropped} rows with NaN")
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features de momentum."""
        close = df['close']
        
        if HAS_PANDAS_TA:
            # RSI
            df['rsi_14'] = ta.rsi(close, length=self.rsi_period_slow)
            df['rsi_7'] = ta.rsi(close, length=self.rsi_period_fast)
            
            # MACD
            macd = ta.macd(close, fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
            if macd is not None:
                df['macd'] = macd.iloc[:, 0]  # MACD line
                df['macd_signal'] = macd.iloc[:, 2]  # Signal line
                df['macd_hist'] = macd.iloc[:, 1]  # Histogram
            else:
                df['macd'] = df['macd_signal'] = df['macd_hist'] = 0.0
            
            # ROC (Rate of Change)
            df['roc_10'] = ta.roc(close, length=10)
        else:
            # Fallback manual
            df['rsi_14'] = self._calculate_rsi_manual(close, self.rsi_period_slow)
            df['rsi_7'] = self._calculate_rsi_manual(close, self.rsi_period_fast)
            df['macd'] = close.ewm(span=12).mean() - close.ewm(span=26).mean()
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            df['roc_10'] = close.pct_change(10) * 100
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features de volatilidade."""
        high, low, close = df['high'], df['low'], df['close']
        
        if HAS_PANDAS_TA:
            # ATR
            df['atr_14'] = ta.atr(high, low, close, length=self.atr_period)
            
            # Bollinger Bands
            bb = ta.bbands(close, length=self.bb_period, std=self.bb_std)
            if bb is not None:
                upper = bb.iloc[:, 0]
                lower = bb.iloc[:, 2]
                df['bb_width'] = (upper - lower) / close * 100
            else:
                df['bb_width'] = 0.0
        else:
            # ATR manual
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df['atr_14'] = tr.rolling(self.atr_period).mean()
            
            # BB width manual
            sma = close.rolling(self.bb_period).mean()
            std = close.rolling(self.bb_period).std()
            upper = sma + (self.bb_std * std)
            lower = sma - (self.bb_std * std)
            df['bb_width'] = (upper - lower) / close * 100
        
        # Volatilidade percentual (normalizada)
        df['volatility_pct'] = (df['atr_14'] / close) * 100
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features de volume."""
        close, volume = df['close'], df['volume']
        high, low = df['high'], df['low']
        
        if HAS_PANDAS_TA:
            # OBV
            obv = ta.obv(close, volume)
            if obv is not None:
                # Normalizar OBV (z-score rolling)
                df['obv_norm'] = (obv - obv.rolling(50).mean()) / obv.rolling(50).std()
                df['obv_norm'] = df['obv_norm'].fillna(0)
            else:
                df['obv_norm'] = 0.0
            
            # CMF (Chaikin Money Flow)
            cmf = ta.cmf(high, low, close, volume, length=20)
            df['cmf'] = cmf if cmf is not None else 0.0
        else:
            # OBV manual
            direction = np.sign(close.diff())
            obv = (direction * volume).cumsum()
            df['obv_norm'] = (obv - obv.rolling(50).mean()) / obv.rolling(50).std()
            df['obv_norm'] = df['obv_norm'].fillna(0)
            
            # CMF manual
            mfm = ((close - low) - (high - close)) / (high - low + 1e-10)
            mfv = mfm * volume
            df['cmf'] = mfv.rolling(20).sum() / volume.rolling(20).sum()
        
        return df
    
    def _add_vostok_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features customizadas Vostok (CVD, Entropy)."""
        close, volume = df['close'], df['volume']
        
        # CVD (Cumulative Volume Delta) - Proxy
        # Em dados de candle, estimamos direção pelo close vs open
        if 'open' in df.columns:
            buy_volume = np.where(close >= df['open'], volume, 0)
            sell_volume = np.where(close < df['open'], volume, 0)
        else:
            # Fallback: usar direção do retorno
            direction = np.sign(close.diff())
            buy_volume = np.where(direction >= 0, volume, 0)
            sell_volume = np.where(direction < 0, volume, 0)
        
        cvd = pd.Series(buy_volume - sell_volume, index=df.index).cumsum()
        # Normalizar CVD
        df['cvd'] = (cvd - cvd.rolling(50).mean()) / (cvd.rolling(50).std() + 1e-10)
        df['cvd'] = df['cvd'].fillna(0)
        
        # Shannon Entropy (volatilidade de preço)
        df['entropy'] = self._calculate_entropy(close, period=self.entropy_period)
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features de lag para capturar padrões temporais."""
        close, volume = df['close'], df['volume']
        
        # Retornos defasados
        for lag in self.lag_periods:
            df[f'return_{lag}'] = close.pct_change(lag) * 100
        
        # Volume ratio (volume atual / média do volume)
        vol_mean = volume.rolling(20).mean()
        for lag in self.lag_periods:
            df[f'volume_ratio_{lag}'] = (volume.shift(lag) / vol_mean).fillna(1.0)
        
        return df
    
    def _add_price_action_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features de price action."""
        close = df['close']
        
        # EMA crosses
        ema_20 = close.ewm(span=20).mean()
        ema_50 = close.ewm(span=50).mean()
        
        df['close_vs_ema_20'] = (close - ema_20) / ema_20 * 100
        df['close_vs_ema_50'] = (close - ema_50) / ema_50 * 100
        
        return df
    
    def _calculate_rsi_manual(self, series: pd.Series, period: int) -> pd.Series:
        """Calcula RSI manualmente."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_entropy(self, series: pd.Series, period: int = 14) -> pd.Series:
        """
        Calcula Shannon Entropy para medir "caos" do mercado.
        
        Valor alto = mercado imprevisível
        Valor baixo = tendência clara
        """
        # Retornos
        returns = series.pct_change()
        
        # Bins para discretização
        def calc_entropy(window):
            if len(window.dropna()) < period // 2:
                return 0.5
            
            # Discretizar retornos em bins
            try:
                hist, _ = np.histogram(window.dropna(), bins=10)
                hist = hist / hist.sum()  # Normalizar
                hist = hist[hist > 0]  # Remover zeros
                entropy = -np.sum(hist * np.log2(hist))
                # Normalizar para 0-1
                max_entropy = np.log2(10)
                return entropy / max_entropy
            except:
                return 0.5
        
        entropy = returns.rolling(window=period).apply(calc_entropy, raw=False)
        
        return entropy
    
    def get_feature_columns(self) -> List[str]:
        """Retorna lista de colunas de features."""
        return FEATURE_COLUMNS.copy()
    
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Seleciona apenas as colunas de features (para inferência)."""
        available = [c for c in FEATURE_COLUMNS if c in df.columns]
        return df[available]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Função utilitária para gerar features rapidamente."""
    generator = FeatureGenerator()
    return generator.generate(df)


def get_feature_names() -> List[str]:
    """Retorna lista de nomes de features."""
    return FEATURE_COLUMNS.copy()
