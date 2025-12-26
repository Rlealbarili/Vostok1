"""
VOSTOK V2 :: LightGBM Strategy Engine
======================================
Motor de predição baseado em LightGBM para sinais de trading.

Features:
- Carrega modelo pré-treinado (ou opera em modo PASSIVE se não existir)
- Gera features em tempo real via FeatureGenerator
- Threshold dinâmico para sinais (configurável)

Arquiteto: Petrovich | Operador: Vostok
"""

import logging
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    lgb = None

from ..features.engineering import FeatureGenerator, FEATURE_COLUMNS

logger = logging.getLogger("lightgbm_engine")


# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_MODEL_PATH = "models/v2/lgbm_model.txt"
DEFAULT_CONFIDENCE_THRESHOLD = 0.70
MIN_DATAFRAME_ROWS = 100


# ============================================================================
# ENUMS & DATA CLASSES
# ============================================================================

class Signal(str, Enum):
    """Sinais de trading."""
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


class EngineMode(str, Enum):
    """Modos de operação do engine."""
    ACTIVE = "ACTIVE"    # Modelo carregado, fazendo predições
    PASSIVE = "PASSIVE"  # Modelo não encontrado, retorna NEUTRAL


@dataclass
class PredictionResult:
    """Resultado de uma predição."""
    signal: Signal
    confidence: float
    probability_long: float
    probability_short: float
    features_snapshot: Dict[str, float]
    mode: EngineMode
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dict."""
        return {
            "signal": self.signal.value,
            "confidence": round(self.confidence, 4),
            "probability_long": round(self.probability_long, 4),
            "probability_short": round(self.probability_short, 4),
            "mode": self.mode.value,
            "features": {k: round(v, 4) for k, v in self.features_snapshot.items()},
        }


# ============================================================================
# LIGHTGBM STRATEGY CLASS
# ============================================================================

class LightGBMStrategy:
    """
    Motor de estratégia baseado em LightGBM.
    
    Uso:
        strategy = LightGBMStrategy(model_path="models/v2/lgbm_model.txt")
        
        if strategy.mode == EngineMode.PASSIVE:
            logger.warning("Model not found, training required")
        
        result = strategy.predict(df_ohlcv)
        
        if result.signal == Signal.LONG and result.confidence > 0.65:
            # Execute long trade
    """
    
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        feature_columns: Optional[List[str]] = None,
    ):
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.feature_columns = feature_columns or FEATURE_COLUMNS.copy()
        
        # Feature generator
        self.feature_generator = FeatureGenerator()
        
        # Model state
        self.model: Optional[lgb.Booster] = None
        self.mode = EngineMode.PASSIVE
        
        # Statistics
        self.prediction_count = 0
        self.long_signals = 0
        self.short_signals = 0
        
        # Load model
        self._load_model()
    
    def _load_model(self) -> None:
        """Tenta carregar o modelo do disco."""
        if not HAS_LIGHTGBM:
            logger.error("⚠️ LightGBM not installed. Run: pip install lightgbm")
            self.mode = EngineMode.PASSIVE
            return
        
        if not self.model_path.exists():
            logger.warning(
                f"⚠️ Model not found at {self.model_path}. "
                "Training required. Operating in PASSIVE mode."
            )
            self.mode = EngineMode.PASSIVE
            return
        
        try:
            self.model = lgb.Booster(model_file=str(self.model_path))
            self.mode = EngineMode.ACTIVE
            
            # Get feature names from model
            model_features = self.model.feature_name()
            if model_features:
                self.feature_columns = model_features
            
            logger.info(
                f"✅ LightGBM model loaded from {self.model_path} | "
                f"Features: {len(self.feature_columns)}"
            )
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            self.mode = EngineMode.PASSIVE
    
    def predict(self, df: pd.DataFrame) -> PredictionResult:
        """
        Gera predição para o dataframe fornecido.
        
        Args:
            df: DataFrame com OHLCV (usa última linha para predição)
            
        Returns:
            PredictionResult com sinal, confiança e features
        """
        # Cold start / Passive mode
        if self.mode == EngineMode.PASSIVE:
            return PredictionResult(
                signal=Signal.NEUTRAL,
                confidence=0.0,
                probability_long=0.5,
                probability_short=0.5,
                features_snapshot={},
                mode=EngineMode.PASSIVE,
            )
        
        # Verificar dados mínimos
        if len(df) < MIN_DATAFRAME_ROWS:
            logger.warning(f"Insufficient data: {len(df)} rows (need {MIN_DATAFRAME_ROWS}+)")
            return self._create_neutral_result()
        
        try:
            # 1. Gerar features
            df_features = self.feature_generator.generate(df)
            
            if len(df_features) == 0:
                logger.warning("No valid rows after feature generation")
                return self._create_neutral_result()
            
            # 2. Selecionar última linha
            last_row = df_features.iloc[-1]
            
            # 3. Preparar features para predição
            features_dict = {}
            feature_values = []
            
            for col in self.feature_columns:
                if col in last_row:
                    val = float(last_row[col])
                    # Handle NaN/Inf
                    if np.isnan(val) or np.isinf(val):
                        val = 0.0
                    features_dict[col] = val
                    feature_values.append(val)
                else:
                    features_dict[col] = 0.0
                    feature_values.append(0.0)
            
            # 4. Rodar inferência
            X = np.array([feature_values])
            probabilities = self.model.predict(X)
            
            # LightGBM binary classifier retorna P(class=1)
            if len(probabilities.shape) == 1:
                prob_long = float(probabilities[0])
                prob_short = 1.0 - prob_long
            else:
                # Multi-class (improvável para este caso)
                prob_long = float(probabilities[0][1])
                prob_short = float(probabilities[0][0])
            
            # 5. Determinar sinal
            signal, confidence = self._determine_signal(prob_long, prob_short)
            
            # 6. Atualizar estatísticas
            self.prediction_count += 1
            if signal == Signal.LONG:
                self.long_signals += 1
            elif signal == Signal.SHORT:
                self.short_signals += 1
            
            result = PredictionResult(
                signal=signal,
                confidence=confidence,
                probability_long=prob_long,
                probability_short=prob_short,
                features_snapshot=features_dict,
                mode=EngineMode.ACTIVE,
            )
            
            # Log
            emoji = "🟢" if signal == Signal.LONG else "🔴" if signal == Signal.SHORT else "⚪"
            logger.info(
                f"{emoji} Prediction: {signal.value} | "
                f"Confidence: {confidence:.2%} | "
                f"P(Long)={prob_long:.3f}"
            )
            
            return result
            
        except Exception as e:
            logger.exception(f"Prediction error: {e}")
            return self._create_neutral_result()
    
    def _determine_signal(
        self, 
        prob_long: float, 
        prob_short: float
    ) -> tuple[Signal, float]:
        """
        Determina o sinal baseado nas probabilidades e threshold.
        
        Returns:
            (Signal, confidence)
        """
        # Confidence é a probabilidade da classe vencedora
        if prob_long > prob_short:
            confidence = prob_long
            if confidence >= self.confidence_threshold:
                return Signal.LONG, confidence
        else:
            confidence = prob_short
            if confidence >= self.confidence_threshold:
                return Signal.SHORT, confidence
        
        # Abaixo do threshold = NEUTRAL
        return Signal.NEUTRAL, max(prob_long, prob_short)
    
    def _create_neutral_result(self) -> PredictionResult:
        """Cria resultado neutral para fallback."""
        return PredictionResult(
            signal=Signal.NEUTRAL,
            confidence=0.5,
            probability_long=0.5,
            probability_short=0.5,
            features_snapshot={},
            mode=self.mode,
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas de uso."""
        return {
            "mode": self.mode.value,
            "model_path": str(self.model_path),
            "prediction_count": self.prediction_count,
            "long_signals": self.long_signals,
            "short_signals": self.short_signals,
            "neutral_signals": self.prediction_count - self.long_signals - self.short_signals,
            "confidence_threshold": self.confidence_threshold,
            "feature_count": len(self.feature_columns),
        }
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """Atualiza threshold de confiança."""
        if 0.5 <= threshold <= 0.95:
            self.confidence_threshold = threshold
            logger.info(f"Confidence threshold updated to {threshold:.2%}")
        else:
            logger.warning(f"Invalid threshold: {threshold}. Must be between 0.5 and 0.95")
    
    def is_active(self) -> bool:
        """Verifica se o engine está em modo ativo."""
        return self.mode == EngineMode.ACTIVE


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_strategy(
    model_path: str = DEFAULT_MODEL_PATH,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> LightGBMStrategy:
    """Factory function para criar LightGBMStrategy."""
    return LightGBMStrategy(
        model_path=model_path,
        confidence_threshold=threshold,
    )


def quick_predict(df: pd.DataFrame, model_path: str = DEFAULT_MODEL_PATH) -> Dict:
    """Predição rápida para uso em scripts."""
    strategy = LightGBMStrategy(model_path=model_path)
    result = strategy.predict(df)
    return result.to_dict()


# ============================================================================
# MAIN (EXAMPLE USAGE)
# ============================================================================

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
    )
    
    print("=" * 60)
    print("🎯 LightGBM Strategy Engine - Test Mode")
    print("=" * 60)
    
    # Criar engine
    strategy = LightGBMStrategy()
    
    print(f"\nMode: {strategy.mode.value}")
    print(f"Threshold: {strategy.confidence_threshold:.0%}")
    print(f"Features: {len(strategy.feature_columns)}")
    
    if strategy.mode == EngineMode.PASSIVE:
        print("\n⚠️ Model not found. Train the model first:")
        print("   python -m src.v2.training.train_lgbm")
    else:
        print("\n✅ Model loaded and ready for predictions")
    
    # Test with dummy data
    print("\n--- Creating dummy data for feature generation test ---")
    
    np.random.seed(42)
    n = 200
    
    dummy_df = pd.DataFrame({
        'open': 100 + np.random.randn(n).cumsum(),
        'high': 101 + np.random.randn(n).cumsum(),
        'low': 99 + np.random.randn(n).cumsum(),
        'close': 100 + np.random.randn(n).cumsum(),
        'volume': np.random.randint(1000, 10000, n),
    })
    
    # Ensure high >= open, close, low
    dummy_df['high'] = dummy_df[['open', 'high', 'low', 'close']].max(axis=1)
    dummy_df['low'] = dummy_df[['open', 'high', 'low', 'close']].min(axis=1)
    
    # Generate features
    generator = strategy.feature_generator
    df_with_features = generator.generate(dummy_df)
    
    print(f"\nInput rows: {len(dummy_df)}")
    print(f"Output rows after feature generation: {len(df_with_features)}")
    print(f"Feature columns generated: {len([c for c in df_with_features.columns if c in FEATURE_COLUMNS])}")
    
    # Make prediction (will be NEUTRAL in PASSIVE mode)
    result = strategy.predict(dummy_df)
    
    print(f"\n--- Prediction Result ---")
    print(f"Signal: {result.signal.value}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Mode: {result.mode.value}")
    
    print("\n" + "=" * 60)
    print("Test complete!")
