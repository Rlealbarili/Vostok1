"""
VOSTOK V2 :: Core Orchestrator Engine
======================================
Cérebro central do sistema de trading Vostok V2.

HIERARQUIA DE DECISÃO (STRICT FLOW):
1. DEFESA (Prioridade 0): RegimeFilter - Mercado seguro?
2. INTELIGÊNCIA (Prioridade 1): CryptoBuffett - Veto estrutural?
3. ATAQUE (Prioridade 2): LightGBMStrategy - Sinal com alta confiança?

Pipeline: Market Data → Regime → Sentiment → ML → Decision

Arquiteto: Petrovich | Operador: Vostok
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd

# V2 Components
from .filters.regime import RegimeFilter, RegimeStatus, AllowedDirection
from .sentiment.buffett import CryptoBuffett, TradePermission
from .strategy.lightgbm_engine import LightGBMStrategy, Signal, EngineMode

logger = logging.getLogger("vostok_v2_engine")


# ============================================================================
# ENUMS & DATA CLASSES
# ============================================================================

class Action(str, Enum):
    """Ações de decisão do engine."""
    EXECUTE = "EXECUTE"   # Executar trade
    HOLD = "HOLD"         # Não operar (baixa confiança)
    WAIT = "WAIT"         # Aguardar (veto temporário)
    SKIP = "SKIP"         # Pular (regime bloqueado)


@dataclass
class EngineDecision:
    """Resultado da análise do engine."""
    action: Action
    direction: Optional[str] = None  # "LONG" ou "SHORT"
    confidence: float = 0.0
    reason: str = ""
    origin: str = "VostokV2Engine"
    
    # Contexto
    regime_status: str = ""
    regime_reason: str = ""
    allowed_direction: str = ""
    buffett_verdict: str = ""
    buffett_permission: str = ""
    ml_signal: str = ""
    ml_confidence: float = 0.0
    
    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dict."""
        return {
            "action": self.action.value,
            "direction": self.direction,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
            "origin": self.origin,
            "context": {
                "regime_status": self.regime_status,
                "regime_reason": self.regime_reason,
                "allowed_direction": self.allowed_direction,
                "buffett_verdict": self.buffett_verdict,
                "buffett_permission": self.buffett_permission,
                "ml_signal": self.ml_signal,
                "ml_confidence": round(self.ml_confidence, 4),
            },
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class EngineStats:
    """Estatísticas do engine."""
    cycles: int = 0
    executes: int = 0
    holds: int = 0
    waits: int = 0
    skips: int = 0
    regime_locks: int = 0
    buffett_vetos: int = 0
    ml_neutrals: int = 0
    trend_filters: int = 0


# ============================================================================
# VOSTOK V2 ENGINE
# ============================================================================

class VostokV2Engine:
    """
    Orquestrador central do sistema de trading Vostok V2.
    
    Hierarquia de Decisão:
    1. RegimeFilter (DEFESA) - Bloqueia se mercado lateral/morto
    2. CryptoBuffett (INTELIGÊNCIA) - Veta riscos estruturais
    3. LightGBMStrategy (ATAQUE) - Gera sinal com confiança
    
    Uso:
        engine = VostokV2Engine()
        decision = await engine.analyze_market(df_m1, df_h1, news="BTC crashes 10%")
        
        if decision.action == Action.EXECUTE:
            # Execute trade em decision.direction
    """
    
    def __init__(
        self,
        model_path: str = "models/v2/lgbm_model.txt",
        confidence_threshold: float = 0.70,
        ollama_host: str = "llm_engine",
    ):
        # Initialize components
        self.regime_filter = RegimeFilter()
        self.buffett = CryptoBuffett(ollama_host=ollama_host)
        self.strategy = LightGBMStrategy(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
        )
        
        # Statistics
        self.stats = EngineStats()
        
        # State
        self._last_decision: Optional[EngineDecision] = None
        self._running = False
        
        # Log initialization
        logger.info("=" * 60)
        logger.info("🚀 VOSTOK V2 ENGINE INITIALIZED")
        logger.info("=" * 60)
        logger.info(f"  RegimeFilter: ADX>{self.regime_filter.adx_min_threshold}, CHOP<{self.regime_filter.chop_threshold}")
        logger.info(f"  CryptoBuffett: {ollama_host}")
        logger.info(f"  LightGBM: {self.strategy.mode.value} (threshold={confidence_threshold:.0%})")
        logger.info("=" * 60)
    
    async def analyze_market(
        self,
        df_m1: pd.DataFrame,
        df_h1: Optional[pd.DataFrame] = None,
        news_context: Optional[str] = None,
    ) -> EngineDecision:
        """
        Analisa o mercado e retorna uma decisão.
        
        Args:
            df_m1: DataFrame M1 com OHLCV (execução)
            df_h1: DataFrame H1 com OHLCV (tendência macro, opcional)
            news_context: Headline de notícia para análise Buffett (opcional)
            
        Returns:
            EngineDecision com action, direction, confidence e contexto
        """
        self.stats.cycles += 1
        
        decision = EngineDecision(
            action=Action.HOLD,
            reason="Initial state",
        )
        
        # =====================================================================
        # STEP 1: DEFESA - Regime Filter (Prioridade 0)
        # =====================================================================
        logger.debug("Step 1: Checking Regime...")
        
        regime_result = self.regime_filter.check_regime(df_m1, df_h1)
        
        decision.regime_status = regime_result.status.value
        decision.regime_reason = regime_result.reason.value
        decision.allowed_direction = regime_result.allowed_direction.value
        
        if regime_result.status == RegimeStatus.LOCKED:
            self.stats.skips += 1
            self.stats.regime_locks += 1
            
            decision.action = Action.SKIP
            decision.reason = f"Regime LOCKED: {regime_result.reason.value}"
            
            logger.info(
                f"🔴 SKIP | Regime: {regime_result.reason.value} | "
                f"CHOP={regime_result.choppiness:.1f}, ADX={regime_result.adx:.1f}"
            )
            
            self._last_decision = decision
            return decision
        
        # =====================================================================
        # STEP 2: INTELIGÊNCIA - Crypto Buffett (Prioridade 1)
        # =====================================================================
        buffett_permission = TradePermission.PROCEED
        
        if news_context:
            logger.debug("Step 2: Consulting Crypto Buffett...")
            
            buffett_analysis = await self.buffett.analyze_news(news_context)
            buffett_permission = buffett_analysis.get_permission()
            
            decision.buffett_verdict = buffett_analysis.buffett_verdict
            decision.buffett_permission = buffett_permission.value
            
            if buffett_permission == TradePermission.HARD_VETO:
                self.stats.waits += 1
                self.stats.buffett_vetos += 1
                
                decision.action = Action.WAIT
                decision.reason = f"Buffett VETO: {buffett_analysis.reasoning}"
                
                logger.warning(
                    f"🛑 WAIT | Buffett Veto: {buffett_analysis.buffett_verdict} | "
                    f"Score: {buffett_analysis.sentiment_score:+.2f}"
                )
                
                self._last_decision = decision
                return decision
        
        # =====================================================================
        # STEP 3: ATAQUE - LightGBM Strategy (Prioridade 2)
        # =====================================================================
        logger.debug("Step 3: Getting ML Prediction...")
        
        ml_result = self.strategy.predict(df_m1)
        
        decision.ml_signal = ml_result.signal.value
        decision.ml_confidence = ml_result.confidence
        
        # Check if model is in passive mode
        if ml_result.mode == EngineMode.PASSIVE:
            self.stats.holds += 1
            
            decision.action = Action.HOLD
            decision.reason = "ML Model in PASSIVE mode (training required)"
            
            logger.warning("⚪ HOLD | ML Model PASSIVE - training required")
            
            self._last_decision = decision
            return decision
        
        # Check if signal is neutral (low confidence)
        if ml_result.signal == Signal.NEUTRAL:
            self.stats.holds += 1
            self.stats.ml_neutrals += 1
            
            decision.action = Action.HOLD
            decision.reason = f"Low confidence signal ({ml_result.confidence:.1%})"
            
            logger.info(
                f"⚪ HOLD | ML: NEUTRAL | "
                f"Confidence: {ml_result.confidence:.1%}"
            )
            
            self._last_decision = decision
            return decision
        
        # =====================================================================
        # STEP 4: FUSION - Aplicar filtro de tendência H1
        # =====================================================================
        logger.debug("Step 4: Applying Trend Filter...")
        
        ml_direction = ml_result.signal.value  # "LONG" ou "SHORT"
        allowed = regime_result.allowed_direction
        
        # Verificar se a direção é permitida pelo filtro de tendência
        if allowed != AllowedDirection.BOTH and allowed != AllowedDirection.NONE:
            if ml_direction != allowed.value:
                self.stats.holds += 1
                self.stats.trend_filters += 1
                
                decision.action = Action.HOLD
                decision.reason = f"Trend filter mismatch ({allowed.value} only)"
                
                logger.info(
                    f"⚪ HOLD | Trend Filter | "
                    f"ML={ml_direction}, Allowed={allowed.value}"
                )
                
                self._last_decision = decision
                return decision
        
        # =====================================================================
        # STEP 5: EXECUTE - Todas as verificações passaram
        # =====================================================================
        self.stats.executes += 1
        
        decision.action = Action.EXECUTE
        decision.direction = ml_direction
        decision.confidence = ml_result.confidence
        decision.reason = "All filters passed"
        decision.origin = "LightGBM + V2 Filters"
        
        # Boost info se Buffett disse "Buy the Fear"
        if buffett_permission == TradePermission.BOOST_LONG:
            decision.reason = "All filters passed + Buffett BOOST"
        
        logger.info(
            f"🟢 EXECUTE | {ml_direction} | "
            f"Confidence: {ml_result.confidence:.1%} | "
            f"Regime: {regime_result.status.value}"
        )
        
        self._last_decision = decision
        return decision
    
    def analyze_market_sync(
        self,
        df_m1: pd.DataFrame,
        df_h1: Optional[pd.DataFrame] = None,
        news_context: Optional[str] = None,
    ) -> EngineDecision:
        """
        Versão síncrona do analyze_market para uso em scripts.
        """
        return asyncio.get_event_loop().run_until_complete(
            self.analyze_market(df_m1, df_h1, news_context)
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de uso."""
        return {
            "cycles": self.stats.cycles,
            "executes": self.stats.executes,
            "holds": self.stats.holds,
            "waits": self.stats.waits,
            "skips": self.stats.skips,
            "regime_locks": self.stats.regime_locks,
            "buffett_vetos": self.stats.buffett_vetos,
            "ml_neutrals": self.stats.ml_neutrals,
            "trend_filters": self.stats.trend_filters,
            "ml_mode": self.strategy.mode.value,
        }
    
    def get_last_decision(self) -> Optional[EngineDecision]:
        """Retorna a última decisão."""
        return self._last_decision
    
    def reset_stats(self):
        """Reseta estatísticas."""
        self.stats = EngineStats()
        logger.info("Statistics reset")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_engine(
    model_path: str = "models/v2/lgbm_model.txt",
    confidence_threshold: float = 0.70,
    ollama_host: str = "llm_engine",
) -> VostokV2Engine:
    """Factory function para criar VostokV2Engine."""
    return VostokV2Engine(
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        ollama_host=ollama_host,
    )


async def quick_analysis(
    df_m1: pd.DataFrame,
    df_h1: Optional[pd.DataFrame] = None,
    news: Optional[str] = None,
) -> Dict[str, Any]:
    """Análise rápida para uso em scripts."""
    engine = VostokV2Engine(ollama_host="localhost")
    decision = await engine.analyze_market(df_m1, df_h1, news)
    return decision.to_dict()


# ============================================================================
# MAIN (TEST MODE)
# ============================================================================

async def main():
    """Test mode para o engine."""
    import numpy as np
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
    )
    
    print("=" * 60)
    print("🧠 VOSTOK V2 ENGINE - Test Mode")
    print("=" * 60)
    
    # Create dummy data
    np.random.seed(42)
    n = 200
    
    df_m1 = pd.DataFrame({
        'open': 100 + np.random.randn(n).cumsum(),
        'high': 101 + np.random.randn(n).cumsum(),
        'low': 99 + np.random.randn(n).cumsum(),
        'close': 100 + np.random.randn(n).cumsum(),
        'volume': np.random.randint(1000, 10000, n),
    })
    
    df_m1['high'] = df_m1[['open', 'high', 'low', 'close']].max(axis=1)
    df_m1['low'] = df_m1[['open', 'high', 'low', 'close']].min(axis=1)
    
    # Create engine (using localhost for testing)
    engine = VostokV2Engine(ollama_host="localhost")
    
    print("\n--- Test 1: Market Analysis (No News) ---")
    decision1 = await engine.analyze_market(df_m1)
    print(f"Action: {decision1.action.value}")
    print(f"Reason: {decision1.reason}")
    print(f"ML Signal: {decision1.ml_signal}, Confidence: {decision1.ml_confidence:.1%}")
    
    print("\n--- Engine Statistics ---")
    stats = engine.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("Test complete!")


if __name__ == "__main__":
    asyncio.run(main())
