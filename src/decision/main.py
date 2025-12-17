"""
VOSTOK-1 :: Decision Engine v2.0 (ML-Powered)
==============================================
Motor de Decisão com integração do modelo Random Forest treinado.
Substitui regras simples por inferência ML para maior precisão.

Arquiteto: Petrovich | Operador: Vostok
Stack: Python 3.11 + asyncio + redis-py + scikit-learn
"""

import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import redis.asyncio as aioredis

# ML imports (optional - fallback if not available)
try:
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    joblib = None

# ============================================================================
# CONFIGURAÇÃO DE LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "module": "%(name)s", "message": "%(message)s"}',
    datefmt='%Y-%m-%dT%H:%M:%S%z',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("decision")

# ============================================================================
# CONFIGURAÇÕES
# ============================================================================
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
SIGNAL_STREAM = os.getenv("SIGNAL_STREAM", "stream:signals:tech")
ORDER_STREAM = os.getenv("ORDER_STREAM", "stream:orders:execute")
CONSUMER_GROUP = os.getenv("CONSUMER_GROUP", "decision_group")
CONSUMER_NAME = os.getenv("CONSUMER_NAME", "decision_worker_1")

# Diretórios
DATA_DIR = Path(os.getenv("DATA_DIR", "/app/data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", "/app/models"))
MODEL_FILE = MODELS_DIR / "sniper_v1.pkl"
TRAINING_DIR = DATA_DIR / "training"
DATASET_FILE = TRAINING_DIR / "dataset.jsonl"

# ML Threshold - só dispara se confiança > 60%
ML_CONFIDENCE_THRESHOLD = float(os.getenv("ML_CONFIDENCE_THRESHOLD", 0.60))

# Parâmetros da estratégia (fallback)
RSI_OVERSOLD = 35
RSI_OVERBOUGHT = 65

# Parâmetros do Triple Barrier
TP_ATR_MULT = 2.0  # Take Profit = ATR * 2.0
SL_ATR_MULT = 1.0  # Stop Loss = ATR * 1.0
MAX_BARS = 120     # Barreira temporal

# Features na MESMA ORDEM do treinamento
FEATURE_NAMES = ["rsi", "cvd", "entropy", "volatility_atr", "funding_rate"]


# ============================================================================
# ENUMS E DATACLASSES
# ============================================================================
class TradeAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class TradeLabel(int, Enum):
    LOSS = 0
    WIN = 1


@dataclass
class VirtualTrade:
    """Representa um trade virtual para rotulagem."""
    id: str
    timestamp: int
    action: TradeAction
    entry_price: float
    tp_price: float
    sl_price: float
    max_bars: int
    bars_elapsed: int = 0
    features: dict[str, float] = field(default_factory=dict)
    label: TradeLabel | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    pnl_percent: float | None = None

    def check_barriers(self, current_price: float) -> bool:
        """Verifica se alguma barreira foi atingida."""
        self.bars_elapsed += 1
        
        if self.action == TradeAction.BUY:
            if current_price >= self.tp_price:
                self.label = TradeLabel.WIN
                self.exit_price = current_price
                self.exit_reason = "TP"
                self.pnl_percent = ((current_price - self.entry_price) / self.entry_price) * 100
                return True
            if current_price <= self.sl_price:
                self.label = TradeLabel.LOSS
                self.exit_price = current_price
                self.exit_reason = "SL"
                self.pnl_percent = ((current_price - self.entry_price) / self.entry_price) * 100
                return True
        
        if self.bars_elapsed >= self.max_bars:
            self.label = TradeLabel.LOSS
            self.exit_price = current_price
            self.exit_reason = "TIME"
            self.pnl_percent = ((current_price - self.entry_price) / self.entry_price) * 100
            return True
        
        return False


# ============================================================================
# ML MODEL LOADER
# ============================================================================
class MLPredictor:
    """
    Carrega e executa inferência com o modelo Random Forest treinado.
    Fallback para regras simples se modelo não disponível.
    """

    def __init__(self) -> None:
        self.model = None
        self.feature_names: list[str] = []
        self.metrics: dict = {}
        self.ml_enabled = False
        self._load_model()

    def _load_model(self) -> None:
        """Carrega o modelo do disco."""
        if not ML_AVAILABLE:
            logger.warning("⚠️  scikit-learn/joblib não disponível. Usando fallback.")
            return
        
        if not MODEL_FILE.exists():
            logger.warning(f"⚠️  Modelo não encontrado: {MODEL_FILE}. Usando fallback.")
            return
        
        try:
            model_data = joblib.load(MODEL_FILE)
            self.model = model_data.get('model')
            self.feature_names = model_data.get('feature_names', FEATURE_NAMES)
            self.metrics = model_data.get('metrics', {})
            self.ml_enabled = True
            
            logger.info("🧠 Modelo ML carregado com sucesso!")
            logger.info(f"   Precision: {self.metrics.get('precision', 'N/A')}")
            logger.info(f"   Features: {self.feature_names}")
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelo: {e}")
            self.ml_enabled = False

    def predict(self, signal: dict[str, Any]) -> tuple[bool, float, str]:
        """
        Executa inferência no sinal.
        
        Returns:
            (should_trade, confidence, reason)
        """
        if not self.ml_enabled:
            return self._fallback_predict(signal)
        
        try:
            # Montar vetor de features na ordem correta
            feature_vector = self._extract_features(signal)
            
            if feature_vector is None:
                return False, 0.0, "Invalid features"
            
            # Reshape para 2D (sklearn espera)
            X = np.array([feature_vector]).reshape(1, -1)
            
            # Obter probabilidade da classe 1 (Win)
            probabilities = self.model.predict_proba(X)
            confidence = probabilities[0][1]  # Probabilidade de Win
            
            # Verificar threshold
            if confidence >= ML_CONFIDENCE_THRESHOLD:
                reason = f"ML Strategy | Conf: {confidence:.2f}"
                return True, confidence, reason
            else:
                return False, confidence, f"Below threshold ({confidence:.2f} < {ML_CONFIDENCE_THRESHOLD})"
            
        except Exception as e:
            logger.error(f"Erro na inferência ML: {e}")
            return self._fallback_predict(signal)

    def _extract_features(self, signal: dict[str, Any]) -> list[float] | None:
        """Extrai features do sinal na ordem correta para o modelo."""
        try:
            features = []
            
            for feat_name in self.feature_names:
                # Mapeamento de nomes (sinal pode ter nomes diferentes)
                value = None
                
                if feat_name == "rsi":
                    value = signal.get('rsi')
                elif feat_name == "cvd":
                    value = signal.get('cvd_absolute') or signal.get('cvd')
                elif feat_name == "entropy":
                    value = signal.get('entropy')
                elif feat_name == "volatility_atr":
                    value = signal.get('volatility_atr') or signal.get('atr')
                elif feat_name == "funding_rate":
                    value = signal.get('funding_rate', 0)
                else:
                    value = signal.get(feat_name, 0)
                
                # Converter e validar
                if value is None:
                    value = 0.0
                
                feat_value = float(value)
                if not np.isfinite(feat_value):
                    feat_value = 0.0
                
                features.append(feat_value)
            
            return features
            
        except Exception as e:
            logger.warning(f"Erro ao extrair features: {e}")
            return None

    def _fallback_predict(self, signal: dict[str, Any]) -> tuple[bool, float, str]:
        """Lógica de fallback (regras simples) quando ML não disponível."""
        try:
            rsi = float(signal.get('rsi', 50)) if signal.get('rsi') else 50.0
            cvd = float(signal.get('cvd_absolute', 0)) if signal.get('cvd_absolute') else 0.0
            
            # Regra simples: RSI oversold + CVD positivo
            if rsi < RSI_OVERSOLD and cvd > 0:
                return True, 0.50, f"Fallback: RSI({rsi:.1f}) < {RSI_OVERSOLD} & CVD > 0"
            
            return False, 0.0, "No fallback trigger"
            
        except Exception:
            return False, 0.0, "Fallback error"


# ============================================================================
# TRIPLE BARRIER LABELER
# ============================================================================
class TripleBarrierLabeler:
    """Implementa o método Triple Barrier para rotulagem de dados."""

    def __init__(self, dataset_path: Path = DATASET_FILE) -> None:
        self.dataset_path = dataset_path
        self.open_trades: list[VirtualTrade] = []
        self.closed_trades: int = 0
        self.wins: int = 0
        self.losses: int = 0
        
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.dataset_path.exists():
            self.dataset_path.touch()

    def create_trade(
        self,
        action: TradeAction,
        signal: dict[str, Any],
        confidence: float = 0.0,
        reason: str = ""
    ) -> VirtualTrade:
        """Cria um novo trade virtual com barreiras definidas."""
        timestamp = int(signal.get('timestamp', 0))
        entry_price = float(signal.get('close', 0))
        atr = float(signal.get('volatility_atr', 50)) if signal.get('volatility_atr') else 50.0
        
        # Barreiras dinâmicas baseadas em ATR
        tp_price = entry_price + (atr * TP_ATR_MULT)
        sl_price = entry_price - (atr * SL_ATR_MULT)
        
        features = {
            'rsi': float(signal.get('rsi', 0)) if signal.get('rsi') else 0.0,
            'cvd': float(signal.get('cvd_absolute', 0)) if signal.get('cvd_absolute') else 0.0,
            'entropy': float(signal.get('entropy', 0)) if signal.get('entropy') else 0.0,
            'volatility_atr': atr,
            'funding_rate': float(signal.get('funding_rate', 0)) if signal.get('funding_rate') else 0.0,
            'ml_confidence': confidence,
        }
        
        trade = VirtualTrade(
            id=f"{action.value}_{timestamp}",
            timestamp=timestamp,
            action=action,
            entry_price=entry_price,
            tp_price=tp_price,
            sl_price=sl_price,
            max_bars=MAX_BARS,
            features=features
        )
        
        self.open_trades.append(trade)
        
        logger.info(
            f"🎯 Trade Virtual | {action.value} @ {entry_price:.2f} | "
            f"TP: {tp_price:.2f} | SL: {sl_price:.2f} | "
            f"Conf: {confidence:.2f} | {reason}"
        )
        
        return trade

    def update_trades(self, current_price: float) -> list[VirtualTrade]:
        """Atualiza todos os trades abertos."""
        closed = []
        still_open = []
        
        for trade in self.open_trades:
            if trade.check_barriers(current_price):
                closed.append(trade)
                self.closed_trades += 1
                
                if trade.label == TradeLabel.WIN:
                    self.wins += 1
                else:
                    self.losses += 1
                
                self._persist_trade(trade)
                
                emoji = "✅" if trade.label == TradeLabel.WIN else "❌"
                logger.info(
                    f"{emoji} Trade Finalizado: {trade.exit_reason} | "
                    f"PnL: {trade.pnl_percent:+.2f}% | "
                    f"Win Rate: {self.win_rate:.1f}%"
                )
            else:
                still_open.append(trade)
        
        self.open_trades = still_open
        return closed

    def _persist_trade(self, trade: VirtualTrade) -> None:
        """Persiste o trade rotulado."""
        record = {
            "timestamp": trade.timestamp,
            "features": trade.features,
            "action": trade.action.value,
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "exit_reason": trade.exit_reason,
            "outcome_label": trade.label.value if trade.label else None,
            "pnl_percent": round(trade.pnl_percent, 4) if trade.pnl_percent else None,
            "source": "live_ml_decision",
        }
        
        try:
            with open(self.dataset_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record) + '\n')
        except IOError as e:
            logger.error(f"Erro ao persistir trade: {e}")

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return (self.wins / total * 100) if total > 0 else 0.0


# ============================================================================
# DECISION PROCESSOR v2.0 (ML-POWERED)
# ============================================================================
class DecisionProcessor:
    """
    Orquestra o fluxo de decisão com ML:
    1. Lê sinais do stream:signals:tech
    2. Extrai features e consulta modelo ML
    3. Se confidence >= threshold, publica ordem
    4. Monitora trades via Triple Barrier
    """

    def __init__(self) -> None:
        self.redis: aioredis.Redis | None = None
        self.predictor = MLPredictor()
        self.labeler = TripleBarrierLabeler()
        self.running = False
        self.last_price: float = 0.0
        self.signals_evaluated: int = 0
        self.signals_triggered: int = 0
        self.last_signal_time: int = 0
        self.cooldown_bars: int = 5

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
            await self.redis.xgroup_create(SIGNAL_STREAM, CONSUMER_GROUP, id='$', mkstream=True)
            logger.info(f"Consumer group '{CONSUMER_GROUP}' criado")
        except aioredis.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.info(f"Consumer group '{CONSUMER_GROUP}' já existe")
            else:
                raise

    async def publish_order(self, signal: dict[str, Any], confidence: float, reason: str) -> None:
        """Publica ordem no stream de execução."""
        entry_price = float(signal.get('close', 0))
        atr = float(signal.get('volatility_atr', 50)) if signal.get('volatility_atr') else 50.0
        entropy = float(signal.get('entropy', 0)) if signal.get('entropy') else 0.0
        
        order = {
            "signal": "BUY",
            "confidence": str(round(confidence, 4)),
            "entry_price": str(round(entry_price, 2)),
            "stop_loss": str(round(entry_price - (atr * SL_ATR_MULT), 2)),
            "take_profit": str(round(entry_price + (atr * TP_ATR_MULT), 2)),
            "reason": f"{reason} | Entropy: {entropy:.2f}",
            "timestamp": str(signal.get('timestamp', 0)),
            "atr": str(round(atr, 2)),
        }
        
        await self.redis.xadd(ORDER_STREAM, order)
        
        logger.info(
            f"📤 ORDEM PUBLICADA | BUY @ {entry_price:.2f} | "
            f"TP: {order['take_profit']} | SL: {order['stop_loss']} | "
            f"Conf: {confidence:.2%}"
        )

    async def process_signal(self, signal: dict[str, Any]) -> None:
        """Processa um sinal do stream."""
        self.signals_evaluated += 1
        
        # Atualizar último preço
        close_price = signal.get('close')
        if close_price:
            try:
                self.last_price = float(close_price)
            except (ValueError, TypeError):
                pass
        
        # Atualizar trades abertos
        if self.last_price > 0:
            self.labeler.update_trades(self.last_price)
        
        # Cooldown: evitar sinais muito próximos
        timestamp = int(signal.get('timestamp', 0))
        if timestamp - self.last_signal_time < (self.cooldown_bars * 60000):
            return
        
        # Consultar modelo ML
        should_trade, confidence, reason = self.predictor.predict(signal)
        
        if should_trade:
            self.signals_triggered += 1
            self.last_signal_time = timestamp
            
            # Publicar ordem
            await self.publish_order(signal, confidence, reason)
            
            # Criar trade virtual para labeling
            self.labeler.create_trade(
                TradeAction.BUY, signal, confidence, reason
            )

    async def consume_stream(self) -> None:
        """Loop principal de consumo."""
        while self.running:
            try:
                messages = await self.redis.xreadgroup(
                    groupname=CONSUMER_GROUP,
                    consumername=CONSUMER_NAME,
                    streams={SIGNAL_STREAM: '>'},
                    count=10,
                    block=1000
                )
                
                if messages:
                    for stream_name, stream_messages in messages:
                        for message_id, data in stream_messages:
                            await self.process_signal(data)
                            await self.redis.xack(SIGNAL_STREAM, CONSUMER_GROUP, message_id)
                            
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
        logger.info("VOSTOK-1 :: Decision Engine v2.0 (ML-Powered)")
        logger.info("=" * 60)
        logger.info(f"ML Enabled: {self.predictor.ml_enabled}")
        logger.info(f"Confidence Threshold: {ML_CONFIDENCE_THRESHOLD:.0%}")
        logger.info(f"Signal Stream: {SIGNAL_STREAM}")
        logger.info(f"Order Stream: {ORDER_STREAM}")
        logger.info(f"Model: {MODEL_FILE}")
        logger.info("=" * 60)
        
        self.running = True
        await self.connect_redis()
        await self.setup_consumer_group()
        await self.consume_stream()

    async def stop(self) -> None:
        """Para o processador."""
        logger.info("Parando Decision Engine...")
        self.running = False
        
        if self.redis:
            await self.redis.close()
        
        hit_rate = (self.signals_triggered / self.signals_evaluated * 100) if self.signals_evaluated > 0 else 0
        
        logger.info(
            f"Decision parado. "
            f"Avaliados: {self.signals_evaluated} | "
            f"Disparados: {self.signals_triggered} ({hit_rate:.1f}%) | "
            f"Win Rate: {self.labeler.win_rate:.1f}%"
        )


# ============================================================================
# ENTRY POINT
# ============================================================================
async def main() -> None:
    processor = DecisionProcessor()
    try:
        await processor.start()
    except KeyboardInterrupt:
        await processor.stop()


if __name__ == "__main__":
    asyncio.run(main())
