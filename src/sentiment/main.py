"""
VOSTOK-1 :: Sentiment Analysis Module
======================================
Análise de sentimento de notícias cripto usando LLM local (Ollama/Qwen).
Publica scores no Redis Stream para integração com Decision Engine.

Arquiteto: Petrovich | Operador: Vostok
Stack: Python 3.11 + Ollama + Redis
"""

import asyncio
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from typing import Any

import redis.asyncio as aioredis
import requests

# ============================================================================
# CONFIGURAÇÃO DE LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("sentiment")

# ============================================================================
# CONFIGURAÇÕES
# ============================================================================
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
SENTIMENT_STREAM = os.getenv("SENTIMENT_STREAM", "stream:signals:sentiment")

# Ollama LLM Configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "llm_engine")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", 11434))
OLLAMA_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/generate"
MODEL_NAME = os.getenv("LLM_MODEL", "qwen2.5:7b-instruct")

# News API
CRYPTOPANIC_API_KEY = os.getenv("CRYPTOPANIC_API_KEY", "")
CRYPTOPANIC_URL = "https://cryptopanic.com/api/v1/posts/"

# Timing
ANALYSIS_INTERVAL = int(os.getenv("ANALYSIS_INTERVAL", 900))  # 15 min default

# ============================================================================
# SYSTEM PROMPT - A DOUTRINA (Elite Hedge Fund Analyst)
# ============================================================================
SYSTEM_PROMPT = """ROLE: Elite Crypto Market Analyst (Hedge Fund Tier).

MISSION: Analyze news headlines and determine immediate market sentiment for Bitcoin (BTC).

RULES:
1. IGNORE FUD/NOISE: Generic opinion pieces = Neutral (0.0).
2. WEIGH REGULATION: SEC/Gov/CFTC news has 2x weight on sentiment.
3. DETECT INSTITUTIONAL FLOW: BlackRock/Fidelity/ETF news is High Impact.
4. PRIORITIZE FACTS: On-chain data > Rumors. Exchange flows > Twitter.
5. TIME SENSITIVITY: News older than 1h = reduced impact.

SCORING SCALE:
- Strong Bullish: +0.8 to +1.0 (ETF approval, major adoption)
- Bullish: +0.3 to +0.7 (institutional buy, positive regulation)
- Neutral: -0.2 to +0.2 (noise, irrelevant, opinion)
- Bearish: -0.7 to -0.3 (exchange hack, negative regulation)
- Strong Bearish: -1.0 to -0.8 (major ban, systemic failure)

OUTPUT FORMAT: JSON ONLY, no explanation, no markdown:
{"sentiment_score": <float>, "summary": "<one-line summary>", "confidence": <float 0-1>}"""


# ============================================================================
# MOCK NEWS DATA (Fallback when no API key)
# ============================================================================
MOCK_HEADLINES = [
    "Bitcoin holds steady above $85,000 amid market uncertainty",
    "SEC Commissioner hints at clearer crypto regulations in 2025",
    "BlackRock Bitcoin ETF sees record inflows of $500M",
    "Whale alert: 10,000 BTC moved from exchange to cold wallet",
    "Federal Reserve maintains interest rates, crypto markets react",
]


# ============================================================================
# NEWS FETCHER
# ============================================================================
class NewsFetcher:
    """Busca notícias de cripto de fontes externas."""

    def __init__(self, api_key: str = "") -> None:
        self.api_key = api_key
        self.session = requests.Session()
        self.session.timeout = 10

    def fetch_headlines(self, limit: int = 5) -> list[str]:
        """
        Busca headlines recentes.
        Retorna mock data se não houver API key.
        """
        if not self.api_key:
            logger.info("📰 Usando mock headlines (sem API key)")
            return MOCK_HEADLINES[:limit]

        try:
            params = {
                "auth_token": self.api_key,
                "currencies": "BTC",
                "kind": "news",
                "filter": "important",
                "public": "true",
            }
            
            response = self.session.get(CRYPTOPANIC_URL, params=params)
            response.raise_for_status()
            
            data = response.json()
            results = data.get("results", [])
            
            headlines = [item.get("title", "") for item in results[:limit]]
            logger.info(f"📰 {len(headlines)} headlines fetched from CryptoPanic")
            
            return headlines if headlines else MOCK_HEADLINES[:limit]
            
        except Exception as e:
            logger.warning(f"⚠️  Erro ao buscar notícias: {e}")
            return MOCK_HEADLINES[:limit]


# ============================================================================
# LLM ANALYZER
# ============================================================================
class LLMAnalyzer:
    """Analisa sentimento usando Ollama/Qwen."""

    def __init__(self) -> None:
        self.model = MODEL_NAME
        self.session = requests.Session()

    def analyze(self, headlines: list[str]) -> dict[str, Any] | None:
        """
        Envia headlines para o LLM e extrai análise de sentimento.
        Configuração: temperatura 0.1 (máxima precisão), ctx 4096.
        """
        if not headlines:
            return None

        # Formatar headlines para análise
        headlines_text = "\n".join([f"- {h}" for h in headlines])
        
        prompt = f"""Analyze these Bitcoin news headlines and provide sentiment:

{headlines_text}

Remember: Output JSON ONLY with sentiment_score, summary, and confidence."""

        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": SYSTEM_PROMPT,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Máxima precisão, zero criatividade
                "num_ctx": 4096,     # Janela de contexto estendida
            },
        }

        try:
            logger.info(f"🧠 Consultando LLM ({self.model})...")
            start_time = time.perf_counter()
            
            response = self.session.post(
                OLLAMA_URL,
                json=payload,
                timeout=120,  # LLM pode demorar
            )
            response.raise_for_status()
            
            elapsed = time.perf_counter() - start_time
            result = response.json()
            
            raw_response = result.get("response", "")
            logger.info(f"🧠 LLM respondeu em {elapsed:.2f}s")
            
            # Parse JSON da resposta
            return self._parse_response(raw_response)
            
        except requests.exceptions.ConnectionError:
            logger.error("❌ LLM Engine não está acessível. Verifique se vostok_llm está rodando.")
            return None
        except Exception as e:
            logger.error(f"❌ Erro ao consultar LLM: {e}")
            return None

    def _parse_response(self, raw: str) -> dict[str, Any] | None:
        """Extrai JSON da resposta do LLM."""
        try:
            # Tentar parse direto
            return json.loads(raw.strip())
        except json.JSONDecodeError:
            pass

        # Tentar extrair JSON do texto
        json_match = re.search(r'\{[^{}]+\}', raw)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        logger.warning(f"⚠️  Não foi possível parsear resposta: {raw[:200]}")
        return None


# ============================================================================
# SENTIMENT PROCESSOR
# ============================================================================
class SentimentProcessor:
    """Processador principal de sentimento."""

    def __init__(self) -> None:
        self.redis: aioredis.Redis | None = None
        self.news_fetcher = NewsFetcher(CRYPTOPANIC_API_KEY)
        self.analyzer = LLMAnalyzer()
        self.running = False
        self.analyses_done = 0

    async def connect_redis(self) -> None:
        """Conecta ao Redis."""
        self.redis = aioredis.Redis(
            host=REDIS_HOST, port=REDIS_PORT, decode_responses=True
        )
        await self.redis.ping()
        logger.info(f"✅ Redis conectado: {REDIS_HOST}:{REDIS_PORT}")

    async def publish_sentiment(self, analysis: dict[str, Any]) -> None:
        """Publica análise de sentimento no Redis Stream."""
        payload = {
            "timestamp": str(int(datetime.now(timezone.utc).timestamp() * 1000)),
            "sentiment_score": str(analysis.get("sentiment_score", 0)),
            "summary": str(analysis.get("summary", "")),
            "confidence": str(analysis.get("confidence", 0.5)),
            "model": MODEL_NAME,
            "source": "ollama",
        }
        
        await self.redis.xadd(SENTIMENT_STREAM, payload, maxlen=1000)
        self.analyses_done += 1
        
        score = float(analysis.get("sentiment_score", 0))
        emoji = "🔥" if score > 0.3 else "❄️" if score < -0.3 else "⚖️"
        
        logger.info(
            f"{emoji} Sentiment #{self.analyses_done} | "
            f"Score: {score:+.2f} | "
            f"Confidence: {analysis.get('confidence', 0):.2f} | "
            f"Summary: {analysis.get('summary', '')[:50]}..."
        )

    async def run_analysis_cycle(self) -> None:
        """Executa um ciclo de análise."""
        logger.info("=" * 60)
        logger.info("🔄 Iniciando ciclo de análise de sentimento...")
        
        # Buscar headlines
        headlines = self.news_fetcher.fetch_headlines(limit=5)
        
        if not headlines:
            logger.warning("⚠️  Nenhuma headline disponível")
            return
        
        for h in headlines:
            logger.info(f"   📰 {h[:60]}...")
        
        # Analisar com LLM
        analysis = self.analyzer.analyze(headlines)
        
        if analysis:
            await self.publish_sentiment(analysis)
        else:
            logger.warning("⚠️  Análise não retornou resultado válido")

    async def run_loop(self) -> None:
        """Loop principal de análise periódica."""
        while self.running:
            try:
                await self.run_analysis_cycle()
            except Exception as e:
                logger.exception(f"❌ Erro no ciclo: {e}")
            
            # Aguardar próximo ciclo
            logger.info(f"⏰ Próxima análise em {ANALYSIS_INTERVAL // 60} minutos...")
            await asyncio.sleep(ANALYSIS_INTERVAL)

    async def start(self) -> None:
        """Inicia o processador."""
        logger.info("")
        logger.info("╔══════════════════════════════════════════════════════════════╗")
        logger.info("║   VOSTOK-1 :: SENTIMENT ANALYSIS MODULE                     ║")
        logger.info("║   Elite Crypto Market Analyst (Hedge Fund Tier)             ║")
        logger.info("╚══════════════════════════════════════════════════════════════╝")
        logger.info("")
        logger.info(f"LLM: {MODEL_NAME} @ {OLLAMA_URL}")
        logger.info(f"Output Stream: {SENTIMENT_STREAM}")
        logger.info(f"Analysis Interval: {ANALYSIS_INTERVAL}s ({ANALYSIS_INTERVAL // 60} min)")
        logger.info("")
        
        self.running = True
        await self.connect_redis()
        await self.run_loop()

    async def stop(self) -> None:
        """Para o processador."""
        logger.info("Parando Sentiment Processor...")
        self.running = False
        if self.redis:
            await self.redis.close()


# ============================================================================
# ENTRY POINT
# ============================================================================
async def main() -> None:
    processor = SentimentProcessor()
    try:
        await processor.start()
    except KeyboardInterrupt:
        await processor.stop()


if __name__ == "__main__":
    asyncio.run(main())
