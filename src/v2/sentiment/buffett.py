"""
VOSTOK V2 :: Crypto Buffett Sentiment Analyzer
===============================================
Implementação do prompt "Crypto Buffett" para análise contrarian de sentimento.

Persona: Warren Buffett / Howard Marks adaptado para Cripto
Filosofia: "Be fearful when others are greedy, and greedy when others are fearful."

Arquiteto: Petrovich
"""

import json
import logging
from dataclasses import dataclass
from enum import Enum

import httpx

logger = logging.getLogger("crypto_buffett")


class BuffettVerdict(str, Enum):
    """Vereditos do Crypto Buffett."""
    IGNORE = "Ignore"
    BUY_THE_FEAR = "Buy the Fear"
    SELL_THE_NEWS = "Sell the News"
    STRUCTURAL_RISK = "Structural Risk"


@dataclass
class BuffettAnalysis:
    """Resultado da análise do Crypto Buffett."""
    classification: str  # "Regulatory FUD", "Protocol Upgrade", "Macro Noise", "Adoption"
    sentiment_score: float  # -1.0 to 1.0
    impact_duration: str  # "Short-term", "Medium-term", "Structural"
    verdict: BuffettVerdict
    reasoning: str


# The Crypto Buffett Prompt
BUFFETT_SYSTEM_PROMPT = """You are "Crypto Buffett," a cold, analytical value investor focused on long-term cryptocurrency fundamentals.
Your goal is to separate MARKET NOISE from STRUCTURAL SIGNAL.

CORE PRINCIPLES:
1. Price volatility ≠ fundamental change. Ignore short-term noise.
2. Be contrarian: "Be fearful when others are greedy, and greedy when others are fearful."
3. Structural changes (Regulation, Protocol Upgrades) > Temporary FUD.

ANALYSIS PROCESS (Chain of Thought):
1. CLASSIFY: Is this Regulatory, Technical, Adoption, or just Price Action?
2. ASSESS IMPACT: Does this change the network effect or security in 3-5 years?
3. VERDICT: Would a patient value investor care?

OUTPUT FORMAT (JSON ONLY):
{
  "classification": "Regulatory FUD" | "Protocol Upgrade" | "Macro Noise" | "Adoption",
  "sentiment_score": float (-1.0 to 1.0),
  "impact_duration": "Short-term" | "Medium-term" | "Structural",
  "buffett_verdict": "Ignore" | "Buy the Fear" | "Sell the News" | "Structural Risk",
  "reasoning": "Max 20 words explanation"
}"""


class CryptoBuffett:
    """
    Analisador de sentimento baseado na persona Crypto Buffett.
    
    Usa Ollama (Qwen 2.5) para análise contrarian de notícias.
    """
    
    def __init__(
        self,
        ollama_host: str = "llm_engine",
        ollama_port: int = 11434,
        model: str = "qwen2.5:7b-instruct",
        timeout: float = 30.0,
    ):
        self.base_url = f"http://{ollama_host}:{ollama_port}"
        self.model = model
        self.timeout = timeout
        
        # Cache do último resultado
        self.last_analysis: BuffettAnalysis | None = None
    
    async def analyze(self, headlines: list[str]) -> BuffettAnalysis:
        """
        Analisa headlines de notícias usando o Crypto Buffett.
        
        Args:
            headlines: Lista de manchetes recentes
            
        Returns:
            BuffettAnalysis com veredito e score
        """
        if not headlines:
            return BuffettAnalysis(
                classification="Macro Noise",
                sentiment_score=0.0,
                impact_duration="Short-term",
                verdict=BuffettVerdict.IGNORE,
                reasoning="No headlines to analyze",
            )
        
        # Formatar prompt
        headlines_text = "\n".join(f"- {h}" for h in headlines[:5])  # Max 5
        user_prompt = f"Analyze these crypto headlines:\n{headlines_text}"
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "system": BUFFETT_SYSTEM_PROMPT,
                        "prompt": user_prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.1,
                            "num_ctx": 4096,
                        },
                    },
                )
                response.raise_for_status()
                
                result = response.json()
                raw_response = result.get("response", "{}")
                
                # Parse JSON
                analysis = self._parse_response(raw_response)
                self.last_analysis = analysis
                
                logger.info(
                    f"Buffett Analysis: {analysis.verdict.value} "
                    f"(score: {analysis.sentiment_score:.2f})"
                )
                
                return analysis
                
        except Exception as e:
            logger.error(f"Buffett analysis failed: {e}")
            return BuffettAnalysis(
                classification="Error",
                sentiment_score=0.0,
                impact_duration="Short-term",
                verdict=BuffettVerdict.IGNORE,
                reasoning=f"Analysis failed: {str(e)[:50]}",
            )
    
    def _parse_response(self, raw: str) -> BuffettAnalysis:
        """Parse JSON response from LLM."""
        try:
            # Tentar extrair JSON do response
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = raw[start:end]
                data = json.loads(json_str)
                
                # Map verdict string to enum
                verdict_str = data.get("buffett_verdict", "Ignore")
                verdict = BuffettVerdict(verdict_str)
                
                return BuffettAnalysis(
                    classification=data.get("classification", "Unknown"),
                    sentiment_score=float(data.get("sentiment_score", 0.0)),
                    impact_duration=data.get("impact_duration", "Short-term"),
                    verdict=verdict,
                    reasoning=data.get("reasoning", "")[:100],
                )
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse Buffett response: {e}")
        
        # Fallback
        return BuffettAnalysis(
            classification="Parse Error",
            sentiment_score=0.0,
            impact_duration="Short-term",
            verdict=BuffettVerdict.IGNORE,
            reasoning="Could not parse LLM response",
        )
    
    def should_veto(self, analysis: BuffettAnalysis) -> tuple[bool, str]:
        """
        Verifica se deve vetar o trade baseado na análise.
        
        Returns:
            (should_veto, reason)
        """
        if analysis.verdict == BuffettVerdict.STRUCTURAL_RISK:
            return True, "Structural Risk detected by Buffett"
        
        if analysis.sentiment_score < -0.8:
            return True, f"Extreme Fear: {analysis.sentiment_score:.2f}"
        
        return False, ""
