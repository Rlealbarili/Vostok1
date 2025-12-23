"""
VOSTOK V2 :: Crypto Buffett Intelligence
==========================================
Análise fundamentalista contrarian usando LLM local (Qwen 2.5).

Filosofia:
- "Be fearful when others are greedy, and greedy when others are fearful."
- Ignorar ruído de preço, focar em mudanças ESTRUTURAIS
- Economic Moat + Margin of Safety adaptados para Cripto

Persona: Warren Buffett / Howard Marks

Arquiteto: Petrovich | Operador: Vostok
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Literal

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

try:
    from pydantic import BaseModel, Field, ValidationError
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    # Fallback simples se Pydantic não estiver disponível
    BaseModel = object
    Field = lambda **kwargs: None
    ValidationError = Exception

logger = logging.getLogger("crypto_buffett")


# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_OLLAMA_HOST = "llm_engine"
DEFAULT_OLLAMA_PORT = 11434
DEFAULT_MODEL = "qwen2.5:7b-instruct"
DEFAULT_TIMEOUT = 45.0

# The Soul of the System
BUFFETT_SYSTEM_PROMPT = """You are 'Crypto Buffett', a cold, analytical value investor.
You ignore price volatility (noise) and focus ONLY on structural changes (Regulation, Protocol, Adoption).
Use Chain of Thought to separate FUD from FACT.

CORE PRINCIPLES:
1. Price volatility ≠ fundamental change. A 10% crash means NOTHING without structural damage.
2. Be contrarian: "Be fearful when others are greedy, and greedy when others are fearful."
3. Structural changes (Regulation, Protocol Upgrades) matter more than temporary FUD.
4. Economic Moat: Does this news affect Bitcoin's network effect, security, or adoption?
5. Margin of Safety: Panic creates opportunity if fundamentals are intact.

ANALYSIS PROCESS (Chain of Thought):
1. CLASSIFY: Is this Regulatory, Technical/Protocol, Adoption, or just Price Action/Noise?
2. ASSESS MOAT: Does this threaten the network effect, security, or decentralization?
3. ASSESS IMPACT: Will this matter in 3-5 years? Or is it temporary?
4. VERDICT: What would a patient value investor do?

OUTPUT FORMAT (JSON ONLY, no markdown, no explanation):
{
  "sentiment_score": <float between -1.0 and 1.0>,
  "impact_duration": "Short-term Noise" | "Medium-term" | "Structural Change",
  "buffett_verdict": "Ignore" | "Buy the Fear" | "Sell the News" | "Structural Risk",
  "reasoning": "<max 150 chars explaining your verdict>"
}"""


# ============================================================================
# ENUMS & PYDANTIC MODELS
# ============================================================================

class BuffettVerdict(str, Enum):
    """Vereditos possíveis do Crypto Buffett."""
    IGNORE = "Ignore"
    BUY_THE_FEAR = "Buy the Fear"
    SELL_THE_NEWS = "Sell the News"
    STRUCTURAL_RISK = "Structural Risk"


class ImpactDuration(str, Enum):
    """Duração do impacto da notícia."""
    SHORT_TERM = "Short-term Noise"
    MEDIUM_TERM = "Medium-term"
    STRUCTURAL = "Structural Change"


class TradePermission(str, Enum):
    """Permissão de trade baseada no veredito."""
    HARD_VETO = "HARD_VETO"       # Não operar de jeito nenhum
    BOOST_LONG = "BOOST_LONG"     # Oportunidade contrarian
    PROCEED = "PROCEED"           # Seguir modelo matemático


# Pydantic model para validação rigorosa
if HAS_PYDANTIC:
    class BuffettAnalysisModel(BaseModel):
        """Schema Pydantic para validação do JSON do LLM."""
        sentiment_score: float = Field(ge=-1.0, le=1.0)
        impact_duration: Literal["Short-term Noise", "Medium-term", "Structural Change"]
        buffett_verdict: Literal["Ignore", "Buy the Fear", "Sell the News", "Structural Risk"]
        reasoning: str = Field(max_length=200)


@dataclass
class BuffettAnalysis:
    """Resultado da análise do Crypto Buffett."""
    sentiment_score: float
    impact_duration: str
    buffett_verdict: str
    reasoning: str
    
    # Metadados
    raw_response: str = ""
    parse_success: bool = True
    
    def get_permission(self) -> TradePermission:
        """Retorna a permissão de trade baseada no veredito."""
        return CryptoBuffett.get_trade_permission(self.buffett_verdict)
    
    def to_dict(self) -> dict:
        """Converte para dict."""
        return {
            "sentiment_score": self.sentiment_score,
            "impact_duration": self.impact_duration,
            "buffett_verdict": self.buffett_verdict,
            "reasoning": self.reasoning,
            "permission": self.get_permission().value,
        }


# ============================================================================
# CRYPTO BUFFETT CLASS
# ============================================================================

class CryptoBuffett:
    """
    Analisador de sentimento baseado na persona do Value Investor.
    
    Usa Ollama (Qwen 2.5) local para análise contrarian de notícias.
    Aplica framework de Value Investing (Economic Moat, Margin of Safety).
    
    Uso:
        buffett = CryptoBuffett()
        result = await buffett.analyze_news("Bitcoin crashes 10% on liquidation cascade")
        
        if result.get_permission() == TradePermission.HARD_VETO:
            logger.warning("VETO: Structural Risk detected")
            return
        elif result.get_permission() == TradePermission.BOOST_LONG:
            logger.info("Contrarian opportunity: Buy the Fear")
    """
    
    def __init__(
        self,
        ollama_host: str = DEFAULT_OLLAMA_HOST,
        ollama_port: int = DEFAULT_OLLAMA_PORT,
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.base_url = f"http://{ollama_host}:{ollama_port}"
        self.model = model
        self.timeout = timeout
        
        # Cache do último resultado
        self._last_analysis: BuffettAnalysis | None = None
        
        logger.info(
            f"CryptoBuffett initialized: {self.base_url}, model={model}"
        )
    
    async def analyze_news(self, headline: str) -> BuffettAnalysis:
        """
        Analisa uma headline de notícia usando o framework Crypto Buffett.
        
        Args:
            headline: Manchete da notícia a ser analisada
            
        Returns:
            BuffettAnalysis com veredito e score
        """
        if not headline or not headline.strip():
            return self._create_neutral_result("Empty headline")
        
        if not HAS_HTTPX:
            logger.error("httpx not installed, returning neutral")
            return self._create_neutral_result("httpx not available")
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "system": BUFFETT_SYSTEM_PROMPT,
                        "prompt": f"Analyze this crypto headline:\n\n\"{headline}\"",
                        "stream": False,
                        "options": {
                            "temperature": 0.1,  # Máxima precisão
                            "num_ctx": 4096,
                            "num_predict": 256,  # Limitar output
                        },
                    },
                )
                response.raise_for_status()
                
                result = response.json()
                raw_response = result.get("response", "")
                
                # Parse e validação
                analysis = self._parse_and_validate(raw_response, headline)
                self._last_analysis = analysis
                
                # Log
                emoji = self._get_verdict_emoji(analysis.buffett_verdict)
                logger.info(
                    f"{emoji} Buffett: {analysis.buffett_verdict} "
                    f"(score: {analysis.sentiment_score:+.2f}) | "
                    f"{analysis.reasoning[:50]}..."
                )
                
                return analysis
                
        except httpx.TimeoutException:
            logger.warning(f"Ollama timeout ({self.timeout}s)")
            return self._create_neutral_result("LLM timeout")
        except httpx.ConnectError:
            logger.error(f"Cannot connect to Ollama at {self.base_url}")
            return self._create_neutral_result("LLM connection failed")
        except Exception as e:
            logger.exception(f"Buffett analysis failed: {e}")
            return self._create_neutral_result(f"Error: {str(e)[:50]}")
    
    async def analyze_multiple(self, headlines: list[str]) -> BuffettAnalysis:
        """
        Analisa múltiplas headlines e retorna o veredito mais severo.
        
        Args:
            headlines: Lista de manchetes
            
        Returns:
            BuffettAnalysis com o veredito mais restritivo
        """
        if not headlines:
            return self._create_neutral_result("No headlines")
        
        # Concatenar para análise conjunta
        combined = "\n".join(f"- {h}" for h in headlines[:5])  # Max 5
        
        if not HAS_HTTPX:
            return self._create_neutral_result("httpx not available")
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "system": BUFFETT_SYSTEM_PROMPT,
                        "prompt": f"Analyze these crypto headlines as a whole:\n\n{combined}",
                        "stream": False,
                        "options": {
                            "temperature": 0.1,
                            "num_ctx": 4096,
                        },
                    },
                )
                response.raise_for_status()
                
                result = response.json()
                raw_response = result.get("response", "")
                
                return self._parse_and_validate(raw_response, "Multiple headlines")
                
        except Exception as e:
            logger.exception(f"Multi-headline analysis failed: {e}")
            return self._create_neutral_result(f"Error: {str(e)[:50]}")
    
    def _parse_and_validate(self, raw_response: str, context: str) -> BuffettAnalysis:
        """Parse e valida o JSON do LLM com Pydantic."""
        
        # Tentar extrair JSON do response
        json_str = self._extract_json(raw_response)
        
        if not json_str:
            logger.warning(f"No JSON found in response for: {context[:30]}")
            return self._create_neutral_result("No JSON in response", raw_response)
        
        try:
            data = json.loads(json_str)
            
            # Validar com Pydantic se disponível
            if HAS_PYDANTIC:
                validated = BuffettAnalysisModel(**data)
                return BuffettAnalysis(
                    sentiment_score=validated.sentiment_score,
                    impact_duration=validated.impact_duration,
                    buffett_verdict=validated.buffett_verdict,
                    reasoning=validated.reasoning,
                    raw_response=raw_response,
                    parse_success=True,
                )
            else:
                # Validação manual básica
                return BuffettAnalysis(
                    sentiment_score=max(-1.0, min(1.0, float(data.get("sentiment_score", 0)))),
                    impact_duration=data.get("impact_duration", "Short-term Noise"),
                    buffett_verdict=data.get("buffett_verdict", "Ignore"),
                    reasoning=str(data.get("reasoning", ""))[:200],
                    raw_response=raw_response,
                    parse_success=True,
                )
                
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            return self._create_neutral_result("JSON parse error", raw_response)
        except ValidationError as e:
            logger.warning(f"Pydantic validation error: {e}")
            return self._create_neutral_result("Validation error", raw_response)
        except Exception as e:
            logger.warning(f"Parse error: {e}")
            return self._create_neutral_result(str(e), raw_response)
    
    def _extract_json(self, text: str) -> str | None:
        """Extrai JSON de uma string que pode conter markdown ou texto extra."""
        # Tentar encontrar JSON diretamente
        start = text.find("{")
        end = text.rfind("}") + 1
        
        if start >= 0 and end > start:
            return text[start:end]
        
        # Tentar remover markdown code blocks
        code_block_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        match = re.search(code_block_pattern, text)
        if match:
            inner = match.group(1)
            start = inner.find("{")
            end = inner.rfind("}") + 1
            if start >= 0 and end > start:
                return inner[start:end]
        
        return None
    
    def _create_neutral_result(
        self, 
        reason: str, 
        raw_response: str = ""
    ) -> BuffettAnalysis:
        """Cria resultado neutro (Ignore) para fallback."""
        return BuffettAnalysis(
            sentiment_score=0.0,
            impact_duration="Short-term Noise",
            buffett_verdict="Ignore",
            reasoning=f"Fallback: {reason}",
            raw_response=raw_response,
            parse_success=False,
        )
    
    def _get_verdict_emoji(self, verdict: str) -> str:
        """Retorna emoji para o veredito."""
        emoji_map = {
            "Ignore": "⚪",
            "Buy the Fear": "🟢",
            "Sell the News": "🟡",
            "Structural Risk": "🔴",
        }
        return emoji_map.get(verdict, "⚪")
    
    @staticmethod
    def get_trade_permission(verdict: str) -> TradePermission:
        """
        Converte veredito Buffett em permissão de trade.
        
        Args:
            verdict: O buffett_verdict retornado pela análise
            
        Returns:
            TradePermission indicando a ação a tomar
        """
        if verdict == "Structural Risk":
            return TradePermission.HARD_VETO
        elif verdict == "Buy the Fear":
            return TradePermission.BOOST_LONG
        else:
            # "Ignore" ou "Sell the News" = seguir modelo matemático
            return TradePermission.PROCEED
    
    def get_last_analysis(self) -> BuffettAnalysis | None:
        """Retorna a última análise (cache)."""
        return self._last_analysis


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

async def quick_buffett_check(
    headline: str,
    ollama_host: str = DEFAULT_OLLAMA_HOST,
) -> dict:
    """
    Verificação rápida de Buffett para uma headline.
    
    Returns:
        dict com verdict, score e permission
    """
    buffett = CryptoBuffett(ollama_host=ollama_host)
    result = await buffett.analyze_news(headline)
    return result.to_dict()


def get_permission_from_verdict(verdict: str) -> str:
    """Helper function para obter permissão do veredito."""
    return CryptoBuffett.get_trade_permission(verdict).value


# ============================================================================
# MAIN (EXAMPLE USAGE)
# ============================================================================

async def main():
    """Exemplo de uso do CryptoBuffett."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
    )
    
    print("=" * 60)
    print("🎩 CRYPTO BUFFETT - Value Investing Intelligence")
    print("=" * 60)
    
    # Testar com headlines de exemplo
    test_headlines = [
        "Bitcoin crashes 10% on liquidation cascade",
        "SEC sues major exchange for fraud",
        "New meme coin surges 500%",
        "Ethereum completes major upgrade reducing fees by 90%",
        "Russia bans all cryptocurrency transactions",
    ]
    
    # Usar localhost para teste local
    buffett = CryptoBuffett(ollama_host="localhost")
    
    for headline in test_headlines:
        print(f"\n📰 Headline: \"{headline}\"")
        print("-" * 50)
        
        result = await buffett.analyze_news(headline)
        
        print(f"  Score:    {result.sentiment_score:+.2f}")
        print(f"  Duration: {result.impact_duration}")
        print(f"  Verdict:  {result.buffett_verdict}")
        print(f"  Reason:   {result.reasoning}")
        print(f"  Permission: {result.get_permission().value}")
        print()
    
    print("=" * 60)
    print("Test complete!")


if __name__ == "__main__":
    asyncio.run(main())
