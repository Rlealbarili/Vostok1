# 📡 Vostok On-Chain Module: Long-Range Radar

> **Objetivo:** Detectar movimentos de baleias e congestionamento de rede ANTES que afetem o preço.  
> **Natureza:** Radar de Longo Alcance (assíncrono, não-bloqueante).  
> **Status:** 🟡 Em Planejamento  
> **Data:** 2025-12-22  

---

## 1. Arquitetura do Serviço

### 1.1 Características

| Aspecto | Especificação |
|---------|---------------|
| **Container** | `vostok_onchain` |
| **Linguagem** | Python 3.11 + asyncio |
| **Ciclo** | Loop a cada 5-10 minutos |
| **Integração** | Redis (não-bloqueante) |
| **Impacto no Trading** | Zero latência no loop de 1 min |

### 1.2 Diagrama de Fluxo

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Blockchain.com │────▶│  vostok_onchain  │────▶│     Redis       │
│  Mempool.space  │     │   (Radar Loop)   │     │ vostok:context  │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
                                                 ┌─────────────────┐
                                                 │ Decision Engine │
                                                 │   (LightGBM)    │
                                                 └─────────────────┘
```

---

## 2. Fontes de Dados (APIs Gratuitas)

### 2.1 APIs Primárias

| API | Endpoint | Dados |
|-----|----------|-------|
| **Blockchain.com** | `api.blockchain.info` | Exchange flows, Large Tx |
| **Mempool.space** | `mempool.space/api` | Taxas, Congestionamento |
| **Blockchair** | `api.blockchair.com` | Whale transactions |

### 2.2 Rate Limits

| API | Limite Gratuito | Intervalo Recomendado |
|-----|-----------------|----------------------|
| Blockchain.com | 300 req/5min | 10 min |
| Mempool.space | 100 req/min | 5 min |
| Blockchair | 30 req/min | 10 min |

---

## 3. Métricas & Lógica de Negócio

### 3.1 Exchange Inflow (Fluxo para Exchanges)

```python
class ExchangeInflowMonitor:
    """
    Monitora fluxo de BTC entrando em exchanges.
    Alto inflow = Pressão de venda iminente.
    """
    BINANCE_ADDRESSES = [
        "bc1qm34lsc65zpw79lxes69zkqmk6ee3ewf0j77s3h",
        "3JZq4atUahhuA9rLhXLMhhTo133J9rF97j",
        # ... mais endereços conhecidos
    ]
    
    THRESHOLD_BTC_1H = 500  # 500 BTC em 1h = alerta
    THRESHOLD_BTC_4H = 2000 # 2000 BTC em 4h = veto
    
    async def check(self) -> dict:
        inflow_1h = await self.get_inflow(hours=1)
        inflow_4h = await self.get_inflow(hours=4)
        
        if inflow_4h > self.THRESHOLD_BTC_4H:
            return {
                "risk_level": "CRITICAL",
                "action": "VETO_LONG",
                "reason": f"Exchange Inflow: {inflow_4h} BTC em 4h"
            }
        elif inflow_1h > self.THRESHOLD_BTC_1H:
            return {
                "risk_level": "HIGH",
                "action": "REDUCE_SIZE",
                "reason": f"Exchange Inflow: {inflow_1h} BTC em 1h"
            }
        return {"risk_level": "NORMAL", "action": "PROCEED"}
```

---

### 3.2 Mempool Congestion (Taxas de Rede)

```python
class MempoolMonitor:
    """
    Monitora congestionamento da mempool Bitcoin.
    Taxas altas = Alta volatilidade esperada.
    """
    # Taxas em sat/vB
    THRESHOLD_HIGH = 100      # Congestionamento moderado
    THRESHOLD_CRITICAL = 300  # Congestionamento severo
    
    async def check(self) -> dict:
        # Mempool.space API
        response = await self.fetch("https://mempool.space/api/v1/fees/recommended")
        fastest_fee = response.get("fastestFee", 0)
        
        if fastest_fee > self.THRESHOLD_CRITICAL:
            return {
                "risk_level": "CRITICAL",
                "action": "PAUSE_5MIN",
                "reason": f"Mempool congestionada: {fastest_fee} sat/vB"
            }
        elif fastest_fee > self.THRESHOLD_HIGH:
            return {
                "risk_level": "HIGH",
                "action": "REDUCE_SIZE",
                "reason": f"Taxas elevadas: {fastest_fee} sat/vB"
            }
        return {"risk_level": "NORMAL", "action": "PROCEED"}
```

---

### 3.3 Whale Alert (Transações Grandes)

```python
class WhaleMonitor:
    """
    Detecta transações de baleias (>$10M USD).
    Movimento de baleia = Possível manipulação.
    """
    THRESHOLD_USD = 10_000_000  # $10M
    COOLDOWN_MINUTES = 5
    
    async def check(self) -> dict:
        large_txs = await self.get_recent_large_transactions()
        
        for tx in large_txs:
            if tx['value_usd'] > self.THRESHOLD_USD:
                # Determinar direção
                if tx['to_exchange']:
                    action = "VETO_LONG"
                    direction = "para Exchange (venda)"
                else:
                    action = "VETO_SHORT"
                    direction = "de Exchange (compra)"
                
                return {
                    "risk_level": "CRITICAL",
                    "action": action,
                    "reason": f"Whale Tx: ${tx['value_usd']/1e6:.1f}M {direction}",
                    "cooldown_until": int(time.time()) + (self.COOLDOWN_MINUTES * 60)
                }
        
        return {"risk_level": "NORMAL", "action": "PROCEED"}
```

---

## 4. Estrutura de Arquivos

```
src/
└── onchain/
    ├── __init__.py
    ├── main.py           # Entry point (async loop)
    ├── radar.py          # Classe principal OnChainRadar
    ├── monitors/
    │   ├── __init__.py
    │   ├── exchange_flow.py
    │   ├── mempool.py
    │   └── whale.py
    ├── apis/
    │   ├── __init__.py
    │   ├── blockchain.py
    │   └── mempool_space.py
    └── requirements.txt
```

---

## 5. Script Principal (`src/onchain/radar.py`)

```python
"""
VOSTOK-1 :: On-Chain Radar Module
=================================
Long-range detection of whale movements and network congestion.
Runs asynchronously, does NOT block the main trading loop.

Arquiteto: Petrovich | Operador: Vostok
Stack: Python 3.11 + aiohttp + redis-py
"""

import asyncio
import json
import logging
from datetime import datetime, timezone

import aiohttp
import redis.asyncio as aioredis

from monitors.exchange_flow import ExchangeInflowMonitor
from monitors.mempool import MempoolMonitor
from monitors.whale import WhaleMonitor

logger = logging.getLogger("onchain")

# Configuração
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_KEY = "vostok:context:onchain"
SCAN_INTERVAL = int(os.getenv("ONCHAIN_INTERVAL", 300))  # 5 min default


class OnChainRadar:
    """
    Radar On-Chain para detecção antecipada de movimentos de mercado.
    """
    
    def __init__(self):
        self.redis: aioredis.Redis | None = None
        self.session: aiohttp.ClientSession | None = None
        self.running = False
        
        # Monitores
        self.monitors = [
            ExchangeInflowMonitor(),
            MempoolMonitor(),
            WhaleMonitor(),
        ]
    
    async def connect(self):
        """Inicializa conexões."""
        self.redis = aioredis.Redis(host=REDIS_HOST, decode_responses=True)
        self.session = aiohttp.ClientSession()
        await self.redis.ping()
        logger.info("✅ On-Chain Radar conectado")
    
    async def scan(self) -> dict:
        """Executa scan completo de todos os monitores."""
        results = []
        
        for monitor in self.monitors:
            try:
                result = await monitor.check()
                results.append(result)
            except Exception as e:
                logger.warning(f"⚠️ Erro no monitor {monitor.__class__.__name__}: {e}")
        
        # Agregar resultados (pior caso prevalece)
        return self.aggregate_results(results)
    
    def aggregate_results(self, results: list[dict]) -> dict:
        """Agrega resultados de múltiplos monitores."""
        risk_priority = {"CRITICAL": 3, "HIGH": 2, "NORMAL": 1}
        
        worst_result = {"risk_level": "NORMAL", "action": "PROCEED"}
        
        for result in results:
            if risk_priority.get(result.get("risk_level"), 0) > \
               risk_priority.get(worst_result.get("risk_level"), 0):
                worst_result = result
        
        worst_result["timestamp"] = int(datetime.now(timezone.utc).timestamp())
        worst_result["scan_count"] = len(results)
        
        return worst_result
    
    async def publish(self, context: dict):
        """Publica contexto On-Chain no Redis."""
        await self.redis.set(REDIS_KEY, json.dumps(context))
        await self.redis.expire(REDIS_KEY, SCAN_INTERVAL * 2)  # TTL
        
        emoji = "🔴" if context["risk_level"] == "CRITICAL" else \
                "🟡" if context["risk_level"] == "HIGH" else "🟢"
        
        logger.info(
            f"{emoji} On-Chain Update | "
            f"Risk: {context['risk_level']} | "
            f"Action: {context['action']}"
        )
    
    async def run(self):
        """Loop principal do radar."""
        logger.info("=" * 60)
        logger.info("VOSTOK-1 :: On-Chain Radar (Long-Range Detection)")
        logger.info("=" * 60)
        logger.info(f"Scan Interval: {SCAN_INTERVAL}s ({SCAN_INTERVAL//60} min)")
        logger.info(f"Redis Key: {REDIS_KEY}")
        logger.info("=" * 60)
        
        await self.connect()
        self.running = True
        
        while self.running:
            try:
                context = await self.scan()
                await self.publish(context)
            except Exception as e:
                logger.exception(f"❌ Erro no scan: {e}")
            
            await asyncio.sleep(SCAN_INTERVAL)


async def main():
    radar = OnChainRadar()
    await radar.run()


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 6. Integração Redis

### 6.1 Chave de Contexto

| Chave | Tipo | TTL | Descrição |
|-------|------|-----|-----------|
| `vostok:context:onchain` | JSON String | 10 min | Último scan On-Chain |

### 6.2 Schema do Payload

```json
{
  "risk_level": "CRITICAL" | "HIGH" | "NORMAL",
  "action": "VETO_LONG" | "VETO_SHORT" | "REDUCE_SIZE" | "PAUSE_5MIN" | "PROCEED",
  "reason": "Descrição curta do motivo",
  "cooldown_until": 1708992000,  // opcional, Unix timestamp
  "timestamp": 1708992000,
  "scan_count": 3
}
```

---

## 7. Integração com Decision Engine

### 7.1 Leitura do Contexto On-Chain

```python
# Em src/decision/engine.py

async def get_onchain_context() -> dict | None:
    """Lê o contexto On-Chain do Redis."""
    try:
        data = await redis.get("vostok:context:onchain")
        if data:
            return json.loads(data)
    except Exception:
        pass
    return None


async def apply_onchain_filter(signal: str, confidence: float) -> tuple[str, float]:
    """
    Aplica filtro On-Chain ao sinal de trading.
    
    Returns:
        (new_signal, new_confidence)
    """
    context = await get_onchain_context()
    
    if not context:
        return signal, confidence  # Sem dados, segue normal
    
    action = context.get("action", "PROCEED")
    
    # VETO: Cancelar trade
    if action == "VETO_LONG" and signal == "BUY":
        logger.warning(f"🛑 On-Chain VETO: {context.get('reason')}")
        return "HOLD", 0.0
    
    if action == "VETO_SHORT" and signal == "SELL":
        logger.warning(f"🛑 On-Chain VETO: {context.get('reason')}")
        return "HOLD", 0.0
    
    # PAUSE: Cooldown ativo
    if action == "PAUSE_5MIN":
        cooldown_until = context.get("cooldown_until", 0)
        if time.time() < cooldown_until:
            logger.info("⏸️ On-Chain PAUSE ativo")
            return "HOLD", 0.0
    
    # REDUCE_SIZE: Diminuir confiança
    if action == "REDUCE_SIZE":
        logger.info(f"📉 On-Chain: Reduzindo size (reason: {context.get('reason')})")
        return signal, confidence * 0.5  # 50% do size normal
    
    return signal, confidence
```

---

## 8. Docker Compose

```yaml
# Adicionar ao docker-compose.yml

  onchain:
    build:
      context: .
      dockerfile: Dockerfile.onchain
    container_name: vostok_onchain
    restart: unless-stopped
    env_file:
      - .env
    environment:
      REDIS_HOST: redis
      REDIS_PORT: 6379
      ONCHAIN_INTERVAL: 300  # 5 minutos
    depends_on:
      redis:
        condition: service_healthy
    networks:
      - vostok_net
```

---

## 9. Plano de Implementação

### FASE 1: Setup Básico 🔴

| Task | Descrição |
|------|-----------|
| 1.1 | Criar estrutura `src/onchain/` |
| 1.2 | Implementar `MempoolMonitor` (mais simples) |
| 1.3 | Testar escrita no Redis |

### FASE 2: Monitores Avançados 🔴

| Task | Descrição |
|------|-----------|
| 2.1 | Implementar `WhaleMonitor` (Blockchair API) |
| 2.2 | Implementar `ExchangeInflowMonitor` |
| 2.3 | Calibrar thresholds com dados históricos |

### FASE 3: Integração 🔴

| Task | Descrição |
|------|-----------|
| 3.1 | Adicionar `apply_onchain_filter()` ao Decision Engine |
| 3.2 | Logs de auditoria (trades vetados por On-Chain) |
| 3.3 | Dashboard: painel On-Chain no Monitor TUI |

---

## 10. Métricas de Sucesso

| Métrica | Alvo |
|---------|------|
| **Latência do Scan** | < 5s |
| **Uptime** | > 99% |
| **Trades vetados corretamente** | Taxa de acerto > 70% |
| **Falsos positivos** | < 10% |

---

## 11. Riscos e Mitigações

| Risco | Mitigação |
|-------|-----------|
| API rate limit | Cache agressivo, fallback entre APIs |
| Dados atrasados | TTL no Redis, ignorar dados > 15 min |
| Falsos vetos | Log todos os vetos, revisar semanalmente |

---

*Documento de Arquitetura - Vostok On-Chain Radar Module*  
*Última atualização: 2025-12-22*
