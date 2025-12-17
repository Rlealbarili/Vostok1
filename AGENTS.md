# AGENTS.MD - Vostok-1 Knowledge Base

> **Status do Sistema:** 🟢 MONITOR TUI OPERACIONAL
> **Última Atualização:** 2024-12-17T10:00:00-03:00
> **Engenheiro Chefe:** Petrovich
> **Operador:** Vostok

## 1. Missão
Sistema de trading autônomo com Data Labeling e monitoramento em tempo real.

## 2. Arquitetura
```
Binance WS → Ingestor → Quant → Decision → Dataset
                  ↓
              Monitor TUI
```

## 3. Estado Atual
- [x] Infraestrutura Docker (Redis 7 + TimescaleDB PG16)
- [x] **Ingestor Sniper**: watch_trades + watch_funding_rate ✅
- [x] **Quant Sniper**: CVD + Entropia + ATR + Parkinson ✅
- [x] **Decision Engine**: TripleBarrierLabeler ✅
- [x] **Monitor TUI**: Dashboard Rich ✅
- [ ] Módulo Sentiment (Qwen)
- [ ] Módulo Executor

## 4. Monitor TUI (2024-12-17) ✅
- Dashboard em tempo real com biblioteca Rich
- Market Intelligence: Price, RSI, CVD, ATR, Funding
- Regime Panel: Entropy (CHAOS MODE alert)
- Dataset Log: últimos trades rotulados

**Executar:** `docker compose run --rm monitor`

## 5. Containers
| Container | Status | Função |
|-----------|--------|--------|
| vostok_redis | Healthy | Event Bus |
| vostok_timescale | Healthy | Cold Storage |
| vostok_ingestor | Healthy | Trades + Funding |
| vostok_quant | Healthy | OHLCV + Indicators |
| vostok_decision | Healthy | Data Labeling |
| vostok_monitor | Interativo | TUI Dashboard |

## 6. APIs Configuradas (.env)
- Binance (Futures)
- Coinglass
- CryptoPanic
- NewsAPI

## 7. Diretrizes
- Nunca comitar `.env`
- Type hints obrigatórios
- Logs JSON estruturados