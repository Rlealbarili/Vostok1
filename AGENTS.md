# AGENTS.MD - Vostok-1 Knowledge Base

> **Status do Sistema:** 🟢 SNIPER UPGRADE OPERACIONAL
> **Última Atualização:** 2024-12-16T17:10:00-03:00
> **Engenheiro Chefe:** Petrovich
> **Operador:** Vostok

## 1. Missão
Sistema de trading autônomo de baixa latência com detecção de regime e order flow sintético.

## 2. Arquitetura Sniper
```
Binance WS (Trades+Funding) → Ingestor → stream:market:btc_usdt → Quant → stream:signals:tech
```

## 3. Estado Atual
- [x] Infraestrutura Docker (Redis 7 + TimescaleDB PG16)
- [x] **Ingestor Sniper**: watch_trades + watch_funding_rate ✅
- [x] **Quant Sniper**: CVD + Entropia + ATR + Parkinson ✅
- [ ] Módulo Sentiment (Qwen)
- [ ] Módulo Decision (Motor)
- [ ] Módulo Executor

## 4. Sprint Sniper (2024-12-16) ✅

### Ingestor Upgrade
- `watch_trades` + `watch_funding_rate` concorrentes
- Payload com campo `type` (trade/funding)
- Funding Rate: `9.884e-05` capturado

### Quant Upgrade
- CVD (Cumulative Volume Delta): buy_vol - sell_vol
- Entropia de Shannon (detector de ruído)
- ATR (Average True Range)
- Volatilidade de Parkinson (High/Low)
- Calc time: **0.22ms** (target < 2ms ✓)

### Validação
```
✔ stream:signals:tech → 924+ sinais
✔ CVD: -5.12 | ATR: 42.74 | Parkinson: 0.0363
✔ Funding Rate: 9.884e-05
```

## 5. Payload `stream:signals:tech`
| Campo | Descrição |
|-------|-----------|
| open, high, low, close, volume | OHLCV |
| cvd_absolute, buy_volume, sell_volume | Order Flow |
| entropy | Detector de ruído (0-1) |
| volatility_atr, volatility_parkinson | Regime |
| funding_rate | Taxa de funding |
| rsi, macd, macd_signal, macd_hist | Momentum |
| bb_upper, bb_middle, bb_lower | Volatilidade |

## 6. Containers
| Container | Status | Função |
|-----------|--------|--------|
| vostok_redis | Healthy | Event Bus |
| vostok_timescale | Healthy | Cold Storage |
| vostok_ingestor | Healthy | Trades + Funding |
| vostok_quant | Healthy | OHLCV + Sniper Metrics |

## 7. Diretrizes
- Nunca comitar chaves de API
- Type hints obrigatórios
- Logs JSON estruturados