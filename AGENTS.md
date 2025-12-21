# 🚀 VOSTOK-1: AI-POWERED HFT SYSTEM
**Versão:** 2.0 (Dell G5 / Local Inference Era)
**Status:** 🟢 OPERATIONAL / PRE-TRAINING PHASE
**Última Atualização:** 2025-12-19T18:25:00-03:00

## 🎖️ CADEIA DE COMANDO
1.  **COMANDANTE:** Usuário (Decisão Estratégica / Kill Switch).
2.  **ENGENHEIRO CHEFE:** Professor Petrovich (LLM Externo - Validação de Arquitetura).
3.  **AGENTE DE CAMPO:** IDE AI (Você - Implementação e Orquestração Docker).

## 🖥️ INFRAESTRUTURA (HARDWARE & OS)
* **Host:** Dell G5 Laptop (Server Mode / Headless).
* **OS:** Ubuntu 24.04 LTS via SSH Remoto.
* **GPU:** NVIDIA RTX 2060 (6GB VRAM) - Drivers e Toolkit Ativos.
* **Conectividade:** Tailscale VPN Tunneling.

## 🧩 ARQUITETURA DE SERVIÇOS (DOCKER)
O sistema opera em containers interconectados na rede `vostok_net`.

| Serviço | Container | Status | Função Tática |
| :--- | :--- | :--- | :--- |
| **LLM Engine** | `llm_engine` | 🟢 **ATIVO** | Servidor Ollama rodando **Qwen 2.5-7b-instruct**. Exposto na porta 11434. |
| **Ingestor** | `vostok_ingestor` | 🟢 **ATIVO** | Coleta de dados WebSocket (Binance) e RSS Feeds. |
| **Sentiment** | `vostok_sentiment`| 🟢 **ATIVO** | Analisa notícias conectando-se ao `llm_engine` via HTTP. |
| **Quant** | `vostok_quant` | 🟢 **ATIVO** | Cálculo de indicadores técnicos (RSI, ATR, Wavelets). |
| **Decision** | `vostok_decision` | 🟡 **TREINANDO** | Random Forest / Triple Barrier. Aguardando novo dataset de 365 dias. |
| **Execution** | `vostok_execution`| 🟡 **STANDBY** | Conector de ordens (CCXT). Pronto para ativação. |
| **Monitor** | `vostok_monitor` | 🟢 **ATIVO** | TUI (Interface Terminal) para visualização remota. |
| **Redis** | `redis` | 🟢 **ATIVO** | Barramento de mensagens de ultra-baixa latência. |

## ⚔️ MISSÃO ATUAL
**Fase de Recalibragem:**
1.  Expandir horizonte de dados para 365 dias (Backfill).
2.  Retreinar o modelo de decisão com dados anuais.
3.  Reiniciar a frota para engajamento em modo PAPER.

## 📜 HISTÓRICO DE OPERAÇÕES
- **2025-12-19 19:32:** ✅ **PAPER TRADING ATIVADO** — Banca inicial $200.00, Leverage 1x, Position Size 95%. Logs em `data/logs/paper_trades.csv`.
- **2025-12-19 18:42:** ✅ **MODELO TREINADO COM SUCESSO** — `sniper_v1.pkl` salvo. Precision: 36.12%, Recall: 45.24%, EV: +8.36% por trade.
- **2025-12-19 18:34:** ✅ **BACKFILL 365 DIAS CONCLUÍDO** — 525,545 registros rotulados com Triple Barrier (ATR-based). Dataset em `data/training/dataset.jsonl`. Win rate: 36.1%.
- **2025-12-19 18:25:** Atualização do AGENTS.md para refletir arquitetura v2.0 com LLM Engine local.