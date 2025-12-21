# 🧠 Vostok Sentiment V2: "The Crypto Buffett"

> **Objetivo:** Transformar o módulo de sentimento de um "leitor de manchetes reativo" em um "analista fundamentalista contrarian".  
> **Persona:** Warren Buffett / Howard Marks adaptado para Cripto.  
> **Status:** 🟡 Em Planejamento  
> **Data:** 2025-12-21  

---

## 1. O System Prompt Otimizado (Copiar e Colar)

Este prompt foi desenhado para modelos 7B (Qwen 2.5) rodando localmente.

```text
You are "Crypto Buffett," a cold, analytical value investor focused on long-term cryptocurrency fundamentals.
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
}
```

---

## 2. Lógica de Integração (Python)

Como o código Python deve interpretar o JSON do Buffett:

```python
def apply_sentiment_logic(trade_signal: str, sentiment_data: dict) -> tuple[str, str]:
    """
    Aplica a lógica do Crypto Buffett aos sinais de trade.
    
    Returns:
        tuple: (action, reason)
        - action: "VETO" | "PROCEED" | "BOOST"
        - reason: Explicação da decisão
    """
    verdict = sentiment_data.get('buffett_verdict')
    score = sentiment_data.get('sentiment_score', 0)

    # CASO 1: Risco Estrutural (VETO ABSOLUTO)
    # Ex: Banimento governamental, Hack de protocolo
    if verdict == "Structural Risk" or score < -0.8:
        return "VETO", "Risco estrutural detectado. Protegendo capital."

    # CASO 2: Ruído de Curto Prazo (IGNORAR SENTIMENTO)
    # Ex: "Bitcoin cai 5%", "Elon Musk tweeta"
    if verdict == "Ignore":
        return "PROCEED", "Ruído ignorado. Seguindo modelo matemático."

    # CASO 3: Oportunidade Contrarian (BUY THE FEAR)
    # Ex: Pânico irracional do mercado sem mudança de fundamento
    if verdict == "Buy the Fear" and trade_signal == "BUY":
        return "BOOST", "Medo irracional detectado. Aumentar posição?"

    # CASO 4: Sell the News (Cautela)
    # Ex: Evento positivo já precificado
    if verdict == "Sell the News" and trade_signal == "BUY":
        return "PROCEED", "Evento já precificado. Posição normal."

    # Padrão: Seguir o score numérico como filtro suave
    return "PROCEED", "Sentimento dentro dos limites normais."
```

---

## 3. Exemplos de Calibragem (Few-Shot)

### 3.1 Buy the Fear (Oportunidade Contrarian)

```json
// Input: "Bitcoin crashes 10% on liquidation cascade."
{
  "classification": "Macro Noise",
  "sentiment_score": -0.1,
  "impact_duration": "Short-term",
  "buffett_verdict": "Buy the Fear",
  "reasoning": "Liquidação técnica. Fundamentos intactos. Oportunidade."
}
```

**Ação:** Se modelo diz `BUY` → **BOOST** (considerar aumentar posição)

---

### 3.2 Structural Risk (Veto Absoluto)

```json
// Input: "SEC sues major exchange for fraud."
{
  "classification": "Regulatory FUD",
  "sentiment_score": -0.9,
  "impact_duration": "Structural",
  "buffett_verdict": "Structural Risk",
  "reasoning": "Risco regulatório sistêmico. Evitar exposição."
}
```

**Ação:** **VETO** (cancelar qualquer trade)

---

### 3.3 Ignore (Ruído de Mercado)

```json
// Input: "New meme coin surges 500%."
{
  "classification": "Macro Noise",
  "sentiment_score": 0.0,
  "impact_duration": "Short-term",
  "buffett_verdict": "Ignore",
  "reasoning": "Especulação sem valor intrínseco. Irrelevante."
}
```

**Ação:** **PROCEED** (seguir modelo matemático)

---

### 3.4 Protocol Upgrade (Positivo Estrutural)

```json
// Input: "Ethereum completes major upgrade reducing fees by 90%."
{
  "classification": "Protocol Upgrade",
  "sentiment_score": 0.7,
  "impact_duration": "Structural",
  "buffett_verdict": "Buy the Fear",
  "reasoning": "Melhoria fundamental. Aguardar correção para entry."
}
```

---

## 4. Matriz de Decisão

| Verdict | Score Range | Ação | Descrição |
|---------|-------------|------|-----------|
| **Structural Risk** | < -0.8 | 🔴 VETO | Cancelar trade imediatamente |
| **Buy the Fear** | -0.5 a 0.0 | 🟢 BOOST | Oportunidade contrarian |
| **Ignore** | -0.3 a +0.3 | ⚪ PROCEED | Seguir modelo matemático |
| **Sell the News** | > +0.5 | 🟡 PROCEED | Cautela, não aumentar |

---

## 5. Implementação no Vostok

### 5.1 Arquivo: `src/sentiment/main.py`

Substituir o `SYSTEM_PROMPT` atual pelo prompt do Crypto Buffett (Seção 1).

### 5.2 Arquivo: `src/decision/engine.py`

Adicionar chamada para `apply_sentiment_logic()` após receber sinal do modelo ML:

```python
# Pseudo-código
if ml_signal == "BUY":
    sentiment = await get_latest_sentiment()
    action, reason = apply_sentiment_logic("BUY", sentiment)
    
    if action == "VETO":
        logger.warning(f"🛑 VETO: {reason}")
        return  # Não executa trade
    elif action == "BOOST":
        logger.info(f"🚀 BOOST: {reason}")
        # Opcional: aumentar position size
```

### 5.3 Novo Campo no CSV

Adicionar coluna `sentiment_verdict` em `paper_trades.csv` para auditoria.

---

## 6. Métricas de Sucesso

| Métrica | Atual | Alvo V2 |
|---------|-------|---------|
| **Trades vetados por Structural Risk** | 0 | Quando necessário |
| **False Vetos** | N/A | < 5% |
| **Win Rate em "Buy the Fear"** | N/A | > 45% |
| **Drawdown após eventos negativos** | ~5% | < 2% |

---

## 7. Riscos e Mitigações

| Risco | Mitigação |
|-------|-----------|
| Veto excessivo | Log todos os vetos, revisar semanalmente |
| LLM alucinando | Validar JSON antes de usar |
| Latência do LLM | Cache de 15 min, não bloquear trades |

---

## 8. Referências

1. "The Most Important Thing" - Howard Marks
2. Berkshire Hathaway Letters to Shareholders
3. "Thinking, Fast and Slow" - Daniel Kahneman
4. Market Psychology in Crypto - Glassnode Research

---

*Documento de Estratégia - Vostok Sentiment V2 "Crypto Buffett"*  
*Última atualização: 2025-12-21*
