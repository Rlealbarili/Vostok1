# 🚀 Vostok V2: Architecture Blueprint & Implementation Plan

> **Objetivo:** Elevar a Precision de ~34% para 40%+ e reduzir o Drawdown em mercados laterais.  
> **Status:** 🟡 Em Planejamento  
> **Data:** 2025-12-21  

---

## 1. Os 4 Pilares da Evolução (Core Upgrades)

### A. Migração de Modelo (The Brain)

| Aspecto | Atual | Novo |
|---------|-------|------|
| **Algoritmo** | Random Forest (Scikit-Learn) | **LightGBM** (Gradient Boosting) |
| **Inferência** | ~50ms | **<5ms** |

**Justificativa:** Inferência mais rápida (<5ms), melhor captura de não-linearidade e tratamento de ruído via GOSS (Gradient-based One-Side Sampling).

**Validação:** Obrigatório uso de **Purged K-Fold Cross-Validation** para evitar look-ahead bias.

---

### B. Filtro de Tendência MTF (The Compass)

**Conceito:** Consenso de tendência Multi-Timeframe.

| Timeframe | Indicador | Função |
|-----------|-----------|--------|
| **H1 (1 Hora)** | EMA 200 | Tendência Primária |
| **M15 (15 Min)** | SuperTrend | Confirmação (opcional) |

**Regras de Direção:**
- Se Preço > EMA 200 (H1) → **Apenas LONG**
- Se Preço < EMA 200 (H1) → **Apenas SHORT**

---

### C. Filtro de Mercado Morto (The Shield)

| Aspecto | Atual | Novo |
|---------|-------|------|
| **Filtro** | Nenhum | **Choppiness Index (CHOP)** + ADX |
| **Problema** | Suscetível a taxas em consolidação | Bloqueia trades laterais |

**Regra de Ouro:**
- Se `CHOP > 61.8` (Timeframe M1): **DESLIGAR TRADING** (Mercado Lateral/Dead Market)
- Se `ADX < 15`: Confirmar mercado morto

---

### D. Fusão de Sentimento (The Judge)

| Aspecto | Atual | Novo |
|---------|-------|------|
| **Integração** | Soft (apenas log) | **Hard Veto (Pós-Processamento)** |

**Lógica:** O score do LLM (Qwen) NÃO entra no modelo numérico. Ele atua como um **"Disjuntor"**.

**Exemplo de Veto:**
- Se Modelo diz `COMPRA` mas Sentimento é `EXTREME FEAR (-0.8)` → **VETO (Cancel Trade)**

---

## 2. Snippets de Implementação (Reference)

### 2.1 LightGBM Params (Anti-Overfitting)

```python
params = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.01,      # Lento para generalizar
    'num_leaves': 31,           # Árvores rasas
    'max_depth': 5,             # Evitar complexidade
    'min_data_in_leaf': 200,    # Robustez ao ruído
    'lambda_l1': 0.5,           # Regularização L1
    'lambda_l2': 1.0,           # Regularização L2
    'device': 'gpu'             # RTX 2060
}
```

---

### 2.2 Choppiness Index (Python)

```python
def calculate_choppiness(high, low, close, period=14):
    """
    Calcula o Choppiness Index.
    Valores > 61.8 indicam mercado lateral (evitar trades).
    Valores < 38.2 indicam tendência forte (bom para trades).
    """
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr_sum = tr.rolling(period).sum()
    range_hl = high.rolling(period).max() - low.rolling(period).min()
    
    # Fórmula: 100 * Log10(SumATR / Range) / Log10(Period)
    chop = 100 * np.log10(atr_sum / range_hl) / np.log10(period)
    return chop
```

---

### 2.3 Purged K-Fold Cross-Validation

```python
class PurgedKFold:
    """
    K-Fold que remove observações adjacentes ao split
    para evitar data leakage em séries temporais.
    """
    def __init__(self, n_splits=5, purge_gap=10):
        self.n_splits = n_splits
        self.purge_gap = purge_gap  # Candles a purgar entre folds
    
    def split(self, X):
        n_samples = len(X)
        fold_size = n_samples // self.n_splits
        
        for i in range(self.n_splits):
            test_start = i * fold_size
            test_end = (i + 1) * fold_size
            
            # Purge: remover amostras próximas ao teste
            train_idx = list(range(0, test_start - self.purge_gap))
            train_idx += list(range(test_end + self.purge_gap, n_samples))
            test_idx = list(range(test_start, test_end))
            
            yield train_idx, test_idx
```

---

## 3. Plano de Implementação (Roadmap)

### FASE 1: Filtros de Proteção (Imediato) 🟡

| Task | Arquivo | Descrição |
|------|---------|-----------|
| 1.1 | `src/quant/indicators.py` | Implementar `calculate_choppiness()` |
| 1.2 | `src/decision/engine.py` | Adicionar lógica: `if CHOP > 61.8: skip_trade()` |
| 1.3 | Teste | Validar redução de trades em consolidação |

**Meta:** Reduzir trades laterais e custos de taxa.

---

### FASE 2: Treinamento LightGBM (Offline) 🔴

| Task | Arquivo | Descrição |
|------|---------|-----------|
| 2.1 | `src/trainer/train_lgbm.py` | Criar script de treino LightGBM |
| 2.2 | `src/trainer/purged_kfold.py` | Implementar classe PurgedKFold |
| 2.3 | `models/sniper_v2.txt` | Gerar novo modelo (formato LightGBM) |
| 2.4 | Validação | Comparar métricas RF vs LightGBM |

**Meta:** Precision > 40% no hold-out set.

---

### FASE 3: Integração Final 🔴

| Task | Arquivo | Descrição |
|------|---------|-----------|
| 3.1 | `src/decision/engine.py` | Substituir RF por LightGBM |
| 3.2 | `src/decision/engine.py` | Ativar Veto de Sentimento (Hard Fusion) |
| 3.3 | `docker-compose.yml` | Atualizar dependências (lightgbm-gpu) |
| 3.4 | A/B Test | Paper Trading V1 vs V2 |

**Meta:** Drawdown < 5% em Paper Trading.

---

## 4. Métricas de Sucesso

| Métrica | V1 (Atual) | V2 (Alvo) |
|---------|------------|-----------|
| **Precision** | 34-36% | **40%+** |
| **Win Rate (Paper)** | 34.57% | **38%+** |
| **Trades em Consolidação** | Alto | **Reduzir 50%** |
| **Drawdown Máximo** | ~5% | **< 3%** |
| **Latência de Inferência** | ~50ms | **< 10ms** |

---

## 5. Dependências

```bash
# Adicionar ao requirements.txt do trainer
lightgbm>=4.0.0
optuna>=3.0.0  # Hyperparameter tuning
```

```dockerfile
# Dockerfile.trainer - GPU Support
RUN pip install lightgbm --install-option=--gpu
```

---

## 6. Riscos e Mitigações

| Risco | Probabilidade | Impacto | Mitigação |
|-------|--------------|---------|-----------|
| Overfitting LightGBM | Média | Alto | Purged K-Fold + Early Stopping |
| Filtro CHOP muito agressivo | Baixa | Médio | Calibrar threshold (61.8 vs 55) |
| Veto de Sentimento excessivo | Média | Médio | Log antes de ativar hard veto |
| GPU Memory (LightGBM) | Baixa | Baixo | Batch processing |

---

## 7. Referências

1. LightGBM Documentation - https://lightgbm.readthedocs.io
2. Advances in Financial Machine Learning (Marcos López de Prado)
3. Choppiness Index - Dreiss, E.W.
4. Multi-Timeframe Analysis - Elder, A.

---

*Documento de Planejamento - Vostok V2 Architecture Blueprint*  
*Última atualização: 2025-12-21*
