# **Documento de Design de Projeto (DDP): Sistema Autônomo de Trading "Vostok-1"**

ID do Documento: DDP-VOSTOK-GENESIS  
Versão: 1.0 (Greenfield)  
Classificação: Crítico / Alta Frequência  
Arquiteto Chefe: Professor Anatoly Petrovich

## **1.0 Princípios Fundamentais (Axiomas do Projeto)**

Para garantir que não repitamos erros do passado, este sistema obedecerá a três leis imutáveis:

1. **Lei da Segregação Temporal:** O processamento de sinais lentos (LLM/Notícias \- segundos) jamais deve bloquear o processamento de sinais rápidos (Preço/Indicadores \- milissegundos).  
2. **Lei do Determinismo:** O motor de decisão não "acha"; ele calcula. Dada a mesma entrada (Input A \+ Input B), ele deve gerar *sempre* a mesma saída (Ação C), sem alucinações.  
3. **Lei da Sobrevivência (Fail-Safe):** Se um módulo falhar, o sistema deve entrar em modo de "Liquidação Defensiva" ou "Congelamento Seguro", nunca em comportamento errático.

## **2.0 Arquitetura de Alto Nível: O Reator de Eventos**

Abandonamos o modelo HTTP/REST (Request-Response). O "Vostok-1" é uma arquitetura orientada a eventos (Event-Driven) baseada em **Streams**.

O núcleo central não é um banco de dados, nem um orquestrador Python. É o **Redis Streams**. Ele atua como o sistema nervoso central, garantindo que cada evento (tick de preço, notícia, ordem) seja gravado sequencialmente e processado na ordem exata de chegada.

### **O Fluxo de Dados (Pipeline)**

Snippet de código

graph LR  
    A\[Exchange\] \--\>|WebSocket| B(Módulo: Ingestão de Mercado)  
    C\[Web/RSS/Twitter\] \--\>|Polling/Push| D(Módulo: Sentimento AI)  
      
    B \--\>|Stream: market\_ticks| E{Redis Streams}  
    D \--\>|Stream: sentiment\_signals| E  
      
    E \--\>|Leitura Rápida| F(Módulo: Processador Quant)  
    F \--\>|Stream: tech\_signals| E  
      
    E \--\>|Consolidação| G(Módulo: Motor de Decisão)  
    G \--\>|Stream: execution\_orders| E  
      
    E \--\>|Leitura Crítica| H(Módulo: Executor de Ordens)  
    H \--\>|API REST| A

## **3.0 Detalhamento dos Microserviços (Os Módulos)**

Cada módulo é um container Docker isolado, escrito em Python 3.11+ (com partes críticas otimizadas).

### **3.1. Módulo Ingestor\_Mercado (O Reflexo)**

* **Função:** Ouvir o mercado.  
* **Comportamento:** Conecta-se via WebSocket (Binance/Bybit/Alpaca) e "escuta" o Order Book e Trades.  
* **Saída:** Normaliza os dados e escreve no Redis Stream stream:market:btc\_usdt.  
* **Latência Alvo:** \< 50ms (Exchange \-\> Redis).  
* **Bibliotecas:** ccxt.pro (versão assíncrona) ou websockets.

### **3.2. Módulo Sentimento\_AI (O Intelecto)**

* **Função:** Ler o mundo não estruturado.  
* **Comportamento:**  
  1. Coleta notícias de APIs (CryptoPanic, Twitter, NewsAPI).  
  2. Utiliza o **Qwen 14B (Local)** para analisar o texto.  
  3. Classifica em: Score (-1.0 a \+1.0), Relevância (0 a 10\) e Impacto Estimado.  
* **Saída:** Escreve no Redis Stream stream:sentiment:global.  
* **Latência Alvo:** 2 a 10 segundos (Assíncrono \- não bloqueia o trade).  
* **Técnica:** RAG leve (apenas contexto imediato de 24h).

### **3.3. Módulo Processador\_Quant (A Calculadora)**

* **Função:** Transformar preço em estatística.  
* **Comportamento:**  
  1. Lê stream:market:btc\_usdt em tempo real.  
  2. Constrói velas (Candles) em memória (1s, 1m, 5m).  
  3. Aplica **TA-Lib** para calcular RSI, MACD, Bollinger Bands, ATR.  
* **Saída:** Escreve no Redis Stream stream:signals:tech.  
* **Latência Alvo:** \< 10ms.

### **3.4. Módulo Motor\_Decisao (O Juiz)**

* **Função:** A síntese final.  
* **Comportamento:**  
  1. Sincroniza os streams (Sinal Técnico Rápido \+ Sinal de Sentimento Lento).  
  2. Aplica a **Árvore de Decisão** (Strategy Tree) ou Modelo ML.  
  3. Calcula o *Position Sizing* (Gestão de Risco baseada no saldo e volatilidade).  
* **Saída:** Se houver oportunidade, escreve no Redis Stream stream:orders:new.  
* **Regra de Ouro:** Se o Sentimento for \< \-0.5, ignora sinais de Compra Técnica (Filtro de Tendência).

### **3.5. Módulo Executor\_Ordens (O Sniper)**

* **Função:** Execução cirúrgica.  
* **Comportamento:**  
  1. Lê stream:orders:new.  
  2. Envia a ordem para a Exchange via API privada.  
  3. Monitora o *status* (Preenchida, Parcial, Rejeitada).  
  4. Gerencia Stop-Loss dinâmico (Trailing Stop) localmente para velocidade.  
* **Saída:** Escreve logs de transação no Banco de Dados.

## **4.0 Camada de Persistência (Memória)**

Não usaremos bancos relacionais lentos para a operação em tempo real.

1. **Hot Storage (Redis):** Mantém o estado atual, filas de eventos e janelas de dados recentes (últimas 24h).  
2. **Cold Storage (TimescaleDB):** Banco de dados baseado em PostgreSQL, otimizado para séries temporais.  
   * Armazena: Histórico OHLCV eterno, Logs de execução, Snapshots de sentimento.  
   * Uso: Backtesting e retreinamento de modelos.  
3. **Vector Store (PGVector integrado ao Timescale):** Armazena embeddings de notícias para o RAG do módulo de Sentimento.

## **5.0 Stack Tecnológica Otimizada**

Esqueça o FastAPI para o núcleo. O núcleo são scripts Python Asyncio puros rodando como *daemons*.

* **Linguagem:** Python 3.11 (com uvloop para performance próxima a Go/Rust).  
* **Mensageria:** Redis 7.0 (Streams).  
* **Banco de Dados:** TimescaleDB (Time-series \+ Relacional \+ Vetorial).  
* **Análise Quant:** Pandas, NumPy, TA-Lib, Numba (para compilação JIT de cálculos pesados).  
* **IA Local:** Ollama (servindo Qwen 14B) ou vLLM (para maior throughput).  
* **Backtesting:** VectorBT Pro (simulação vetorial de alta velocidade).

## **6.0 Roteiro de Implementação (Roadmap)**

Não construiremos tudo de uma vez. Seguiremos a doutrina da "Evolução Estável".

### **Fase 1: A Fundação (Semana 1-2)**

* Setup do Docker Compose com Redis e TimescaleDB.  
* Criação do Ingestor\_Mercado: Gravar dados reais no Redis e Timescale.  
* Objetivo: Ter um banco de dados populado com dados de alta qualidade.

### **Fase 2: O Cérebro Analítico (Semana 3-4)**

* Criação do Processador\_Quant e Sentimento\_AI.  
* Visualização dos sinais em um Dashboard (Grafana conectado ao TimescaleDB).  
* Objetivo: Ver o RSI e o Sentimento do Qwen plotados em tempo real num gráfico.

### **Fase 3: O Simulador (Semana 5\)**

* Implementação do Motor\_Decisao em modo "Paper Trading" (Simulado).  
* O bot "opera" mas não envia ordens reais; apenas registra o que *teria* feito.  
* Objetivo: Validar a estratégia sem perder dinheiro.

### **Fase 4: O "Go Live" (Semana 6+)**

* Ativação do Executor\_Ordens com capital mínimo ($50-$100).  
* Monitoramento 24/7.

---

Comandante, esta é a planta da "Vostok-1". Ela é modular, resiliente e brutalmente eficiente. Não há gordura, apenas músculo e nervos.

**Sua aprovação é necessária para iniciarmos a codificação da Fase 1 (Docker e Ingestão). Devo prosseguir?**