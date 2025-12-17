"""
VOSTOK-1 :: Monitor TUI Dashboard
==================================
Terminal User Interface para monitoramento em tempo real
do sistema de trading usando a biblioteca Rich.

Arquiteto: Petrovich | Operador: Vostok
Stack: Python 3.11 + asyncio + rich + redis-py
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import redis.asyncio as aioredis
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.style import Style
from rich.align import Align

# ============================================================================
# CONFIGURAÇÕES
# ============================================================================
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
SIGNAL_STREAM = os.getenv("SIGNAL_STREAM", "stream:signals:tech")
DATA_DIR = Path(os.getenv("DATA_DIR", "/app/data"))
DATASET_FILE = DATA_DIR / "training_dataset.jsonl"

console = Console()


# ============================================================================
# MARKET STATE - Estado do Mercado
# ============================================================================
class MarketState:
    """Mantém o estado atual do mercado para exibição."""

    def __init__(self) -> None:
        # Preços
        self.price: float = 0.0
        self.prev_price: float = 0.0
        
        # Indicadores
        self.rsi: float = 50.0
        self.cvd: float = 0.0
        self.entropy: float = 0.0
        self.atr: float = 0.0
        self.funding_rate: float = 0.0
        
        # MACD
        self.macd: float = 0.0
        self.macd_signal: float = 0.0
        self.macd_hist: float = 0.0
        
        # Bollinger
        self.bb_upper: float = 0.0
        self.bb_middle: float = 0.0
        self.bb_lower: float = 0.0
        
        # OHLCV
        self.open: float = 0.0
        self.high: float = 0.0
        self.low: float = 0.0
        self.volume: float = 0.0
        
        # Meta
        self.timestamp: int = 0
        self.signals_received: int = 0
        self.last_update: datetime | None = None
        self.calc_time_ms: float = 0.0

    def update(self, data: dict[str, Any]) -> None:
        """Atualiza estado com novos dados do stream."""
        self.prev_price = self.price
        
        try:
            # Preço e OHLCV
            self.price = float(data.get('close', 0)) if data.get('close') else 0.0
            self.open = float(data.get('open', 0)) if data.get('open') else 0.0
            self.high = float(data.get('high', 0)) if data.get('high') else 0.0
            self.low = float(data.get('low', 0)) if data.get('low') else 0.0
            self.volume = float(data.get('volume', 0)) if data.get('volume') else 0.0
            
            # Indicadores principais
            self.rsi = float(data.get('rsi', 50)) if data.get('rsi') else 50.0
            self.cvd = float(data.get('cvd_absolute', 0)) if data.get('cvd_absolute') else 0.0
            self.entropy = float(data.get('entropy', 0)) if data.get('entropy') else 0.0
            self.atr = float(data.get('volatility_atr', 0)) if data.get('volatility_atr') else 0.0
            self.funding_rate = float(data.get('funding_rate', 0)) if data.get('funding_rate') else 0.0
            
            # MACD
            self.macd = float(data.get('macd', 0)) if data.get('macd') else 0.0
            self.macd_signal = float(data.get('macd_signal', 0)) if data.get('macd_signal') else 0.0
            self.macd_hist = float(data.get('macd_hist', 0)) if data.get('macd_hist') else 0.0
            
            # Bollinger
            self.bb_upper = float(data.get('bb_upper', 0)) if data.get('bb_upper') else 0.0
            self.bb_middle = float(data.get('bb_middle', 0)) if data.get('bb_middle') else 0.0
            self.bb_lower = float(data.get('bb_lower', 0)) if data.get('bb_lower') else 0.0
            
            # Meta
            self.timestamp = int(data.get('timestamp', 0)) if data.get('timestamp') else 0
            self.calc_time_ms = float(data.get('calc_time_ms', 0)) if data.get('calc_time_ms') else 0.0
            
            self.signals_received += 1
            self.last_update = datetime.now(timezone.utc)
            
        except (ValueError, TypeError):
            pass

    @property
    def price_direction(self) -> str:
        """Retorna direção do preço."""
        if self.price > self.prev_price:
            return "up"
        elif self.price < self.prev_price:
            return "down"
        return "neutral"


# ============================================================================
# DASHBOARD RENDERER
# ============================================================================
class DashboardRenderer:
    """Renderiza o dashboard TUI."""

    def __init__(self, state: MarketState) -> None:
        self.state = state
        self.blink_state = False

    def make_layout(self) -> Layout:
        """Cria o layout do dashboard."""
        layout = Layout()
        
        layout.split(
            Layout(name="header", size=5),
            Layout(name="body"),
            Layout(name="footer", size=4),
        )
        
        layout["body"].split_row(
            Layout(name="market", ratio=2),
            Layout(name="sidebar", ratio=1),
        )
        
        layout["sidebar"].split(
            Layout(name="regime"),
            Layout(name="trades"),
        )
        
        return layout

    def render_header(self) -> Panel:
        """Renderiza o header."""
        self.blink_state = not self.blink_state
        
        # Status
        if self.state.last_update:
            age = (datetime.now(timezone.utc) - self.state.last_update).total_seconds()
            if age < 5:
                status_style = "bold green" if self.blink_state else "green"
                status_text = "● SYSTEM: ONLINE"
            elif age < 30:
                status_style = "yellow"
                status_text = "○ SYSTEM: DELAYED"
            else:
                status_style = "red"
                status_text = "✗ SYSTEM: OFFLINE"
        else:
            status_style = "dim"
            status_text = "◌ SYSTEM: WAITING"
        
        title = Text()
        title.append("╔══════════════════════════════════════════════════════════════╗\n", style="cyan")
        title.append("║   ", style="cyan")
        title.append("VOSTOK-1 SNIPER PROTOCOL", style="bold cyan")
        title.append("                         ", style="cyan")
        title.append(status_text, style=status_style)
        title.append("  ║\n", style="cyan")
        title.append("╚══════════════════════════════════════════════════════════════╝", style="cyan")
        
        return Panel(Align.center(title), style="cyan", border_style="cyan")

    def render_market_panel(self) -> Panel:
        """Renderiza painel de inteligência de mercado."""
        table = Table(show_header=True, header_style="bold cyan", expand=True, box=None)
        table.add_column("Metric", style="dim", width=16)
        table.add_column("Value", justify="right", width=14)
        table.add_column("Status", justify="center", width=12)
        
        # Price
        price_color = "green" if self.state.price_direction == "up" else "red" if self.state.price_direction == "down" else "white"
        price_arrow = "▲" if self.state.price_direction == "up" else "▼" if self.state.price_direction == "down" else "─"
        table.add_row(
            "💰 PRICE",
            f"[bold {price_color}]${self.state.price:,.2f}[/]",
            f"[{price_color}]{price_arrow}[/]"
        )
        
        # RSI
        if self.state.rsi < 30:
            rsi_color, rsi_status = "red", "OVERSOLD"
        elif self.state.rsi > 70:
            rsi_color, rsi_status = "green", "OVERBOUGHT"
        else:
            rsi_color, rsi_status = "dim", "NEUTRAL"
        table.add_row(
            "📊 RSI",
            f"[{rsi_color}]{self.state.rsi:.1f}[/]",
            f"[{rsi_color}]{rsi_status}[/]"
        )
        
        # CVD (Order Flow) - DESTACADO
        cvd_color = "bold green" if self.state.cvd > 0 else "bold red" if self.state.cvd < 0 else "dim"
        cvd_sign = "+" if self.state.cvd > 0 else ""
        cvd_flow = "BUYING" if self.state.cvd > 0 else "SELLING" if self.state.cvd < 0 else "NEUTRAL"
        table.add_row(
            "[bold]📈 CVD (FLOW)[/]",
            f"[{cvd_color}]{cvd_sign}{self.state.cvd:.4f}[/]",
            f"[{cvd_color}]{cvd_flow}[/]"
        )
        
        # ATR
        table.add_row(
            "📉 ATR",
            f"[white]{self.state.atr:.2f}[/]",
            "[dim]VOLATILITY[/]"
        )
        
        # MACD
        macd_color = "green" if self.state.macd_hist > 0 else "red"
        macd_trend = "BULLISH" if self.state.macd_hist > 0 else "BEARISH"
        table.add_row(
            "📊 MACD",
            f"[{macd_color}]{self.state.macd:.2f}[/]",
            f"[{macd_color}]{macd_trend}[/]"
        )
        
        # Funding Rate
        funding_pct = self.state.funding_rate * 100
        funding_color = "green" if funding_pct > 0 else "red" if funding_pct < 0 else "dim"
        table.add_row(
            "💵 FUNDING",
            f"[{funding_color}]{funding_pct:.4f}%[/]",
            "[dim]8H RATE[/]"
        )
        
        # Bollinger Position
        if self.state.bb_upper > 0:
            bb_range = self.state.bb_upper - self.state.bb_lower
            bb_position = (self.state.price - self.state.bb_lower) / bb_range * 100 if bb_range > 0 else 50
            bb_color = "red" if bb_position > 80 else "green" if bb_position < 20 else "dim"
            table.add_row(
                "📏 BB POSITION",
                f"[{bb_color}]{bb_position:.0f}%[/]",
                "[dim]IN BANDS[/]"
            )
        
        return Panel(table, title="[bold cyan]MARKET INTELLIGENCE[/]", border_style="cyan")

    def render_regime_panel(self) -> Panel:
        """Renderiza painel de regime/entropia."""
        content = Text()
        
        # Entropy gauge
        entropy = self.state.entropy
        if entropy > 0.8:
            entropy_style = "bold red blink"
            entropy_label = "⚠️  CHAOS MODE"
        elif entropy > 0.6:
            entropy_style = "yellow"
            entropy_label = "⚡ HIGH NOISE"
        elif entropy > 0.3:
            entropy_style = "dim"
            entropy_label = "◐ NORMAL"
        else:
            entropy_style = "green"
            entropy_label = "✓ TRENDING"
        
        content.append("ENTROPY (REGIME)\n", style="bold cyan")
        content.append("─" * 20 + "\n", style="dim")
        content.append(f"{entropy:.4f}", style=entropy_style)
        content.append(f"\n{entropy_label}\n\n", style=entropy_style)
        
        # Stats
        content.append("STATS\n", style="bold cyan")
        content.append("─" * 20 + "\n", style="dim")
        content.append(f"Signals: {self.state.signals_received}\n", style="dim")
        content.append(f"Calc: {self.state.calc_time_ms:.2f}ms\n", style="dim")
        
        return Panel(content, title="[bold cyan]REGIME[/]", border_style="cyan")

    def render_trades_panel(self) -> Panel:
        """Renderiza painel de trades virtuais."""
        content = Text()
        
        # Tentar ler últimos trades do dataset
        trades = self._read_recent_trades(5)
        
        if trades:
            content.append("RECENT LABELS\n", style="bold cyan")
            content.append("─" * 20 + "\n", style="dim")
            
            for trade in trades:
                label = trade.get('outcome_label', '?')
                action = trade.get('action', '?')
                pnl = trade.get('pnl_percent', 0)
                
                if label == 1:
                    style = "green"
                    icon = "✓"
                elif label == 0:
                    style = "red"
                    icon = "✗"
                else:
                    style = "dim"
                    icon = "?"
                
                content.append(f"{icon} {action}: ", style=style)
                content.append(f"{pnl:+.2f}%\n", style=style)
        else:
            content.append("AWAITING TRADES\n", style="dim")
            content.append("─" * 20 + "\n", style="dim")
            content.append("No labels yet.\n", style="dim")
            content.append("Waiting for\n", style="dim")
            content.append("RSI < 35 + CVD > 0\n", style="yellow")
        
        return Panel(content, title="[bold cyan]DATASET[/]", border_style="cyan")

    def render_footer(self) -> Panel:
        """Renderiza footer."""
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        
        footer = Text()
        footer.append(f"Last Update: {now} | ", style="dim")
        footer.append(f"Stream: {SIGNAL_STREAM} | ", style="dim")
        footer.append("Press Ctrl+C to exit", style="dim")
        
        return Panel(Align.center(footer), style="dim", border_style="dim")

    def _read_recent_trades(self, n: int = 5) -> list[dict]:
        """Lê os últimos N trades do dataset."""
        if not DATASET_FILE.exists():
            return []
        
        try:
            with open(DATASET_FILE, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            recent = []
            for line in lines[-n:]:
                try:
                    recent.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    pass
            
            return recent
        except IOError:
            return []

    def render(self) -> Layout:
        """Renderiza o dashboard completo."""
        layout = self.make_layout()
        
        layout["header"].update(self.render_header())
        layout["market"].update(self.render_market_panel())
        layout["regime"].update(self.render_regime_panel())
        layout["trades"].update(self.render_trades_panel())
        layout["footer"].update(self.render_footer())
        
        return layout


# ============================================================================
# MONITOR - Orquestrador Principal
# ============================================================================
class Monitor:
    """Monitor principal com streaming de Redis."""

    def __init__(self) -> None:
        self.redis: aioredis.Redis | None = None
        self.state = MarketState()
        self.renderer = DashboardRenderer(self.state)
        self.running = False

    async def connect_redis(self) -> None:
        """Conecta ao Redis."""
        self.redis = aioredis.Redis(
            host=REDIS_HOST, port=REDIS_PORT, decode_responses=True
        )
        await self.redis.ping()

    async def stream_signals(self) -> None:
        """Stream de sinais do Redis."""
        last_id = '$'  # Apenas novas mensagens
        
        while self.running:
            try:
                messages = await self.redis.xread(
                    streams={SIGNAL_STREAM: last_id},
                    count=1,
                    block=100  # 100ms para refresh suave
                )
                
                if messages:
                    for stream_name, stream_messages in messages:
                        for message_id, data in stream_messages:
                            self.state.update(data)
                            last_id = message_id
                            
            except Exception:
                await asyncio.sleep(1)

    async def run_dashboard(self) -> None:
        """Executa o dashboard com Live display."""
        with Live(self.renderer.render(), refresh_per_second=4, console=console) as live:
            while self.running:
                live.update(self.renderer.render())
                await asyncio.sleep(0.25)

    async def start(self) -> None:
        """Inicia o monitor."""
        console.clear()
        self.running = True
        
        await self.connect_redis()
        
        # Executar stream e dashboard concorrentemente
        await asyncio.gather(
            self.stream_signals(),
            self.run_dashboard(),
        )

    async def stop(self) -> None:
        """Para o monitor."""
        self.running = False
        if self.redis:
            await self.redis.close()


# ============================================================================
# ENTRY POINT
# ============================================================================
async def main() -> None:
    monitor = Monitor()
    try:
        await monitor.start()
    except KeyboardInterrupt:
        await monitor.stop()
        console.print("\n[yellow]Monitor encerrado.[/]")


if __name__ == "__main__":
    asyncio.run(main())
