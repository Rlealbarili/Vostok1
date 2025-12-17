"""
VOSTOK-1 :: Monitor TUI Dashboard
==================================
Terminal User Interface para monitoramento em tempo real.
Path atualizado: /app/data/training/dataset.jsonl

Arquiteto: Petrovich | Operador: Vostok
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
from rich.align import Align

# ============================================================================
# CONFIGURAÇÕES
# ============================================================================
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
SIGNAL_STREAM = os.getenv("SIGNAL_STREAM", "stream:signals:tech")
DATA_DIR = Path(os.getenv("DATA_DIR", "/app/data"))
TRAINING_DIR = DATA_DIR / "training"
DATASET_FILE = TRAINING_DIR / "dataset.jsonl"

# Tolerâncias de status (segundos)
ONLINE_TOLERANCE = 5
DELAYED_TOLERANCE = 30

console = Console()


# ============================================================================
# MARKET STATE
# ============================================================================
class MarketState:
    def __init__(self) -> None:
        self.price: float = 0.0
        self.prev_price: float = 0.0
        self.rsi: float = 50.0
        self.cvd: float = 0.0
        self.entropy: float = 0.0
        self.atr: float = 0.0
        self.funding_rate: float = 0.0
        self.macd: float = 0.0
        self.macd_hist: float = 0.0
        self.volume: float = 0.0
        self.signals_received: int = 0
        self.last_update_utc: datetime | None = None
        self.calc_time_ms: float = 0.0

    def update(self, data: dict[str, Any]) -> None:
        self.prev_price = self.price
        try:
            self.price = float(data.get('close', 0)) if data.get('close') else 0.0
            self.rsi = float(data.get('rsi', 50)) if data.get('rsi') else 50.0
            self.cvd = float(data.get('cvd_absolute', 0)) if data.get('cvd_absolute') else 0.0
            self.entropy = float(data.get('entropy', 0)) if data.get('entropy') else 0.0
            self.atr = float(data.get('volatility_atr', 0)) if data.get('volatility_atr') else 0.0
            self.funding_rate = float(data.get('funding_rate', 0)) if data.get('funding_rate') else 0.0
            self.macd = float(data.get('macd', 0)) if data.get('macd') else 0.0
            self.macd_hist = float(data.get('macd_hist', 0)) if data.get('macd_hist') else 0.0
            self.volume = float(data.get('volume', 0)) if data.get('volume') else 0.0
            self.calc_time_ms = float(data.get('calc_time_ms', 0)) if data.get('calc_time_ms') else 0.0
            self.signals_received += 1
            self.last_update_utc = datetime.now(timezone.utc)
        except (ValueError, TypeError):
            pass

    @property
    def price_direction(self) -> str:
        if self.price > self.prev_price:
            return "up"
        elif self.price < self.prev_price:
            return "down"
        return "neutral"


# ============================================================================
# DASHBOARD RENDERER
# ============================================================================
class DashboardRenderer:
    def __init__(self, state: MarketState) -> None:
        self.state = state
        self.blink_state = False

    def make_layout(self) -> Layout:
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
            Layout(name="dataset"),
        )
        return layout

    def render_header(self) -> Panel:
        self.blink_state = not self.blink_state
        
        if self.state.last_update_utc:
            now_utc = datetime.now(timezone.utc)
            age_seconds = (now_utc - self.state.last_update_utc).total_seconds()
            
            if age_seconds < ONLINE_TOLERANCE:
                status_style = "bold green" if self.blink_state else "green"
                status_text = "● ONLINE"
            elif age_seconds < DELAYED_TOLERANCE:
                status_style = "yellow"
                status_text = f"○ SYNC ({age_seconds:.0f}s)"
            else:
                status_style = "red"
                status_text = "✗ OFFLINE"
        else:
            status_style = "dim"
            status_text = "◌ WAITING"
        
        title = Text()
        title.append("╔══════════════════════════════════════════════════════════════╗\n", style="cyan")
        title.append("║   ", style="cyan")
        title.append("VOSTOK-1 SNIPER PROTOCOL", style="bold cyan")
        title.append("               ", style="cyan")
        title.append(status_text, style=status_style)
        title.append("  ║\n", style="cyan")
        title.append("╚══════════════════════════════════════════════════════════════╝", style="cyan")
        
        return Panel(Align.center(title), style="cyan", border_style="cyan")

    def render_market_panel(self) -> Panel:
        table = Table(show_header=True, header_style="bold cyan", expand=True, box=None)
        table.add_column("Metric", style="dim", width=16)
        table.add_column("Value", justify="right", width=14)
        table.add_column("Status", justify="center", width=12)
        
        pc = "green" if self.state.price_direction == "up" else "red" if self.state.price_direction == "down" else "white"
        pa = "▲" if self.state.price_direction == "up" else "▼" if self.state.price_direction == "down" else "─"
        table.add_row("💰 PRICE", f"[bold {pc}]${self.state.price:,.2f}[/]", f"[{pc}]{pa}[/]")
        
        if self.state.rsi < 30:
            rc, rs = "red", "OVERSOLD"
        elif self.state.rsi > 70:
            rc, rs = "green", "OVERBOUGHT"
        else:
            rc, rs = "dim", "NEUTRAL"
        table.add_row("📊 RSI", f"[{rc}]{self.state.rsi:.1f}[/]", f"[{rc}]{rs}[/]")
        
        cc = "bold green" if self.state.cvd > 0 else "bold red" if self.state.cvd < 0 else "dim"
        cs = "+" if self.state.cvd > 0 else ""
        cf = "BUYING" if self.state.cvd > 0 else "SELLING" if self.state.cvd < 0 else "NEUTRAL"
        table.add_row("[bold]📈 CVD (FLOW)[/]", f"[{cc}]{cs}{self.state.cvd:.4f}[/]", f"[{cc}]{cf}[/]")
        
        table.add_row("📉 ATR", f"[white]{self.state.atr:.2f}[/]", "[dim]VOLATILITY[/]")
        
        mc = "green" if self.state.macd_hist > 0 else "red"
        mt = "BULLISH" if self.state.macd_hist > 0 else "BEARISH"
        table.add_row("📊 MACD", f"[{mc}]{self.state.macd:.2f}[/]", f"[{mc}]{mt}[/]")
        
        fp = self.state.funding_rate * 100
        fc = "green" if fp > 0 else "red" if fp < 0 else "dim"
        table.add_row("💵 FUNDING", f"[{fc}]{fp:.4f}%[/]", "[dim]8H RATE[/]")
        
        table.add_row("📦 VOLUME", f"[white]{self.state.volume:.4f}[/]", "[dim]BTC[/]")
        
        return Panel(table, title="[bold cyan]MARKET INTELLIGENCE[/]", border_style="cyan")

    def render_regime_panel(self) -> Panel:
        content = Text()
        
        entropy = self.state.entropy
        if entropy > 0.8:
            es, el = "bold red blink", "⚠️  CHAOS MODE"
        elif entropy > 0.6:
            es, el = "yellow", "⚡ HIGH NOISE"
        elif entropy > 0.3:
            es, el = "dim", "◐ NORMAL"
        else:
            es, el = "green", "✓ TRENDING"
        
        content.append("ENTROPY (REGIME)\n", style="bold cyan")
        content.append("─" * 18 + "\n", style="dim")
        content.append(f"{entropy:.4f}", style=es)
        content.append(f"\n{el}\n\n", style=es)
        
        content.append("PERFORMANCE\n", style="bold cyan")
        content.append("─" * 18 + "\n", style="dim")
        content.append(f"Signals: {self.state.signals_received}\n", style="dim")
        content.append(f"Calc: {self.state.calc_time_ms:.2f}ms\n", style="dim")
        
        return Panel(content, title="[bold cyan]REGIME[/]", border_style="cyan")

    def render_dataset_panel(self) -> Panel:
        content = Text()
        
        dataset_lines = self._count_dataset_lines()
        dataset_exists = DATASET_FILE.exists()
        trades = self._read_recent_trades(3)
        
        content.append("DATASET STATUS\n", style="bold cyan")
        content.append("─" * 18 + "\n", style="dim")
        
        if not dataset_exists:
            content.append("📁 NOT FOUND\n", style="red")
            content.append("Run setup.sh\n", style="dim")
        elif dataset_lines == 0:
            content.append("📁 INITIALIZED\n", style="yellow")
            content.append("AWAITING TRADES\n", style="dim")
        else:
            content.append(f"📁 SIZE: ", style="dim")
            content.append(f"{dataset_lines} lines\n", style="bold green")
        
        content.append("\n")
        
        if trades:
            content.append("RECENT LABELS\n", style="bold cyan")
            content.append("─" * 18 + "\n", style="dim")
            
            for trade in trades:
                label = trade.get('outcome_label', '?')
                action = trade.get('action', '?')
                pnl = trade.get('pnl_percent', 0)
                
                if label == 1:
                    style, icon = "green", "✓"
                elif label == 0:
                    style, icon = "red", "✗"
                else:
                    style, icon = "dim", "?"
                
                content.append(f"{icon} {action}: ", style=style)
                content.append(f"{pnl:+.2f}%\n", style=style)
        else:
            content.append("RSI<35 + CVD>0\n", style="yellow")
        
        return Panel(content, title="[bold cyan]DATASET[/]", border_style="cyan")

    def render_footer(self) -> Panel:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        footer = Text()
        footer.append(f"Time: {now} | ", style="dim")
        footer.append(f"Stream: {SIGNAL_STREAM} | ", style="dim")
        footer.append("Ctrl+C to exit", style="dim")
        return Panel(Align.center(footer), style="dim", border_style="dim")

    def _count_dataset_lines(self) -> int:
        if not DATASET_FILE.exists():
            return 0
        try:
            with open(DATASET_FILE, 'r', encoding='utf-8') as f:
                return sum(1 for line in f if line.strip())
        except IOError:
            return 0

    def _read_recent_trades(self, n: int = 3) -> list[dict]:
        if not DATASET_FILE.exists():
            return []
        try:
            with open(DATASET_FILE, 'r', encoding='utf-8') as f:
                lines = [l for l in f.readlines() if l.strip()]
            return [json.loads(line.strip()) for line in lines[-n:]]
        except (IOError, json.JSONDecodeError):
            return []

    def render(self) -> Layout:
        layout = self.make_layout()
        layout["header"].update(self.render_header())
        layout["market"].update(self.render_market_panel())
        layout["regime"].update(self.render_regime_panel())
        layout["dataset"].update(self.render_dataset_panel())
        layout["footer"].update(self.render_footer())
        return layout


# ============================================================================
# MONITOR
# ============================================================================
class Monitor:
    def __init__(self) -> None:
        self.redis: aioredis.Redis | None = None
        self.state = MarketState()
        self.renderer = DashboardRenderer(self.state)
        self.running = False

    async def connect_redis(self) -> None:
        self.redis = aioredis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        await self.redis.ping()

    async def stream_signals(self) -> None:
        last_id = '$'
        while self.running:
            try:
                messages = await self.redis.xread(
                    streams={SIGNAL_STREAM: last_id},
                    count=1,
                    block=100
                )
                if messages:
                    for _, stream_messages in messages:
                        for message_id, data in stream_messages:
                            self.state.update(data)
                            last_id = message_id
            except Exception:
                await asyncio.sleep(1)

    async def run_dashboard(self) -> None:
        with Live(self.renderer.render(), refresh_per_second=4, console=console) as live:
            while self.running:
                live.update(self.renderer.render())
                await asyncio.sleep(0.25)

    async def start(self) -> None:
        console.clear()
        self.running = True
        await self.connect_redis()
        await asyncio.gather(self.stream_signals(), self.run_dashboard())

    async def stop(self) -> None:
        self.running = False
        if self.redis:
            await self.redis.close()


async def main() -> None:
    monitor = Monitor()
    try:
        await monitor.start()
    except KeyboardInterrupt:
        await monitor.stop()
        console.print("\n[yellow]Monitor encerrado.[/]")


if __name__ == "__main__":
    asyncio.run(main())
