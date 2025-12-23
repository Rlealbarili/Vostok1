#!/usr/bin/env python3
"""
VOSTOK V2 :: Status Monitor
===========================
Mostra o status do Paper Trading V2 a partir do CSV de logs.

Uso:
    python check_v2.py           # Status resumido
    python check_v2.py --full    # Todos os trades
    python check_v2.py --tail 10 # Últimos 10 trades
    python check_v2.py --live    # Atualiza a cada 5s
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


LOG_FILE = Path("data/logs/v2_paper_trades.csv")
INITIAL_BALANCE = 200.0


def load_trades():
    """Carrega trades do CSV."""
    if not LOG_FILE.exists():
        return None
    
    try:
        df = pd.read_csv(LOG_FILE)
        return df
    except Exception as e:
        print(f"Error loading trades: {e}")
        return None


def print_banner():
    """Banner simples."""
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║           📊 VOSTOK V2 PAPER TRADING STATUS                  ║")
    print("╚══════════════════════════════════════════════════════════════╝")


def print_summary(df):
    """Imprime resumo das operações."""
    if df is None or len(df) == 0:
        print("\n  ⚪ Nenhum trade registrado ainda.")
        print(f"  📂 Log file: {LOG_FILE}")
        return
    
    total_trades = len(df)
    wins = len(df[df['result'] == 'TP'])
    losses = len(df[df['result'] == 'SL'])
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0
    
    # Balance
    if 'balance' in df.columns:
        current_balance = df['balance'].iloc[-1]
    else:
        current_balance = INITIAL_BALANCE + df['pnl_usd'].sum()
    
    pnl_total = current_balance - INITIAL_BALANCE
    pnl_pct = pnl_total / INITIAL_BALANCE * 100
    
    # Time range
    if 'timestamp' in df.columns:
        first_trade = df['timestamp'].iloc[0]
        last_trade = df['timestamp'].iloc[-1]
    else:
        first_trade = "N/A"
        last_trade = "N/A"
    
    # Stats
    if len(df) > 0 and 'pnl_pct' in df.columns:
        avg_win = df[df['pnl_usd'] > 0]['pnl_usd'].mean() if wins > 0 else 0
        avg_loss = df[df['pnl_usd'] < 0]['pnl_usd'].mean() if losses > 0 else 0
        best_trade = df['pnl_usd'].max()
        worst_trade = df['pnl_usd'].min()
    else:
        avg_win = avg_loss = best_trade = worst_trade = 0
    
    print()
    print("  ═══════════════════════════════════════════════════════")
    print(f"  📈 RESUMO")
    print("  ═══════════════════════════════════════════════════════")
    print()
    print(f"    Balance Inicial:   ${INITIAL_BALANCE:.2f}")
    print(f"    Balance Atual:     ${current_balance:.2f}")
    print(f"    P&L Total:         ${pnl_total:+.2f} ({pnl_pct:+.2f}%)")
    print()
    print(f"    Total de Trades:   {total_trades}")
    print(f"    Wins (TP):         {wins}")
    print(f"    Losses (SL):       {losses}")
    print(f"    Win Rate:          {win_rate:.1f}%")
    print()
    print(f"    Avg Win:           ${avg_win:+.2f}")
    print(f"    Avg Loss:          ${avg_loss:+.2f}")
    print(f"    Best Trade:        ${best_trade:+.2f}")
    print(f"    Worst Trade:       ${worst_trade:+.2f}")
    print()
    print(f"    Primeiro Trade:    {first_trade}")
    print(f"    Último Trade:      {last_trade}")
    print("  ═══════════════════════════════════════════════════════")


def print_trades(df, n=None):
    """Imprime lista de trades."""
    if df is None or len(df) == 0:
        print("\n  Nenhum trade registrado.")
        return
    
    if n is not None:
        df = df.tail(n)
    
    print()
    print("  ═══════════════════════════════════════════════════════════════════════════")
    print("  HISTÓRICO DE TRADES")
    print("  ═══════════════════════════════════════════════════════════════════════════")
    print()
    print(f"  {'#':<4} {'Direção':<6} {'Entry':<12} {'Exit':<12} {'P&L':<12} {'Resultado':<8} {'Balance'}")
    print("  " + "-" * 75)
    
    for i, row in df.iterrows():
        direction = row.get('direction', 'N/A')
        entry = row.get('entry', 0)
        exit_price = row.get('exit', 0)
        pnl = row.get('pnl_usd', 0)
        pnl_pct = row.get('pnl_pct', 0)
        result = row.get('result', 'N/A')
        balance = row.get('balance', 0)
        
        emoji = "✅" if result == 'TP' else "❌"
        
        print(f"  {len(df) - len(df) + i + 1:<4} {direction:<6} ${entry:<10.2f} ${exit_price:<10.2f} ${pnl:+8.2f} ({pnl_pct:+.1f}%) {emoji} {result:<6} ${balance:.2f}")
    
    print()


def check_bot_running():
    """Verifica se o bot está rodando."""
    import subprocess
    
    try:
        result = subprocess.run(
            ["pgrep", "-f", "paper_live.py"],
            capture_output=True,
            text=True,
        )
        pids = result.stdout.strip().split('\n')
        pids = [p for p in pids if p]
        
        if pids:
            print(f"\n  🟢 BOT ATIVO (PIDs: {', '.join(pids)})")
            return True
        else:
            print("\n  🔴 BOT OFFLINE")
            return False
    except:
        print("\n  ⚪ Status do bot: desconhecido")
        return None


def live_mode():
    """Modo live - atualiza a cada 5s."""
    import subprocess
    
    try:
        while True:
            subprocess.run(["clear"])
            print_banner()
            check_bot_running()
            df = load_trades()
            print_summary(df)
            print("\n  (Atualizando a cada 5s... Ctrl+C para sair)")
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n\n  👋 Bye!")


def main():
    parser = argparse.ArgumentParser(description="Vostok V2 Status Monitor")
    parser.add_argument("--full", "-f", action="store_true", help="Show all trades")
    parser.add_argument("--tail", "-t", type=int, default=None, help="Show last N trades")
    parser.add_argument("--live", "-l", action="store_true", help="Live update mode")
    
    args = parser.parse_args()
    
    if not HAS_PANDAS:
        print("Error: pandas not installed")
        print("Run: pip install pandas")
        sys.exit(1)
    
    if args.live:
        live_mode()
        return
    
    print_banner()
    check_bot_running()
    
    df = load_trades()
    print_summary(df)
    
    if args.full:
        print_trades(df)
    elif args.tail:
        print_trades(df, args.tail)
    
    print()


if __name__ == "__main__":
    main()
