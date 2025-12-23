#!/usr/bin/env python3
"""
VOSTOK V2 :: Start Paper Trading Bot
=====================================
Script para iniciar o bot de paper trading V2.

Uso:
    python start_v2.py              # Inicia com balance $200
    python start_v2.py --balance 500  # Inicia com balance custom
    python start_v2.py --dry-run    # Só mostra status, não opera
"""

import argparse
import asyncio
import logging
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)7s | %(message)s',
    datefmt='%H:%M:%S',
)

# Reduzir ruído
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("lightgbm").setLevel(logging.WARNING)


def print_banner():
    """Imprime banner do bot."""
    banner = """
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   ██╗   ██╗ ██████╗ ███████╗████████╗ ██████╗ ██╗  ██╗    ██╗   ██╗ ██████╗ ║
║   ██║   ██║██╔═══██╗██╔════╝╚══██╔══╝██╔═══██╗██║ ██╔╝    ██║   ██║╚════██╗║
║   ██║   ██║██║   ██║███████╗   ██║   ██║   ██║█████╔╝     ██║   ██║ █████╔╝║
║   ╚██╗ ██╔╝██║   ██║╚════██║   ██║   ██║   ██║██╔═██╗     ╚██╗ ██╔╝██╔═══╝ ║
║    ╚████╔╝ ╚██████╔╝███████║   ██║   ╚██████╔╝██║  ██╗     ╚████╔╝ ███████╗║
║     ╚═══╝   ╚═════╝ ╚══════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝      ╚═══╝  ╚══════╝║
║                                                                      ║
║                    🚀 PAPER TRADING BOT 🚀                           ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def check_model():
    """Verifica se o modelo está treinado."""
    import os
    model_path = "models/v2/lgbm_model.txt"
    
    if not os.path.exists(model_path):
        print("❌ ERROR: Model not found!")
        print(f"   Expected: {model_path}")
        print()
        print("   Train the model first:")
        print("   python train_v2.py data/training/ohlcv_btc_365d.csv")
        print()
        return False
    
    size_mb = os.path.getsize(model_path) / 1e6
    print(f"✅ Model found: {model_path} ({size_mb:.1f} MB)")
    return True


def check_dependencies():
    """Verifica dependências."""
    missing = []
    
    try:
        import requests
    except ImportError:
        missing.append("requests")
    
    try:
        import pandas
    except ImportError:
        missing.append("pandas")
    
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    
    try:
        import lightgbm
    except ImportError:
        missing.append("lightgbm")
    
    if missing:
        print("❌ Missing dependencies:")
        for dep in missing:
            print(f"   - {dep}")
        print()
        print("   Install with:")
        print(f"   pip install {' '.join(missing)}")
        return False
    
    return True


async def run_bot(balance: float, ollama_host: str, dry_run: bool):
    """Executa o bot."""
    from src.v2.execution.paper_live import PaperTradingBot
    
    if dry_run:
        print("\n🧪 DRY RUN MODE - Showing single cycle only\n")
        
        bot = PaperTradingBot(
            initial_balance=balance,
            ollama_host=ollama_host,
        )
        await bot.run_cycle()
        print("\n")
        return
    
    bot = PaperTradingBot(
        initial_balance=balance,
        ollama_host=ollama_host,
    )
    
    await bot.run_forever()


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Vostok V2 Paper Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--balance", "-b",
        type=float,
        default=200.0,
        help="Initial balance in USD (default: 200)",
    )
    
    parser.add_argument(
        "--ollama-host",
        type=str,
        default="localhost",
        help="Ollama host for Buffett AI (default: localhost)",
    )
    
    parser.add_argument(
        "--dry-run", "-d",
        action="store_true",
        help="Run single cycle only (for testing)",
    )
    
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip dependency and model checks",
    )
    
    args = parser.parse_args()
    
    # Banner
    print_banner()
    
    # Checks
    if not args.skip_checks:
        print("Checking prerequisites...")
        print()
        
        if not check_dependencies():
            sys.exit(1)
        
        if not check_model():
            sys.exit(1)
        
        print()
    
    # Run
    try:
        asyncio.run(run_bot(
            balance=args.balance,
            ollama_host=args.ollama_host,
            dry_run=args.dry_run,
        ))
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
