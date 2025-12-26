#!/usr/bin/env python3
"""
VOSTOK V2 :: Buffett Probe - Diagnostic Tool
==============================================
Script de diagnóstico para testar a infraestrutura de IA (Ollama)
e a classe CryptoBuffett isoladamente.

Uso:
    python -m src.v2.sentiment.diagnose_buffett
    python src/v2/sentiment/diagnose_buffett.py
    python src/v2/sentiment/diagnose_buffett.py --host localhost
    python src/v2/sentiment/diagnose_buffett.py --host host.docker.internal

Arquiteto: Petrovich | Operador: Vostok
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Try imports
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False


# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 11434
EXPECTED_MODEL = "qwen2.5:7b-instruct"
TEST_HEADLINE = "Bitcoin transaction fees drop to record lows as adoption increases."
TIMEOUT = 60.0


# ============================================================================
# DIAGNOSTIC FUNCTIONS
# ============================================================================

def print_header():
    """Print diagnostic header."""
    print()
    print("=" * 70)
    print("🎩 BUFFETT PROBE - CryptoBuffett Diagnostic Tool")
    print("=" * 70)
    print()


def print_step(step: int, description: str):
    """Print step header."""
    print(f"┌{'─' * 68}┐")
    print(f"│ STEP {step}: {description:<59} │")
    print(f"└{'─' * 68}┘")


def print_result(success: bool, message: str):
    """Print result with emoji."""
    emoji = "🟢" if success else "🔴"
    status = "OK" if success else "FAIL"
    print(f"  {emoji} [{status}] {message}")


def print_info(message: str):
    """Print info message."""
    print(f"  ℹ️  {message}")


def print_warn(message: str):
    """Print warning message."""
    print(f"  ⚠️  {message}")


# ============================================================================
# STEP 1: TEST OLLAMA CONNECTIVITY
# ============================================================================

def test_ollama_connectivity(host: str, port: int) -> bool:
    """
    Step 1: Test if Ollama server is reachable.
    """
    print_step(1, "OLLAMA CONNECTIVITY TEST")
    
    if not HAS_REQUESTS:
        print_result(False, "requests library not installed")
        print_info("Run: pip install requests")
        return False
    
    base_url = f"http://{host}:{port}"
    print_info(f"Testing connection to: {base_url}")
    
    try:
        start = time.time()
        response = requests.get(f"{base_url}/", timeout=10)
        elapsed = (time.time() - start) * 1000
        
        if response.status_code == 200:
            print_result(True, f"OLLAMA ONLINE (responded in {elapsed:.0f}ms)")
            print_info(f"Response: {response.text.strip()[:50]}")
            return True
        else:
            print_result(False, f"Unexpected status code: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print_result(False, "OLLAMA UNREACHABLE - Connection refused")
        print_info(f"Is Ollama running? Try: ollama serve")
        print_info(f"Or check if host '{host}' is correct")
        return False
    except requests.exceptions.Timeout:
        print_result(False, "OLLAMA TIMEOUT - Server not responding")
        return False
    except Exception as e:
        print_result(False, f"Connection error: {e}")
        return False


# ============================================================================
# STEP 2: VERIFY MODEL AVAILABILITY
# ============================================================================

def test_model_availability(host: str, port: int) -> bool:
    """
    Step 2: Check if the expected model is available.
    """
    print()
    print_step(2, "MODEL AVAILABILITY CHECK")
    
    base_url = f"http://{host}:{port}"
    print_info(f"Checking for model: {EXPECTED_MODEL}")
    
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=10)
        response.raise_for_status()
        
        data = response.json()
        models = data.get("models", [])
        
        print_info(f"Found {len(models)} model(s) installed:")
        
        model_found = False
        for model in models:
            name = model.get("name", "unknown")
            size = model.get("size", 0) / 1e9  # Convert to GB
            print(f"      • {name} ({size:.1f} GB)")
            
            # Check if this is our expected model (partial match)
            if EXPECTED_MODEL.split(":")[0] in name:
                model_found = True
        
        if model_found:
            print_result(True, f"Model '{EXPECTED_MODEL}' is available")
            return True
        else:
            print_result(False, f"MODEL MISSING: {EXPECTED_MODEL}")
            print_info(f"Run: ollama pull {EXPECTED_MODEL}")
            return False
            
    except Exception as e:
        print_result(False, f"Failed to check models: {e}")
        return False


# ============================================================================
# STEP 3: TEST CLASS INSTANTIATION
# ============================================================================

def test_class_instantiation(host: str, port: int) -> object:
    """
    Step 3: Test importing and instantiating CryptoBuffett class.
    """
    print()
    print_step(3, "CLASS INSTANTIATION TEST")
    
    print_info("Importing CryptoBuffett class...")
    
    try:
        from src.v2.sentiment.buffett import CryptoBuffett, TradePermission
        print_result(True, "Import successful")
    except ImportError as e:
        print_result(False, f"Import failed: {e}")
        print_info("Check if src/v2/sentiment/buffett.py exists")
        return None
    except Exception as e:
        print_result(False, f"Import error: {e}")
        return None
    
    print_info("Instantiating CryptoBuffett...")
    
    try:
        buffett = CryptoBuffett(
            ollama_host=host,
            ollama_port=port,
            model=EXPECTED_MODEL,
            timeout=TIMEOUT,
        )
        print_result(True, "Class instantiated successfully")
        print_info(f"  Base URL: {buffett.base_url}")
        print_info(f"  Model: {buffett.model}")
        print_info(f"  Timeout: {buffett.timeout}s")
        return buffett
    except Exception as e:
        print_result(False, f"Instantiation failed: {e}")
        return None


# ============================================================================
# STEP 4: TEST INFERENCE
# ============================================================================

async def test_inference(buffett: object) -> bool:
    """
    Step 4: Test actual inference with the LLM.
    """
    print()
    print_step(4, "INFERENCE TEST")
    
    print_info(f"Test headline: \"{TEST_HEADLINE}\"")
    print_info("Sending to LLM... (this may take 30-60 seconds)")
    print()
    
    try:
        start = time.time()
        result = await buffett.analyze_news(TEST_HEADLINE)
        elapsed = time.time() - start
        
        print_result(True, f"Inference completed in {elapsed:.1f}s")
        print()
        
        # Print raw response
        print("  ┌─ RAW RESPONSE ─────────────────────────────────────────┐")
        if result.raw_response:
            # Truncate if too long
            raw = result.raw_response[:500]
            for line in raw.split('\n'):
                print(f"  │ {line[:60]:<60} │")
        print("  └──────────────────────────────────────────────────────────┘")
        print()
        
        # Print parsed result
        print("  ┌─ PARSED RESULT ────────────────────────────────────────┐")
        print(f"  │ Sentiment Score: {result.sentiment_score:+.2f}                                  │")
        print(f"  │ Impact Duration: {result.impact_duration:<40} │")
        print(f"  │ Buffett Verdict: {result.buffett_verdict:<40} │")
        print(f"  │ Trade Permission: {result.get_permission().value:<39} │")
        print(f"  │ Parse Success: {result.parse_success}                                      │")
        print("  └──────────────────────────────────────────────────────────┘")
        print()
        print(f"  📝 Reasoning: {result.reasoning}")
        print()
        
        if result.parse_success:
            print_result(True, "Inference and parsing successful!")
        else:
            print_warn("Parse failed - LLM returned unexpected format")
            print_info("The system fell back to neutral (Ignore) verdict")
        
        return result.parse_success
        
    except Exception as e:
        print_result(False, f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# MAIN
# ============================================================================

async def run_diagnostics(host: str, port: int):
    """Run all diagnostic steps."""
    print_header()
    
    results = {
        "connectivity": False,
        "model": False,
        "class": False,
        "inference": False,
    }
    
    # Step 1: Connectivity
    results["connectivity"] = test_ollama_connectivity(host, port)
    
    if not results["connectivity"]:
        print()
        print("=" * 70)
        print("🔴 DIAGNOSTIC ABORTED: Ollama not reachable")
        print("=" * 70)
        print()
        print("   Possible solutions:")
        print("   1. Start Ollama: ollama serve")
        print("   2. Check if running in Docker: use --host host.docker.internal")
        print("   3. Check firewall/ports")
        print()
        return results
    
    # Step 2: Model
    results["model"] = test_model_availability(host, port)
    
    if not results["model"]:
        print()
        print("=" * 70)
        print("🔴 DIAGNOSTIC ABORTED: Model not available")
        print("=" * 70)
        print()
        print(f"   Run: ollama pull {EXPECTED_MODEL}")
        print()
        return results
    
    # Step 3: Class instantiation
    buffett = test_class_instantiation(host, port)
    results["class"] = buffett is not None
    
    if not results["class"]:
        print()
        print("=" * 70)
        print("🔴 DIAGNOSTIC ABORTED: Class instantiation failed")
        print("=" * 70)
        return results
    
    # Step 4: Inference
    results["inference"] = await test_inference(buffett)
    
    # Summary
    print()
    print("=" * 70)
    print("📊 DIAGNOSTIC SUMMARY")
    print("=" * 70)
    
    all_passed = all(results.values())
    
    for step, passed in results.items():
        emoji = "✅" if passed else "❌"
        print(f"   {emoji} {step.upper()}")
    
    print()
    if all_passed:
        print("🟢 ALL TESTS PASSED - CryptoBuffett is fully operational!")
    else:
        print("🔴 SOME TESTS FAILED - Check the output above for details")
    print("=" * 70)
    print()
    
    return results


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Buffett Probe - CryptoBuffett Diagnostic Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python diagnose_buffett.py                    # Test localhost
    python diagnose_buffett.py --host llm_engine  # Test Docker container
    python diagnose_buffett.py --host 192.168.1.100 --port 11434
        """
    )
    
    parser.add_argument(
        "--host", "-H",
        type=str,
        default=DEFAULT_HOST,
        help=f"Ollama host (default: {DEFAULT_HOST})",
    )
    
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=DEFAULT_PORT,
        help=f"Ollama port (default: {DEFAULT_PORT})",
    )
    
    args = parser.parse_args()
    
    # Run diagnostics
    asyncio.run(run_diagnostics(args.host, args.port))


if __name__ == "__main__":
    main()
