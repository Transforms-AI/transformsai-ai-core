#!/usr/bin/env python3
"""Test script to verify cli_debug parameter works for stderr output."""

from transformsai_ai_core.central_logger import get_logger


def test_cli_debug_false():
    """Test logging with cli_debug=False (default - INFO and above)."""
    print("\n" + "=" * 60)
    print("TEST 1: cli_debug=False (should show INFO, WARNING, ERROR only)")
    print("=" * 60 + "\n")
    
    logger = get_logger("TestCLIdebugFalse", cli_debug=False)
    
    logger.debug("This is a DEBUG message - should NOT appear in stderr")
    logger.info("This is an INFO message - should appear in stderr")
    logger.warning("This is a WARNING message - should appear in stderr")
    logger.error("This is an ERROR message - should appear in stderr")


def test_cli_debug_true():
    """Test logging with cli_debug=True (should show DEBUG and above)."""
    print("\n" + "=" * 60)
    print("TEST 2: cli_debug=True (should show DEBUG, INFO, WARNING, ERROR)")
    print("=" * 60 + "\n")
    
    logger = get_logger("TestCLIdebugTrue", cli_debug=True)
    
    logger.debug("This is a DEBUG message - should appear in stderr")
    logger.info("This is an INFO message - should appear in stderr")
    logger.warning("This is a WARNING message - should appear in stderr")
    logger.error("This is an ERROR message - should appear in stderr")


def test_multiple_loggers():
    """Test multiple loggers with different names and cli_debug settings."""
    print("\n" + "=" * 60)
    print("TEST 3: Multiple loggers with cli_debug=False then True")
    print("=" * 60 + "\n")
    
    # Logger with cli_debug=False
    logger1 = get_logger("ModuleA", cli_debug=False)
    logger1.info("ModuleA - INFO message with cli_debug=False")
    logger1.debug("ModuleA - DEBUG message with cli_debug=False (should NOT appear)")
    
    # Logger with cli_debug=True (should override to debug)
    logger2 = get_logger("ModuleB", cli_debug=True)
    logger2.info("ModuleB - INFO message with cli_debug=True")
    logger2.debug("ModuleB - DEBUG message with cli_debug=True (should appear)")
    logger2.warning("ModuleB - WARNING message with cli_debug=True")


if __name__ == "__main__":
    test_cli_debug_false()
    test_cli_debug_true()
    test_multiple_loggers()
    
    print("\n" + "=" * 60)
    print("TESTS COMPLETE - Check stderr output above")
    print("=" * 60 + "\n")
