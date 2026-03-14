#!/usr/bin/env python3
"""Test script to verify debug parameter works for stderr output."""

from transformsai_ai_core.central_logger import get_logger, set_global_console_level


def test_debug_false():
    """Test logging with debug=False (default - INFO and above)."""
    print("\n" + "=" * 60)
    print("TEST 1: debug=False (should show INFO, WARNING, ERROR only)")
    print("=" * 60 + "\n")
    
    logger = get_logger("TestDebugFalse", debug=False)
    
    logger.debug("This is a DEBUG message - should NOT appear in stderr")
    logger.info("This is an INFO message - should appear in stderr")
    logger.warning("This is a WARNING message - should appear in stderr")
    logger.error("This is an ERROR message - should appear in stderr")


def test_debug_true():
    """Test logging with debug=True (should show DEBUG and above)."""
    print("\n" + "=" * 60)
    print("TEST 2: debug=True (should show DEBUG, INFO, WARNING, ERROR)")
    print("=" * 60 + "\n")
    
    logger = get_logger("TestDebugTrue", debug=True)
    
    logger.debug("This is a DEBUG message - should appear in stderr")
    logger.info("This is an INFO message - should appear in stderr")
    logger.warning("This is a WARNING message - should appear in stderr")
    logger.error("This is an ERROR message - should appear in stderr")


def test_multiple_loggers():
    """Test multiple loggers with different names and debug settings."""
    print("\n" + "=" * 60)
    print("TEST 3: Multiple loggers with debug=False then True")
    print("=" * 60 + "\n")
    
    # Logger with debug=False
    logger1 = get_logger("ModuleA", debug=False)
    logger1.info("ModuleA - INFO message with debug=False")
    logger1.debug("ModuleA - DEBUG message with debug=False (should NOT appear)")
    
    # Logger with debug=True (should override to debug)
    logger2 = get_logger("ModuleB", debug=True)
    logger2.info("ModuleB - INFO message with debug=True")
    logger2.debug("ModuleB - DEBUG message with debug=True (should appear)")
    logger2.warning("ModuleB - WARNING message with debug=True")


def test_global_console_level():
    """Test global console level control."""
    print("\n" + "=" * 60)
    print("TEST 4: Global console level control")
    print("=" * 60 + "\n")
    
    # Set global to DEBUG
    set_global_console_level("DEBUG")
    global_debug_logger = get_logger("GlobalDebug")
    global_debug_logger.debug("This DEBUG appears (global DEBUG)")
    global_debug_logger.info("This INFO appears (global DEBUG)")
    
    # Reset to INFO
    set_global_console_level("INFO")
    info_only_logger = get_logger("InfoOnly")
    info_only_logger.debug("This DEBUG does NOT appear (global INFO)")
    info_only_logger.info("This INFO appears (global INFO)")


if __name__ == "__main__":
    test_debug_false()
    test_debug_true()
    test_multiple_loggers()
    test_global_console_level()
    
    print("\n" + "=" * 60)
    print("TESTS COMPLETE - Check stderr output above")
    print("=" * 60 + "\n")
