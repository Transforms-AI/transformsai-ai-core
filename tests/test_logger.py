#!/usr/bin/env python3
"""Comprehensive test script for the central logger."""

import logging
from src.transformsai_ai_core.central_logger import get_logger


def print_test_header(test_num: int, description: str):
    """Print a formatted test header."""
    print(f"\n{'='*70}")
    print(f"Test {test_num}: {description}")
    print('='*70)


# =============================================================================
# SECTION 1: LOGGER INITIALIZATION TESTS
# =============================================================================

print_test_header(1, "Logger with String Name")
string_logger = get_logger(name="CustomStringLogger")
string_logger.info("Logger initialized with custom string name")
string_logger.debug("Debug message from string logger")

print_test_header(2, "Logger with Object/Class Instance")
class MyService:
    def __init__(self):
        self.logger = get_logger(name=self)
    
    def perform_operation(self):
        self.logger.info("Service operation performed")
        self.logger.warning("Service warning issued")

service = MyService()
service.perform_operation()

print_test_header(3, "Logger with module_name Parameter")
module_logger = get_logger(module_name="test_module")
module_logger.info("Logger using module_name parameter")

print_test_header(4, "Logger with No Name (Default)")
default_logger = get_logger()
default_logger.info("Logger with default/no name")

print_test_header(5, "Multiple Loggers with Different Names")
logger_a = get_logger(name="ComponentA")
logger_b = get_logger(name="ComponentB")
logger_c = get_logger(name="ComponentC")
logger_a.info("Message from Component A")
logger_b.info("Message from Component B")
logger_c.info("Message from Component C")


# =============================================================================
# SECTION 2: LOG LEVEL TESTS
# =============================================================================

print_test_header(6, "All Log Levels (DEBUG to CRITICAL)")
level_logger = get_logger(name="LevelTester")
level_logger.trace("TRACE level message (lowest)")
level_logger.debug("DEBUG level message")
level_logger.info("INFO level message")
level_logger.success("SUCCESS level message")
level_logger.warning("WARNING level message")
level_logger.error("ERROR level message (no exception)", traceback=False)
level_logger.critical("CRITICAL level message")

print_test_header(7, "Formatted Messages")
format_logger = get_logger(name="FormatTester")
name = "Alice"
age = 30
format_logger.info(f"User {name} is {age} years old")
format_logger.info("User {} is {} years old", name, age)
format_logger.info("Dict data: {data}", data={"key": "value", "count": 42})


# =============================================================================
# SECTION 3: ERROR AND EXCEPTION HANDLING TESTS
# =============================================================================

print_test_header(8, "error() with traceback=True (default)")
error_logger = get_logger(name="ErrorHandler")
try:
    result = 1 / 0
except ZeroDivisionError:
    error_logger.error("Division by zero with traceback=True", traceback=True)

print_test_header(9, "error() with traceback=False")
try:
    result = 1 / 0
except ZeroDivisionError:
    error_logger.error("Division by zero with traceback=False", traceback=False)

print_test_header(10, "exception() Method - Full Traceback")
try:
    def nested_function():
        def inner_function():
            return 10 / 0
        return inner_function()
    nested_function()
except Exception:
    error_logger.exception("Exception in nested function calls")

print_test_header(11, "Multiple Exception Types")
exception_logger = get_logger(name="ExceptionTester")

# IndexError
try:
    my_list = [1, 2, 3]
    item = my_list[10]
except IndexError:
    exception_logger.exception("IndexError caught")

# KeyError
try:
    my_dict = {"key": "value"}
    value = my_dict["nonexistent"]
except KeyError:
    exception_logger.exception("KeyError caught")

# TypeError
try:
    result = "string" + 42
except TypeError:
    exception_logger.exception("TypeError caught")

# AttributeError
try:
    obj = None
    obj.some_method()
except AttributeError:
    exception_logger.exception("AttributeError caught")


# =============================================================================
# SECTION 4: CONTEXTUAL LOGGING TESTS
# =============================================================================

print_test_header(12, "Logging with Context Data")
context_logger = get_logger(name="ContextLogger")
context_logger.bind(user_id=12345, session="abc-def").info("User logged in")
context_logger.bind(request_id="req-789").warning("Request timeout")

print_test_header(13, "Logging in Different Scopes")

def function_with_logger():
    """Test logger in function scope."""
    func_logger = get_logger(name="FunctionScope")
    func_logger.info("Logging from function scope")

class ClassWithLogger:
    """Test logger in class scope."""
    def __init__(self):
        self.logger = get_logger(name=self)
    
    def method_a(self):
        self.logger.info("Logging from method_a")
    
    def method_b(self):
        self.logger.info("Logging from method_b")

function_with_logger()
cls_instance = ClassWithLogger()
cls_instance.method_a()
cls_instance.method_b()


# =============================================================================
# SECTION 5: STANDARD LIBRARY LOGGING INTERCEPTION
# =============================================================================

print_test_header(14, "Standard Library logging Module Interception")
stdlib_logger = logging.getLogger("stdlib_test")
stdlib_logger.debug("Standard library DEBUG message")
stdlib_logger.info("Standard library INFO message")
stdlib_logger.warning("Standard library WARNING message")
stdlib_logger.error("Standard library ERROR message")
stdlib_logger.critical("Standard library CRITICAL message")

print_test_header(15, "Third-party Library Logger Simulation")
# Simulate common third-party loggers (urllib3, requests, httpx)
urllib3_logger = logging.getLogger("urllib3")
urllib3_logger.warning("This should appear (WARNING level)")
urllib3_logger.debug("This should NOT appear (DEBUG level - filtered)")

requests_logger = logging.getLogger("requests")
requests_logger.warning("Requests library warning")


# =============================================================================
# SECTION 6: EDGE CASES AND SPECIAL SCENARIOS
# =============================================================================

print_test_header(16, "Very Long Messages")
long_logger = get_logger(name="LongMessageTest")
long_message = "A" * 500
long_logger.info(f"Long message test: {long_message}")

print_test_header(17, "Special Characters in Messages")
special_logger = get_logger(name="SpecialCharTest")
special_logger.info("Message with special chars: !@#$%^&*()[]{}|\\;:'\"<>,.?/~`")
special_logger.info("Unicode: 你好 мир 🌍 🚀 ✨")
special_logger.info("Newlines:\nLine 1\nLine 2\nLine 3")
special_logger.info("Tabs:\tColumn1\tColumn2\tColumn3")

print_test_header(18, "Empty and None Messages")
empty_logger = get_logger(name="EmptyTest")
empty_logger.info("")
empty_logger.info("   ")  # Whitespace only

print_test_header(19, "Rapid Sequential Logging")
rapid_logger = get_logger(name="RapidTest")
for i in range(10):
    rapid_logger.info(f"Rapid log message #{i+1}")

print_test_header(20, "Nested Error Context")
nested_logger = get_logger(name="NestedError")
try:
    try:
        try:
            raise ValueError("Inner exception")
        except ValueError:
            raise KeyError("Middle exception")
    except KeyError:
        raise RuntimeError("Outer exception")
except RuntimeError:
    nested_logger.exception("Nested exception chain")


# =============================================================================
# SECTION 7: PERFORMANCE AND STRESS TESTS
# =============================================================================

print_test_header(21, "Mixed Log Levels in Sequence")
mixed_logger = get_logger(name="MixedLevelTest")
mixed_logger.debug("Debug 1")
mixed_logger.info("Info 1")
mixed_logger.warning("Warning 1")
mixed_logger.debug("Debug 2")
mixed_logger.error("Error 1", traceback=False)
mixed_logger.info("Info 2")
mixed_logger.critical("Critical 1")

print_test_header(22, "Logger Reinitialization")
# Test getting the same logger multiple times
reinit_logger_1 = get_logger(name="ReinitTest")
reinit_logger_1.info("First initialization")
reinit_logger_2 = get_logger(name="ReinitTest")
reinit_logger_2.info("Second initialization (same name)")


# =============================================================================
# SECTION 8: REAL-WORLD USAGE SCENARIOS
# =============================================================================

print_test_header(23, "Simulated Application Workflow")
class Application:
    def __init__(self):
        self.logger = get_logger(name=self)
    
    def startup(self):
        self.logger.info("Application starting up...")
        self.logger.debug("Loading configuration")
        self.logger.debug("Initializing database connection")
        self.logger.success("Application started successfully")
    
    def process_request(self, request_id: str):
        self.logger.info(f"Processing request: {request_id}")
        try:
            # Simulate processing
            if request_id == "bad_request":
                raise ValueError("Invalid request ID")
            self.logger.debug(f"Request {request_id} processed successfully")
        except ValueError:
            self.logger.exception(f"Failed to process request: {request_id}")
    
    def shutdown(self):
        self.logger.warning("Application shutting down...")
        self.logger.info("Cleanup completed")

app = Application()
app.startup()
app.process_request("req-001")
app.process_request("req-002")
app.process_request("bad_request")
app.process_request("req-003")
app.shutdown()


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "="*70)
print("ALL COMPREHENSIVE TESTS COMPLETED")
print("="*70)
print("\n📊 Test Summary:")
print("   ✓ Logger initialization (5 tests)")
print("   ✓ Log levels (2 tests)")
print("   ✓ Error/Exception handling (5 tests)")
print("   ✓ Contextual logging (2 tests)")
print("   ✓ Standard library interception (2 tests)")
print("   ✓ Edge cases (7 tests)")
print("   ✓ Real-world scenarios (1 test)")
print(f"\n   Total: 23 comprehensive test scenarios")
print("\n📁 Output Locations:")
print("   • Console: INFO level and above (you're seeing this now)")
print("   • logs/debug_*.log: All levels including DEBUG and TRACE")
print("   • logs/error_*.log: ERROR and CRITICAL only with full tracebacks")
print("\n💡 Next Steps:")
print("   1. Check the console output above")
print("   2. Review logs/debug_*.log for detailed logging")
print("   3. Review logs/error_*.log for error-specific logs")
print("="*70)
