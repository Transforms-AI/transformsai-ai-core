import time
import sys
from transformsai_ai_core.central_logger import get_logger, logger

def run_demo():
    # 1. Initialize the logger
    # In a real app, you'd call this once at the very start of your main script.
    # Set debug=True to see DEBUG logs in the console.
    log = get_logger(name="MainPipeline", debug=True)

    print("\n" + "="*60)
    print("        COMPUTER VISION LOGGER DEMO")
    print("="*60 + "\n")

    # 2. Standard Logging Levels
    log.info("Checking system requirements...")
    time.sleep(0.2)
    log.success("GPU detected: NVIDIA RTX 4090")
    log.debug("CUDA version: 12.1")
    
    # This TRACE log won't show in console, but will be in the .jsonl file
    log.trace("Raw tensor shapes: [64, 3, 224, 224]")

    # 3. Structured Logging (The "Logdy" Superpower)
    # By using .bind(), these keys become top-level columns in Logdy!
    log.info("Starting training loop...")
    
    for epoch in range(1, 4):
        # Simulate training metrics
        mAP = 0.5 + (epoch * 0.1)
        loss = 1.0 / epoch
        
        # We 'bind' extra data to the log entry
        epoch_log = log.bind(epoch=epoch, mAP=mAP, loss=loss)
        epoch_log.info(f"Completed epoch {epoch}")
        time.sleep(0.3)

    # 4. Handling Warnings and Errors
    log.warning("High memory usage detected (85%)")
    
    # 5. Exception Handling (Automatic Tracebacks)
    log.info("Attempting to load a corrupted image...")
    try:
        # Simulate an error
        result = 10 / 0
    except ZeroDivisionError:
        log.exception("Failed to process image batch!")

    # 6. Finalizing
    print("\n" + "-"*60)
    # Force the background thread to finish writing to the file
    logger.complete() 
    
    from transformsai_ai_core.central_logger import _current_log_path
    print(f"DONE! Logs written to: {_current_log_path}")
    print("-"*60)
    
    print("\nQuick Tip for Logdy:")
    print(f"Run this command in another terminal to see your logs live:")
    print(f"  logdy follow {_current_log_path}")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_demo()