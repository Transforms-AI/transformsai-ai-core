import time
import sys
from transformsai_ai_core.central_logger import get_logger


logger = get_logger()
logger.info("This should be seen in file and stderr")
logger.debug("only file")
logger.trace("nowhere")


logger = get_logger(cli_sink_level="TRACE", file_sink_level="INFO")
logger.info("Both")
logger.debug("only stderr")
logger.trace("only stderr")


# Test auto rotation log info many times
max_logs = 100000
for i in range(max_logs):
    logger.info(f"Log number {i}")
    time.sleep(0.0001)