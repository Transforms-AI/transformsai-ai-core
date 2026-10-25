from transformsai_ai_core.central_logger import configure_logging, get_logger


logger = get_logger()
logger.info("This should be seen in file and stderr")
logger.debug("only file")
logger.trace("nowhere")


configure_logging(console_level="TRACE", file_level="INFO", secrets=["hunter2-secret"])
logger.info("Both")
logger.debug("only stderr")
logger.trace("only stderr")
logger.error("error")
logger.info("redacted: rtsp://admin:hunter2-secret@10.0.0.5/stream and hunter2-secret")

try:
    raise ValueError("This is a test error")
except ValueError:
    logger.error("An error occurred")


# # Test auto rotation log info many times
# max_logs = 100000
# for i in range(max_logs):
#     logger.info(f"Log number {i}")
#     time.sleep(0.0001)
