import os
import sys
import json
import glob
import unittest
import io
from contextlib import redirect_stderr
import transformsai_ai_core.central_logger as cv_logger
from loguru import logger

class TestCentralLogger(unittest.TestCase):
    
    def setUp(self):
        cv_logger._is_configured = False
        cv_logger._current_log_path = None
        cv_logger._start_time_obj = None
        logger.remove()

    def tearDown(self):
        logger.remove()
        if os.path.exists(".core-logs"):
            for f in glob.glob(".core-logs/*.jsonl"):
                try:
                    os.remove(f)
                except Exception:
                    pass

    def _read_jsonl_logs(self):
        # We need to wait for the enqueue thread to finish writing to file
        logger.complete() 
        logs = []
        if os.path.exists(cv_logger._current_log_path):
            with open(cv_logger._current_log_path, 'r') as f:
                for line in f:
                    if line.strip():
                        logs.append(json.loads(line))
        return logs

    def test_blind_naming(self):
        log = cv_logger.get_logger()
        log.info("Testing blind name")
        logs = self._read_jsonl_logs()
        # Now top-level key
        self.assertIn(logs[0]['logger_name'], ['test_logger', '__main__'])

    def test_explicit_naming(self):
        log = cv_logger.get_logger(name="DataAugmentationWorker")
        log.info("Testing explicit name")
        logs = self._read_jsonl_logs()
        self.assertEqual(logs[0]['logger_name'], "DataAugmentationWorker")

    def test_debug_mode_false(self):
        stderr_capture = io.StringIO()
        with redirect_stderr(stderr_capture):
            log = cv_logger.get_logger(debug=False)
            log.trace("This is a trace")
            log.debug("This is a debug")
            log.info("This is an info")
            logger.complete()

        console_output = stderr_capture.getvalue()
        self.assertIn("This is an info", console_output)
        self.assertNotIn("This is a debug", console_output)
        
        logs = self._read_jsonl_logs()
        self.assertEqual(len(logs), 3)
        self.assertEqual(logs[0]['level'], "TRACE")

    def test_debug_mode_true(self):
        stderr_capture = io.StringIO()
        with redirect_stderr(stderr_capture):
            log = cv_logger.get_logger(debug=True)
            log.debug("This is a debug")
            logger.complete()

        console_output = stderr_capture.getvalue()
        self.assertIn("This is a debug", console_output)

    def test_file_creation_and_json_format(self):
        log = cv_logger.get_logger()
        log.warning("Testing formatting")
        
        path = str(cv_logger._current_log_path)
        self.assertTrue("_IN_PROGRESS.jsonl" in path)
        
        logs = self._read_jsonl_logs()
        # Test flat keys
        self.assertIn("level", logs[0])
        self.assertIn("timestamp", logs[0])
        self.assertEqual(logs[0]['level'], "WARNING")

    def test_atexit_file_rename(self):
        log = cv_logger.get_logger()
        log.info("Testing rename")
        
        original_path = str(cv_logger._current_log_path)
        logger.remove() # Close everything
        cv_logger._rename_log_at_exit()
        
        new_files = glob.glob(".core-logs/run_*_to_*.jsonl")
        self.assertTrue(len(new_files) >= 1)

if __name__ == '__main__':
    unittest.main(verbosity=2)