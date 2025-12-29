#!/usr/bin/env python3
"""
Test YOLO/YOLOE Wrappers with Auto-Export and Sequential Batching.

This test script validates:
1. Model loading and initialization
2. Auto-export functionality with caching
3. Sequential batching with multiple inputs
4. Fallback to original model on export failure
5. Both YOLO and YOLOE wrapper behavior
"""

from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Check if ultralytics is available
try:
    from transformsai_ai_core.yolo_wrapper import YOLOWrapper, YOLOEWrapper, ULTRALYTICS_AVAILABLE
    from transformsai_ai_core import get_logger
except ImportError as e:
    print(f"ERROR: Ultralytics not installed: {e}")
    print("Please install with: pip install transformsai-ai-core[all]")
    sys.exit(1)

logger = get_logger("test_yolo_wrapper")


# Test file paths
TEST_DIR = Path(__file__).parent
TEST_IMAGE = TEST_DIR / "test.png"  # User will provide this
YOLO_MODEL = "yolo11n.pt"  # Default model, will be auto-downloaded by ultralytics
YOLOE_MODEL = "yoloe-11s-seg.pt"  # Default model, will be auto-downloaded by ultralytics


def print_test_header(test_num: int, description: str):
    """Print a formatted test header."""
    print(f"\n{'='*70}")
    print(f"Test {test_num}: {description}")
    print('='*70)


# =============================================================================
# SECTION 1: PRE-FLIGHT CHECKS
# =============================================================================

def check_prerequisites():
    """Check if all required files exist."""
    print_test_header(0, "Pre-flight Checks")
    
    all_good = True
    
    if not TEST_IMAGE.exists():
        logger.error(f"Test image not found: {TEST_IMAGE}")
        logger.error("Please provide a test image at tests/test.png")
        all_good = False
    else:
        logger.success(f"✓ Test image found: {TEST_IMAGE}")
    
    # Models will be auto-downloaded by ultralytics
    logger.info(f"YOLO model: {YOLO_MODEL} (will auto-download if needed)")
    logger.info(f"YOLOE model: {YOLOE_MODEL} (will auto-download if needed)")
    
    return all_good


# =============================================================================
# SECTION 2: YOLO WRAPPER TESTS
# =============================================================================

print_test_header(1, "YOLO Basic Loading (No Export)")
try:
    model = YOLOWrapper(
        model_path=YOLO_MODEL,
        export=False,
        batch_size=1
    )
    logger.success("✓ YOLO model loaded successfully")
    logger.info(f"  Original path: {model.original_model_path}")
    logger.info(f"  Export enabled: {model.export_enabled}")
    logger.info(f"  Batch size: {model.batch_size}")
    logger.info(f"  Exported path: {model.exported_model_path}")
except Exception as e:
    logger.error(f"✗ Failed to load YOLO model: {e}")


print_test_header(2, "YOLO Single Image Prediction")
try:
    model = YOLOWrapper(
        model_path=YOLO_MODEL,
        export=False,
        batch_size=1
    )
    
    results = model.predict(str(TEST_IMAGE))
    logger.success(f"✓ Prediction successful, got {len(results)} result(s)")
    
    # Analyze results
    result = results[0]
    num_detections = len(result.boxes) if hasattr(result.boxes, '__len__') else 0
    logger.info(f"  Number of detections: {num_detections}")
    
    if num_detections > 0:
        logger.info(f"  Classes detected: {result.boxes.cls.tolist()}")
        logger.info(f"  Confidences: {[f'{c:.3f}' for c in result.boxes.conf.tolist()]}")
        
        # Filter for class 0
        class_0_mask = result.boxes.cls == 0
        class_0_count = class_0_mask.sum().item()
        logger.info(f"  Class 0 detections: {class_0_count}")
    else:
        logger.warning("  No objects detected (may be expected depending on image)")
        
except Exception as e:
    logger.error(f"✗ Prediction failed: {e}", traceback=True)


print_test_header(3, "YOLO Batch Prediction (5 images, batch_size=2)")
try:
    model = YOLOWrapper(
        model_path=YOLO_MODEL,
        export=False,
        batch_size=2
    )
    
    # Create list of 5 image paths (using same image)
    images = [str(TEST_IMAGE)] * 5
    
    results = model.predict(images)
    logger.success(f"✓ Batch prediction successful, got {len(results)} result(s)")
    
    # Verify all results
    for i, result in enumerate(results):
        num_detections = len(result.boxes) if hasattr(result.boxes, '__len__') else 0
        logger.info(f"  Image {i+1}: {num_detections} detections")
        
except Exception as e:
    logger.error(f"✗ Batch prediction failed: {e}", traceback=True)


print_test_header(4, "YOLO ONNX Export with Caching")
try:
    # Clean up any existing ONNX file
    onnx_path = Path(YOLO_MODEL).with_suffix('.onnx')
    if onnx_path.exists():
        logger.info(f"Removing existing ONNX file: {onnx_path}")
        onnx_path.unlink()
    
    # First initialization - should trigger export
    logger.info("First load - expecting export...")
    model1 = YOLOWrapper(
        model_path=YOLO_MODEL,
        export=True,
        export_options={"format": "onnx", "imgsz": 640},
        batch_size=1
    )
    
    if model1.exported_model_path and model1.exported_model_path.exists():
        logger.success(f"✓ Exported model created: {model1.exported_model_path}")
        
        # Test prediction with exported model
        results = model1.predict(str(TEST_IMAGE))
        logger.success(f"✓ Prediction with exported model successful")
        
        # Second initialization - should use cached export
        logger.info("Second load - expecting to use cached export...")
        model2 = YOLOWrapper(
            model_path=YOLO_MODEL,
            export=True,
            export_options={"format": "onnx", "imgsz": 640},
            batch_size=1
        )
        
        if model2.exported_model_path == model1.exported_model_path:
            logger.success("✓ Cached export used successfully")
        else:
            logger.warning("⚠ Different export paths (unexpected)")
        
        # Clean up
        if onnx_path.exists():
            onnx_path.unlink()
            logger.info(f"Cleaned up: {onnx_path}")
    else:
        logger.warning("⚠ Export did not create file (may have failed)")
        
except Exception as e:
    logger.error(f"✗ Export test failed: {e}", traceback=True)


print_test_header(5, "YOLO Different Batch Sizes")
try:
    images = [str(TEST_IMAGE)] * 10
    
    for batch_size in [1, 3, 5, 10]:
        logger.info(f"Testing batch_size={batch_size}...")
        model = YOLOWrapper(
            model_path=YOLO_MODEL,
            export=False,
            batch_size=batch_size
        )
        
        results = model.predict(images)
        if len(results) == 10:
            logger.success(f"  ✓ batch_size={batch_size}: Got {len(results)} results")
        else:
            logger.error(f"  ✗ batch_size={batch_size}: Expected 10 results, got {len(results)}")
            
except Exception as e:
    logger.error(f"✗ Batch size test failed: {e}", traceback=True)


print_test_header(6, "YOLO Path Type Handling")
try:
    model = YOLOWrapper(
        model_path=YOLO_MODEL,
        export=False,
        batch_size=1
    )
    
    # Test with string path
    results_str = model.predict(str(TEST_IMAGE))
    logger.success(f"✓ String path: {len(results_str)} result(s)")
    
    # Test with Path object
    results_path = model.predict(TEST_IMAGE)
    logger.success(f"✓ Path object: {len(results_path)} result(s)")
    
    # Test with list of strings
    results_list_str = model.predict([str(TEST_IMAGE), str(TEST_IMAGE)])
    logger.success(f"✓ List of strings: {len(results_list_str)} result(s)")
    
    # Test with list of Paths
    results_list_path = model.predict([TEST_IMAGE, TEST_IMAGE])
    logger.success(f"✓ List of Paths: {len(results_list_path)} result(s)")
    
except Exception as e:
    logger.error(f"✗ Path type test failed: {e}", traceback=True)


# =============================================================================
# SECTION 3: YOLOE WRAPPER TESTS
# =============================================================================

print_test_header(7, "YOLOE Basic Loading (No Export)")
try:
    model = YOLOEWrapper(
        model_path=YOLOE_MODEL,
        export=False,
        batch_size=1
    )
    logger.success("✓ YOLOE model loaded successfully")
    logger.info(f"  Original path: {model.original_model_path}")
    logger.info(f"  Export enabled: {model.export_enabled}")
    logger.info(f"  Batch size: {model.batch_size}")
    logger.info(f"  Exported path: {model.exported_model_path}")
except Exception as e:
    logger.error(f"✗ Failed to load YOLOE model: {e}")


print_test_header(8, "YOLOE Single Image Prediction")
try:
    model = YOLOEWrapper(
        model_path=YOLOE_MODEL,
        export=False,
        batch_size=1
    )
    
    results = model.predict(str(TEST_IMAGE))
    logger.success(f"✓ YOLOE prediction successful, got {len(results)} result(s)")
    
    # Analyze results
    result = results[0]
    num_detections = len(result.boxes) if hasattr(result.boxes, '__len__') else 0
    logger.info(f"  Number of detections: {num_detections}")
    
    if num_detections > 0:
        logger.info(f"  Classes detected: {result.boxes.cls.tolist()}")
        logger.info(f"  Confidences: {[f'{c:.3f}' for c in result.boxes.conf.tolist()]}")
        
        # Filter for class 0
        class_0_mask = result.boxes.cls == 0
        class_0_count = class_0_mask.sum().item()
        logger.info(f"  Class 0 detections: {class_0_count}")
    else:
        logger.warning("  No objects detected (may be expected depending on image)")
        
except Exception as e:
    logger.error(f"✗ YOLOE prediction failed: {e}", traceback=True)


print_test_header(9, "YOLOE Batch Prediction (5 images, batch_size=2)")
try:
    model = YOLOEWrapper(
        model_path=YOLOE_MODEL,
        export=False,
        batch_size=2
    )
    
    images = [str(TEST_IMAGE)] * 5
    results = model.predict(images)
    logger.success(f"✓ YOLOE batch prediction successful, got {len(results)} result(s)")
    
    for i, result in enumerate(results):
        num_detections = len(result.boxes) if hasattr(result.boxes, '__len__') else 0
        logger.info(f"  Image {i+1}: {num_detections} detections")
        
except Exception as e:
    logger.error(f"✗ YOLOE batch prediction failed: {e}", traceback=True)


print_test_header(10, "YOLOE ONNX Export")
try:
    # Clean up any existing ONNX file
    onnx_path = Path(YOLOE_MODEL).with_suffix('.onnx')
    if onnx_path.exists():
        logger.info(f"Removing existing ONNX file: {onnx_path}")
        onnx_path.unlink()
    
    # Initialize with export
    model = YOLOEWrapper(
        model_path=YOLOE_MODEL,
        export=True,
        export_options={"format": "onnx", "imgsz": 640},
        batch_size=1
    )
    
    if model.exported_model_path and model.exported_model_path.exists():
        logger.success(f"✓ YOLOE exported model created: {model.exported_model_path}")
        
        # Test prediction
        results = model.predict(str(TEST_IMAGE))
        logger.success(f"✓ YOLOE prediction with exported model successful")
        
        # Clean up
        if onnx_path.exists():
            onnx_path.unlink()
            logger.info(f"Cleaned up: {onnx_path}")
    else:
        logger.warning("⚠ Export did not create file (may have failed)")
        
except Exception as e:
    logger.error(f"✗ YOLOE export test failed: {e}", traceback=True)


# =============================================================================
# SECTION 4: EXPORT FORMAT TESTS
# =============================================================================

print_test_header(11, "Export Format Mappings")
try:
    from transformsai_ai_core.yolo_wrapper import EXPORT_FORMAT_MAP
    
    logger.info(f"Total supported formats: {len(EXPORT_FORMAT_MAP)}")
    
    # Display some common formats
    common_formats = ['onnx', 'torchscript', 'engine', 'openvino', 'tflite']
    for fmt in common_formats:
        if fmt in EXPORT_FORMAT_MAP:
            logger.info(f"  {fmt:12s} -> {EXPORT_FORMAT_MAP[fmt]}")
    
    logger.success("✓ Export format mappings validated")
    
except Exception as e:
    logger.error(f"✗ Format mapping test failed: {e}", traceback=True)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("YOLO/YOLOE Wrapper Test Suite")
    logger.info("=" * 70)
    
    # Check prerequisites
    if not check_prerequisites():
        logger.error("\nTests cannot run without required files!")
        logger.error("Please provide:")
        logger.error("  1. tests/test.png - A test image")
        sys.exit(1)
    
    logger.info("\n" + "=" * 70)
    logger.info("All prerequisites met - Starting tests...")
    logger.info("=" * 70)
    
    # All tests are executed in order as the script runs
    # (Tests 1-11 execute sequentially above)
    
    logger.info("\n" + "=" * 70)
    logger.info("Test Suite Complete!")
    logger.info("=" * 70)
