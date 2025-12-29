"""
YOLO/YOLOE Wrappers with Auto-Export and Sequential Batching.

Provides wrapper classes that extend ultralytics YOLO and YOLOE with:
1. Automatic model export to optimized formats (ONNX, TensorRT, etc.)
2. Smart caching - checks if exported model exists before exporting
3. Sequential batching - chunks inputs by batch_size and aggregates results
4. Fallback loading - uses exported model with fallback to original on error
"""

from pathlib import Path
from typing import Any, Union, List
from .central_logger import get_logger

try:
    from ultralytics import YOLO, YOLOE
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    # Create dummy base classes for type hints when ultralytics not installed
    class YOLO:
        pass
    class YOLOE:
        pass


# Format to file extension/suffix mapping for exported models
EXPORT_FORMAT_MAP = {
    # File-based formats
    'onnx': '.onnx',
    'torchscript': '.torchscript',
    'engine': '.engine',
    'tflite': '.tflite',
    'pb': '.pb',
    'mnn': '.mnn',
    'coreml': '.mlpackage',
    'edgetpu': '_edgetpu.tflite',
    
    # Directory-based formats (append _model suffix)
    'openvino': '_openvino_model',
    'saved_model': '_saved_model',
    'paddle': '_paddle_model',
    'ncnn': '_ncnn_model',
    'imx': '_imx_model',
    'rknn': '_rknn_model',
    'axelera': '_axelera_model',
}


class YOLOWrapper(YOLO):
    """
    YOLO wrapper with auto-export and sequential batching.
    
    Extends ultralytics YOLO with automatic model export to optimized formats
    and custom sequential batching for controlled memory usage.
    """
    
    def __init__(
        self,
        model_dict: dict[str, Any],
        **kwargs
    ):
        """
        Initialize YOLO wrapper with optional export and batching.
        
        Args:
            model_dict: Dictionary containing model configuration
            **kwargs: Additional arguments passed to YOLO constructor
        """
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError(
                "Ultralytics is not installed. "
                "Please install with: pip install transformsai-ai-core[all]"
            )
        
        self.logger = get_logger(self)
        self.original_model_path = Path(model_dict["path"])
        self.task = model_dict.get("load_options", {}).get("task", "detect")
        self.export_enabled = model_dict.get("export", False)
        self.export_options = model_dict.get("export_options", {})
        self.batch_size = model_dict.get("batch", 1)
        self.exported_model_path = None
        
        # Determine model path to load (exported or original)
        model_to_load = self._prepare_model()
        
        # Initialize parent YOLO class
        try:
            super().__init__(model_to_load, task=self.task, **kwargs)
            self.logger.info(f"Loaded YOLO model: {model_to_load}")
        except Exception as e:
            # Fallback to original if exported model fails
            if self.exported_model_path and model_to_load != str(self.original_model_path):
                self.logger.warning(f"Failed to load exported model, falling back to original: {e}")
                super().__init__(str(self.original_model_path), task=self.task, **kwargs)
                self.exported_model_path = None
            else:
                raise
    
    def _prepare_model(self) -> str:
        """
        Prepare model for loading: check for exported version or export if needed.
        
        Returns:
            Path to model file to load (exported or original)
        """
        # If export not enabled, use original model
        if not self.export_enabled:
            return str(self.original_model_path)
        
        # Check if exported model already exists
        export_format = self.export_options.get('format', 'onnx')
        exported_path = self._get_expected_export_path(export_format)
        
        if exported_path and exported_path.exists():
            self.logger.info(f"Found existing exported model: {exported_path}")
            self.exported_model_path = exported_path
            return str(exported_path)
        
        # Export model if it doesn't exist
        self.logger.info(f"Exporting model to {export_format} format...")
        try:
            # Load original model temporarily for export
            temp_model = YOLO(str(self.original_model_path), task=self.task)
            
            # Export with user-provided options
            export_result = temp_model.export(**self.export_options)
            
            # export() returns the path to exported model
            self.exported_model_path = Path(export_result)
            self.logger.info(f"Model exported successfully: {self.exported_model_path}")
            return str(self.exported_model_path)
            
        except Exception as e:
            self.logger.error(f"Export failed, using original model: {e}")
            return str(self.original_model_path)
    
    def _get_expected_export_path(self, export_format: str) -> Union[Path, None]:
        """
        Get expected path for exported model based on format.
        
        Args:
            export_format: Export format (onnx, engine, openvino, etc.)
            
        Returns:
            Expected path to exported model, or None if format unknown
        """
        if export_format not in EXPORT_FORMAT_MAP:
            self.logger.warning(f"Unknown export format: {export_format}")
            return None
        
        suffix = EXPORT_FORMAT_MAP[export_format]
        model_stem = self.original_model_path.stem  # filename without extension
        model_dir = self.original_model_path.parent
        
        # Construct expected path
        if suffix.startswith('_') and suffix.endswith('_model'):
            # Directory-based format
            expected_path = model_dir / f"{model_stem}{suffix}"
        else:
            # File-based format
            expected_path = model_dir / f"{model_stem}{suffix}"
        
        return expected_path
    
    def predict(
        self,
        source: Union[str, Path, List[Union[str, Path]], Any],
        **kwargs
    ) -> List[Any]:
        """
        Run prediction with sequential batching.
        
        Accepts single or multiple inputs, chunks them by batch_size,
        and aggregates results from sequential batches.
        
        Args:
            source: Single source (str/Path/array) or list of sources
            **kwargs: Additional arguments passed to YOLO.predict()
            
        Returns:
            List of Results objects (same as ultralytics)
        """
        # Normalize input to list
        if not isinstance(source, list):
            sources = [source]
        else:
            sources = source
        
        # If batch_size is 1 or sources fit in one batch, process directly
        if self.batch_size >= len(sources):
            return super().predict(sources, **kwargs)
        
        # Sequential batching: chunk inputs and aggregate results
        all_results = []
        for i in range(0, len(sources), self.batch_size):
            batch = sources[i:i + self.batch_size]
            batch_results = super().predict(batch, **kwargs)
            
            # Aggregate results (handle both list and generator)
            if hasattr(batch_results, '__iter__'):
                all_results.extend(batch_results)
            else:
                all_results.append(batch_results)
        
        return all_results


class YOLOEWrapper(YOLOE): 
    """
    YOLOE wrapper with auto-export and sequential batching.
    
    Extends ultralytics YOLOE with automatic model export to optimized formats
    and custom sequential batching for controlled memory usage.
    Supports YOLOE-specific features like text/visual prompts.
    """
    
    def __init__(
        self,
        model_dict: dict[str, Any],
        **kwargs
    ):
        """
        Initialize YOLOE wrapper with optional export and batching.
        
        Args:
            model_dict: Dictionary containing model configuration
            **kwargs: Additional arguments passed to YOLOE constructor
        """
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError(
                "Ultralytics is not installed. "
                "Please install with: pip install transformsai-ai-core[all]"
            )
        
        self.logger = get_logger(self)
        self.original_model_path = Path(model_dict["path"])
        self.export_enabled = model_dict.get("export", False)
        self.export_options = model_dict.get("export_options", {})
        self.batch_size = model_dict.get("batch", 1)
        self.task = model_dict.get("load_options", {}).get("task", "detect")
        self.set_classes = None # TODO: Currently set classes doesn't work, FIX LATER
        self.exported_model_path = None
        
        # Determine model path to load (exported or original)
        model_to_load = self._prepare_model()
        
        # Initialize parent YOLOE class
        try:
            super().__init__(model_to_load, task=self.task, **kwargs)
            if not self.export_enabled and self.set_classes is not None:
                # If the model is not exported, set classes after loading
                super().set_classes(self.set_classes, super().get_text_pe(self.set_classes))
            self.logger.info(f"Loaded YOLOE model: {model_to_load}")
        except Exception as e:
            # Fallback to original if exported model fails
            if self.exported_model_path and model_to_load != str(self.original_model_path):
                self.logger.warning(f"Failed to load exported model, falling back to original: {e}")
                super().__init__(str(self.original_model_path), task=self.task, **kwargs)
                self.exported_model_path = None
            else:
                raise
    
    def _prepare_model(self) -> str:
        """
        Prepare model for loading: check for exported version or export if needed.
        
        Returns:
            Path to model file to load (exported or original)
        """
        # If export not enabled, use original model
        if not self.export_enabled:
            return str(self.original_model_path)
        
        # Check if exported model already exists
        export_format = self.export_options.get('format', 'onnx')
        exported_path = self._get_expected_export_path(export_format)
        
        if exported_path and exported_path.exists():
            self.logger.info(f"Found existing exported model: {exported_path}")
            self.exported_model_path = exported_path
            return str(exported_path)
        
        # Export model if it doesn't exist
        self.logger.info(f"Exporting model to {export_format} format...")
        try:
            # Load original model temporarily for export
            temp_model = YOLOE(str(self.original_model_path), task=self.task)
            
            if self.set_classes is not None:
                temp_model.set_classes(self.set_classes, temp_model.get_text_pe(self.set_classes))
            
            # Export with user-provided options
            export_result = temp_model.export(**self.export_options)
            
            # export() returns the path to exported model
            self.exported_model_path = Path(export_result)
            self.logger.info(f"Model exported successfully: {self.exported_model_path}")
            return str(self.exported_model_path)
            
        except Exception as e:
            self.logger.error(f"Export failed, using original model: {e}")
            return str(self.original_model_path)
    
    def _get_expected_export_path(self, export_format: str) -> Union[Path, None]:
        """
        Get expected path for exported model based on format.
        
        Args:
            export_format: Export format (onnx, engine, openvino, etc.)
            
        Returns:
            Expected path to exported model, or None if format unknown
        """
        if export_format not in EXPORT_FORMAT_MAP:
            self.logger.warning(f"Unknown export format: {export_format}")
            return None
        
        suffix = EXPORT_FORMAT_MAP[export_format]
        model_stem = self.original_model_path.stem  # filename without extension
        model_dir = self.original_model_path.parent
        
        # Construct expected path
        if suffix.startswith('_') and suffix.endswith('_model'):
            # Directory-based format
            expected_path = model_dir / f"{model_stem}{suffix}"
        else:
            # File-based format
            expected_path = model_dir / f"{model_stem}{suffix}"
        
        return expected_path
    
    def predict(
        self,
        source: Union[str, Path, List[Union[str, Path]], Any],
        **kwargs
    ) -> List[Any]:
        """
        Run prediction with sequential batching.
        
        Accepts single or multiple inputs, chunks them by batch_size,
        and aggregates results from sequential batches.
        
        Args:
            source: Single source (str/Path/array) or list of sources
            **kwargs: Additional arguments passed to YOLOE.predict()
            
        Returns:
            List of Results objects (same as ultralytics)
        """
        # Normalize input to list
        if not isinstance(source, list):
            sources = [source]
        else:
            sources = source
        
        # If batch_size is 1 or sources fit in one batch, process directly
        if self.batch_size >= len(sources):
            return super().predict(sources, **kwargs)
        
        # Sequential batching: chunk inputs and aggregate results
        all_results = []
        for i in range(0, len(sources), self.batch_size):
            batch = sources[i:i + self.batch_size]
            batch_results = super().predict(batch, **kwargs)
            
            # Aggregate results (handle both list and generator)
            if hasattr(batch_results, '__iter__'):
                all_results.extend(batch_results)
            else:
                all_results.append(batch_results)
        
        return all_results
