import os
import json
import time
import uuid
import socket
import struct
import threading
import requests
import random
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from typing import Dict, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
import urllib.parse
from .central_logger import get_logger

# --- Information About Script ---
__name__ = "DataUploader with Caching"
__version__ = "5.2.4" 
__author__ = "TransformsAI"

@dataclass
class CacheItem:
    """Represents a cached upload item"""
    uuid: str
    timestamp: float
    url: str
    method: str
    data_payload: Optional[str]
    headers: Dict[str, str]
    # Support multiple files with the same key.
    # Structure will be a list of tuples: [(field_name, file_info_dict), ...]
    cached_files: List[Tuple[str, Dict]] = field(default_factory=list)
    retry_count: int = 0
    is_heartbeat: bool = False

class NetworkUtils:
    """Utility class for network operations"""
    
    def __init__(self):
        self.logger = get_logger(self)
    
    @staticmethod
    def get_mac_address() -> str:
        """Get MAC address with multiple fallback methods"""
        logger = get_logger(name="NetworkUtils")
        try:
            import uuid
            mac = uuid.getnode()
            return ':'.join(['{:02x}'.format((mac >> elements) & 0xff) 
                           for elements in range(0, 2*6, 2)][::-1])
        except Exception as e:
            logger.warning(f"Failed to get MAC address using uuid method: {e}")
        
        try:
            with open('/sys/class/net/eth0/address', 'r') as f:
                mac = f.read().strip()
                logger.debug(f"Retrieved MAC address from eth0: {mac}")
                return mac
        except Exception as e:
            logger.debug(f"Failed to get MAC address from eth0: {e}")
        
        logger.warning("Using fallback MAC address")
        return "00:00:00:00:00:00"
    
    @staticmethod
    def get_ip_address() -> str:
        """Get IP address with interface fallback"""
        logger = get_logger(name="NetworkUtils")
        
        def get_ip(ifname: str) -> Optional[str]:
            try:
                import fcntl
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                ip = socket.inet_ntoa(fcntl.ioctl(
                    s.fileno(), 0x8915,
                    struct.pack('256s', ifname[:15].encode('utf-8'))
                )[20:24])
                logger.debug(f"Retrieved IP from {ifname}: {ip}")
                return ip
            except Exception as e:
                logger.debug(f"Failed to get IP from {ifname}: {e}")
                return None
        
        for iface in ['eth0', 'wlan0', 'enp0s3', 'enp1s0']:
            ip = get_ip(iface)
            if ip:
                return ip
        
        try:
            hostname = socket.gethostname()
            ip = socket.gethostbyname(hostname)
            logger.debug(f"Retrieved IP from hostname resolution: {ip}")
            return ip
        except socket.gaierror as e:
            logger.debug(f"Failed to get IP from hostname: {e}")
        
        logger.warning("Using fallback IP address")
        return "127.0.0.1"

class CacheManager:
    """Handles caching operations with thread safety"""
    
    def __init__(self, cache_file_path: str, cache_files_dir: str, 
                 max_cache_items: int, max_cache_age_seconds: int, 
                 max_cache_retries: int):
        self.cache_file_path = cache_file_path
        self.cache_files_dir = cache_files_dir
        self.max_cache_items = max_cache_items
        self.max_cache_age_seconds = max_cache_age_seconds
        self.max_cache_retries = max_cache_retries
        self.failed_sends_cache: List[CacheItem] = []
        self.cache_lock = threading.RLock()
        
        self.logger = get_logger(self)
        
        self._ensure_cache_directory()
        self._load_cache()
    
    def _ensure_cache_directory(self) -> None:
        """Ensure cache directory exists"""
        if self.cache_files_dir and not os.path.exists(self.cache_files_dir):
            try:
                os.makedirs(self.cache_files_dir, exist_ok=True)
                self.logger.info(f"Created cache directory: {self.cache_files_dir}")
            except Exception as e:
                self.logger.error(f"Failed to create cache directory {self.cache_files_dir}: {e}")
    
    def _load_cache(self) -> None:
        """Load cache from disk"""
        start_time = time.time()
        with self.cache_lock:
            if not os.path.exists(self.cache_file_path):
                self.logger.debug(f"Cache file not found: {self.cache_file_path}")
                return
            
            try:
                with open(self.cache_file_path, 'r') as f:
                    cache_data = json.load(f)
                
                self.failed_sends_cache = [
                    CacheItem(**item) for item in cache_data
                ]
                
                load_time = time.time() - start_time
                self.logger.info(f"Loaded {len(self.failed_sends_cache)} cached items in {load_time:.2f}s")
            except Exception as e:
                self.logger.error(f"Failed to load cache from {self.cache_file_path}: {e}")
                self.failed_sends_cache = []
    
    def _save_cache(self) -> bool:
        """Save cache to disk"""
        start_time = time.time()
        try:
            cache_data = [
                {
                    'uuid': item.uuid,
                    'timestamp': item.timestamp,
                    'url': item.url,
                    'method': item.method,
                    'data_payload': item.data_payload,
                    'headers': item.headers,
                    'cached_files': item.cached_files,
                    'retry_count': item.retry_count,
                    'is_heartbeat': item.is_heartbeat
                }
                for item in self.failed_sends_cache
            ]
            
            with open(self.cache_file_path, 'w') as f:
                json.dump(cache_data, f, indent=2)
            return True
        except Exception as e:
            self.logger.error(f"Failed to save cache: {e}")
            return False
    
    def add_to_cache(self, data_payload: Optional[str], url: str, files_dict: Optional[Dict], 
                    identifier: str, is_heartbeat: bool, method: str = "POST", 
                    headers: Optional[Dict] = None) -> None:
        """Add item to cache"""
        with self.cache_lock:
            # Changed to a list to support multiple files under the same key
            cached_files = []
            
            # Cache files to disk if present
            if files_dict and self.cache_files_dir:
                # Loop to handle both single file tuples and lists of file tuples
                for field_name, value in files_dict.items():
                    files_to_process = value if isinstance(value, list) else [value]
                    
                    for index, file_tuple in enumerate(files_to_process):
                        if len(file_tuple) < 3:
                            continue
                        
                        filename, file_content, mimetype = file_tuple[:3]
                        # Add index to filename to ensure uniqueness in cache
                        cache_filename = f"{identifier}_{field_name}_{index}_{filename}"
                        cache_file_path = os.path.join(self.cache_files_dir, cache_filename)
                        
                        try:
                            # Handle different types of file content
                            content_to_write = None
                            
                            if hasattr(file_content, 'read'):
                                # File-like object
                                if hasattr(file_content, 'seek'):
                                    file_content.seek(0)
                                content_to_write = file_content.read()
                                if hasattr(file_content, 'seek'):
                                    file_content.seek(0)
                            elif isinstance(file_content, (bytes, bytearray)):
                                # Bytes data
                                content_to_write = bytes(file_content)
                            else:
                                # Try to convert to bytes
                                content_to_write = bytes(file_content)
                            
                            # Validate content is not empty
                            if not content_to_write:
                                self.logger.error(f"File content is empty for {filename}, skipping cache")
                                continue
                            
                            # Write to cache file
                            with open(cache_file_path, 'wb') as f:
                                f.write(content_to_write)
                            
                            # Verify file was written correctly
                            if os.path.getsize(cache_file_path) == 0:
                                self.logger.error(f"Cached file is empty after write: {cache_file_path}")
                                os.remove(cache_file_path)
                                continue
                            
                            # Append a tuple of (field_name, file_info) to the list
                            file_info = {
                                'cache_file_path': cache_file_path,
                                'original_filename': filename,
                                'mimetype': mimetype
                            }
                            cached_files.append((field_name, file_info))
                            
                            self.logger.debug(f"Cached file: {filename} -> {cache_file_path} ({len(content_to_write)} bytes)")
                            
                        except Exception as e:
                            self.logger.error(f"Failed to cache file {filename}: {e}")
                            # Clean up partial file if it exists
                            if os.path.exists(cache_file_path):
                                try:
                                    os.remove(cache_file_path)
                                except:
                                    pass
            
            cache_item = CacheItem(
                uuid=identifier,
                timestamp=time.time(),
                url=url,
                method=method,
                data_payload=data_payload,
                headers=headers or {},
                cached_files=cached_files,
                is_heartbeat=is_heartbeat
            )
            
            self.failed_sends_cache.append(cache_item)
            self._enforce_cache_limits()
            self._save_cache()
            
            self.logger.info(f"Cached {identifier} (total: {len(self.failed_sends_cache)})")
    
    def remove_from_cache(self, item: CacheItem) -> None:
        """Remove item from cache and cleanup files"""
        with self.cache_lock:
            if item in self.failed_sends_cache:
                self.failed_sends_cache.remove(item)
                self._cleanup_cached_files(item)
                self._save_cache()
    
    def _cleanup_cached_files(self, item: CacheItem) -> None:
        """Clean up cached files for an item"""
        # Iterate through the list of tuples
        for _, field_data in item.cached_files:
            cache_file_path = field_data.get('cache_file_path')
            if cache_file_path and os.path.exists(cache_file_path):
                try:
                    os.remove(cache_file_path)
                    self.logger.debug(f"Cleaned up cached file: {cache_file_path}")
                except Exception as e:
                    self.logger.error(f"Failed to cleanup cached file {cache_file_path}: {e}")
    
    def _enforce_cache_limits(self) -> None:
        """Enforce cache size and age limits"""
        current_time = time.time()
        items_to_remove = []
        
        # Remove old items
        if self.max_cache_age_seconds > 0:
            for item in self.failed_sends_cache:
                if current_time - item.timestamp > self.max_cache_age_seconds:
                    items_to_remove.append(item)
                    self.logger.debug(f"Marking old item for removal: {item.uuid}")
        
        # Remove excess items (keep newest)
        if self.max_cache_items > 0 and len(self.failed_sends_cache) > self.max_cache_items:
            sorted_items = sorted(self.failed_sends_cache, key=lambda x: x.timestamp)
            excess_count = len(self.failed_sends_cache) - self.max_cache_items
            items_to_remove.extend(sorted_items[:excess_count])
        
        # Clean up items
        removed_count = 0
        for item in items_to_remove:
            if item in self.failed_sends_cache:
                self.failed_sends_cache.remove(item)
                self._cleanup_cached_files(item)
                removed_count += 1
        
        if removed_count > 0:
            self.logger.info(f"Cleaned up {removed_count} cached items")
    
    def get_items_for_retry(self) -> List[CacheItem]:
        """Get items that can be retried"""
        with self.cache_lock:
            return [item for item in self.failed_sends_cache 
                   if item.retry_count < self.max_cache_retries]

class DataUploader:
    """Enhanced DataUploader with caching, retry mechanism, and multiple HTTP methods support"""
    
    def __init__(self, base_url: Optional[str] = None,
                 heartbeat_url: Optional[str] = None,
                 headers: Optional[Dict] = None,
                 secret_keys: Optional[Union[str, List[str]]] = None,
                 secret_key_header: str = "X-Secret-Key",
                 max_workers: int = 5,
                 max_retries: int = 5,
                 retry_delay: int = 1,
                 timeout: int = 300,
                 disable_caching: bool = False,
                 cache_file_path: str = "uploader_cache.json",
                 cache_files_dir: str = "uploader_cached_files",
                 max_cache_retries: int = 5,
                 cache_retry_interval: int = 100,
                 max_cache_items: int = 300,
                 max_cache_age_seconds: int = 24*60*60,
                 source: str = "Frame Processor",
                 project_version: Optional[str] = None):
        
        self.logger = get_logger(self)
        
        # Core configuration
        self.base_url = base_url
        self.heartbeat_url = heartbeat_url
        self.headers = headers or {}

        # Process secret_keys to always be a list of strings
        if isinstance(secret_keys, str):
            self.secret_keys = [secret_keys]
        elif isinstance(secret_keys, list):
            self.secret_keys = secret_keys
        else:
            self.secret_keys = []
            
        self.secret_key_header = secret_key_header
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.source = source
        self.project_version = project_version
        
        # Network information
        self.mac_address = NetworkUtils.get_mac_address()
        self.ip_address = NetworkUtils.get_ip_address()
        try:
            self.hostname = os.uname().nodename
        except OSError:
            self.hostname = "unknown"
        
        self.logger.info(f"DataUploader initialized - MAC: {self.mac_address}, IP: {self.ip_address}")
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.shutting_down_event = threading.Event()
        
        # Cache management
        self.caching_enabled = not disable_caching
        if self.caching_enabled:
            self.cache_manager = CacheManager(
                cache_file_path, cache_files_dir, max_cache_items,
                max_cache_age_seconds, max_cache_retries
            )
            self.cache_retry_interval = cache_retry_interval
            self.cache_management_timer = None
            if cache_retry_interval > 0:
                self._start_cache_management_timer()
            self.logger.info("Caching enabled")
        else:
            self.cache_manager = None
            self.logger.info("Caching disabled")
    
    def _requires_caching(func: Callable) -> Callable:
        """Decorator to check if caching is enabled"""
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not self.caching_enabled or not self.cache_manager:
                return
            return func(self, *args, **kwargs)
        return wrapper
    
    def _build_url(self, base_url: Optional[str], endpoint_path: str, 
                   url_params: Optional[Dict] = None) -> str:
        """Build complete URL from components"""
        if base_url:
            url = base_url.rstrip('/') + '/' + endpoint_path.lstrip('/')
        elif self.base_url:
            url = self.base_url.rstrip('/') + '/' + endpoint_path.lstrip('/')
        else:
            raise ValueError("No base URL provided")
        
        if url_params:
            url += '?' + urllib.parse.urlencode(url_params)
        
        self.logger.debug(f"Built URL: {url}")
        return url
    
    def _get_random_secret_key(self) -> Optional[str]:
        """Get a random secret key from the list"""
        if not self.secret_keys:
            return None
        return random.choice(self.secret_keys)
    
    def _add_secret_key_to_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Add a randomly selected secret key to headers if not already present"""
        headers_copy = dict(headers)
        
        # Check if secret key header already exists
        if self.secret_key_header in headers_copy:
            return headers_copy
        
        # Add random secret key if not present and we have keys available
        secret_key = self._get_random_secret_key()
        if secret_key:
            headers_copy[self.secret_key_header] = secret_key
        
        return headers_copy
    
    def _determine_content_type_and_prepare_data(self, content_type: str, data: Optional[Dict], 
                                               files: Optional[Dict], method: str) -> Tuple[Optional[str], Optional[List], Dict]:
        """Determine content type and prepare data/files"""
        if content_type == "auto":
            content_type = "form-data" if files else ("json" if data else "form-data")
        
        data_payload = None
        # Changed to a list to support multiple files with the same key
        files_prepared = None
        request_headers = dict(self.headers)
        
        # Add random secret key to headers
        request_headers = self._add_secret_key_to_headers(request_headers)
        
        if content_type == "json":
            if data:
                data_payload = json.dumps(data)
                request_headers['Content-Type'] = 'application/json'
        elif content_type == "form-data":
            data_payload = data
            if files:
                # Prepare a list of tuples for requests
                files_prepared = []
                for field_name, value in files.items():
                    # Check if the value is a list of files or a single file
                    files_to_process = value if isinstance(value, list) else [value]
                    
                    for file_tuple in files_to_process:
                        if len(file_tuple) >= 2:
                            filename, file_content = file_tuple[:2]
                            mimetype = file_tuple[2] if len(file_tuple) >= 3 else 'application/octet-stream'
                            
                            if hasattr(file_content, 'read'):
                                files_prepared.append((field_name, (filename, file_content, mimetype)))
                            else:
                                import io
                                files_prepared.append((field_name, (filename, io.BytesIO(file_content), mimetype)))
        
        return data_payload, files_prepared, request_headers
    
    def _make_http_request(self, url: str, method: str, headers: Dict, 
                          data: Optional[str], files: Optional[List], timeout: int) -> requests.Response: # files type hint
        """Make HTTP request using specified method"""
        prepared_data = data
        if method == "GET" and data:
            # For GET requests, convert data to URL params
            try:
                data_dict = json.loads(data) if isinstance(data, str) else data
                url += '?' + urllib.parse.urlencode(data_dict)
                prepared_data = None
            except (json.JSONDecodeError, TypeError):
                pass
        
        # Method mapping for cleaner code
        method_map = {
            "GET": lambda: requests.get(url, headers=headers, timeout=timeout),
            # Pass the `files` list directly - requests will handle it correctly
            "POST": lambda: requests.post(url, headers=headers, data=prepared_data, files=files, timeout=timeout),
            "PATCH": lambda: requests.patch(url, headers=headers, data=prepared_data, files=files, timeout=timeout),
            "PUT": lambda: requests.put(url, headers=headers, data=prepared_data, files=files, timeout=timeout),
            "DELETE": lambda: requests.delete(url, headers=headers, timeout=timeout)
        }
        
        if method not in method_map:
            raise ValueError(f"Unsupported HTTP method: {method}")
        
        return method_map[method]()
    
    def _send_data_core(self, data_payload: Optional[str], url: str, files: Optional[List],
                        identifier: str, is_heartbeat: bool, method: str, headers: Dict,
                        cache_entry: Optional[CacheItem] = None, dont_cache: bool = False,
                        original_files_dict: Optional[Dict] = None) -> Tuple[bool, List[str], Optional[str]]:
        """Core data sending logic with retry mechanism"""
        start_time = time.time()
        messages = []
        single_request_timeout = max(5, self.timeout // (self.max_retries + 1))
        endpoint = url.split('/')[-1] if '/' in url else url
        
        for attempt in range(self.max_retries + 1):
            if time.time() - start_time > self.timeout:
                msg = f"Overall timeout exceeded for {identifier}"
                self.logger.warning(msg)
                messages.append(msg)
                break
            
            try:
                # Reset seek on file-like objects in the list of tuples
                if files:
                    for _, file_tuple in files:  # The first element is the field name
                        if len(file_tuple) > 1:
                            file_content = file_tuple[1]
                            if hasattr(file_content, 'seek'):
                                file_content.seek(0)

                attempt_start = time.time()
                response = self._make_http_request(url, method, headers, data_payload, files, single_request_timeout)
                
                success_codes = [200, 201, 204, 202]
                
                if response.status_code in success_codes:
                    # Success logging: time, attempts, endpoint, response code
                    total_time = time.time() - start_time
                    msg = f"✓ {identifier} sent in {total_time:.2f}s (attempt {attempt + 1}) to {endpoint} [{response.status_code}]"
                    self.logger.info(msg)
                    messages.append(msg)
                    
                    # Remove from cache if this was a retry
                    if cache_entry and self.cache_manager:
                        self.cache_manager.remove_from_cache(cache_entry)
                    
                    return True, messages, response.text
                else:
                    # Build error message with better formatting
                    error_msg = f"HTTP {response.status_code} for {identifier} (attempt {attempt + 1}/{self.max_retries + 1}) - {method} {endpoint}"
                    
                    # Add response details for debugging
                    if response.text:
                        content_type = response.headers.get('Content-Type', '')
                        
                        # Handle HTML responses (likely error pages)
                        if 'text/html' in content_type.lower():
                            # Extract title if present
                            import re
                            title_match = re.search(r'<title>(.*?)</title>', response.text, re.IGNORECASE | re.DOTALL)
                            title = title_match.group(1).strip() if title_match else "Unknown error"
                            error_msg += f" | Error: {title}"
                        else:
                            # For JSON or plain text, show limited content
                            response_preview = response.text[:200].replace('\n', ' ').replace('\r', '')
                            if len(response.text) > 200:
                                response_preview += "..."
                            error_msg += f" | Response: {response_preview}"
                    
                    self.logger.error(error_msg)
                    messages.append(error_msg)
                    
            except requests.exceptions.Timeout:
                error_msg = f"Timeout for {identifier} (attempt {attempt + 1}) - {method} {endpoint} after {single_request_timeout}s"
                self.logger.error(error_msg)
                messages.append(error_msg)
                
            except requests.exceptions.ConnectionError as e:
                error_msg = f"Connection error for {identifier} (attempt {attempt + 1}) - {method} {endpoint}: {str(e)}"
                self.logger.error(error_msg)
                messages.append(error_msg)
                
            except Exception as e:
                error_msg = f"Error for {identifier} (attempt {attempt + 1}) - {method} {endpoint}: {str(e)}"
                self.logger.error(error_msg)
                messages.append(error_msg)
            
            if attempt < self.max_retries:
                delay = self.retry_delay * (2 ** attempt)
                delay = min(delay, 10)
                self.logger.debug(f"Waiting {delay}s before retry {attempt + 2}")
                time.sleep(delay)

        # All retries failed. Cache if allowed.
        if not dont_cache and not is_heartbeat and self.cache_manager:
            self.cache_manager.add_to_cache(
                data_payload=data_payload,
                url=url,
                files_dict=original_files_dict, # Use the original dict for caching
                identifier=identifier,
                is_heartbeat=is_heartbeat,
                method=method,
                headers=headers
            )
            msg = f"FAILED: {identifier} after {self.max_retries + 1} attempts. Added to cache."
            self.logger.error(msg)
            messages.append(msg)
        else:
            final_failure_msg = f"FAILED: {identifier} after {self.max_retries + 1} attempts (not cached)"
            self.logger.error(final_failure_msg)
            messages.append(final_failure_msg)
        
        return False, messages, None
    
    def _send_data_thread(self, data_payload: Optional[str], url: str, files_prepared: Optional[List],
                        messages: List[str], identifier: str, is_heartbeat: bool, 
                        method: str, headers: Dict, cache_entry: Optional[CacheItem] = None,
                        original_files_dict: Optional[Dict] = None) -> List[str]:
        """Thread wrapper for core sending logic"""
        success, core_messages, response_text = self._send_data_core(
            data_payload=data_payload, 
            url=url, 
            files=files_prepared, 
            identifier=identifier, 
            is_heartbeat=is_heartbeat, 
            method=method, 
            headers=headers, 
            cache_entry=cache_entry,
            original_files_dict=original_files_dict
        )
        messages.extend(core_messages)
        return messages
    
    def _thread_done_callback(self, future, identifier: str, data_type_description: str) -> None:
        """Callback for completed threads"""
        try:
            future.result()
        except Exception as e:
            self.logger.error(f"Thread error for {identifier}: {e}")
                
    def send_data(self, data: Optional[Dict] = None, heartbeat: bool = False, 
                files: Optional[Dict] = None, base_url: Optional[str] = None,
                method: str = "POST", content_type: str = "auto", 
                endpoint_path: str = "", url_params: Optional[Dict] = None) -> None:
        """Sends data asynchronously in a background thread.

        This method queues the data for sending and returns immediately. It is the
        preferred method for non-blocking operations. If the request fails, it will
        be automatically cached for a later retry (if caching is enabled).

        Args:
            data: A dictionary of the data payload to send.
            heartbeat: If True, marks this as a heartbeat request, which may use a
                different URL and is not cached on failure.
            files: A dictionary of files to upload. The format is:
                {
                    'field_name': ('filename.jpg', b'file_content', 'image/jpeg'),
                    'multiple_files': [
                        ('file1.txt', b'content1', 'text/plain'),
                        ('file2.txt', b'content2', 'text/plain')
                    ]
                }
            base_url: An optional URL to override the instance's default base_url.
            method: The HTTP method to use (e.g., "POST", "GET", "PUT").
            content_type: The content type ("json", "form-data"). Defaults to "auto",
                which selects "json" if data is present, "form-data" if files are.
            endpoint_path: The specific API path to append to the base URL.
            url_params: A dictionary of query parameters to append to the URL.
        """
        messages = []
        base_id = "Heartbeat" if heartbeat else "DataUpload"
        identifier = f"{base_id}-{uuid.uuid4()}"
        
        try:
            # Determine URL
            if heartbeat and self.heartbeat_url:
                if "http" in self.heartbeat_url:
                    url_to_use = self.heartbeat_url
                else:
                    # If heartbeat_url is a relative path, combine with base_url
                    if base_url:
                        url_to_use = self._build_url(base_url, self.heartbeat_url)
                    else:
                        url_to_use = self._build_url(self.base_url, self.heartbeat_url)
            else:
                url_to_use = self._build_url(base_url, endpoint_path, url_params if method.upper() == "GET" else None)
            
            # Prepare data and headers
            data_payload, files_prepared, request_headers = self._determine_content_type_and_prepare_data(
                content_type, data, files, method
            )
            
            # Submit to thread pool
            future = self.executor.submit(
                self._send_data_thread, 
                data_payload=data_payload, 
                url=url_to_use, 
                files_prepared=files_prepared,
                messages=messages, 
                identifier=identifier, 
                is_heartbeat=heartbeat, 
                method=method.upper(), 
                headers=request_headers,
                original_files_dict=files
            )
            future.add_done_callback(
                lambda f: self._thread_done_callback(f, identifier, "async_send")
            )
            
        except Exception as e:
            self.logger.error(f"Failed to submit {identifier}: {e}")
                
    def send_data_sync(self, data: Optional[Dict] = None, heartbeat: bool = False,
                    files: Optional[Dict] = None, base_url: Optional[str] = None,
                    method: str = "POST", content_type: str = "auto",
                    endpoint_path: str = "", url_params: Optional[Dict] = None,
                    dont_cache: bool = False) -> Optional[str]:
        """Sends data synchronously and waits for a response.

        This method blocks until the request (including any retries) is complete.
        It is useful when you need to know the result of the upload immediately.

        Args:
            data: A dictionary of the data payload to send.
            heartbeat: If True, marks this as a heartbeat request.
            files: A dictionary of files to upload. See `send_data` for format.
            base_url: An optional URL to override the instance's default base_url.
            method: The HTTP method to use (e.g., "POST", "GET", "PUT").
            content_type: The content type ("json", "form-data", or "auto").
            endpoint_path: The specific API path to append to the base URL.
            url_params: A dictionary of query parameters to append to the URL.
            dont_cache: If True, this request will not be cached even if it fails
                and caching is globally enabled.

        Returns:
            The text content of the server's response if the request was
            successful, otherwise None.
        """
        base_id = "Heartbeat" if heartbeat else "DataUpload"
        identifier = f"{base_id}-{uuid.uuid4()}"
        
        try:
            # Determine URL
            if heartbeat and self.heartbeat_url:
                url_to_use = self.heartbeat_url
            else:
                url_to_use = self._build_url(base_url, endpoint_path, url_params if method.upper() == "GET" else None)
            
            # Prepare data and headers
            data_payload, files_prepared, request_headers = self._determine_content_type_and_prepare_data(
                content_type, data, files, method
            )
            
            # Send synchronously
            success, messages, response_text = self._send_data_core(
                data_payload=data_payload, 
                url=url_to_use, 
                files=files_prepared, 
                identifier=identifier,
                is_heartbeat=heartbeat, 
                method=method.upper(), 
                headers=request_headers, 
                dont_cache=dont_cache,
                original_files_dict=files
            )
            
            return response_text if success else None
            
        except Exception as e:
            self.logger.error(f"Sync send failed for {identifier}: {e}")
            return None
    
    def send_heartbeat(
        self, 
        sn: str, 
        timestamp: str,
        live_url: Optional[str] = None,
        method: str = "POST",
        extra_data: Optional[Dict] = {},
        status_log: str = "Heartbeat received successfully."
        ) -> None:
        """Send heartbeat with device information"""
        
        heartbeat_data = {
            "sn" : sn,
            "time": timestamp,
            "status_log": status_log,
            "mac_address": self.mac_address,
            "ip_address": self.ip_address,
            "hostname": self.hostname,
            "source": self.source
        }
        
        heartbeat_data.update(extra_data)
        
        if self.project_version:
            heartbeat_data["version"] = self.project_version
        
        if live_url:
            heartbeat_data["live_url"] = live_url
        
        self.send_data(
            data=heartbeat_data,
            heartbeat=True,
            method=method.upper(),
        )    
        
        return
    
    @_requires_caching
    def _start_cache_management_timer(self) -> None:
        """Start periodic cache management"""
        if self.shutting_down_event.is_set() or self.cache_retry_interval <= 0:
            return
        
        self.cache_management_timer = threading.Timer(
            self.cache_retry_interval, self._periodic_cache_management_task
        )
        self.cache_management_timer.daemon = True
        self.cache_management_timer.start()
        self.logger.info(f"Cache management timer started (interval: {self.cache_retry_interval}s)")
    
    @_requires_caching
    def _periodic_cache_management_task(self) -> None:
        """Periodic cache management task"""
        if self.shutting_down_event.is_set():
            return
        
        self.logger.debug("Running periodic cache management")
        self._retry_failed_sends()
        
        if not self.shutting_down_event.is_set():
            self._start_cache_management_timer()
    
    @_requires_caching
    def _retry_failed_sends(self) -> None:
        """Retry failed sends from cache"""
        if not self.cache_manager:
            return
            
        items_to_retry = self.cache_manager.get_items_for_retry()
        total_cached = len(self.cache_manager.failed_sends_cache)
        
        if not items_to_retry:
            if total_cached > 0:
                self.logger.info(f"Cache check: {total_cached} items cached, 0 eligible for retry")
            return
        
        self.logger.info(f"Cache check: {total_cached} total cached, retrying {len(items_to_retry)} items")
        
        for item in items_to_retry:
            try:
                # Prepare files from the cached list of tuples
                files_for_retry = None
                if item.cached_files:
                    files_for_retry = []
                    for field_name, file_info in item.cached_files:
                        cache_file_path = file_info.get('cache_file_path')
                        if cache_file_path and os.path.exists(cache_file_path):
                            try:
                                with open(cache_file_path, 'rb') as f:
                                    file_content = f.read()
                                
                                # Validate file content is not empty
                                if not file_content:
                                    self.logger.error(f"Cached file is empty: {cache_file_path}")
                                    continue
                                
                                # Create BytesIO object for proper file handling
                                import io
                                file_obj = io.BytesIO(file_content)
                                
                                # Append tuple to the list
                                files_for_retry.append((
                                    field_name,
                                    (
                                        file_info.get('original_filename', 'unknown'),
                                        file_obj,
                                        file_info.get('mimetype', 'application/octet-stream')
                                    )
                                ))
                                
                                self.logger.debug(f"Loaded cached file: {cache_file_path} ({len(file_content)} bytes)")
                                
                            except Exception as e:
                                self.logger.error(f"Failed to load cached file {cache_file_path}: {e}")
                                continue
                        else:
                            self.logger.warning(f"Cached file not found: {cache_file_path}")
                            continue
                
                # Skip retry if no valid files were loaded
                if item.cached_files and not files_for_retry:
                    self.logger.error(f"No valid cached files found for {item.uuid}, skipping retry")
                    continue
                
                # Increment retry count before attempting
                item.retry_count += 1
                self.cache_manager._save_cache()
                
                # Attempt to send (don't cache again on failure)
                success, messages, response_text = self._send_data_core(
                    item.data_payload, item.url, files_for_retry,
                    item.uuid, item.is_heartbeat, item.method, item.headers, 
                    cache_entry=item, dont_cache=True
                )
                
                if success:
                    self.logger.info(f"✓ Cache retry successful: {item.uuid}")
                else:
                    if item.retry_count >= self.cache_manager.max_cache_retries:
                        self.logger.warning(f"Max retries exceeded for {item.uuid}, removing from cache")
                        self.cache_manager.remove_from_cache(item)
                
            except Exception as e:
                self.logger.error(f"Cache retry error for {item.uuid}: {e}")

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the uploader and cleanup resources"""
        self.logger.info("Shutting down DataUploader")
        self.shutting_down_event.set()
        
        # Cancel timer
        if hasattr(self, 'cache_management_timer') and self.cache_management_timer:
            self.cache_management_timer.cancel()
        
        # Shutdown executor
        if wait:
            self.executor.shutdown(wait=True)
            self.logger.info("Shutdown complete")
        else:
            self.executor.shutdown(wait=False)
            self.logger.info("Shutdown initiated")