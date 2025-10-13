import os
import sys
import time
import json
import tempfile
import shutil
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from io import BytesIO
import requests

# Assuming the DataUploader code is in a file accessible by the test runner
from transformsai_ai_core.datasend import (
    DataUploader,
    CacheManager,
    NetworkUtils,
    CacheItem
)

@pytest.fixture
def temp_cache_dir():
    """Create a temporary cache directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestNetworkUtils:
    """Test NetworkUtils class"""
    
    def test_get_mac_address(self):
        """Test MAC address retrieval"""
        mac = NetworkUtils.get_mac_address()
        assert mac is not None
        assert isinstance(mac, str)
        assert len(mac) == 17  # Format: XX:XX:XX:XX:XX:XX
    
    def test_get_ip_address(self):
        """Test IP address retrieval"""
        ip = NetworkUtils.get_ip_address()
        assert ip is not None
        assert isinstance(ip, str)
        assert len(ip.split('.')) == 4  # IPv4 format


class TestCacheManager:
    """Test CacheManager class"""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def cache_manager(self, temp_cache_dir):
        """Create CacheManager instance"""
        cache_file = os.path.join(temp_cache_dir, "test_cache.json")
        cache_files_dir = os.path.join(temp_cache_dir, "cached_files")
        return CacheManager(
            cache_file_path=cache_file,
            cache_files_dir=cache_files_dir,
            max_cache_items=10,
            max_cache_age_seconds=3600,
            max_cache_retries=3
        )
    
    def test_cache_directory_creation(self, cache_manager):
        """Test cache directory is created"""
        assert os.path.exists(cache_manager.cache_files_dir)
    
    def test_add_to_cache_without_files(self, cache_manager):
        """Test adding item to cache without files"""
        cache_manager.add_to_cache(
            data_payload='{"test": "data"}',
            url="http://test.com/api",
            files_dict=None,
            identifier="test-uuid-1",
            is_heartbeat=False,
            method="POST",
            headers={"Content-Type": "application/json"}
        )
        
        assert len(cache_manager.failed_sends_cache) == 1
        assert cache_manager.failed_sends_cache[0].uuid == "test-uuid-1"
    
    def test_add_to_cache_with_single_file(self, cache_manager):
        """Test adding item to cache with a single file"""
        file_content = b"Test file content"
        files_dict = {
            "image": ("test.jpg", file_content, "image/jpeg")
        }
        
        cache_manager.add_to_cache(
            data_payload='{"test": "data"}',
            url="http://test.com/api",
            files_dict=files_dict,
            identifier="test-uuid-2",
            is_heartbeat=False,
            method="POST"
        )
        
        assert len(cache_manager.failed_sends_cache) == 1
        item = cache_manager.failed_sends_cache[0]
        assert len(item.cached_files) == 1
        
        # Verify file was cached
        field_name, file_info = item.cached_files[0]
        assert field_name == "image"
        assert os.path.exists(file_info['cache_file_path'])
    
    def test_add_to_cache_with_multiple_files_same_field(self, cache_manager):
        """Test adding item to cache with multiple files for same field"""
        file1_content = b"File 1 content"
        file2_content = b"File 2 content"
        files_dict = {
            "images": [
                ("file1.jpg", file1_content, "image/jpeg"),
                ("file2.jpg", file2_content, "image/jpeg")
            ]
        }
        
        cache_manager.add_to_cache(
            data_payload='{"test": "data"}',
            url="http://test.com/api",
            files_dict=files_dict,
            identifier="test-uuid-3",
            is_heartbeat=False,
            method="POST"
        )
        
        assert len(cache_manager.failed_sends_cache) == 1
        item = cache_manager.failed_sends_cache[0]
        assert len(item.cached_files) == 2
        
        # Verify both files were cached
        for field_name, file_info in item.cached_files:
            assert field_name == "images"
            assert os.path.exists(file_info['cache_file_path'])
    
    def test_add_to_cache_with_multiple_fields(self, cache_manager):
        """Test adding item to cache with multiple file fields"""
        files_dict = {
            "image": ("photo.jpg", b"Photo content", "image/jpeg"),
            "document": ("doc.pdf", b"PDF content", "application/pdf")
        }
        
        cache_manager.add_to_cache(
            data_payload='{"test": "data"}',
            url="http://test.com/api",
            files_dict=files_dict,
            identifier="test-uuid-4",
            is_heartbeat=False,
            method="POST"
        )
        
        assert len(cache_manager.failed_sends_cache) == 1
        item = cache_manager.failed_sends_cache[0]
        assert len(item.cached_files) == 2
    
    def test_remove_from_cache(self, cache_manager):
        """Test removing item from cache"""
        cache_manager.add_to_cache(
            data_payload='{"test": "data"}',
            url="http://test.com/api",
            files_dict=None,
            identifier="test-uuid-5",
            is_heartbeat=False,
            method="POST"
        )
        
        item = cache_manager.failed_sends_cache[0]
        cache_manager.remove_from_cache(item)
        
        assert len(cache_manager.failed_sends_cache) == 0
    
    def test_remove_from_cache_with_files(self, cache_manager):
        """Test removing item from cache cleans up files"""
        files_dict = {
            "image": ("test.jpg", b"Test content", "image/jpeg")
        }
        
        cache_manager.add_to_cache(
            data_payload='{"test": "data"}',
            url="http://test.com/api",
            files_dict=files_dict,
            identifier="test-uuid-6",
            is_heartbeat=False,
            method="POST"
        )
        
        item = cache_manager.failed_sends_cache[0]
        field_name, file_info = item.cached_files[0]
        cache_file_path = file_info['cache_file_path']
        
        assert os.path.exists(cache_file_path)
        
        cache_manager.remove_from_cache(item)
        
        assert not os.path.exists(cache_file_path)
    
    def test_enforce_cache_limits_by_count(self, cache_manager):
        """Test cache limits enforcement by item count"""
        # Add more items than max_cache_items (10)
        for i in range(15):
            cache_manager.add_to_cache(
                data_payload=f'{{"test": "data{i}"}}',
                url="http://test.com/api",
                files_dict=None,
                identifier=f"test-uuid-{i}",
                is_heartbeat=False,
                method="POST"
            )
        
        # Should only keep newest 10 items
        assert len(cache_manager.failed_sends_cache) == 10
    
    def test_enforce_cache_limits_by_age(self, temp_cache_dir):
        """Test cache limits enforcement by age"""
        cache_manager = CacheManager(
            cache_file_path=os.path.join(temp_cache_dir, "test_cache.json"),
            cache_files_dir=os.path.join(temp_cache_dir, "cached_files"),
            max_cache_items=100,
            max_cache_age_seconds=1,  # 1 second
            max_cache_retries=3
        )
        
        cache_manager.add_to_cache(
            data_payload='{"test": "data"}',
            url="http://test.com/api",
            files_dict=None,
            identifier="test-uuid-old",
            is_heartbeat=False,
            method="POST"
        )
        
        # Wait for item to expire
        time.sleep(2)
        
        # Add another item to trigger cleanup
        cache_manager.add_to_cache(
            data_payload='{"test": "data2"}',
            url="http://test.com/api",
            files_dict=None,
            identifier="test-uuid-new",
            is_heartbeat=False,
            method="POST"
        )
        
        # Old item should be removed
        assert len(cache_manager.failed_sends_cache) == 1
        assert cache_manager.failed_sends_cache[0].uuid == "test-uuid-new"
    
    def test_get_items_for_retry(self, cache_manager):
        """Test getting items eligible for retry"""
        # Add items with different retry counts
        for i in range(5):
            cache_manager.add_to_cache(
                data_payload=f'{{"test": "data{i}"}}',
                url="http://test.com/api",
                files_dict=None,
                identifier=f"test-uuid-{i}",
                is_heartbeat=False,
                method="POST"
            )
            cache_manager.failed_sends_cache[-1].retry_count = i
        
        # Should only get items with retry_count < max_cache_retries (3)
        items = cache_manager.get_items_for_retry()
        assert len(items) == 3
    
    def test_save_and_load_cache(self, cache_manager):
        """Test saving and loading cache from disk"""
        # Add items
        cache_manager.add_to_cache(
            data_payload='{"test": "data"}',
            url="http://test.com/api",
            files_dict=None,
            identifier="test-uuid-save",
            is_heartbeat=False,
            method="POST"
        )
        
        # Create new cache manager with same file path
        cache_file_path = cache_manager.cache_file_path
        cache_files_dir = cache_manager.cache_files_dir
        
        new_cache_manager = CacheManager(
            cache_file_path=cache_file_path,
            cache_files_dir=cache_files_dir,
            max_cache_items=10,
            max_cache_age_seconds=3600,
            max_cache_retries=3
        )
        
        # Should load the saved cache
        assert len(new_cache_manager.failed_sends_cache) == 1
        assert new_cache_manager.failed_sends_cache[0].uuid == "test-uuid-save"


class TestDataUploader:
    """Test DataUploader class"""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def uploader(self, temp_cache_dir):
        """Create DataUploader instance"""
        cache_file = os.path.join(temp_cache_dir, "test_cache.json")
        cache_files_dir = os.path.join(temp_cache_dir, "cached_files")
        
        uploader_instance = DataUploader(
            base_url="http://test.com/api",
            heartbeat_url="http://test.com/heartbeat",
            max_workers=2,
            max_retries=2,
            retry_delay=0.1,
            timeout=10,
            disable_caching=False,
            cache_file_path=cache_file,
            cache_files_dir=cache_files_dir,
            cache_retry_interval=0  # Disable periodic retry for tests
        )
        yield uploader_instance
        uploader_instance.shutdown(wait=True) # Ensure shutdown
    
    @pytest.fixture
    def uploader_no_cache(self):
        """Create DataUploader instance without caching"""
        uploader_instance = DataUploader(
            base_url="http://test.com/api",
            disable_caching=True,
            max_retries=2,
            retry_delay=0.1
        )
        yield uploader_instance
        uploader_instance.shutdown(wait=True)

    
    def test_initialization(self, uploader):
        """Test DataUploader initialization"""
        assert uploader.base_url == "http://test.com/api"
        assert uploader.heartbeat_url == "http://test.com/heartbeat"
        assert uploader.caching_enabled is True
        assert uploader.cache_manager is not None
    
    def test_initialization_without_cache(self, uploader_no_cache):
        """Test DataUploader initialization without caching"""
        assert uploader_no_cache.caching_enabled is False
        assert uploader_no_cache.cache_manager is None
    
    def test_secret_key_handling_single(self):
        """Test secret key handling with single key"""
        uploader = DataUploader(
            base_url="http://test.com/api",
            secret_keys="test-secret-key",
            disable_caching=True
        )
        assert uploader.secret_keys == ["test-secret-key"]
    
    def test_secret_key_handling_multiple(self):
        """Test secret key handling with multiple keys"""
        keys = ["key1", "key2", "key3"]
        uploader = DataUploader(
            base_url="http://test.com/api",
            secret_keys=keys,
            disable_caching=True
        )
        assert uploader.secret_keys == keys
    
    def test_build_url_with_base_url(self, uploader):
        """Test URL building with base URL"""
        url = uploader._build_url(None, "/endpoint")
        assert url == "http://test.com/api/endpoint"
    
    def test_build_url_with_override(self, uploader):
        """Test URL building with override base URL"""
        url = uploader._build_url("http://other.com", "/endpoint")
        assert url == "http://other.com/endpoint"
    
    def test_build_url_with_params(self, uploader):
        """Test URL building with query parameters"""
        url = uploader._build_url(None, "/endpoint", {"key": "value", "foo": "bar"})
        assert "http://test.com/api/endpoint?" in url
        assert "key=value" in url
        assert "foo=bar" in url
    
    @patch('requests.post')
    def test_send_data_post_json_success(self, mock_post, uploader):
        """Test successful POST request with JSON data"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = '{"success": true}'
        mock_post.return_value = mock_response
        
        result = uploader.send_data_sync(
            data={"test": "data"},
            endpoint_path="/test",
            method="POST",
            content_type="json"
        )
        
        assert result == '{"success": true}'
        assert mock_post.called
    
    @patch('requests.post')
    def test_send_data_post_form_data_success(self, mock_post, uploader):
        """Test successful POST request with form data"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = '{"success": true}'
        mock_post.return_value = mock_response
        
        result = uploader.send_data_sync(
            data={"test": "data"},
            endpoint_path="/test",
            method="POST",
            content_type="form-data"
        )
        
        assert result == '{"success": true}'
        assert mock_post.called
    
    @patch('requests.get')
    def test_send_data_get_success(self, mock_get, uploader):
        """Test successful GET request"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = '{"result": "data"}'
        mock_get.return_value = mock_response
        
        result = uploader.send_data_sync(
            endpoint_path="/test",
            method="GET"
        )
        
        assert result == '{"result": "data"}'
        assert mock_get.called
    
    @patch('requests.get')
    def test_send_data_get_with_params(self, mock_get, uploader):
        """Test GET request with URL parameters"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = '{"result": "data"}'
        mock_get.return_value = mock_response
        
        result = uploader.send_data_sync(
            endpoint_path="/test",
            method="GET",
            url_params={"key": "value", "page": "1"}
        )
        
        assert result == '{"result": "data"}'
        assert mock_get.called
    
    @patch('requests.patch')
    def test_send_data_patch_success(self, mock_patch, uploader):
        """Test successful PATCH request"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = '{"updated": true}'
        mock_patch.return_value = mock_response
        
        result = uploader.send_data_sync(
            data={"field": "updated_value"},
            endpoint_path="/test/123",
            method="PATCH",
            content_type="json"
        )
        
        assert result == '{"updated": true}'
        assert mock_patch.called
    
    @patch('requests.put')
    def test_send_data_put_success(self, mock_put, uploader):
        """Test successful PUT request"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = '{"replaced": true}'
        mock_put.return_value = mock_response
        
        result = uploader.send_data_sync(
            data={"complete": "new_data"},
            endpoint_path="/test/123",
            method="PUT",
            content_type="json"
        )
        
        assert result == '{"replaced": true}'
        assert mock_put.called
    
    @patch('requests.delete')
    def test_send_data_delete_success(self, mock_delete, uploader):
        """Test successful DELETE request"""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_response.text = ''
        mock_delete.return_value = mock_response
        
        result = uploader.send_data_sync(
            endpoint_path="/test/123",
            method="DELETE"
        )
        
        assert result == ''
        assert mock_delete.called
    
    @patch('requests.post')
    def test_send_data_with_single_file(self, mock_post, uploader):
        """Test POST request with single file"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = '{"uploaded": true}'
        mock_post.return_value = mock_response
        
        file_content = b"Test file content"
        files = {
            "image": ("test.jpg", file_content, "image/jpeg")
        }
        
        result = uploader.send_data_sync(
            data={"description": "Test image"},
            files=files,
            endpoint_path="/upload",
            method="POST"
        )
        
        assert result == '{"uploaded": true}'
        assert mock_post.called
    
    @patch('requests.post')
    def test_send_data_with_multiple_files_same_field(self, mock_post, uploader):
        """Test POST request with multiple files for same field"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = '{"uploaded": 2}'
        mock_post.return_value = mock_response
        
        files = {
            "images": [
                ("file1.jpg", b"Content 1", "image/jpeg"),
                ("file2.jpg", b"Content 2", "image/jpeg")
            ]
        }
        
        result = uploader.send_data_sync(
            files=files,
            endpoint_path="/upload-multiple",
            method="POST"
        )
        
        assert result == '{"uploaded": 2}'
        assert mock_post.called
    
    @patch('requests.post')
    def test_send_data_with_multiple_file_fields(self, mock_post, uploader):
        """Test POST request with multiple file fields"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = '{"success": true}'
        mock_post.return_value = mock_response
        
        files = {
            "image": ("photo.jpg", b"Photo data", "image/jpeg"),
            "document": ("doc.pdf", b"PDF data", "application/pdf")
        }
        
        result = uploader.send_data_sync(
            data={"title": "Mixed upload"},
            files=files,
            endpoint_path="/upload",
            method="POST"
        )
        
        assert result == '{"success": true}'
        assert mock_post.called
    
    @patch('requests.post')
    def test_send_data_with_file_like_object(self, mock_post, uploader):
        """Test POST request with file-like object"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = '{"uploaded": true}'
        mock_post.return_value = mock_response
        
        file_obj = BytesIO(b"File content from BytesIO")
        files = {
            "file": ("data.txt", file_obj, "text/plain")
        }
        
        result = uploader.send_data_sync(
            files=files,
            endpoint_path="/upload",
            method="POST"
        )
        
        assert result == '{"uploaded": true}'
        assert mock_post.called
    
    @patch('requests.post')
    def test_send_data_retry_on_failure(self, mock_post, uploader):
        """Test retry mechanism on failure"""
        # First two attempts fail, third succeeds
        mock_response_fail = Mock()
        mock_response_fail.status_code = 500
        mock_response_fail.text = 'Server error'
        
        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.text = '{"success": true}'
        
        mock_post.side_effect = [
            mock_response_fail,
            mock_response_fail,
            mock_response_success
        ]
        
        result = uploader.send_data_sync(
            data={"test": "data"},
            endpoint_path="/test",
            method="POST"
        )
        
        assert result == '{"success": true}'
        assert mock_post.call_count == 3
    
    @patch('requests.post')
    def test_send_data_timeout(self, mock_post, uploader):
        """Test timeout handling"""
        import requests
        mock_post.side_effect = requests.exceptions.Timeout()
        
        result = uploader.send_data_sync(
            data={"test": "data"},
            endpoint_path="/test",
            method="POST"
        )
        
        assert result is None
        assert mock_post.call_count == uploader.max_retries + 1
    
    @patch('requests.post')
    def test_send_data_connection_error(self, mock_post, uploader):
        """Test connection error handling"""
        import requests
        mock_post.side_effect = requests.exceptions.ConnectionError()
        
        result = uploader.send_data_sync(
            data={"test": "data"},
            endpoint_path="/test",
            method="POST"
        )
        
        assert result is None
        assert mock_post.call_count == uploader.max_retries + 1
    
    @patch('requests.post')
    def test_send_data_adds_to_cache_on_failure(self, mock_post, uploader):
        """Test that failed requests are cached"""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = 'Server error'
        mock_post.return_value = mock_response
        
        initial_cache_size = len(uploader.cache_manager.failed_sends_cache)
        
        result = uploader.send_data_sync(
            data={"test": "data"},
            endpoint_path="/test",
            method="POST",
            dont_cache=False # Explicitly enable caching on failure
        )
        
        assert result is None
        assert len(uploader.cache_manager.failed_sends_cache) == initial_cache_size + 1
    
    @patch('requests.post')
    def test_send_heartbeat(self, mock_post, uploader):
        """Test sending heartbeat"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = '{"received": true}'
        mock_post.return_value = mock_response
        
        uploader.send_heartbeat(
            sn="test-device-001",
            timestamp="2025-01-01T00:00:00Z",
            status_log="System operational"
        )
        
        # Give async operation time to complete
        time.sleep(0.5)
        
        assert mock_post.called
        call_args = mock_post.call_args
        
        # Check that heartbeat data includes device info
        if call_args[1].get('data'):
            data = json.loads(call_args[1]['data']) if isinstance(call_args[1]['data'], str) else call_args[1]['data']
            assert 'sn' in data
            assert 'mac_address' in data
            assert 'ip_address' in data
    
    @patch('requests.post')
    def test_send_heartbeat_with_live_url(self, mock_post, uploader):
        """Test sending heartbeat with live URL"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = '{"received": true}'
        mock_post.return_value = mock_response
        
        uploader.send_heartbeat(
            sn="test-device-001",
            timestamp="2025-01-01T00:00:00Z",
            live_url="rtsp://example.com/stream",
            status_log="Streaming"
        )
        
        time.sleep(0.5)
        assert mock_post.called
    
    def test_async_send_data(self, uploader):
        """Test asynchronous send_data method"""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = '{"success": true}'
            mock_post.return_value = mock_response
            
            uploader.send_data(
                data={"test": "async"},
                endpoint_path="/test",
                method="POST"
            )
            
            # Wait for async operation
            time.sleep(0.5)
            
            assert mock_post.called
    
    def test_secret_key_added_to_headers(self):
        """Test that secret key is added to request headers"""
        uploader = DataUploader(
            base_url="http://test.com/api",
            secret_keys=["test-key-123"],
            disable_caching=True
        )
        
        headers = {"Content-Type": "application/json"}
        result_headers = uploader._add_secret_key_to_headers(headers)
        
        assert "X-Secret-Key" in result_headers
        assert result_headers["X-Secret-Key"] == "test-key-123"
    
    def test_secret_key_not_overridden(self):
        """Test that existing secret key in headers is not overridden"""
        uploader = DataUploader(
            base_url="http://test.com/api",
            secret_keys=["default-key"],
            disable_caching=True
        )
        
        headers = {"X-Secret-Key": "custom-key"}
        result_headers = uploader._add_secret_key_to_headers(headers)
        
        assert result_headers["X-Secret-Key"] == "custom-key"
    
    def test_random_secret_key_selection(self):
        """Test random selection from multiple secret keys"""
        keys = ["key1", "key2", "key3"]
        uploader = DataUploader(
            base_url="http://test.com/api",
            secret_keys=keys,
            disable_caching=True
        )
        
        selected_keys = set()
        for _ in range(20):  # Try multiple times
            key = uploader._get_random_secret_key()
            selected_keys.add(key)
        
        # With 20 tries, we should see multiple different keys
        assert len(selected_keys) > 1
        assert all(key in keys for key in selected_keys)
    
    def test_shutdown(self, uploader):
        """Test uploader shutdown"""
        uploader.shutdown(wait=False)
        assert uploader.shutting_down_event.is_set()
    
    def test_content_type_auto_detection_json(self, uploader):
        """Test automatic content type detection for JSON"""
        data_payload, files_prepared, headers = uploader._determine_content_type_and_prepare_data(
            content_type="auto",
            data={"test": "data"},
            files=None,
            method="POST"
        )
        
        assert "application/json" in headers.get("Content-Type", "")
    
    def test_content_type_auto_detection_form_data(self, uploader):
        """Test automatic content type detection for form data with files"""
        files = {
            "image": ("test.jpg", b"content", "image/jpeg")
        }
        
        data_payload, files_prepared, headers = uploader._determine_content_type_and_prepare_data(
            content_type="auto",
            data={"test": "data"},
            files=files,
            method="POST"
        )
        
        assert files_prepared is not None
        assert len(files_prepared) == 1


class TestCoverageGaps:
    """
    New tests to fill identified gaps in coverage, focusing on async
    failure, caching, and retry mechanisms.
    """

    @pytest.fixture
    def uploader(self, temp_cache_dir):
        """Create a DataUploader instance for these tests."""
        cache_file = os.path.join(temp_cache_dir, "test_cache.json")
        cache_files_dir = os.path.join(temp_cache_dir, "cached_files")
        
        uploader_instance = DataUploader(
            base_url="http://test.com/api",
            max_workers=1,
            max_retries=1, # Keep retries low for faster tests
            retry_delay=0.1,
            disable_caching=False,
            cache_file_path=cache_file,
            cache_files_dir=cache_files_dir,
            cache_retry_interval=0
        )
        yield uploader_instance
        uploader_instance.shutdown(wait=True)

    @patch('requests.post')
    def test_async_failure_with_files_is_cached(self, mock_post, uploader):
        """
        Verify that a failing ASYNCHRONOUS request with files correctly
        adds the item and files to the cache.
        """
        # 1. Setup: Mock request to always fail
        mock_post.side_effect = requests.exceptions.ConnectionError("Network down")
        
        files = {"image": ("test.jpg", b"file-content", "image/jpeg")}
        data = {"description": "test"}

        # 2. Action: Send data asynchronously
        uploader.send_data(data=data, files=files, endpoint_path="/upload")
        
        # Wait for the async thread and its retries to complete
        time.sleep(1) 

        # 3. Assertions
        # Check that the item was added to the cache
        assert len(uploader.cache_manager.failed_sends_cache) == 1
        cached_item = uploader.cache_manager.failed_sends_cache[0]
        
        # Check that the cached data is correct
        payload_to_check = json.loads(cached_item.data_payload) if isinstance(cached_item.data_payload, str) else cached_item.data_payload
        assert payload_to_check == data
        
        # Check that the file was cached to disk
        assert len(cached_item.cached_files) == 1
        field_name, file_info = cached_item.cached_files[0]
        cached_file_path = file_info['cache_file_path']
        
        assert field_name == "image"
        assert os.path.exists(cached_file_path)
        
        with open(cached_file_path, 'rb') as f:
            assert f.read() == b"file-content"

    @patch('requests.post')
    def test_end_to_end_cache_retry_with_files(self, mock_post, uploader):
        """
        Test the full workflow:
        1. An async request with files fails and is cached.
        2. The retry mechanism successfully sends it later.
        3. The cache is cleared upon success.
        """
        # --- Part 1: Failure and Caching ---
        
        # Mock the request to fail initially
        mock_post.side_effect = requests.exceptions.Timeout("Request timed out")
        
        files = {"doc": ("report.pdf", b"pdf-data", "application/pdf")}
        
        uploader.send_data(files=files, endpoint_path="/submit")
        time.sleep(1) # Wait for failure and caching

        # Assert that the item and file are in the cache
        assert len(uploader.cache_manager.failed_sends_cache) == 1
        cached_item = uploader.cache_manager.failed_sends_cache[0]
        _, file_info = cached_item.cached_files[0]
        cached_file_path = file_info['cache_file_path']
        assert os.path.exists(cached_file_path)
        
        # --- Part 2: Success on Retry ---

        # Change the mock to succeed
        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.text = '{"status": "ok"}'
        mock_post.side_effect = None # Clear the side effect
        mock_post.return_value = mock_response_success
        
        # Manually trigger the retry mechanism
        uploader._retry_failed_sends()
        time.sleep(0.5) # Allow retry to process

        # Assert that the cache is now empty
        assert len(uploader.cache_manager.failed_sends_cache) == 0
        
        # Assert that the cached file on disk has been cleaned up
        assert not os.path.exists(cached_file_path)
        
        # Assert that the successful request was made
        assert mock_post.call_count > 1 # Initial failed calls + 1 successful retry
        
    @patch('requests.post')
    def test_failing_heartbeat_is_not_cached(self, mock_post, uploader):
        """Verify that a failing heartbeat does not get added to the cache."""
        mock_post.side_effect = requests.exceptions.ConnectionError("Network down")
        
        uploader.send_heartbeat(sn="hb-test-01", timestamp="now")
        time.sleep(1) # Wait for async operation to fail

        assert len(uploader.cache_manager.failed_sends_cache) == 0

    def test_empty_file_is_not_cached(self, uploader):
        """Verify that a file with empty content is not saved to the cache directory."""
        files_dict = {"empty_file": ("empty.txt", b"", "text/plain")}
        
        uploader.cache_manager.add_to_cache(
            data_payload='{}',
            url="http://test.com/api",
            files_dict=files_dict,
            identifier="test-empty-file",
            is_heartbeat=False,
            method="POST"
        )
        
        # The cache item itself is created
        assert len(uploader.cache_manager.failed_sends_cache) == 1
        cached_item = uploader.cache_manager.failed_sends_cache[0]
        
        # But the list of cached files should be empty
        assert len(cached_item.cached_files) == 0
        
        # And no file should have been created in the cache directory
        cached_files_dir = uploader.cache_manager.cache_files_dir
        assert len(os.listdir(cached_files_dir)) == 0
        
    @patch('requests.post')
    def test_cache_retry_sends_correct_file_data(self, mock_post, uploader):
        """
        Verify that the retry mechanism correctly reads a cached file and
        includes its content in the subsequent HTTP request.
        """
        # 1. Setup: Manually create a cached state.
        file_content = b"this-is-the-exact-file-content"
        original_filename = "report.pdf"
        field_name = "document"
        
        # Create a dummy cache file on disk, just as the real process would.
        cache_files_dir = uploader.cache_manager.cache_files_dir
        # Use a unique name to avoid conflicts
        cached_file_path = os.path.join(cache_files_dir, "manual_cache_file_123.pdf")
        with open(cached_file_path, 'wb') as f:
            f.write(file_content)

        # Create the CacheItem object that points to this file.
        cached_item = CacheItem(
            uuid="test-retry-uuid-123",
            timestamp=time.time(),
            url="http://test.com/api/submit_report",
            method="POST",
            data_payload='{"report_id": "xyz"}',
            headers={},
            cached_files=[
                (field_name, {
                    'cache_file_path': cached_file_path,
                    'original_filename': original_filename,
                    'mimetype': 'application/pdf'
                })
            ]
        )
        
        # Manually insert the item into the cache manager.
        uploader.cache_manager.failed_sends_cache.append(cached_item)
        
        # Configure the mock to succeed.
        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_post.return_value = mock_response_success

        # 2. Action: Trigger the retry mechanism.
        uploader._retry_failed_sends()
        time.sleep(0.5) # Allow the retry to complete

        # 3. Assertions: Inspect the arguments of the mock call.
        mock_post.assert_called_once() # Ensure it was called exactly once.
        
        # Get the arguments that requests.post was called with.
        call_args = mock_post.call_args
        
        # The 'files' argument is in the keyword arguments ('kwargs').
        sent_files = call_args.kwargs.get('files')
        assert sent_files is not None, "The 'files' argument was not found in the request."
        
        # The 'requests' library prepares files as a list of tuples:
        # [(field_name, (filename, file_object, mimetype)), ...]
        assert len(sent_files) == 1
        sent_field_name, file_tuple = sent_files[0]
        
        sent_filename = file_tuple[0]
        sent_file_object = file_tuple[1]
        
        # Verify the details of the sent file.
        assert sent_field_name == field_name
        assert sent_filename == original_filename
        
        # This is the most important check: read the content of the file object
        # that was sent and verify it matches the original content.
        sent_content = sent_file_object.read()
        assert sent_content == file_content