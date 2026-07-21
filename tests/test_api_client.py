import os
import json
import shutil
import tempfile
from datetime import timedelta
from io import BytesIO

import pytest
import requests

from transformsai_ai_core.api_client import ApiClient, Response, EndpointProfile


# --------------------------------------------------------------------------- helpers
class FakeResp:
    """Minimal stand-in for requests.Response (what ApiClient._wrap touches)."""

    def __init__(self, status_code=200, text="ok", headers=None, content=None,
                 url="http://test.local/x"):
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}
        self.content = content if content is not None else text.encode("utf-8")
        self.url = url
        self.elapsed = timedelta(seconds=0.01)


@pytest.fixture
def cache_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


def make_client(cache_dir=None, **kw):
    kw.setdefault("base_url", "http://test.local/api")
    kw.setdefault("retry_backoff", 0)        # no real sleeping in tests
    kw.setdefault("retry_backoff_max", 0)
    kw.setdefault("cache_retry_interval", 0)  # no background timer
    if cache_dir is not None:
        kw["cache_dir"] = cache_dir
    else:
        kw.setdefault("cache_enabled", False)
    return ApiClient(**kw)


# --------------------------------------------------------------------------- verbs
@pytest.mark.parametrize("verb", ["get", "post", "put", "patch", "delete"])
def test_verb_success(verb):
    client = make_client()
    client._session.request = lambda *a, **k: FakeResp(200, text='{"value": 42}')
    resp = getattr(client, verb)("thing")
    assert isinstance(resp, Response)
    assert resp.ok
    assert resp.status_code == 200
    assert resp.text == '{"value": 42}'
    assert resp.json() == {"value": 42}
    assert resp.attempts == 1
    client.shutdown()


def test_arbitrary_verb():
    client = make_client()
    captured = {}

    def fake(method, url, **k):
        captured["method"] = method
        return FakeResp(200, text="reported")

    client._session.request = fake
    resp = client.request("REPORT", "endpoint")
    assert captured["method"] == "REPORT"
    assert resp.text == "reported"
    client.shutdown()


def test_response_shape():
    client = make_client()
    client._session.request = lambda *a, **k: FakeResp(
        201, text="created", headers={"X-Foo": "bar"}, content=b"created",
        url="http://test.local/api/thing",
    )
    resp = client.post("thing")
    assert resp.status_code == 201
    assert resp.ok is True
    assert resp.headers["X-Foo"] == "bar"
    assert resp.content == b"created"
    assert resp.url == "http://test.local/api/thing"
    assert resp.elapsed > 0
    assert resp.from_cache_retry is False
    client.shutdown()


# --------------------------------------------------------------------------- files
def test_file_upload_multiple_per_field_and_filelike():
    client = make_client()
    seen = {}

    def fake(method, url, **k):
        seen["files"] = k.get("files")
        return FakeResp(200)

    client._session.request = fake
    client.post("up", files={
        "imgs": [("a.jpg", b"A", "image/jpeg"), ("b.jpg", b"B", "image/jpeg")],
        "doc": ("d.txt", BytesIO(b"hello"), "text/plain"),
    })
    fields = [f[0] for f in seen["files"]]
    assert fields.count("imgs") == 2
    assert fields.count("doc") == 1
    client.shutdown()


# --------------------------------------------------------------------------- retry
def test_retry_on_connection_error_attempt_count():
    client = make_client(max_retries=3)
    client._session.request = lambda *a, **k: (_ for _ in ()).throw(
        requests.exceptions.ConnectionError("down"))
    resp = client.post("x")
    assert resp.ok is False
    assert resp.status_code == 0
    assert resp.attempts == 4  # max_retries + 1
    client.shutdown()


def test_retry_on_timeout_attempt_count():
    client = make_client(max_retries=2)
    client._session.request = lambda *a, **k: (_ for _ in ()).throw(
        requests.exceptions.Timeout("slow"))
    resp = client.post("x")
    assert resp.attempts == 3
    client.shutdown()


def test_retry_on_status_503_retried():
    client = make_client(max_retries=2)
    client._session.request = lambda *a, **k: FakeResp(503, text="busy")
    resp = client.post("x")
    assert resp.ok is False
    assert resp.attempts == 3
    client.shutdown()


def test_4xx_fails_fast():
    client = make_client(max_retries=3)
    calls = {"n": 0}

    def fake(*a, **k):
        calls["n"] += 1
        return FakeResp(400, text="bad")

    client._session.request = fake
    resp = client.post("x")
    assert resp.ok is False
    assert resp.attempts == 1
    assert calls["n"] == 1
    client.shutdown()


# --------------------------------------------------------------------------- caching policy
def test_cache_false_not_cached(cache_dir):
    client = make_client(cache_dir=cache_dir, max_retries=0)
    client._session.request = lambda *a, **k: FakeResp(503)
    client.post("x", cache=False)
    assert client.cache.pending == 0
    client.shutdown()


def test_failed_get_not_cached_auto(cache_dir):
    client = make_client(cache_dir=cache_dir, max_retries=0)
    client._session.request = lambda *a, **k: FakeResp(503)
    client.get("x")
    assert client.cache.pending == 0
    client.shutdown()


def test_failed_post_cached_auto(cache_dir):
    client = make_client(cache_dir=cache_dir, max_retries=0)
    client._session.request = lambda *a, **k: FakeResp(503)
    client.post("x", json={"a": 1})
    assert client.cache.pending == 1
    client.shutdown()


def test_empty_file_not_cached(cache_dir):
    client = make_client(cache_dir=cache_dir, max_retries=0)
    client._session.request = lambda *a, **k: FakeResp(503)
    client.post("up", files={"f": ("empty.txt", b"", "text/plain")})
    assert client.cache.pending == 0
    client.shutdown()


# --------------------------------------------------------------------------- end-to-end cache retry
def test_end_to_end_cache_retry_with_files(cache_dir):
    client = make_client(cache_dir=cache_dir, max_retries=0)
    payload = b"exact-bytes-payload"

    # Phase 1: every request fails -> request is cached to disk.
    client._session.request = lambda *a, **k: FakeResp(503)
    client.post("up", json={"meta": "m"}, files={"img": ("p.jpg", payload, "image/jpeg")})
    assert client.cache.pending == 1

    folders = [e for e in os.scandir(cache_dir) if e.is_dir() and not e.name.startswith(".tmp_")]
    assert len(folders) == 1
    folder = folders[0].path
    assert os.path.exists(os.path.join(folder, "meta.json"))
    files_on_disk = os.listdir(os.path.join(folder, "files"))
    assert len(files_on_disk) == 1

    # Phase 2: flip to success, capture exactly what is sent on retry.
    sent = {}

    def fake_ok(method, url, **k):
        sent["files"] = k.get("files")
        sent["json"] = k.get("json")
        return FakeResp(200)

    client._session.request = fake_ok
    client._retry_pending()

    assert client.cache.pending == 0
    assert not os.path.exists(folder)  # folder rmtree'd on success
    # exact bytes preserved through the cache round-trip
    field, (filename, content, mimetype) = sent["files"][0]
    assert field == "img"
    assert filename == "p.jpg"
    assert content == payload
    assert sent["json"] == {"meta": "m"}
    client.shutdown()


def test_max_cache_retries_drops_request(cache_dir):
    client = make_client(cache_dir=cache_dir, max_retries=0, max_cache_retries=2)
    client._session.request = lambda *a, **k: FakeResp(503)
    client.post("x", json={"a": 1})
    assert client.cache.pending == 1
    # retry repeatedly; after max_cache_retries the folder is dropped
    for _ in range(5):
        client._retry_pending()
        if client.cache.pending == 0:
            break
    assert client.cache.pending == 0
    client.shutdown()


# --------------------------------------------------------------------------- persistent (non-expiring) entries
def test_persist_does_not_cache_on_success(cache_dir):
    client = make_client(cache_dir=cache_dir, max_retries=0)
    client._session.request = lambda *a, **k: FakeResp(200)
    resp = client.post("x", json={"a": 1}, persist=True)
    assert resp.ok
    assert client.cache.pending == 0  # persist changes expiry, not *when* we cache
    client.shutdown()


def test_persist_caches_on_failure_and_is_marked(cache_dir):
    client = make_client(cache_dir=cache_dir, max_retries=0)
    client._session.request = lambda *a, **k: FakeResp(503)
    client.post("x", json={"a": 1}, persist=True)
    entries = client.list_cached()
    assert len(entries) == 1
    assert entries[0]["persistent"] is True
    client.shutdown()


def test_persist_retried_forever_past_max_cache_retries(cache_dir):
    client = make_client(cache_dir=cache_dir, max_retries=0, max_cache_retries=1)
    client._session.request = lambda *a, **k: FakeResp(503)
    client.post("x", json={"a": 1}, persist=True)
    client.post("y", json={"a": 2})  # transient control: dropped once retries run out
    for _ in range(5):
        client._retry_pending()
    entries = client.list_cached()
    assert len(entries) == 1
    assert entries[0]["persistent"] is True
    assert entries[0]["retry_count"] > 1  # kept getting retried
    client.shutdown()


def test_persist_removed_once_it_finally_succeeds(cache_dir):
    client = make_client(cache_dir=cache_dir, max_retries=0)
    client._session.request = lambda *a, **k: FakeResp(503)
    client.post("x", json={"a": 1}, persist=True)
    assert client.cache.pending == 1
    client._session.request = lambda *a, **k: FakeResp(200)
    client._retry_pending()
    assert client.cache.pending == 0  # delivered, so it's done
    client.shutdown()


def test_persist_exempt_from_age_limit(cache_dir):
    client = make_client(cache_dir=cache_dir, max_retries=0, max_cache_age_seconds=60)
    client._session.request = lambda *a, **k: FakeResp(503)
    client.post("x", json={"a": 1}, persist=True)
    client.post("y", json={"a": 2})
    client.cache._folder_epoch = lambda fid: 0.0  # pretend everything is ancient
    assert len(client.cache.eligible()) == 1      # transient aged out, persistent survived
    assert client.list_cached()[0]["persistent"] is True
    client.shutdown()


def test_item_cap_sacrifices_transient_before_persistent(cache_dir):
    client = make_client(cache_dir=cache_dir, max_retries=0, max_cache_items=1)
    client._session.request = lambda *a, **k: FakeResp(503)
    client.post("x", json={"a": 1}, persist=True)  # oldest, but persistent
    client.post("y", json={"a": 2})
    entries = client.list_cached()
    assert len(entries) == 1
    assert entries[0]["persistent"] is True  # transient 'y' evicted instead
    client.shutdown()


def test_item_cap_still_bounds_persistent_entries(cache_dir):
    client = make_client(cache_dir=cache_dir, max_retries=0, max_cache_items=2)
    client._session.request = lambda *a, **k: FakeResp(503)
    for i in range(5):
        client.post(f"x{i}", json={"a": i}, persist=True)
    assert client.cache.pending == 2  # cap is a disk backstop, applies to everyone
    client.shutdown()


def test_remove_cached_deletes_persistent_entry(cache_dir):
    client = make_client(cache_dir=cache_dir, max_retries=0)
    client._session.request = lambda *a, **k: FakeResp(503)
    client.post("x", json={"a": 1}, persist=True)
    client.remove_cached(client.list_cached()[0]["id"])
    assert client.cache.pending == 0
    client.shutdown()


def test_endpoint_profile_persist_true(cache_dir):
    client = make_client(cache_dir=cache_dir, max_retries=0)
    client.register_endpoint("critical", {"path": "t/", "persist": True})
    client._session.request = lambda *a, **k: FakeResp(503)
    client.send("critical", json={"x": 1})
    assert client.list_cached()[0]["persistent"] is True
    client.shutdown()


def test_endpoint_profile_persist_overridden_per_call(cache_dir):
    client = make_client(cache_dir=cache_dir, max_retries=0)
    client.register_endpoint("critical", {"path": "t/", "persist": True})
    client._session.request = lambda *a, **k: FakeResp(503)
    client.send("critical", json={"x": 1}, persist=False)
    assert client.list_cached()[0]["persistent"] is False
    client.shutdown()


# --------------------------------------------------------------------------- startup scan
def test_corrupt_and_tmp_folders_skipped_on_scan(cache_dir):
    # corrupt meta
    bad = os.path.join(cache_dir, "20200101T000000000000_deadbeef")
    os.makedirs(bad)
    with open(os.path.join(bad, "meta.json"), "w") as f:
        f.write("{not valid json")
    # in-progress tmp folder
    os.makedirs(os.path.join(cache_dir, ".tmp_inprogress"))

    client = make_client(cache_dir=cache_dir)
    assert client.cache.pending == 0  # both skipped, no crash
    client.shutdown()


# --------------------------------------------------------------------------- session / pooling
def test_session_pooling_no_double_retry():
    client = make_client()
    assert isinstance(client._session, requests.Session)
    for scheme in ("http://", "https://"):
        adapter = client._session.get_adapter(scheme + "test.local")
        assert adapter.max_retries.total == 0  # urllib3 must not double-retry
    client.shutdown()


def test_session_reused_across_calls():
    client = make_client()
    calls = []
    client._session.request = lambda *a, **k: calls.append(1) or FakeResp(200)
    client.get("a")
    client.post("b")
    assert len(calls) == 2  # same pooled session handles both
    client.shutdown()


# --------------------------------------------------------------------------- endpoint profiles
def test_register_endpoint_string_and_send():
    client = make_client()
    client.register_endpoint("sentiment", "cameras/create-sentiment/")
    captured = {}

    def fake(method, url, **k):
        captured["method"] = method
        captured["url"] = url
        return FakeResp(200)

    client._session.request = fake
    client.send("sentiment", json={"x": 1})
    assert captured["method"] == "POST"
    assert captured["url"].endswith("cameras/create-sentiment/")
    client.shutdown()


def test_register_endpoint_dict_and_send():
    client = make_client()
    client.register_endpoint("report", {"path": "r/", "method": "PUT", "cache": False})
    captured = {}

    def fake(method, url, **k):
        captured["method"] = method
        return FakeResp(200)

    client._session.request = fake
    resp = client.send("report", json={"x": 1})
    assert captured["method"] == "PUT"
    assert resp.ok
    client.shutdown()


def test_endpoint_profile_cache_false_overrides(cache_dir):
    client = make_client(cache_dir=cache_dir, max_retries=0)
    client.register_endpoint("heartbeat", {"path": "hb/", "cache": False})
    client._session.request = lambda *a, **k: FakeResp(503)
    client.send("heartbeat", json={"alive": True})
    assert client.cache.pending == 0  # profile cache=False wins
    client.shutdown()


def test_endpoints_passed_in_constructor():
    client = make_client(endpoints={"a": "path-a/", "b": {"path": "path-b/", "method": "DELETE"}})
    assert "a" in client._endpoints
    assert client._endpoints["b"].method == "DELETE"
    client.shutdown()


# --------------------------------------------------------------------------- async
def test_async_returns_future():
    client = make_client()
    client._session.request = lambda *a, **k: FakeResp(200, text="async-ok")
    fut = client.post("x", async_=True)
    resp = fut.result(timeout=5)
    assert resp.ok
    assert resp.text == "async-ok"
    client.shutdown()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
