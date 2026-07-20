import os
import json
import time
import uuid
import random
import shutil
import threading
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
from requests.adapters import HTTPAdapter

from .central_logger import get_logger

# --- Information About Script ---
__name__ = "Generic ApiClient"
__author__ = "TransformsAI"

# Types
FileTuple = Tuple[str, Any, str]  # (filename, content, mimetype)
NormalizedFile = Tuple[str, FileTuple]  # (field, (filename, content, mimetype))


@dataclass(frozen=True)
class Response:
    """Uniform, backend-agnostic result for every request.

    Returned by sync calls and resolved by async ``Future``s. A total failure
    (timeout / connection error / unexpected exception) yields
    ``Response(status_code=0, ok=False, ...)`` rather than raising, so callers
    never touch a raw ``requests.Response``.
    """

    status_code: int
    headers: Dict[str, str]
    text: str
    content: bytes
    url: str
    ok: bool
    elapsed: float
    attempts: int
    from_cache_retry: bool = False

    def json(self) -> Any:
        """Parse the response body as JSON."""
        return json.loads(self.text)


@dataclass
class EndpointProfile:
    """A named, reusable request shape (the dynamic backend)."""

    path: str
    method: str = "POST"
    headers: Optional[Dict[str, str]] = None
    content_type: Optional[str] = None
    cache: Optional[bool] = None
    persist: Optional[bool] = None


def _read_content(content: Any) -> bytes:
    """Read arbitrary file content (bytes / file-like / str) into bytes, seek-resetting file-likes."""
    if hasattr(content, "read"):
        if hasattr(content, "seek"):
            content.seek(0)
        raw = content.read()
        if hasattr(content, "seek"):
            content.seek(0)
        return raw if isinstance(raw, (bytes, bytearray)) else str(raw).encode("utf-8")
    if isinstance(content, (bytes, bytearray)):
        return bytes(content)
    if isinstance(content, str):
        return content.encode("utf-8")
    return bytes(content)


def _sanitize(name: str) -> str:
    """Make a filename safe to use as a single path component."""
    return os.path.basename(str(name)).replace(os.sep, "_") or "file"


class _DirCache:
    """Directory-per-request cache: one folder per pending request.

    Atomic (write to ``.tmp_<id>/`` then ``os.replace``), corruption-tolerant
    (a bad folder is skipped, never wiping the cache), O(1) add/remove (no
    global manifest rewrite). The in-memory state is only the list of folder
    ids; request bytes are read lazily at retry time to keep memory flat.

    Entries added with ``persistent=True`` are reusable saved requests rather
    than transient failure retries: they are exempt from age/item-count/
    retry-count eviction, excluded from the automatic background retry loop,
    and never removed on success. They stay on disk completely intact until
    an explicit ``remove()`` call — the point being that they can be reused
    (via ``ApiClient.resend_cached``) any number of times, whenever needed.
    """

    SCHEMA = 1

    def __init__(self, cache_dir: str, max_cache_items: int, max_cache_age_seconds: int,
                 max_cache_retries: int, logger):
        self.cache_dir = cache_dir
        self.max_cache_items = max_cache_items
        self.max_cache_age_seconds = max_cache_age_seconds
        self.max_cache_retries = max_cache_retries
        self.logger = logger
        self._lock = threading.RLock()
        self._ids: List[str] = []
        os.makedirs(self.cache_dir, exist_ok=True)
        self._rebuild()

    # --- startup reload ---
    def _rebuild(self) -> None:
        """Scan the cache dir on startup; skip ``.tmp_*`` and any corrupt folder."""
        with self._lock:
            self._ids = []
            try:
                entries = list(os.scandir(self.cache_dir))
            except OSError as e:
                self.logger.error(f"Failed to scan cache dir {self.cache_dir}: {e}")
                return
            for entry in entries:
                if not entry.is_dir() or entry.name.startswith(".tmp_"):
                    continue
                meta_path = os.path.join(entry.path, "meta.json")
                try:
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                    if not isinstance(meta, dict) or "id" not in meta:
                        raise ValueError("missing required fields")
                    self._ids.append(entry.name)
                except Exception as e:
                    self.logger.warning(f"Skipping corrupt cache folder {entry.name}: {e}")
            self._ids.sort()
            if self._ids:
                self.logger.info(f"Reloaded {len(self._ids)} pending request(s) from {self.cache_dir}")
            self._enforce_limits()

    # --- helpers ---
    def _folder(self, fid: str) -> str:
        return os.path.join(self.cache_dir, fid)

    def _folder_epoch(self, fid: str) -> Optional[float]:
        """Derive creation time from the folder-name timestamp prefix (ids-only memory)."""
        ts = fid.split("_", 1)[0]
        try:
            return datetime.strptime(ts, "%Y%m%dT%H%M%S%f").replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            return None

    def _read_meta(self, fid: str) -> Optional[dict]:
        try:
            with open(os.path.join(self._folder(fid), "meta.json"), "r") as f:
                return json.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to read meta for {fid}: {e}")
            return None

    def _write_meta(self, folder: str, meta: dict) -> None:
        path = os.path.join(folder, "meta.json")
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(meta, f)
        os.replace(tmp, path)

    # --- add ---
    def add(self, prepared: dict, persistent: bool = False) -> Optional[str]:
        """Persist a request to a new folder. Returns the folder id (or None if not cached).

        ``persistent=True`` marks this as a saved, reusable request: it is kept
        completely intact on disk (no age/count/retry-count eviction, never
        auto-removed on success) until explicitly deleted.
        """
        with self._lock:
            created = time.time()
            fid = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}_{uuid.uuid4().hex[:8]}"
            tmp = self._folder(".tmp_" + fid)
            final = self._folder(fid)
            try:
                os.makedirs(tmp, exist_ok=True)

                files_meta: List[dict] = []
                normalized = prepared.get("files")
                if normalized:
                    files_dir = os.path.join(tmp, "files")
                    os.makedirs(files_dir, exist_ok=True)
                    for index, (field_name, (filename, content, mimetype)) in enumerate(normalized):
                        raw = _read_content(content)
                        if not raw:
                            # empty-content guard: do not cache a request carrying an empty file
                            self.logger.error(f"Empty file content for '{filename}'; not caching request")
                            shutil.rmtree(tmp, ignore_errors=True)
                            return None
                        disk_name = f"{index}__{field_name}__{_sanitize(filename)}"
                        with open(os.path.join(files_dir, disk_name), "wb") as f:
                            f.write(raw)
                        files_meta.append({
                            "index": index, "field": field_name, "filename": filename,
                            "mimetype": mimetype, "file": disk_name,
                        })

                body_kind = "none"
                if prepared.get("json") is not None:
                    with open(os.path.join(tmp, "body.json"), "w") as f:
                        json.dump(prepared["json"], f)
                    body_kind = "json"
                elif prepared.get("data") is not None:
                    data = prepared["data"]
                    if isinstance(data, dict):
                        body_kind = "form"
                    else:
                        raw = data if isinstance(data, (bytes, bytearray)) else str(data).encode("utf-8")
                        with open(os.path.join(tmp, "body.bin"), "wb") as f:
                            f.write(raw)
                        body_kind = "form_raw"

                meta = {
                    "schema": self.SCHEMA,
                    "id": fid,
                    "name": prepared.get("name"),
                    "method": prepared["method"],
                    "url": prepared["url"],
                    # auth header deliberately NOT persisted — re-rotated on each retry
                    "headers": prepared.get("headers") or {},
                    "params": prepared.get("params"),
                    "content_type": prepared.get("content_type"),
                    "body_kind": body_kind,
                    "form": prepared["data"] if body_kind == "form" else None,
                    "files": files_meta,
                    "created_at": created,
                    "retry_count": 0,
                    "last_attempt_at": None,
                    "persistent": persistent,
                }
                self._write_meta(tmp, meta)
                os.replace(tmp, final)
                self._ids.append(fid)
                self._ids.sort()
                self.logger.info(f"Cached failed request {fid} (pending: {len(self._ids)})")
                self._enforce_limits()
                return fid
            except Exception as e:
                self.logger.error(f"Failed to cache request: {e}")
                shutil.rmtree(tmp, ignore_errors=True)
                return None

    # --- load (lazy) ---
    def load(self, fid: str) -> Optional[dict]:
        """Reconstruct a prepared-request dict from disk (reads file bytes lazily)."""
        folder = self._folder(fid)
        meta = self._read_meta(fid)
        if meta is None:
            return None
        try:
            files: Optional[List[NormalizedFile]] = None
            if meta.get("files"):
                files = []
                for fm in meta["files"]:
                    with open(os.path.join(folder, "files", fm["file"]), "rb") as f:
                        raw = f.read()
                    files.append((fm["field"], (fm["filename"], raw, fm["mimetype"])))

            json_body = None
            data = None
            body_kind = meta.get("body_kind", "none")
            if body_kind == "json":
                with open(os.path.join(folder, "body.json"), "r") as f:
                    json_body = json.load(f)
            elif body_kind == "form":
                data = meta.get("form")
            elif body_kind == "form_raw":
                with open(os.path.join(folder, "body.bin"), "rb") as f:
                    data = f.read()

            return {
                "name": meta.get("name"),
                "method": meta["method"],
                "url": meta["url"],
                "headers": meta.get("headers") or {},
                "params": meta.get("params"),
                "content_type": meta.get("content_type"),
                "json": json_body,
                "data": data,
                "files": files,
                "timeout": None,
            }
        except Exception as e:
            self.logger.warning(f"Failed to load cached request {fid}: {e}")
            return None

    def retry_count(self, fid: str) -> int:
        meta = self._read_meta(fid)
        return int(meta.get("retry_count", 0)) if meta else self.max_cache_retries

    def is_persistent(self, fid: str) -> bool:
        """Saved-for-reuse entries are exempt from every automatic eviction path."""
        meta = self._read_meta(fid)
        return bool(meta.get("persistent", False)) if meta else False

    def bump_retry(self, fid: str) -> None:
        with self._lock:
            meta = self._read_meta(fid)
            if meta is None:
                return
            meta["retry_count"] = int(meta.get("retry_count", 0)) + 1
            meta["last_attempt_at"] = time.time()
            try:
                self._write_meta(self._folder(fid), meta)
            except Exception as e:
                self.logger.warning(f"Failed to persist retry_count for {fid}: {e}")

    def eligible(self) -> List[str]:
        """Non-persistent folder ids still within age and retry-count limits.

        Persistent (saved-for-reuse) entries are deliberately excluded: they
        are only ever sent via an explicit ``ApiClient.resend_cached`` call,
        never by the automatic background retry loop.
        """
        with self._lock:
            self._enforce_limits()
            out = []
            for fid in list(self._ids):
                if self.is_persistent(fid):
                    continue
                if self.retry_count(fid) < self.max_cache_retries:
                    out.append(fid)
            return out

    def list(self) -> List[dict]:
        """Lightweight metadata for every cached request (no file bytes read).

        Powers the first-class reuse API: inspect what's saved, then act on
        an id via ``ApiClient.resend_cached`` / ``remove_cached``.
        """
        with self._lock:
            out = []
            for fid in self._ids:
                meta = self._read_meta(fid)
                if meta is None:
                    continue
                out.append({
                    "id": fid,
                    "name": meta.get("name"),
                    "method": meta.get("method"),
                    "url": meta.get("url"),
                    "persistent": meta.get("persistent", False),
                    "retry_count": meta.get("retry_count", 0),
                    "created_at": meta.get("created_at"),
                })
            return out

    def remove(self, fid: str) -> None:
        with self._lock:
            self._remove_nolock(fid)

    def _remove_nolock(self, fid: str) -> None:
        shutil.rmtree(self._folder(fid), ignore_errors=True)
        if fid in self._ids:
            self._ids.remove(fid)

    def _enforce_limits(self) -> None:
        """Caller must hold the lock. Drop aged-out (any number) then excess (oldest-first).

        Persistent entries never participate in either check.
        """
        if self.max_cache_age_seconds > 0:
            now = time.time()
            for fid in [f for f in self._ids if not self.is_persistent(f)]:
                created = self._folder_epoch(fid)
                if created is not None and now - created > self.max_cache_age_seconds:
                    self.logger.debug(f"Dropping aged-out cached request {fid}")
                    self._remove_nolock(fid)
        transient = [f for f in self._ids if not self.is_persistent(f)]  # _ids sorted -> oldest first
        if self.max_cache_items > 0 and len(transient) > self.max_cache_items:
            excess = len(transient) - self.max_cache_items
            for fid in transient[:excess]:
                self.logger.debug(f"Dropping excess cached request {fid}")
                self._remove_nolock(fid)

    @property
    def pending(self) -> int:
        with self._lock:
            return len(self._ids)


class ApiClient:
    """Generic, pooled HTTP client with caching + auto-retry.

    An ease-of-use wrapper for any send/receive (GET/POST/PUT/PATCH/DELETE or
    any verb). No transformsai-specific formatting and no special heartbeat
    path — a heartbeat is just ``post(..., cache=False)``. Backed by a single
    pooled ``requests.Session`` (keep-alive); async is thread-pool based.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 30,
        success_codes: Tuple[int, ...] = (200, 201, 202, 204),
        default_content_type: str = "auto",
        auth_keys: Optional[Union[str, List[str]]] = None,
        auth_header: str = "X-Secret-Key",
        max_retries: int = 3,
        retry_backoff: float = 1.0,
        retry_backoff_max: float = 30,
        retry_on_status: Tuple[int, ...] = (408, 429, 500, 502, 503, 504),
        max_workers: Optional[int] = None,
        cache_enabled: bool = True,
        cache_dir: str = ".core-api-cache",
        cache_retry_interval: int = 100,
        max_cache_items: int = 300,
        max_cache_age_seconds: int = 86400,
        max_cache_retries: int = 5,
        endpoints: Optional[Dict[str, Any]] = None,
        pool_connections: int = 10,
        pool_maxsize: int = 10,
    ):
        self.logger = get_logger(self)

        self.base_url = base_url
        self.headers = dict(headers) if headers else {}
        self.timeout = timeout
        self.success_codes = tuple(success_codes)
        self.default_content_type = default_content_type

        if isinstance(auth_keys, str):
            self.auth_keys = [auth_keys]
        elif isinstance(auth_keys, list):
            self.auth_keys = list(auth_keys)
        else:
            self.auth_keys = []
        self.auth_header = auth_header

        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.retry_backoff_max = retry_backoff_max
        self.retry_on_status = tuple(retry_on_status)

        if max_workers is None:
            max_workers = min(2, os.cpu_count() or 2)
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # One pooled Session; we own retries, so urllib3 must not double-retry.
        self._session = requests.Session()
        adapter = HTTPAdapter(pool_connections=pool_connections, pool_maxsize=pool_maxsize, max_retries=0)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

        self._endpoints: Dict[str, EndpointProfile] = {}
        if endpoints:
            for name, profile in endpoints.items():
                self.register_endpoint(name, profile)

        self._shutdown_event = threading.Event()
        self.cache_enabled = cache_enabled
        self.cache_retry_interval = cache_retry_interval
        self._timer: Optional[threading.Timer] = None
        if cache_enabled:
            self.cache = _DirCache(cache_dir, max_cache_items, max_cache_age_seconds,
                                   max_cache_retries, self.logger)
            if cache_retry_interval and cache_retry_interval > 0:
                self._arm_timer()
        else:
            self.cache = None

        self.logger.info(
            f"ApiClient ready (workers={max_workers}, cache={'on' if self.cache else 'off'}, "
            f"pool={pool_maxsize})"
        )

    # ------------------------------------------------------------------ URL/body prep
    def _build_url(self, base: Optional[str], path: str) -> str:
        if path.startswith(("http://", "https://")):
            return path
        base = (base or "").rstrip("/")
        if not path:
            if not base:
                raise ValueError("No base_url provided and no absolute path given")
            return base
        if not base:
            raise ValueError("No base_url provided and path is not an absolute URL")
        return base + "/" + path.lstrip("/")

    @staticmethod
    def _normalize_files(files: Optional[Dict[str, Any]]) -> Optional[List[NormalizedFile]]:
        if not files:
            return None
        out: List[NormalizedFile] = []
        for field_name, value in files.items():
            items = value if isinstance(value, list) else [value]
            for ft in items:
                if not isinstance(ft, (tuple, list)) or len(ft) < 2:
                    continue
                filename = ft[0]
                content = ft[1]
                mimetype = ft[2] if len(ft) >= 3 else "application/octet-stream"
                out.append((field_name, (filename, content, mimetype)))
        return out or None

    @staticmethod
    def _files_for_request(normalized: Optional[List[NormalizedFile]]) -> Optional[List[NormalizedFile]]:
        if not normalized:
            return None
        out: List[NormalizedFile] = []
        for field_name, (filename, content, mimetype) in normalized:
            if hasattr(content, "seek"):
                try:
                    content.seek(0)
                except Exception:
                    pass
            out.append((field_name, (filename, content, mimetype)))
        return out

    def _prepare(self, method, path, json_body, data, params, files, headers,
                 base_url, content_type, name) -> dict:
        method = method.upper()
        url = self._build_url(base_url if base_url is not None else self.base_url, path)

        ct = content_type if content_type is not None else self.default_content_type
        normalized = self._normalize_files(files)

        # "json" content-type with a plain dict in `data` and no explicit json -> treat as JSON body.
        if ct == "json" and json_body is None and data is not None and not normalized:
            json_body, data = data, None

        req_headers = dict(self.headers)
        if headers:
            req_headers.update(headers)

        return {
            "name": name,
            "method": method,
            "url": url,
            "headers": req_headers,
            "params": params,
            "content_type": ct,
            "json": json_body,
            "data": data,
            "files": normalized,
            "timeout": None,  # set by request()
        }

    # ------------------------------------------------------------------ auth
    def _apply_auth(self, headers: Dict[str, str]) -> None:
        if self.auth_keys and self.auth_header not in headers:
            headers[self.auth_header] = random.choice(self.auth_keys)

    # ------------------------------------------------------------------ execution
    def _wrap(self, r: requests.Response, attempts: int, from_cache_retry: bool) -> Response:
        return Response(
            status_code=r.status_code,
            headers=dict(r.headers),
            text=r.text,
            content=r.content,
            url=r.url,
            ok=r.status_code in self.success_codes,
            elapsed=r.elapsed.total_seconds() if r.elapsed is not None else 0.0,
            attempts=attempts,
            from_cache_retry=from_cache_retry,
        )

    def _backoff(self, n: int) -> None:
        delay = min(self.retry_backoff * (2 ** n), self.retry_backoff_max)
        self._shutdown_event.wait(delay)

    _REDACT_HEADERS = ("authorization", "cookie", "set-cookie")

    def _redact(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Copy headers with auth/secret values masked (never log credentials)."""
        sensitive = {self.auth_header.lower(), *self._REDACT_HEADERS}
        return {k: ("***" if k.lower() in sensitive else v) for k, v in headers.items()}

    def _log_details(self, prepared: dict, headers: Dict[str, str],
                     resp: Optional[Response] = None, error: Optional[BaseException] = None) -> None:
        """Emit the full request/response context at DEBUG for a failed call."""
        lines = ["Request:",
                 f"  {prepared['method']} {prepared['url']}",
                 f"  params:  {prepared.get('params')}",
                 f"  headers: {self._redact(headers)}"]
        if prepared.get("json") is not None:
            lines.append(f"  json:    {prepared['json']}")
        if prepared.get("data") is not None:
            lines.append(f"  data:    {prepared['data']}")
        files = prepared.get("files")
        if files:
            lines.append(f"  files:   {[fn for _, (fn, _, _) in files]}")
        if resp is not None:
            lines.append("Response:")
            lines.append(f"  status:  {resp.status_code}")
            lines.append(f"  elapsed: {resp.elapsed:.3f}s")
            lines.append(f"  headers: {resp.headers}")
            lines.append(f"  body:    {resp.text}")
        if error is not None:
            lines.append(f"Error: {type(error).__name__}: {error}")
        self.logger.debug("\n".join(lines))

    def _execute(self, prepared: dict, from_cache_retry: bool = False) -> Response:
        """Run the retry loop for one prepared request, returning a uniform Response.

        4xx (outside ``retry_on_status``) fails fast; only ``retry_on_status`` or
        Timeout/ConnectionError are retried with exponential backoff.
        """
        timeout = prepared.get("timeout") or self.timeout
        name = prepared.get("name") or prepared["method"]
        last: Optional[Response] = None

        for n in range(self.max_retries + 1):
            attempts = n + 1
            headers = dict(prepared["headers"])
            self._apply_auth(headers)
            files = self._files_for_request(prepared.get("files"))
            try:
                r = self._session.request(
                    prepared["method"],
                    prepared["url"],
                    params=prepared.get("params"),
                    headers=headers,
                    json=prepared.get("json"),
                    data=prepared.get("data"),
                    files=files,
                    timeout=timeout,
                )
                resp = self._wrap(r, attempts, from_cache_retry)
                if resp.ok:
                    self.logger.info(
                        f"{name}: HTTP {resp.status_code} {prepared['method']} {prepared['url']} "
                        f"({resp.elapsed:.3f}s, attempt {attempts}/{self.max_retries + 1})"
                    )
                    return resp
                last = resp
                if r.status_code in self.retry_on_status and n < self.max_retries:
                    self.logger.warning(
                        f"{name}: HTTP {r.status_code} (attempt {attempts}/{self.max_retries + 1}), retrying"
                    )
                    self._log_details(prepared, headers, resp=resp)
                    self._backoff(n)
                    continue
                # exhausted retries (5xx) or fail fast (4xx)
                self.logger.error(f"{name}: HTTP {r.status_code} -> {prepared['url']} (no retry)")
                self._log_details(prepared, headers, resp=resp)
                return resp
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                last = Response(
                    status_code=0, headers={}, text=str(e), content=b"",
                    url=prepared["url"], ok=False, elapsed=0.0,
                    attempts=attempts, from_cache_retry=from_cache_retry,
                )
                if n < self.max_retries:
                    self.logger.warning(
                        f"{name}: {type(e).__name__} (attempt {attempts}/{self.max_retries + 1}), retrying"
                    )
                    self._backoff(n)
                    continue
                self.logger.error(f"{name}: {type(e).__name__} after {attempts} attempts -> {prepared['url']}")
                self._log_details(prepared, headers, error=e)
            except Exception as e:
                self.logger.error(f"{name}: unexpected error -> {e}")
                self._log_details(prepared, headers, error=e)
                return Response(
                    status_code=0, headers={}, text=str(e), content=b"",
                    url=prepared["url"], ok=False, elapsed=0.0,
                    attempts=attempts, from_cache_retry=from_cache_retry,
                )

        return last if last is not None else Response(
            status_code=0, headers={}, text="", content=b"", url=prepared["url"],
            ok=False, elapsed=0.0, attempts=self.max_retries + 1, from_cache_retry=from_cache_retry,
        )

    def _should_cache(self, method: str, cache: Optional[bool]) -> bool:
        if cache is True:
            return True
        if cache is False:
            return False
        # auto: cache writes on failure, never GET/HEAD/OPTIONS (replaces is_heartbeat)
        return method.upper() not in ("GET", "HEAD", "OPTIONS")

    def _send_and_maybe_cache(self, prepared: dict, cache: Optional[bool], persist: bool = False) -> Response:
        resp = self._execute(prepared)
        if self.cache is not None:
            if persist:
                # keep a permanent, reusable copy regardless of outcome
                self.cache.add(prepared, persistent=True)
            elif not resp.ok and self._should_cache(prepared["method"], cache):
                self.cache.add(prepared)
        return resp

    def _done_callback(self, future: Future) -> None:
        try:
            future.result()
        except Exception as e:
            self.logger.error(f"Async request raised: {e}")

    # ------------------------------------------------------------------ public API
    def request(self, method: str, path: str = "", *, json: Any = None, data: Any = None,
                params: Optional[Dict] = None, files: Optional[Dict] = None,
                headers: Optional[Dict] = None, timeout: Optional[int] = None,
                base_url: Optional[str] = None, content_type: Optional[str] = None,
                cache: Optional[bool] = None, persist: bool = False, async_: bool = False,
                name: Optional[str] = None) -> Union[Response, "Future[Response]"]:
        """Generic single-dispatch request. Any verb works.

        ``async_=False`` (default) blocks through retries and returns a ``Response``;
        ``async_=True`` submits to the executor and returns a ``Future[Response]``.

        ``persist=True`` saves a permanent, reusable copy of this request to the
        cache regardless of success/failure — exempt from age/item-count/
        retry-count eviction, never auto-removed. Reuse it later with
        ``resend_cached(fid)`` (the id is in ``list_cached()``); delete with
        ``remove_cached(fid)``.
        """
        prepared = self._prepare(method, path, json, data, params, files, headers,
                                 base_url, content_type, name)
        prepared["timeout"] = timeout or self.timeout
        if async_:
            future = self.executor.submit(self._send_and_maybe_cache, prepared, cache, persist)
            future.add_done_callback(self._done_callback)
            return future
        return self._send_and_maybe_cache(prepared, cache, persist)

    # convenience verbs
    def get(self, path: str = "", **kw):
        return self.request("GET", path, **kw)

    def post(self, path: str = "", **kw):
        return self.request("POST", path, **kw)

    def put(self, path: str = "", **kw):
        return self.request("PUT", path, **kw)

    def patch(self, path: str = "", **kw):
        return self.request("PATCH", path, **kw)

    def delete(self, path: str = "", **kw):
        return self.request("DELETE", path, **kw)

    # ------------------------------------------------------------------ endpoint profiles
    def register_endpoint(self, name: str, profile: Union[EndpointProfile, str, dict]) -> None:
        """Register a reusable endpoint. Accepts an EndpointProfile, a bare path string, or a dict."""
        if isinstance(profile, EndpointProfile):
            p = profile
        elif isinstance(profile, str):
            p = EndpointProfile(path=profile)
        elif isinstance(profile, dict):
            p = EndpointProfile(
                path=profile["path"],
                method=profile.get("method", "POST"),
                headers=profile.get("headers"),
                content_type=profile.get("content_type"),
                cache=profile.get("cache"),
                persist=profile.get("persist"),
            )
        else:
            raise TypeError(f"Unsupported endpoint profile type: {type(profile)!r}")
        self._endpoints[name] = p

    def send(self, name: str, *, json: Any = None, data: Any = None,
             params: Optional[Dict] = None, files: Optional[Dict] = None,
             headers: Optional[Dict] = None, timeout: Optional[int] = None,
             content_type: Optional[str] = None, cache: Optional[bool] = None,
             persist: Optional[bool] = None,
             async_: bool = False) -> Union[Response, "Future[Response]"]:
        """Resolve a registered endpoint profile and send. Precedence: per-call arg > profile > client default."""
        if name not in self._endpoints:
            raise KeyError(f"Unknown endpoint: {name!r}")
        p = self._endpoints[name]
        merged_headers = dict(p.headers) if p.headers else {}
        if headers:
            merged_headers.update(headers)
        return self.request(
            p.method, p.path,
            json=json, data=data, params=params, files=files,
            headers=merged_headers or None,
            timeout=timeout,
            content_type=content_type if content_type is not None else p.content_type,
            cache=cache if cache is not None else p.cache,
            persist=persist if persist is not None else bool(p.persist),
            async_=async_, name=name,
        )

    # ------------------------------------------------------------------ reusable requests (first-class)
    def save_request(self, method: str, path: str = "", *, json: Any = None, data: Any = None,
                      params: Optional[Dict] = None, files: Optional[Dict] = None,
                      headers: Optional[Dict] = None, content_type: Optional[str] = None,
                      base_url: Optional[str] = None, name: Optional[str] = None) -> Optional[str]:
        """Build and persist a request to disk for later reuse, without sending it now.

        Returns the cache id (or ``None`` if caching is disabled / the save
        failed). The saved request is kept completely intact — exempt from
        age/item-count/retry-count eviction — until you explicitly send it
        with ``resend_cached(fid)`` and/or delete it with ``remove_cached(fid)``.
        """
        if self.cache is None:
            self.logger.warning("Cache is disabled; cannot save a request for reuse")
            return None
        prepared = self._prepare(method, path, json, data, params, files, headers,
                                 base_url, content_type, name)
        prepared["timeout"] = self.timeout
        return self.cache.add(prepared, persistent=True)

    def list_cached(self) -> List[dict]:
        """List every cached request (pending failure-retries and saved-for-reuse), newest last.

        Each entry: ``id``, ``name``, ``method``, ``url``, ``persistent``,
        ``retry_count``, ``created_at``.
        """
        if self.cache is None:
            return []
        return self.cache.list()

    def resend_cached(self, fid: str) -> Optional[Response]:
        """Explicitly (re)send a cached request by id. Returns ``None`` if it doesn't exist.

        Persistent (saved-for-reuse) entries are never removed by this call,
        regardless of outcome, so they can be resent again later. Transient
        (auto-cached failure) entries follow the usual policy: removed on
        success, dropped once ``max_cache_retries`` is exceeded on failure.
        """
        if self.cache is None:
            return None
        prepared = self.cache.load(fid)
        if prepared is None:
            return None
        persistent = self.cache.is_persistent(fid)
        self.cache.bump_retry(fid)
        resp = self._execute(prepared, from_cache_retry=True)
        if persistent:
            return resp
        if resp.ok:
            self.cache.remove(fid)
        elif self.cache.retry_count(fid) >= self.cache.max_cache_retries:
            self.logger.warning(f"Max retries exceeded for {fid}, dropping")
            self.cache.remove(fid)
        return resp

    def remove_cached(self, fid: str) -> None:
        """Explicitly delete a cached request (pending-retry or saved-for-reuse)."""
        if self.cache is not None:
            self.cache.remove(fid)

    # ------------------------------------------------------------------ cache retry loop
    def _arm_timer(self) -> None:
        if self._shutdown_event.is_set() or not self.cache_retry_interval or self.cache_retry_interval <= 0:
            return
        self._timer = threading.Timer(self.cache_retry_interval, self._timer_tick)
        self._timer.daemon = True
        self._timer.start()

    def _timer_tick(self) -> None:
        if self._shutdown_event.is_set():
            return
        try:
            self._retry_pending()
        except Exception as e:
            self.logger.error(f"Cache retry error: {e}")
        if not self._shutdown_event.is_set():
            self._arm_timer()

    def _retry_pending(self) -> None:
        """Retry every eligible cached request once; remove on success or when max retries hit."""
        if self.cache is None:
            return
        eligible = self.cache.eligible()
        if not eligible:
            return
        self.logger.info(f"Retrying {len(eligible)} cached request(s)")
        for fid in eligible:
            if self._shutdown_event.is_set():
                return
            prepared = self.cache.load(fid)
            if prepared is None:
                self.cache.remove(fid)
                continue
            self.cache.bump_retry(fid)
            resp = self._execute(prepared, from_cache_retry=True)
            if resp.ok:
                self.logger.info(f"Cache retry succeeded: {fid}")
                self.cache.remove(fid)
            elif self.cache.retry_count(fid) >= self.cache.max_cache_retries:
                self.logger.warning(f"Max retries exceeded for {fid}, dropping")
                self.cache.remove(fid)

    # ------------------------------------------------------------------ from_config
    @classmethod
    def from_config(cls, cfg, **overrides):
        from .config_loader import init_kwargs
        if hasattr(cfg, "model_dump"):
            cfg = cfg.model_dump()
        else:
            cfg = dict(cfg)
        endpoints = cfg.pop("endpoints", {}) or {}
        kwargs: dict[str, Any] = {
            k: v for k, v in cfg.items()
            if k != "enabled"
        }
        kwargs = {**cfg.get("extras", {}), **kwargs, **overrides}
        client = cls(**init_kwargs(cls, kwargs))
        for name, profile in endpoints.items():
            client.register_endpoint(name, profile)
        return client

    # ------------------------------------------------------------------ lifecycle
    def shutdown(self, wait: bool = True) -> None:
        self.logger.info("Shutting down ApiClient")
        self._shutdown_event.set()
        if self._timer is not None:
            self._timer.cancel()
        self.executor.shutdown(wait=wait)
        try:
            self._session.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown(wait=True)
