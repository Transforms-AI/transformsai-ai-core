"""
Typed config models for the sections core reads: node, cameras, mediamtx, api,
models, logging. Everything else is the project's.

Rules this module follows:
- Defaults live here and only here. Components map these models onto their
  constructors; they never re-state a default.
- Validate only what core reads. Every core object carries a free ``settings: {}``
  that core never inspects, and the root has a free top-level ``settings``.
- Unknown keys are kept (``extra="allow"``) so nothing is lost, and the loader
  reports them (an error in strict mode, a warning in lenient mode).
- Derived values (source URL, stream name/URLs, model path) are computed
  properties; nothing is written back into the config.
"""

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal
from urllib.parse import quote, urlsplit

from annotated_types import Ge, Gt, Le
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)

from . import urls as _urls
from .types import (
    Bitrate,
    Duration,
    DurationOrOff,
    Lowercase,
    NonEmptyStr,
    NonNegativeInt,
    Port,
    PositiveFloat,
    PositiveInt,
    Settings,
    Uppercase,
    is_secret,
    secret_field,
    settings_field,
)

MASK = "***"
_URL_PASSWORD_RE = re.compile(r"(://[^/\s:@]*):([^/\s@]+)@")


def mask_url(url: str) -> str:
    """``scheme://user:password@host`` -> ``scheme://user:***@host``."""
    return _URL_PASSWORD_RE.sub(rf"\1:{MASK}@", url)


def _blank_to_none(value: Any) -> Any:
    return None if isinstance(value, str) and not value.strip() else value


class CoreModel(BaseModel):
    """Base for every core section: lax conversion, defaults validated, unknown keys kept."""

    model_config = ConfigDict(extra="allow", validate_default=True, coerce_numbers_to_str=True)

    @model_validator(mode="before")
    @classmethod
    def _null_is_empty(cls, data: Any) -> Any:
        # `capture:` with no value in YAML means "all defaults", not an error
        return {} if data is None else data


# ==============================================================================
# Node
# ==============================================================================
class NodeConfig(CoreModel):
    """The node: one identity used for stream names, API identity and fleet tooling."""

    sn: NonEmptyStr = Field(description="Node serial number: the one node identity", examples=["node-0001"])
    version: str | None = Field(None, description="Optional node/project version")
    settings: Settings = settings_field("the node")


# ==============================================================================
# Camera sources (discriminated on `type`)
#
# To add a source type: subclass SourceBase with a `type: Literal[...]` tag,
# implement `target`, and add it to SOURCE_TYPES below.
# ==============================================================================
class SourceBase(CoreModel):
    """What a camera reads from."""

    kind: ClassVar[str] = "network"  # VideoCaptureAsync source class: device | file | network
    supports_audio: ClassVar[bool] = False  # can feed `audio.input.type: camera`

    @property
    def target(self) -> str | int:
        """What VideoCaptureAsync opens (contains credentials for network sources)."""
        raise NotImplementedError

    @property
    def display(self) -> str:
        """Log-safe description of the source (credentials masked)."""
        return str(self.target)


class RtspSource(SourceBase):
    """An RTSP camera, addressed by parts. The URL is derived: `source.url`."""

    kind: ClassVar[str] = "network"
    supports_audio: ClassVar[bool] = True

    type: Literal["rtsp"] = Field(description="Source type")
    host: NonEmptyStr = Field(description="Camera host name or IP", examples=["192.168.1.64"])
    port: Port = Field(554, description="RTSP port")
    user: str | None = Field(None, description="RTSP user name")
    password: str | None = secret_field(None, description="RTSP password (secret: URL-encoded in the URL, masked in dumps and logs)")
    path: str = Field("/Streaming/Channels/101", description="Stream path on the camera")
    transport: Annotated[Literal["tcp", "udp"] | None, Lowercase] = Field(
        "tcp", description="RTSP transport: tcp, udp, or null to let FFmpeg negotiate (also used for camera audio)")

    @field_validator("user", "password", mode="before")
    @classmethod
    def _blank(cls, value: Any) -> Any:
        return _blank_to_none(value)

    @field_validator("path")
    @classmethod
    def _leading_slash(cls, value: str) -> str:
        return value if not value or value.startswith("/") else "/" + value

    def _build(self, password: str | None, encode: bool) -> str:
        auth = ""
        if self.user:
            auth = quote(self.user, safe="")
            if password:
                auth += ":" + (quote(password, safe="") if encode else password)
            auth += "@"
        return f"rtsp://{auth}{_urls.url_host(self.host)}:{self.port}{self.path}"

    @property
    def url(self) -> str:
        """Full RTSP URL; user and password are URL-encoded."""
        return self._build(self.password, encode=True)

    @property
    def target(self) -> str:
        return self.url

    @property
    def display(self) -> str:
        return self._build(MASK if self.password else None, encode=False)


class DeviceSource(SourceBase):
    """A local capture device (V4L2 node, udev symlink, or OpenCV index)."""

    kind: ClassVar[str] = "device"

    type: Literal["device"] = Field(description="Source type")
    path: str | None = Field(None, description="Device node, e.g. /dev/video0 or a udev symlink (or use `index`)",
                             examples=["/dev/video0"])
    index: NonNegativeInt | None = Field(None, description="OpenCV device index (instead of `path`)")

    @field_validator("path", mode="before")
    @classmethod
    def _blank(cls, value: Any) -> Any:
        return _blank_to_none(value)

    @model_validator(mode="after")
    def _path_or_index(self) -> "DeviceSource":
        if self.path is not None and self.path.strip().isdigit() and self.index is None:
            self.index, self.path = int(self.path.strip()), None
        if (self.path is None) == (self.index is None):
            raise ValueError("a device source needs exactly one of 'path' or 'index'")
        return self

    @property
    def target(self) -> str | int:
        return self.index if self.index is not None else self.path


class FileSource(SourceBase):
    """A local video file, read at its own frame rate."""

    kind: ClassVar[str] = "file"

    type: Literal["file"] = Field(description="Source type")
    path: NonEmptyStr = Field(description="Video file path", examples=["videos/sample.mp4"])
    loop: bool = Field(True, description="Restart from the beginning at end of file")

    @property
    def target(self) -> str:
        return self.path


class UrlSource(SourceBase):
    """Any URL OpenCV/FFmpeg can open (rtsp://, http://, udp://, ...), given verbatim."""

    kind: ClassVar[str] = "network"
    supports_audio: ClassVar[bool] = True

    type: Literal["url"] = Field(description="Source type")
    url: NonEmptyStr = Field(description="Full source URL; a password in it is treated as a secret",
                             examples=["rtsp://10.0.0.11/stream1"])

    @property
    def password(self) -> str | None:
        try:
            return urlsplit(self.url).password
        except ValueError:
            return None

    @property
    def target(self) -> str:
        return self.url

    @property
    def display(self) -> str:
        return mask_url(self.url)


SOURCE_TYPES: dict[str, type[SourceBase]] = {
    "rtsp": RtspSource,
    "device": DeviceSource,
    "file": FileSource,
    "url": UrlSource,
}
Source = Annotated[RtspSource | DeviceSource | FileSource | UrlSource, Field(discriminator="type")]


def _source_shorthand(value: Any) -> Any:
    """`source: "rtsp://..."` -> a url source, `source: 0` -> device index 0; lower-cases `type`."""
    if isinstance(value, str) and "://" in value:
        return {"type": "url", "url": value}
    if isinstance(value, int) and not isinstance(value, bool):
        return {"type": "device", "index": value}
    if isinstance(value, dict) and isinstance(value.get("type"), str):
        return {**value, "type": value["type"].strip().lower()}
    return value


# ==============================================================================
# Capture (mirrors VideoCaptureAsync)
# ==============================================================================
class RestartConfig(CoreModel):
    """Supervised reconnect policy. Failed opens, read-failure streaks, stalls and exceptions all use it."""

    enabled: bool = Field(True, description="Reconnect automatically after a failure")
    delay: Duration = Field(30.0, description="Backoff ceiling in seconds: delays double from backoff_start up to this")
    backoff_start: DurationOrOff = Field(1.0, description="First reconnect delay in seconds")
    jitter: Annotated[float, Ge(0)] = Field(0.2, description="+/- fraction applied to each delay so cameras don't reconnect in lockstep (capped at 1)")
    reset_after: DurationOrOff = Field(30.0, description="Seconds of healthy streaming before the backoff starts over")
    max_attempts: PositiveInt | None = Field(None, description="Consecutive reconnect attempts before giving up (null = forever)")
    stall_timeout: DurationOrOff | None = Field(None, description="Seconds without a decoded frame before forcing a reconnect (null = auto max(5, 20/fps); 0 = off)")


class CaptureConfig(CoreModel):
    """How VideoCaptureAsync reads this camera. Everything is optional."""

    width: PositiveInt | None = Field(None, description="Capture width in pixels (null = native; needs height too)")
    height: PositiveInt | None = Field(None, description="Capture height in pixels (null = native; needs width too)")
    fps: PositiveFloat | None = Field(None, description="Target frames per second (null = native)")
    buffer_size: PositiveInt = Field(1, description="OpenCV buffer size for network/file sources (1 = minimal lag)")
    max_frame_age: Duration | None = Field(None, description="read() drops frames older than this, e.g. '500ms' (null = never)")
    auto_resize: bool = Field(True, description="Resize in software when the source ignores width/height")
    hw_decode: bool = Field(False, description="Ask OpenCV for hardware-accelerated decoding")
    open_timeout: DurationOrOff = Field(5.0, description="Network open/read timeout in seconds, passed to FFmpeg (0 = FFmpeg default)")
    ffmpeg_options: str | None = Field(None, description="Verbatim OPENCV_FFMPEG_CAPTURE_OPTIONS override")
    health_log_interval: DurationOrOff = Field(60.0, description="Seconds between capture health log lines (0 = off)")
    restart: RestartConfig = Field(default_factory=RestartConfig, description="Reconnect policy")


# ==============================================================================
# Stream (one camera's livestream on the shared MediaMTX server)
# ==============================================================================
class EncoderConfig(CoreModel):
    """FFmpeg video encoder."""

    preset: NonEmptyStr = Field("ultrafast", description="x264-style preset (mapped for hardware encoders)")
    codec: NonEmptyStr = Field("libx264", description="libx264, 'auto' (first available hardware encoder, else libx264), or an FFmpeg H.264 encoder name")
    queue_size: NonNegativeInt = Field(2, description="Async frame queue depth (0 = write synchronously)")


class AlsaAudioInput(CoreModel):
    """A local microphone through ALSA."""

    # Filter values used when `filter.<key>` is not set (4.x behaviour for a MEMS mic)
    filter_defaults: ClassVar[dict[str, Any]] = {"highpass_hz": 20.0, "gain_db": 18.0}

    type: Literal["alsa"] = Field(description="Audio input type")
    device: NonEmptyStr = Field(description="ALSA PCM; use a dsnoop/plug PCM if another process records the same mic",
                                examples=["plughw:1,0"])
    rate: PositiveInt = Field(16000, description="Capture sample rate in Hz")
    channels: PositiveInt = Field(2, description="Channels to capture")
    channel: NonNegativeInt = Field(0, description="Channel kept for the mono stream (0 = left)")

    @field_validator("channel")
    @classmethod
    def _channel_exists(cls, value: int, info) -> int:
        channels = info.data.get("channels")
        if channels is not None and value >= channels:
            raise ValueError(f"must be less than channels ({channels})")
        return value


class CameraAudioInput(CoreModel):
    """The camera's own audio track, read from this camera's source URL (rtsp or url sources)."""

    # Camera audio arrives mono with the camera's own AGC
    filter_defaults: ClassVar[dict[str, Any]] = {"highpass_hz": None, "gain_db": 0.0}

    type: Literal["camera"] = Field(description="Audio input type")


AUDIO_INPUT_TYPES: dict[str, type[CoreModel]] = {"alsa": AlsaAudioInput, "camera": CameraAudioInput}
AudioInput = Annotated[AlsaAudioInput | CameraAudioInput, Field(discriminator="type")]

# Generated example YAML: a key carrying this marker is shown commented out unless the
# example seeds a value, because its "unset" default differs from its null value.
UNSET_MEANS_DEFAULT = {"unset_means_default": True}


class AudioFilter(CoreModel):
    """Audio filter chain, applied in a fixed order: channel select (alsa) -> high-pass -> gain -> limiter -> resample."""

    highpass_hz: Annotated[float, Ge(0)] | None = Field(
        None, json_schema_extra=UNSET_MEANS_DEFAULT, examples=[300],
        description="High-pass cutoff in Hz. Unset: 20 for alsa, off for camera. 0/null = off. Above 20 a 4th-order filter is used")
    gain_db: float | None = Field(
        None, json_schema_extra=UNSET_MEANS_DEFAULT, examples=[18],
        description="Makeup gain in dB (through the limiter). Unset: 18 for alsa, 0 for camera. 0/null = no gain and no limiter")
    limiter: Annotated[float, Gt(0), Le(1)] | None = Field(0.97, description="Limiter ceiling (0-1] applied after gain; null = no limiter")


@dataclass(frozen=True)
class ResolvedAudioFilter:
    """The filter values actually used: per-input defaults applied, 0 normalised to off."""

    highpass_hz: float | None
    gain_db: float
    limiter: float | None


class AudioRetryConfig(CoreModel):
    """Audio is best-effort: a failed audio input drops to video-only and audio is retried on this backoff."""

    backoff: Duration = Field(60.0, description="First audio retry delay in seconds; doubles while failing")
    backoff_max: Duration = Field(900.0, description="Audio retry delay ceiling in seconds")
    reset_after: DurationOrOff = Field(60.0, description="Seconds a run must keep audio before the backoff starts over")


class AudioConfig(CoreModel):
    """Optional live audio track on this camera's stream (Opus)."""

    input: AudioInput = Field(description="Where the audio comes from: {type: alsa, device: ...} or {type: camera}")
    filter: AudioFilter = Field(default_factory=AudioFilter, description="Filter chain")
    bitrate: Bitrate = Field("24k", description="Opus bitrate")
    retry: AudioRetryConfig = Field(default_factory=AudioRetryConfig, description="Retry policy after an audio failure")
    settings: Settings = settings_field("the audio track")

    @field_validator("input", mode="before")
    @classmethod
    def _input_type(cls, value: Any) -> Any:
        if isinstance(value, str):
            return {"type": value.strip().lower()}
        if isinstance(value, dict) and isinstance(value.get("type"), str):
            return {**value, "type": value["type"].strip().lower()}
        return value

    def resolved_filter(self) -> ResolvedAudioFilter:
        """Filter values with this input's defaults applied to every key the config left unset."""
        defaults, chosen = self.input.filter_defaults, self.filter.model_fields_set
        highpass = self.filter.highpass_hz if "highpass_hz" in chosen else defaults["highpass_hz"]
        gain = self.filter.gain_db if "gain_db" in chosen else defaults["gain_db"]
        return ResolvedAudioFilter(highpass_hz=highpass or None, gain_db=float(gain or 0.0),
                                   limiter=self.filter.limiter)


class StreamConfig(CoreModel):
    """This camera's livestream. Size and fps default to the camera's capture values."""

    enabled: bool = Field(True, description="Publish this camera's stream")
    name: NonEmptyStr = Field("{sn}_{camera}", description="Stream name after the fixed 'cam_sn_' prefix; placeholders {sn} (node.sn), {camera} (camera name), {index} (camera position)")
    width: PositiveInt | None = Field(None, description="Stream width (null = capture.width, else the first frame's)")
    height: PositiveInt | None = Field(None, description="Stream height (null = capture.height, else the first frame's)")
    fps: PositiveFloat | None = Field(None, description="Stream fps (null = capture.fps, else 15)")
    bitrate: Bitrate = Field("1500k", description="Video bitrate, e.g. '800k', '1.2M' or bits/s")
    on_demand: bool = Field(False, description="Publish only while a viewer is watching (polls mediamtx demand_url)")
    debug_log_interval: Duration = Field(60.0, description="Seconds between performance debug log lines")
    encoder: EncoderConfig = Field(default_factory=EncoderConfig, description="Video encoder")
    audio: AudioConfig | None = Field(None, description="Optional audio track (null = video only)")
    settings: Settings = settings_field("the stream")

    # Fallback when neither the stream nor the camera's capture sets an fps
    DEFAULT_FPS: ClassVar[float] = 15.0

    @field_validator("name")
    @classmethod
    def _name_placeholders(cls, value: str) -> str:
        return _urls.check_template(value, _urls.STREAM_NAME_FIELDS)


class CameraConfig(CoreModel):
    """One camera: where it reads from, how it captures, and its optional stream."""

    source: Source = Field(description="Where the camera reads from; `type` is rtsp, device, file or url")
    capture: CaptureConfig = Field(default_factory=CaptureConfig, description="Capture options (VideoCaptureAsync)")
    stream: StreamConfig | None = Field(None, description="This camera's livestream (null = none)")
    settings: Settings = settings_field("the camera")

    _name: str = PrivateAttr("")
    _index: int = PrivateAttr(0)

    @field_validator("source", mode="before")
    @classmethod
    def _source(cls, value: Any) -> Any:
        return _source_shorthand(value)

    @field_validator("stream", mode="before")
    @classmethod
    def _stream_shorthand(cls, value: Any) -> Any:
        # `stream: true` = all defaults; `stream: false` = configured but disabled
        if value is True:
            return {}
        if value is False:
            return {"enabled": False}
        return value

    @property
    def name(self) -> str:
        """The camera's key in `cameras` (camera_<i> for the list form)."""
        return self._name

    @property
    def index(self) -> int:
        """The camera's position in `cameras`."""
        return self._index

    def stream_name(self, sn: str) -> str | None:
        """Full stream name ('cam_sn_' + rendered stream.name) for node serial `sn`, or None without a stream block."""
        if self.stream is None:
            return None
        return _urls.render_stream_name(self.stream.name, sn=sn, camera=self.name, index=self.index)

    def stream_size(self) -> tuple[int | None, int | None]:
        """(width, height) for the stream: stream override, else capture, else None (adopted from the first frame)."""
        stream = self.stream or StreamConfig()
        if stream.width and stream.height:
            return stream.width, stream.height
        if self.capture.width and self.capture.height:
            return self.capture.width, self.capture.height
        return None, None

    def stream_fps(self) -> float:
        stream = self.stream or StreamConfig()
        return stream.fps or self.capture.fps or StreamConfig.DEFAULT_FPS


def normalize_cameras(value: Any) -> Any:
    """A list of cameras becomes an ordered map named camera_0, camera_1, ...; null becomes {}."""
    if value is None:
        return {}
    if isinstance(value, list):
        return {f"camera_{i}": camera for i, camera in enumerate(value)}
    return value


# ==============================================================================
# MediaMTX (the stream server shared by every camera's stream)
# ==============================================================================
class DemandConfig(CoreModel):
    """On-demand polling, used by streams with `on_demand: true`."""

    poll_interval: Duration = Field(3.0, description="Seconds between demand polls")
    grace_period: DurationOrOff = Field(10.0, description="Seconds demand must stay off before FFmpeg is stopped")
    timeout: Duration = Field(5.0, description="HTTP timeout per demand poll in seconds")


def _url_template(value: Any) -> Any:
    value = _blank_to_none(value)
    return _urls.check_template(value, _urls.URL_FIELDS) if isinstance(value, str) else value


class MediaMTXConfig(CoreModel):
    """The MediaMTX server every enabled stream publishes to."""

    host: NonEmptyStr | None = Field(None, description="Server host name or IP (required when any stream is enabled)",
                                     examples=["mediamtx.example.com"])
    rtsp_port: Port = Field(8554, description="RTSP port streams are published to")
    webrtc_port: Port = Field(8889, description="WebRTC port (used in the default webrtc_url for a bare-IP host)")
    demand: DemandConfig = Field(default_factory=DemandConfig, description="On-demand polling")
    rtsp_url: str | None = Field(None, description="Publish URL template (null = rtsp://{host}:{rtsp_port}/live/{name})")
    webrtc_url: str | None = Field(None, description="Viewer URL template (null = https://{host}/live/{name}/, or http://{host}:{webrtc_port}/live/{name}/ for a bare IP)")
    demand_url: str | None = Field(None, description="Demand flag URL template (null = https://{host}/demand/{name})")
    settings: Settings = settings_field("the stream server")

    @field_validator("host", mode="before")
    @classmethod
    def _blank(cls, value: Any) -> Any:
        return _blank_to_none(value)

    @field_validator("rtsp_url", "webrtc_url", "demand_url", mode="before")
    @classmethod
    def _templates(cls, value: Any) -> Any:
        return _url_template(value)

    def urls(self, name: str) -> _urls.StreamUrls:
        """The rtsp/webrtc/demand URLs of stream `name`."""
        if not self.host:
            raise ValueError("mediamtx.host is not set")
        return _urls.build_stream_urls(
            self.host, name, rtsp_port=self.rtsp_port, webrtc_port=self.webrtc_port,
            rtsp_url=self.rtsp_url, webrtc_url=self.webrtc_url, demand_url=self.demand_url,
        )


# ==============================================================================
# API client
# ==============================================================================
class RetriesConfig(CoreModel):
    """Per-request retries (inside one call)."""

    max: NonNegativeInt = Field(3, description="Retries after the first attempt (0 = single attempt)")
    backoff: DurationOrOff = Field(1.0, description="First retry delay in seconds; doubles each retry")
    backoff_max: DurationOrOff = Field(30.0, description="Retry delay ceiling in seconds")
    on_status: list[int] = Field(default_factory=lambda: [408, 429, 500, 502, 503, 504], description="HTTP statuses that are retried")


class OutboxConfig(CoreModel):
    """On-disk outbox: failed writes are kept and retried in the background."""

    enabled: bool = Field(True, description="Keep failed writes on disk and retry them")
    dir: str = Field(".core-api-cache", description="Outbox directory")
    retry_interval: DurationOrOff = Field(100.0, description="Seconds between background retry passes (0 = no background retries)")
    max_items: NonNegativeInt | None = Field(300, description="Most requests kept, oldest dropped first (0/null = unbounded)")
    max_age: DurationOrOff | None = Field(86400.0, description="Drop requests older than this, e.g. '24h' (0/null = never)")
    max_retries: NonNegativeInt | None = Field(5, description="Background retries before a request is dropped (null = unbounded; 0 = never retried)")


class PoolConfig(CoreModel):
    """HTTP connection pool and worker threads."""

    connections: PositiveInt = Field(10, description="Connection pools to cache")
    maxsize: PositiveInt = Field(10, description="Connections per pool")
    workers: PositiveInt | None = Field(None, description="Threads for async requests (null = min(2, cpu_count))")


class EndpointConfig(CoreModel):
    """A named request shape. A plain string is shorthand for {path: <string>}."""

    path: NonEmptyStr = Field(description="Path relative to base_url, or an absolute URL")
    method: Annotated[str, Uppercase] = Field("POST", description="HTTP method")
    headers: dict[str, str] | None = Field(None, description="Extra headers for this endpoint")
    content_type: Annotated[Literal["auto", "json", "form"] | None, Lowercase] = Field(None, description="Body encoding (null = client default)")
    cache: bool | None = Field(None, description="Keep failed writes in the outbox (null = client default)")
    persist: bool | None = Field(None, description="Outbox entries never expire by age or retry count")


class ApiConfig(CoreModel):
    """The HTTP API client (ApiClient)."""

    enabled: bool = Field(True, description="Use the API client (the project checks this flag)")
    base_url: str | None = Field(None, description="API base URL (required when enabled)", examples=["https://api.example.com/api/v1/"])
    timeout: Duration = Field(30.0, description="Request timeout in seconds")
    headers: dict[str, str] = Field(default_factory=dict, description="Headers sent with every request")
    auth_keys: list[str] = secret_field(default_factory=list, description="Auth keys, rotated per request (secret)")
    auth_header: NonEmptyStr = Field("X-Secret-Key", description="Header that carries the auth key")
    content_type: Annotated[Literal["auto", "json", "form"], Lowercase] = Field("auto", description="Default body encoding")
    success_codes: list[int] = Field(default_factory=lambda: [200, 201, 202, 204], description="HTTP statuses counted as success")
    retries: RetriesConfig = Field(default_factory=RetriesConfig, description="Per-request retries")
    outbox: OutboxConfig = Field(default_factory=OutboxConfig, description="Disk outbox for failed writes")
    pool: PoolConfig = Field(default_factory=PoolConfig, description="Connection pool and workers")
    endpoints: dict[str, EndpointConfig] = Field(default_factory=dict, description="Named endpoints: name -> path, or a full profile")
    settings: Settings = settings_field("the API client")

    @field_validator("base_url", mode="before")
    @classmethod
    def _blank(cls, value: Any) -> Any:
        return _blank_to_none(value)

    @field_validator("auth_keys", mode="before")
    @classmethod
    def _one_key(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, str):
            return [value] if value.strip() else []
        return value

    @field_validator("headers", mode="before")
    @classmethod
    def _no_headers(cls, value: Any) -> Any:
        return {} if value is None else value

    @field_validator("endpoints", mode="before")
    @classmethod
    def _endpoint_shorthand(cls, value: Any) -> Any:
        if value is None:
            return {}
        if isinstance(value, dict):
            return {name: {"path": p} if isinstance(p, str) else p for name, p in value.items()}
        return value


# ==============================================================================
# Models
# ==============================================================================
class ModelConfig(CoreModel):
    """An inference model. Needs at least one of path, download_key or type."""

    path: str | None = Field(None, description="Model file or directory, relative to the config's directory", examples=["models/yolo11n.pt"])
    download_key: str | None = Field(None, description="Download the model by key")
    type: str | None = Field(None, description="Model category; downloads the latest model of this type")
    task: NonEmptyStr = Field("detect", description="Ultralytics task: detect, segment, classify, pose, obb")
    batch: PositiveInt = Field(1, description="Inference batch size")
    export: bool = Field(False, description="Export to an optimised format on first load")
    export_options: dict[str, Any] = Field(default_factory=dict, description="Passed to ultralytics export() as-is (e.g. format: onnx)")
    set_classes: list[str] | None = Field(None, description="YOLOE only: text prompts used as the class list")
    settings: Settings = settings_field("the model")

    _name: str = PrivateAttr("")
    _base_dir: Path | None = PrivateAttr(None)
    _downloaded_path: Path | None = PrivateAttr(None)

    @field_validator("path", "download_key", "type", mode="before")
    @classmethod
    def _blank(cls, value: Any) -> Any:
        return _blank_to_none(value)

    @field_validator("export_options", mode="before")
    @classmethod
    def _no_options(cls, value: Any) -> Any:
        return {} if value is None else value

    @model_validator(mode="after")
    def _has_a_source(self) -> "ModelConfig":
        if not (self.path or self.download_key or self.type):
            raise ValueError("a model needs one of 'path', 'download_key' or 'type'")
        return self

    @property
    def name(self) -> str:
        return self._name

    @property
    def model_dir(self) -> Path:
        """Where a downloaded model lives: <base_dir>/models/<name>/."""
        return (self._base_dir or Path.cwd()) / "models" / self.name

    @property
    def resolved_path(self) -> Path:
        """The file/directory to load: explicit `path` first, then the downloaded model."""
        if self.path:
            path = Path(self.path).expanduser()
            return path if path.is_absolute() else (self._base_dir or Path.cwd()) / path
        if self._downloaded_path is not None:
            return self._downloaded_path
        expected = self.model_dir / self.name
        if expected.exists():
            return expected
        if self.model_dir.is_dir():
            entries = sorted(e for e in self.model_dir.iterdir() if not e.name.startswith("."))
            if len(entries) == 1:
                return entries[0]
        return expected


# ==============================================================================
# Logging
# ==============================================================================
LOG_LEVELS = ("TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL", "OFF")
_LEVEL_ALIASES = {"WARN": "WARNING", "FATAL": "CRITICAL"}


def _log_level(value: Any) -> Any:
    if isinstance(value, str):
        value = value.strip().upper()
        return _LEVEL_ALIASES.get(value, value)
    return value


LogLevel = Literal["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL", "OFF"]


class LoggingConfig(CoreModel):
    """Core's console and JSONL file sinks."""

    console_level: LogLevel | None = Field("INFO", description="Console sink level (OFF/null = no console sink)")
    file_level: LogLevel | None = Field("DEBUG", description="JSONL file sink level, .core-logs/ (OFF/null = no file sink)")
    settings: Settings = settings_field("logging")

    @field_validator("console_level", "file_level", mode="before")
    @classmethod
    def _level(cls, value: Any) -> Any:
        return _log_level(value)


# ==============================================================================
# Root
# ==============================================================================
def _model_children(model: BaseModel) -> Iterator[tuple[tuple, BaseModel]]:
    """Every nested model under `model` with its key path (dict keys / list indices included)."""
    for name in type(model).model_fields:
        value = getattr(model, name)
        if isinstance(value, BaseModel):
            yield (name,), value
            for sub, child in _model_children(value):
                yield (name, *sub), child
        elif isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, BaseModel):
                    yield (name, key), item
                    for sub, child in _model_children(item):
                        yield (name, key, *sub), child


class Config(CoreModel):
    """A validated config. Build it with `load_config()`; never written back into."""

    node: NodeConfig = Field(description="The node's identity")
    cameras: dict[str, CameraConfig] = Field(default_factory=dict, description="Cameras by name (a plain list is named camera_0, camera_1, ...)")
    mediamtx: MediaMTXConfig | None = Field(None, description="Stream server shared by every camera's stream (null = no streaming)")
    api: ApiConfig | None = Field(None, description="HTTP API client (null = not configured)")
    models: dict[str, ModelConfig] = Field(default_factory=dict, description="Inference models by name")
    logging: LoggingConfig = Field(default_factory=LoggingConfig, description="Core logging sinks")
    settings: Settings = settings_field("the project")

    _base_dir: Path = PrivateAttr(default_factory=Path.cwd)

    @field_validator("cameras", mode="before")
    @classmethod
    def _cameras(cls, value: Any) -> Any:
        return normalize_cameras(value)

    @field_validator("models", mode="before")
    @classmethod
    def _models(cls, value: Any) -> Any:
        return {} if value is None else value

    def model_post_init(self, context: Any) -> None:
        for index, (name, camera) in enumerate(self.cameras.items()):
            camera._name, camera._index = str(name), index
        for name, model in self.models.items():
            model._name = str(name)
            model._base_dir = self._base_dir

    # --- paths -------------------------------------------------------------
    @property
    def base_dir(self) -> Path:
        """Directory relative model paths resolve against (the config file's directory)."""
        return self._base_dir

    def set_base_dir(self, path: str | Path) -> None:
        self._base_dir = Path(path)
        for model in self.models.values():
            model._base_dir = self._base_dir

    # --- cameras / streams -------------------------------------------------
    def camera(self, camera: "str | CameraConfig") -> CameraConfig:
        return camera if isinstance(camera, CameraConfig) else self.cameras[camera]

    def stream_name(self, camera: "str | CameraConfig") -> str | None:
        """Resolved stream name of a camera, or None when it has no stream block."""
        return self.camera(camera).stream_name(self.node.sn)

    def stream_urls(self, camera: "str | CameraConfig") -> _urls.StreamUrls | None:
        """rtsp/webrtc/demand URLs of a camera's stream, or None when it has no stream block."""
        name = self.stream_name(camera)
        if name is None:
            return None
        if self.mediamtx is None:
            raise ValueError("no mediamtx section configured")
        return self.mediamtx.urls(name)

    def enabled_streams(self) -> list[CameraConfig]:
        return [c for c in self.cameras.values() if c.stream is not None and c.stream.enabled]

    # --- secrets -----------------------------------------------------------
    def _secret_paths(self) -> Iterator[tuple[tuple, Any]]:
        """(key path, value) of every secret core knows about (lists yield the whole list)."""
        for path, model in [((), self), *_model_children(self)]:
            for name, info in type(model).model_fields.items():
                if is_secret(info):
                    value = getattr(model, name)
                    if value:
                        yield (*path, name), value

    def secrets(self) -> list[str]:
        """Every secret value: source passwords, API auth keys, URL passwords, and registered project paths."""
        from .registry import iter_matches, secret_patterns

        found: list[str] = []

        def add(value: Any) -> None:
            for item in value if isinstance(value, list) else [value]:
                if item not in (None, "") and str(item) not in found:
                    found.append(str(item))

        for _, value in self._secret_paths():
            add(value)
        for camera in self.cameras.values():
            if isinstance(camera.source, UrlSource) and camera.source.password:
                add(camera.source.password)
        dump = self.model_dump(mode="json")
        for pattern in secret_patterns():
            for _, value in iter_matches(dump, pattern):
                if isinstance(value, (str, int, float, list)):
                    add(value)
        return found

    def masked_dump(self, fmt: Literal["dict", "yaml"] = "dict") -> dict[str, Any] | str:
        """The full config (defaults included) with every secret replaced by '***'."""
        from .registry import iter_matches, secret_patterns, set_path

        data = self.model_dump(mode="json")
        for path, value in self._secret_paths():
            set_path(data, path, [MASK] * len(value) if isinstance(value, list) else MASK)
        for name, camera in self.cameras.items():
            if isinstance(camera.source, UrlSource):
                set_path(data, ("cameras", name, "source", "url"), camera.source.display)
        for pattern in secret_patterns():
            for path, value in list(iter_matches(data, pattern)):
                if value not in (None, ""):
                    set_path(data, path, MASK)
        if fmt == "yaml":
            import yaml
            return yaml.safe_dump(data, sort_keys=False, allow_unicode=True, default_flow_style=False)
        return data

    # --- schema / docs -----------------------------------------------------
    @classmethod
    def schema_json(cls, *, indent: int = 2, strict: bool = True) -> str:
        """JSON Schema of the config. `strict` marks unknown keys invalid outside `settings` (for editors)."""
        import json

        schema = cls.model_json_schema()
        if strict:
            def close(node: Any) -> None:
                if isinstance(node, dict):
                    if "properties" in node and node.get("additionalProperties") is True:
                        node["additionalProperties"] = False
                    for value in node.values():
                        close(value)
                elif isinstance(node, list):
                    for value in node:
                        close(value)
            close(schema)
        return json.dumps(schema, indent=indent)

    @classmethod
    def example_yaml(cls) -> str:
        """A commented example config generated from the field descriptions and defaults."""
        from .example import generate_example
        return generate_example()

    # --- models --------------------------------------------------------------
    def download_models(self, names: list[str] | None = None, *, base_url: str | None = None,
                        force: bool = False) -> dict[str, Path | None]:
        """Download models that have no explicit `path` (by download_key or type). Returns name -> path."""
        from .loader import download_models
        return download_models(self, names=names, base_url=base_url, force=force)
