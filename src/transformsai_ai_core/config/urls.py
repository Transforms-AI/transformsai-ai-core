"""
Stream names and MediaMTX URLs: the one place they are built.

Shared by ``Config.stream_urls()`` and ``MediaMTXStreamer`` so a dashboard link
computed from the config is always the URL the streamer publishes to.
"""

import re
import string
from dataclasses import dataclass

STREAM_NAME_FIELDS = frozenset({"sn", "camera", "index"})
URL_FIELDS = frozenset({"host", "rtsp_port", "webrtc_port", "name"})

# Templates that reproduce 4.x. A null `webrtc_url` picks one of the two by host type.
DEFAULT_RTSP_URL = "rtsp://{host}:{rtsp_port}/live/{name}"
DEFAULT_WEBRTC_URL_HOSTNAME = "https://{host}/live/{name}/"
DEFAULT_WEBRTC_URL_IP = "http://{host}:{webrtc_port}/live/{name}/"
DEFAULT_DEMAND_URL = "https://{host}/demand/{name}"

# A stream name is one MediaMTX path segment (it also names the demand flag file)
_PATH_NAME_RE = re.compile(r"^[A-Za-z0-9_\-.~]+$")

# Every stream name starts with this fixed prefix: the bundled mediamtx-host only runs
# on-demand for paths matching ^live/cam_sn_(.+)$, so a name without it could never go live.
STREAM_NAME_PREFIX = "cam_sn_"

_IPV4_RE = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")
_IPV6_RE = re.compile(r"^\[?([0-9a-fA-F]{0,4}:){2,7}[0-9a-fA-F]{0,4}\]?$")


def is_ip_address(host: str) -> bool:
    """IPv4 or (bracketed or bare) IPv6 literal."""
    return bool(_IPV4_RE.match(host) or _IPV6_RE.match(host))


def url_host(host: str) -> str:
    """Host as it goes into a URL: a bare IPv6 literal gets brackets."""
    if ":" in host and not host.startswith("[") and _IPV6_RE.match(host):
        return f"[{host}]"
    return host


def template_fields(template: str) -> set[str]:
    """Placeholder names used by a ``str.format`` template. Raises ValueError if malformed."""
    names = set()
    for _, field_name, _, _ in string.Formatter().parse(template):
        if field_name is None:
            continue
        if field_name == "" or field_name.isdigit():
            raise ValueError("positional placeholders ('{}') are not allowed; use a name")
        names.add(re.split(r"[.\[]", field_name, maxsplit=1)[0])
    return names


def check_template(template: str, allowed: frozenset) -> str:
    """Validate a template's placeholders against ``allowed``; returns it unchanged."""
    unknown = template_fields(template) - allowed
    if unknown:
        raise ValueError(
            f"unknown placeholder(s) {', '.join('{' + n + '}' for n in sorted(unknown))}; "
            f"allowed: {', '.join('{' + n + '}' for n in sorted(allowed))}"
        )
    return template


def stream_path_name(name: str) -> str:
    """`name` with the fixed 'cam_sn_' prefix, added once (an already-prefixed name is kept)."""
    name = str(name)
    return name if name.startswith(STREAM_NAME_PREFIX) else STREAM_NAME_PREFIX + name


def render_stream_name(template: str, *, sn: str, camera: str, index: int) -> str:
    """Full stream name: 'cam_sn_' + the rendered `stream.name` template."""
    return stream_path_name(template.format(sn=sn, camera=camera, index=index))


def is_valid_path_name(name: str) -> bool:
    return bool(_PATH_NAME_RE.match(name))


@dataclass(frozen=True)
class StreamUrls:
    """Where a stream is published (rtsp), watched (webrtc) and gated (demand)."""

    name: str
    rtsp: str
    webrtc: str
    demand: str


def build_stream_urls(
    host: str,
    name: str,
    *,
    rtsp_port: int,
    webrtc_port: int,
    rtsp_url: str | None = None,
    webrtc_url: str | None = None,
    demand_url: str | None = None,
) -> StreamUrls:
    """Resolve the three URLs of stream ``name`` on server ``host``.

    ``name`` always ends up with the fixed 'cam_sn_' prefix (added if missing).

    A ``None`` template means the 4.x default. Only the webrtc default depends on
    the host: a bare IP is plain HTTP on ``webrtc_port``; a hostname is assumed to
    sit behind a TLS reverse proxy (no port).
    """
    name = stream_path_name(name)
    values = {"host": url_host(host), "rtsp_port": rtsp_port, "webrtc_port": webrtc_port, "name": name}
    if webrtc_url is None:
        webrtc_url = DEFAULT_WEBRTC_URL_IP if is_ip_address(host) else DEFAULT_WEBRTC_URL_HOSTNAME
    return StreamUrls(
        name=name,
        rtsp=(rtsp_url or DEFAULT_RTSP_URL).format(**values),
        webrtc=webrtc_url.format(**values),
        demand=(demand_url or DEFAULT_DEMAND_URL).format(**values),
    )
