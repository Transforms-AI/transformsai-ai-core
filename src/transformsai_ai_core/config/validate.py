"""
The validation engine behind ``load_config`` and ``validate_config``.

Every problem is collected into one ``ValidationReport``; nothing raises here.

- Unknown keys (outside any ``settings``): error in strict mode, warning in
  lenient mode, each with a "did you mean" suggestion.
- An invalid value in an optional key: strict -> error; lenient -> warning and
  the key falls back to its default.
- A missing or invalid required key: always an error. The enclosing camera /
  model / optional block is dropped from the working copy so the rest of the
  config is still checked and reported in the same pass.
- Cross-field checks run on the built config; project validators run on the raw
  (normalised) data.
"""

import copy
import difflib
import re
import types
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel, ValidationError
from pydantic.fields import FieldInfo

from . import registry
from .registry import dotted
from .report import Issue, ValidationReport
from .schema import CameraAudioInput, Config, RtspSource, normalize_cameras
from .types import is_secret
from .urls import DEMAND_NAME_PREFIX, is_valid_path_name

# 4.x keys a 5.0 config should no longer carry, and where each went.
MIGRATION_HINTS = {
    "meta": "replaced by `node` (the serial is node.sn)",
    "advanced": "removed; its sections moved to the top level (api, models, mediamtx) and `settings`",
    "livestream": "replaced by `mediamtx` (server) plus cameras.<name>.stream (per camera)",
    "datasend": "replaced by `api`",
    "timings": "project data: move it under the top-level `settings`",
    "pipeline": "project data: move it under the top-level `settings`",
    "extras": "replaced by `settings`",
    "local": "replaced by `source: {type: device|file, path: ...}`",
    "local_source": "replaced by `source: {type: device|file, path: ...}`",
    "rtsp_source": "replaced by `source: {type: rtsp, host, port, user, password, path}`",
    "rtsp_url": "derived now: camera.source.url",
    "username": "renamed to `user`",
    "ip": "renamed to `host`",
    "rtsp_transport": "moved to source.transport",
    "timestamp": "removed from core: do the overlay in a VideoCaptureAsync frame_hook",
    "auto_restart_on_fail": "renamed to capture.restart.enabled",
    "restart_delay": "renamed to capture.restart.delay",
    "backoff_jitter": "renamed to `jitter`",
    "max_frame_age_ms": "renamed to `max_frame_age` (seconds or e.g. '500ms')",
    "driver": "removed (it was never used)",
    "opencv_backend": "removed (it was never used)",
    "load_options": "removed; `task` is a direct model key now",
    "lib_type": "removed (it was never used)",
    "mediamtx_ip": "renamed to mediamtx.host",
    "camera_sn_id": "derived now: stream.name from node.sn",
    "frame_width": "derived now: stream.width (default: capture.width)",
    "frame_height": "derived now: stream.height (default: capture.height)",
    "hw_encode": "replaced by encoder.codec: auto",
    "demand_poll_interval": "moved to mediamtx.demand.poll_interval",
    "demand_grace_period": "moved to mediamtx.demand.grace_period",
    "demand_timeout": "moved to mediamtx.demand.timeout",
    "max_retries": "moved to api.retries.max",
    "retry_backoff": "moved to api.retries.backoff",
    "retry_backoff_max": "moved to api.retries.backoff_max",
    "retry_on_status": "moved to api.retries.on_status",
    "cache_enabled": "moved to api.outbox.enabled",
    "cache_dir": "moved to api.outbox.dir",
    "cache_retry_interval": "moved to api.outbox.retry_interval",
    "max_cache_items": "moved to api.outbox.max_items",
    "max_cache_age_seconds": "moved to api.outbox.max_age",
    "max_cache_retries": "moved to api.outbox.max_retries",
    "pool_connections": "moved to api.pool.connections",
    "pool_maxsize": "moved to api.pool.maxsize",
    "max_workers": "moved to api.pool.workers",
    "default_content_type": "renamed to `content_type`",
    "token": "removed (nothing read it)",
}

_TAG_ERRORS = ("union_tag_invalid", "union_tag_not_found")


# ==============================================================================
# Walking annotations
# ==============================================================================
def _unwrap(annotation: Any) -> Any:
    """Strip Annotated[...] and Optional[...] (a lone non-None member)."""
    while get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    if get_origin(annotation) in (Union, types.UnionType):
        members = [a for a in get_args(annotation) if a is not type(None)]
        if len(members) == 1:
            return _unwrap(members[0])
    return annotation


def _is_model(annotation: Any) -> bool:
    return isinstance(annotation, type) and issubclass(annotation, BaseModel)


def _tagged(union: Any, discriminator: str, tag: Any) -> type[BaseModel] | None:
    """The member of a discriminated union whose tag literal is `tag`."""
    if isinstance(tag, str):
        tag = tag.strip().lower()
    for member in get_args(union):
        member = _unwrap(member)
        if _is_model(member) and discriminator in member.model_fields:
            literal = _unwrap(member.model_fields[discriminator].annotation)
            if get_origin(literal) is Literal and tag in get_args(literal):
                return member
    return None


def _tags(union: Any, discriminator: str) -> list[str]:
    out = []
    for member in get_args(union):
        member = _unwrap(member)
        if _is_model(member) and discriminator in member.model_fields:
            out += [str(t) for t in get_args(_unwrap(member.model_fields[discriminator].annotation))]
    return out


@dataclass
class _Step:
    key: Any                  # data key: field name, dict key or list index
    field: FieldInfo | None   # set for model fields, None for dict/list entries
    entry_is_model: bool = False


def _walk(loc: tuple) -> list[_Step]:
    """Map a pydantic error location onto data keys (dropping union tag elements)."""
    steps: list[_Step] = []
    annotation: Any = Config
    discriminator: str | None = None
    i = 0
    while i < len(loc):
        part = loc[i]
        core = _unwrap(annotation)
        if discriminator and get_origin(core) in (Union, types.UnionType):
            member = _tagged(core, discriminator, part)
            if member is None:
                break
            annotation, discriminator = member, None
            i += 1
            continue
        if _is_model(core):
            if not isinstance(part, str) or part not in core.model_fields:
                break
            info = core.model_fields[part]
            steps.append(_Step(part, info))
            annotation, discriminator = info.annotation, info.discriminator
            i += 1
            continue
        origin = get_origin(core)
        if origin in (dict, list):
            value_type = get_args(core)[-1] if get_args(core) else Any
            steps.append(_Step(part, None, _is_model(_unwrap(value_type))))
            annotation, discriminator = value_type, None
            i += 1
            continue
        break
    return steps


# ==============================================================================
# Unknown keys
# ==============================================================================
def _unknown_in_model(data: Any, model: type[BaseModel], path: tuple, out: list[Issue]) -> None:
    if not isinstance(data, dict):
        return
    fields = model.model_fields
    for key, value in data.items():
        if key not in fields:
            hint = MIGRATION_HINTS.get(str(key))
            close = difflib.get_close_matches(str(key), list(fields), n=1, cutoff=0.6)
            if hint and not close:
                out.append(Issue(dotted((*path, key)), f"unknown key; 4.x key {hint}"))
            else:
                out.append(Issue(dotted((*path, key)), "unknown key", close[0] if close else None))
            continue
        info = fields[key]
        _unknown_in_value(value, info.annotation, info.discriminator, (*path, key), out)


def _unknown_in_value(value: Any, annotation: Any, discriminator: Any, path: tuple, out: list[Issue]) -> None:
    core = _unwrap(annotation)
    if isinstance(discriminator, str) and get_origin(core) in (Union, types.UnionType):
        if isinstance(value, dict):
            member = _tagged(core, discriminator, value.get(discriminator))
            if member is not None:
                _unknown_in_model(value, member, path, out)
        return
    if _is_model(core):
        _unknown_in_model(value, core, path, out)
        return
    origin = get_origin(core)
    if origin is dict and isinstance(value, dict):
        value_type = get_args(core)[1] if get_args(core) else Any
        for key, item in value.items():
            _unknown_in_value(item, value_type, None, (*path, key), out)
    elif origin is list and isinstance(value, list):
        value_type = get_args(core)[0] if get_args(core) else Any
        for index, item in enumerate(value):
            _unknown_in_value(item, value_type, None, (*path, index), out)


def unknown_keys(data: dict) -> list[Issue]:
    """Every key core doesn't know, outside free sections (settings, export_options, headers)."""
    out: list[Issue] = []
    _unknown_in_model(data, Config, (), out)
    return out


# ==============================================================================
# Building with pruning
# ==============================================================================
def _show(value: Any, limit: int = 60) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _default_text(info: FieldInfo) -> str:
    default = info.get_default(call_default_factory=True)
    if isinstance(default, BaseModel):
        return "its defaults"
    if default is None:
        return "null"
    return _show(default)


def _target(steps: list[_Step], error_type: str) -> tuple[int | None, bool]:
    """(index of the step to prune, fatal?). Index None means nothing can be pruned."""
    if not steps:
        return None, True
    idx = len(steps) - 1
    if steps[idx].field is None and not steps[idx].entry_is_model:
        while idx >= 0 and steps[idx].field is None:  # a scalar list/dict item: judge its field
            idx -= 1
        if idx < 0:
            return None, True
    step = steps[idx]
    if step.field is not None and not step.field.is_required() and error_type not in _TAG_ERRORS:
        return idx, False
    # Fatal: drop the nearest camera/model entry or optional block so the rest is still checked
    for j in range(idx, -1, -1):
        candidate = steps[j]
        if candidate.field is None and candidate.entry_is_model:
            return j, True
        if j < idx and candidate.field is not None and not candidate.field.is_required():
            return j, True
    return None, True


def _message(err: dict) -> tuple[str, str | None]:
    """(message, suggestion) for one pydantic error."""
    error_type = err["type"]
    ctx = err.get("ctx") or {}
    if error_type == "missing":
        return "required key is missing", None
    if error_type in _TAG_ERRORS:
        discriminator = str(ctx.get("discriminator", "type")).strip("'")
        expected = re.findall(r"'([^']*)'", str(ctx.get("expected_tags", "")))
        if error_type == "union_tag_not_found":
            return f"missing '{discriminator}'; expected one of {', '.join(expected)}", None
        tag = str(ctx.get("tag", ""))
        close = difflib.get_close_matches(tag.lower(), expected, n=1, cutoff=0.5)
        return f"unknown {discriminator} {tag!r}; expected one of {', '.join(expected)}", (close[0] if close else None)
    message = err["msg"]
    for prefix in ("Value error, ", "Assertion failed, "):
        message = message.removeprefix(prefix)
    return message, None


def _prune(data: Any, paths: list[tuple]) -> bool:
    """Delete each path (deepest first, skipping ones inside an already-deleted path)."""
    removed = False
    done: list[tuple] = []
    for path in sorted(set(paths), key=len):
        if not any(path[: len(d)] == d for d in done):
            done.append(path)

    # Reverse order so a higher list index goes before a lower one under the same parent
    def order(path: tuple) -> list:
        return [(1, part, "") if isinstance(part, int) else (0, 0, str(part)) for part in path]

    for path in sorted(done, key=order, reverse=True):
        node = data
        try:
            for key in path[:-1]:
                node = node[key]
            del node[path[-1]]
            removed = True
        except (KeyError, IndexError, TypeError):
            continue
    return removed


def _fill_placeholder(work: Any, steps: list[_Step]) -> bool:
    """Put a stand-in where a root-level required key is missing so the rest is still checked.

    The error is already reported, so the config built this way is never handed out.
    """
    if not steps or steps[-1].field is None:
        return False
    info = steps[-1].field
    if info.examples:
        fill: Any = info.examples[0]
    elif _is_model(_unwrap(info.annotation)):
        fill = {}
    else:
        return False
    node = work
    try:
        for step in steps[:-1]:
            node = node[step.key]
        if not isinstance(node, dict):
            return False
        node[steps[-1].key] = fill
    except (KeyError, IndexError, TypeError):
        return False
    return True


def _build(data: dict, strict: bool, report: ValidationReport) -> Config | None:
    work = copy.deepcopy(data)
    placeholders: list[tuple] = []  # stand-ins already reported; errors under them are not new
    for _ in range(25):
        try:
            return Config.model_validate(work)
        except ValidationError as exc:
            prunes: list[tuple] = []
            stop = filled = False
            for err in exc.errors(include_url=False):
                steps = _walk(tuple(err["loc"]))
                keys = tuple(s.key for s in steps)
                message, suggestion = _message(err)
                idx, fatal = _target(steps, err["type"])
                path = dotted(keys)
                if err["type"] in _TAG_ERRORS:
                    path = f"{path}.type" if path else "type"
                secret = bool(steps) and steps[-1].field is not None and is_secret(steps[-1].field)
                shown = "" if err["type"] in ("missing", *_TAG_ERRORS) else f" (got {'***' if secret else _show(err.get('input'))})"
                if any(keys[: len(p)] == p for p in placeholders):
                    pass
                elif fatal or idx is None or strict:
                    report.error(path, message + shown, suggestion)
                else:
                    report.warning(path, f"{message}{shown}; using {_default_text(steps[idx].field)}", suggestion)
                if idx is not None:
                    prunes.append(keys[: idx + 1])
                elif not _fill_placeholder(work, steps):
                    stop = True
                else:
                    filled = True
                    placeholders.append(keys)
            if stop or not (_prune(work, prunes) or filled):
                return None
    report.error("", "config could not be validated (too many nested errors)")
    return None


# ==============================================================================
# Cross-field checks
# ==============================================================================
def _cross_checks(config: Config, report: ValidationReport) -> None:
    seen: dict[str, str] = {}
    for name, camera in config.cameras.items():
        base = f"cameras.{name}"
        source = camera.source
        if isinstance(source, RtspSource) and source.password and not source.user:
            report.warning(f"{base}.source.password", "ignored without a `user`")
        capture = camera.capture
        if (capture.width is None) != (capture.height is None):
            report.warning(f"{base}.capture", "width and height only apply together; set both")

        stream = camera.stream
        if stream is None or not stream.enabled:
            continue
        if (stream.width is None) != (stream.height is None):
            report.warning(f"{base}.stream", "width and height only apply together; set both")
        stream_name = camera.stream_name(config.node.sn)
        if not is_valid_path_name(stream_name):
            report.error(f"{base}.stream.name",
                         f"resolved stream name {stream_name!r} has characters MediaMTX paths don't allow "
                         "(letters, digits and _ - . ~ only)")
        if (stream.on_demand and not stream_name.startswith(DEMAND_NAME_PREFIX)
                and (config.mediamtx is None or config.mediamtx.demand_url is None)):
            report.warning(f"{base}.stream.name",
                           f"on-demand stream {stream_name!r} doesn't start with '{DEMAND_NAME_PREFIX}': the bundled "
                           "mediamtx-host only raises demand flags for live/cam_sn_* paths")
        if stream_name in seen:
            report.error(f"{base}.stream.name",
                         f"stream name {stream_name!r} is also used by camera {seen[stream_name]!r}; two streams "
                         "cannot publish to the same path (use {camera} or {index} in stream.name)")
        else:
            seen[stream_name] = name
        if stream.audio is not None and isinstance(stream.audio.input, CameraAudioInput) and not source.supports_audio:
            report.error(f"{base}.stream.audio.input.type",
                         f"'camera' audio needs an rtsp or url source (this camera's source is {source.type!r})")

    if config.enabled_streams() and (config.mediamtx is None or not config.mediamtx.host):
        report.error("mediamtx.host", "required when a camera stream is enabled")
    if config.api is not None and config.api.enabled and not config.api.base_url:
        report.error("api.base_url", "required when api is enabled")


def _project_checks(data: dict, strict: bool, report: ValidationReport) -> None:
    for validator in registry.validators():
        for path, value in registry.iter_matches(data, validator.pattern):
            for issue in registry.run_check(validator.check, value, path):
                report.add(issue, as_error=strict or validator.required)


# ==============================================================================
# Entry point
# ==============================================================================
def validate_data(data: Any, *, strict: bool) -> tuple[Config | None, ValidationReport]:
    """Validate a parsed config. Returns the built Config (None if it can't be built) and the report."""
    report = ValidationReport()
    if not isinstance(data, dict):
        report.error("", f"a config must be a mapping, got {type(data).__name__}")
        return None, report
    data = copy.deepcopy(data)
    if "cameras" in data:
        data["cameras"] = normalize_cameras(data["cameras"])

    for issue in unknown_keys(data):
        report.add(issue, as_error=strict)
    config = _build(data, strict, report)
    if config is not None:
        _cross_checks(config, report)
    _project_checks(data, strict, report)
    return config, report
