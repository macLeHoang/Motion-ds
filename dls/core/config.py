"""YAML config loading + validated, immutable branch/app specs."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, fields, replace
from types import MappingProxyType
from typing import Any, Mapping, Optional

import yaml

from dls.core.reid import reid_config_errors

ENV_PATTERN = re.compile(r"\$\{(\w+):?([^}]*)\}")

INFERENCE_ELEMENTS = frozenset({"gvadetect", "gvaclassify", "gvainference"})
# Branch configs can come from the API, so only analytics/pass-through elements are allowed:
# no sources/sinks, no gvapython (arbitrary code), no metapublish (network/file writes).
# `pydetect` = Python-decoded detector (SCRFD / YOLO-pose), internal element `vmspydetect`.
PYDETECT = "pydetect"
PYDETECT_KINDS = frozenset({"scrfd", "yolo_pose", "yolo", "yolo_motion", "scrfd_motion", "yolo_pose_motion"})
ALLOWED_ELEMENTS = INFERENCE_ELEMENTS | frozenset({
    "gvatrack", "gvawatermark", "gvafpscounter", "queue", "identity", "videoconvert", "videoscale",
    PYDETECT,
})
FILE_PROPERTIES = ("model", "model-proc", "labels-file")
SINK_TYPES = frozenset({"fakesink", "filesink"})
DEFAULT_MAX_CAMERAS = 8
DEFAULT_MODEL_ROOT = "data/dls/models"
DEFAULT_OUTPUT_DIR = "data/dls/output"


def model_roots() -> list[str]:
    """Directories model files must live under (DLS_MODEL_ROOTS, os.pathsep-separated)."""
    env = os.environ.get("DLS_MODEL_ROOTS")
    if env:
        return [r for r in env.split(os.pathsep) if r]
    return sorted({DEFAULT_MODEL_ROOT, os.environ.get("DLS_MODELS", DEFAULT_MODEL_ROOT)})


def output_dir() -> str:
    return os.environ.get("DLS_OUTPUT_DIR", DEFAULT_OUTPUT_DIR)


def is_within(path: str, roots: list[str]) -> bool:
    real = os.path.realpath(path)
    return any(os.path.commonpath([real, os.path.realpath(r)]) == os.path.realpath(r) for r in roots)


class ConfigError(ValueError):
    """Invalid configuration; `errors` lists every problem found."""

    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


def _freeze(value: Any) -> Any:
    """Recursively convert dicts/lists into read-only mappings/tuples."""
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


def thaw(value: Any) -> Any:
    """Inverse of _freeze — plain dicts/lists (for JSON responses, processors)."""
    if isinstance(value, Mapping):
        return {k: thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [thaw(v) for v in value]
    return value


def expand_env(text: str) -> str:
    """Expand ${VAR:default} using the process environment."""
    return ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), m.group(2) or ""), text)


def load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(expand_env(f.read())) or {}


# ─────────────────────────────────────────────────────────────────────────────
# Branch spec
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ElementSpec:
    type: str
    name: str
    properties: Mapping[str, Any]
    probes: Mapping[str, str]          # pad name → probe name
    when: Optional[str] = None         # camera param flag that must be truthy to include element
    shared: bool = True                # share one model instance across all cameras of the branch

    @property
    def is_inference(self) -> bool:
        return self.type in INFERENCE_ELEMENTS

    @property
    def is_pydetect(self) -> bool:
        return self.type == PYDETECT


@dataclass(frozen=True)
class SinkSpec:
    type: str = "fakesink"
    location: Optional[str] = None     # filesink: may contain {branch} and {camera_id}
    branch: str = ""

    def location_for(self, camera_id: str) -> str:
        template = self.location or os.path.join(output_dir(), "{branch}_{camera_id}.mkv")
        # str.replace, not str.format: the template may come from the API
        return template.replace("{branch}", self.branch).replace("{camera_id}", camera_id)


@dataclass(frozen=True)
class BranchSpec:
    name: str
    elements: tuple[ElementSpec, ...]
    sink: SinkSpec
    processor: Optional[str] = None
    max_cameras: int = DEFAULT_MAX_CAMERAS
    params: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    raw: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    reid: Optional[Mapping[str, Any]] = None      # ReidRemap settings (dls/core/reid.py); None = raw ids

    def elements_for(self, camera_params: Mapping[str, Any]) -> tuple[ElementSpec, ...]:
        """Elements to instantiate for one camera (drops elements whose `when` flag is off)."""
        return tuple(e for e in self.elements if e.when is None or bool(camera_params.get(e.when)))


def _element_errors(i: int, elem: Any, seen: set[str], check_files: bool,
                    roots: list[str]) -> list[str]:
    if not isinstance(elem, Mapping):
        return [f"elements[{i}]: phải là object"]
    errors = []
    etype = elem.get("type")
    if not etype or not isinstance(etype, str):
        errors.append(f"elements[{i}]: thiếu 'type'")
    elif etype not in ALLOWED_ELEMENTS:
        errors.append(f"elements[{i}]: element '{etype}' không được phép; chỉ {sorted(ALLOWED_ELEMENTS)}")
    name = elem.get("name") or f"{etype}{i}"
    if name in seen:
        errors.append(f"elements[{i}]: trùng tên element '{name}'")
    seen.add(name)

    props = elem.get("properties", {}) or {}
    if not isinstance(props, Mapping):
        errors.append(f"elements[{i}]: 'properties' phải là object")
        props = {}
    if etype in INFERENCE_ELEMENTS | {PYDETECT} and not props.get("model"):
        errors.append(f"elements[{i}]: {etype} cần properties.model")
    if etype == PYDETECT and props.get("kind") not in PYDETECT_KINDS:
        errors.append(f"elements[{i}]: pydetect cần properties.kind trong {sorted(PYDETECT_KINDS)}")
    for key in FILE_PROPERTIES:
        path = props.get(key)
        if not path:
            continue
        if not is_within(str(path), roots):
            errors.append(f"elements[{i}]: {key} phải nằm trong {roots}: {path}")
        elif check_files and not os.path.exists(path):
            errors.append(f"elements[{i}]: {key} không tồn tại: {path}")
    if not isinstance(elem.get("probes", {}) or {}, Mapping):
        errors.append(f"elements[{i}]: 'probes' phải là object {{pad: probe_name}}")
    return errors


def _sink_errors(sink: Mapping[str, Any]) -> list[str]:
    if sink.get("type", "fakesink") not in SINK_TYPES:
        return [f"sink.type không hỗ trợ: {sink.get('type')} (chỉ {sorted(SINK_TYPES)})"]
    location = sink.get("location")
    if location and not is_within(str(location), [output_dir()]):
        return [f"sink.location phải nằm trong {output_dir()}: {location}"]
    return []


def validate_branch_config(cfg: Mapping[str, Any], check_files: bool = True,
                           roots: Optional[list[str]] = None) -> list[str]:
    """Return a list of human-readable problems (empty list = valid)."""
    roots = roots if roots is not None else model_roots()
    errors: list[str] = []
    elements = cfg.get("elements")
    if not elements or not isinstance(elements, list):
        errors.append("'elements' phải là list không rỗng")
        elements = []
    seen: set[str] = set()
    for i, elem in enumerate(elements):
        errors.extend(_element_errors(i, elem, seen, check_files, roots))
    errors.extend(_sink_errors(cfg.get("sink", {}) or {}))
    if cfg.get("reid") is not None:
        errors.extend(reid_config_errors(cfg["reid"]))
    max_cams = cfg.get("max_cameras", DEFAULT_MAX_CAMERAS)
    if not isinstance(max_cams, int) or max_cams < 1:
        errors.append("max_cameras phải là số nguyên >= 1")
    return errors


def parse_branch_spec(name: str, cfg: Mapping[str, Any], processor: Optional[str] = None,
                      check_files: bool = True, roots: Optional[list[str]] = None) -> BranchSpec:
    errors = validate_branch_config(cfg, check_files=check_files, roots=roots)
    if errors:
        raise ConfigError(errors)

    elements = tuple(
        ElementSpec(
            type=e["type"],
            name=e.get("name") or f"{e['type']}{i}",
            properties=_freeze(dict(e.get("properties") or {})),
            probes=_freeze(dict(e.get("probes") or {})),
            when=e.get("when"),
            shared=bool(e.get("shared", True)),
        )
        for i, e in enumerate(cfg["elements"])
    )
    sink_cfg = cfg.get("sink") or {}
    sink = SinkSpec(type=sink_cfg.get("type", "fakesink"), location=sink_cfg.get("location"), branch=name)
    return BranchSpec(
        name=name,
        elements=elements,
        sink=sink,
        processor=processor,
        max_cameras=cfg.get("max_cameras", DEFAULT_MAX_CAMERAS),
        params=_freeze(dict(cfg.get("params") or {})),
        raw=_freeze(dict(cfg)),
        reid=_freeze(dict(cfg["reid"])) if cfg.get("reid") is not None else None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# App config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SourceSettings:
    """Per-camera source normalisation (applied before the tee, shared by all branches)."""
    width: int = 1280                 # 0 → keep native size
    height: int = 720
    max_fps: int = 10                 # 0 → no frame-rate cap (videorate drop-only)
    rtsp_latency_ms: int = 200
    rtsp_tcp: bool = True
    file_loop: bool = True            # loop file:// sources (handy for tests / demos)
    queue_size: int = 4               # per-branch leaky queue depth
    reconnect_initial_s: float = 2.0
    reconnect_max_s: float = 30.0
    stall_timeout_s: float = 20.0     # no frames for this long on a live source → reconnect
    chain_stall_timeout_s: float = 60.0   # source healthy but a branch chain silent this long → rebuild
    park_after_failures: int = 5      # consecutive failed reconnects before the camera is parked
    park_s: float = 300.0             # retry interval of a parked camera
    probe_timeout_s: float = 2.0      # TCP probe of the RTSP host:port before each rebuild


@dataclass(frozen=True)
class CpuSettings:
    """CPU limits for the whole pipeline process (see dls/core/cpu.py)."""
    max_cores: float = 0.0            # budget for inference thread pools; 0 = all cores in the affinity set
    cpuset: str = ""                  # hard limit: pin the process to these cores, e.g. "0-1" or "0,2"
    inference_threads: int = 0        # threads per OpenVINO engine; 0 = derived from the budget


@dataclass(frozen=True)
class AppConfig:
    api_host: str = "0.0.0.0"
    api_port: int = 8083
    pipeline_name: str = "dls-pipeline"
    branches: Mapping[str, Any] = field(default_factory=dict)   # name → config path or inline dict
    source: SourceSettings = field(default_factory=SourceSettings)
    device: str = "CPU"
    cpu: CpuSettings = field(default_factory=CpuSettings)


def _source_settings(raw: Mapping[str, Any]) -> SourceSettings:
    known = {f.name for f in fields(SourceSettings)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ConfigError([f"source: khóa không hợp lệ {unknown}; hợp lệ: {sorted(known)}"])
    return SourceSettings(**raw)


CAMERA_SOURCE_KEYS = ("width", "height", "max_fps")


def source_settings_for(base: SourceSettings, params: Mapping[str, Any]) -> SourceSettings:
    """Per-camera override from API/config params: {"source": {"width", "height", "max_fps"}}."""
    override = params.get("source") if params else None
    if not override:
        return base
    if not isinstance(override, Mapping):
        raise ConfigError(["source: phải là object {width, height, max_fps}"])
    errors = [f"source.{k}: không cho phép theo camera (chỉ {list(CAMERA_SOURCE_KEYS)})"
              for k in override if k not in CAMERA_SOURCE_KEYS]
    errors += [f"source.{k}: phải là số nguyên >= 0" for k in CAMERA_SOURCE_KEYS
               if k in override and (type(override[k]) is not int or override[k] < 0)]
    if errors:
        raise ConfigError(errors)
    return replace(base, **{k: override[k] for k in CAMERA_SOURCE_KEYS if k in override})


def load_app_config(path: Optional[str]) -> AppConfig:
    raw = load_yaml(path) if path else {}
    api = raw.get("camera_api", {}) or {}
    pipeline = raw.get("pipeline", {}) or {}
    return AppConfig(
        api_host=str(api.get("host", "0.0.0.0")),
        api_port=int(api.get("port", 8083)),
        pipeline_name=pipeline.get("name", "dls-pipeline"),
        branches=dict(pipeline.get("branches") or {}),
        source=_source_settings(raw.get("source", {}) or {}),
        device=os.environ.get("DLS_DEVICE", raw.get("device", "CPU")),
        cpu=_cpu_settings(raw.get("cpu", {}) or {}),
    )


def _cpu_settings(raw: Mapping[str, Any]) -> CpuSettings:
    """`cpu:` section; env DLS_MAX_CORES / DLS_CPUSET / DLS_INFER_THREADS override it."""
    known = {f.name for f in fields(CpuSettings)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ConfigError([f"cpu: khóa không hợp lệ {unknown}; hợp lệ: {sorted(known)}"])
    return CpuSettings(
        max_cores=float(os.environ.get("DLS_MAX_CORES", raw.get("max_cores", 0)) or 0),
        cpuset=str(os.environ.get("DLS_CPUSET", raw.get("cpuset", "")) or ""),
        inference_threads=int(os.environ.get("DLS_INFER_THREADS", raw.get("inference_threads", 0)) or 0),
    )
