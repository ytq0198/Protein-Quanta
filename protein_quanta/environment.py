"""Privacy-minimized reproducibility environment reporting."""

from importlib import metadata
import platform


DEFAULT_PACKAGES = (
    "numpy",
    "h5py",
    "pandas",
    "torch-geometric",
    "torch-scatter",
    "torch-sparse",
    "torch-cluster",
    "torchdiffeq",
    "torch-ema",
)


def collect_package_versions(package_names=DEFAULT_PACKAGES):
    """Return versions for a fixed package allowlist without local paths."""
    versions = {}
    for name in package_names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _torch_details(torch_module):
    cuda_available = bool(torch_module.cuda.is_available())
    device_count = int(torch_module.cuda.device_count()) if cuda_available else 0
    devices = []
    for index in range(device_count):
        major, minor = torch_module.cuda.get_device_capability(index)
        properties = torch_module.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": str(torch_module.cuda.get_device_name(index)),
                "compute_capability": f"{major}.{minor}",
                "memory_gib": round(float(properties.total_memory) / 1024**3, 3),
            }
        )
    return {
        "version": str(torch_module.__version__),
        "cuda_runtime": getattr(torch_module.version, "cuda", None),
        "cuda_available": cuda_available,
        "device_count": device_count,
        "devices": devices,
    }


def build_environment_report(
    package_versions,
    python_version,
    system,
    release,
    machine,
    torch_module,
):
    """Build a stable schema that excludes hostnames, users, paths, and env vars."""
    return {
        "schema_version": 1,
        "python": python_version,
        "platform": {"system": system, "release": release, "machine": machine},
        "packages": dict(package_versions),
        "torch": _torch_details(torch_module),
    }


def collect_environment():
    """Collect the project allowlist from the active Python environment."""
    import torch

    return build_environment_report(
        package_versions=collect_package_versions(),
        python_version=platform.python_version(),
        system=platform.system(),
        release=platform.release(),
        machine=platform.machine(),
        torch_module=torch,
    )

