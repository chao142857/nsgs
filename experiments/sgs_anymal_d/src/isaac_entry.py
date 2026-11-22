"""Start one headless Isaac instance using an externally installed Isaac Lab and assets."""
import os
from pathlib import Path
import sys


def open_app(device_index, out):
    from isaaclab.app import AppLauncher
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    experience = os.environ.get("SGS_KIT_EXPERIENCE")
    asset_root = os.environ.get("SGS_ASSET_ROOT")
    kit = ["--portable-root", str(out / "kit-portable"),
           "--/app/extensions/registryEnabled=false", "--/renderer/multiGpu/enabled=false",
           "--/renderer/multiGpu/autoEnable=false", f"--/renderer/activeGpu={device_index}",
           "--/physics/cudaDevice=0", "--/plugins/carb.tasking.plugin/threadCount=2",
           f"--/log/file={out / 'kit.log'}"]
    if asset_root:
        for key in ("local", "cloud", "default", "nvidia"):
            kit.append(f"--/persistent/isaac/asset_root/{key}={asset_root}")
    if experience and not Path(experience).is_file():
        raise FileNotFoundError(experience)

    class BoundLauncher(AppLauncher):
        def _resolve_device_settings(self, config):
            super()._resolve_device_settings(config)
            config.update(active_gpu=device_index, physics_gpu=0, multi_gpu=False, limit_cpu_threads=2)

    # Omniverse parses the process command line in addition to explicit kit_args.
    sys.argv[:] = sys.argv[:1]
    return BoundLauncher(headless=True, device="cuda:0", enable_cameras=False, livestream=0,
                         experience=experience or "", kit_args=" ".join(kit)).app
