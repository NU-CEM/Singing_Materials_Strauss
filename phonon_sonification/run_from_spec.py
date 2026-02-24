import yaml
import itertools
import subprocess
import re
from copy import deepcopy
from pathlib import Path
from phonon_dos_sonifier import PhononDOSSonifier

KEY_MAP = {
  "mode": "m",
  "site": "s",
  "temp": "T",
  "mapping": "map",
  "lfo": "lfo",
  "lfo_target": "lfo_tar",
  "duration": "dur"
}

def run_spec(path: str):
    with open(path) as f:
        spec = yaml.safe_load(f)

    globals_cfg = spec.get("globals", {})
    sweeps = spec.get("sweeps", {})
    jobs = spec["jobs"]

    sweep_cases = list(expand_sweeps(sweeps)) or [{}]

    outputs = []

    temp = globals_cfg.get("temp")
    duration = globals_cfg.get("duration", 10.0)

    sonifier = PhononDOSSonifier(
                mp_id=spec["mp_id"],
                duration=duration,
                temperatures=[temp] if temp is not None else None
                )

    for sweep in sweep_cases:
        for job in jobs:
            cfg = merge(globals_cfg, sweep, job)

            output = cfg.get("output") or build_filename(cfg)
            overwrite = cfg.get("overwrite", False)
            output_path = Path(output)

            if output_path.exists() and not overwrite:
                print(f"Skipping existing: {output}")
            else:
                run_job(sonifier, cfg, output)
            outputs.append(output)
             
    if "mix" in spec:

        n = len(outputs)
        mix = spec["mix"]
        
        quiet = mix.get("quiet", False)
        
        weights = mix.get("weights")
        normalise = mix.get("normalise", True)
        mode = mix.get("mode", True)
        overwrite = mix.get("overwrite", False)
        
        cmd = ["ffmpeg"]
        
        if quiet:
            cmd += ["-loglevel", "error"]

        if overwrite:
            cmd.append("-y")
        
        for wav in outputs:
            cmd += ["-i", wav]

        if mode == "super":
            n = len(outputs)
            weights = mix.get("weights")

            if weights:
                w = " ".join(str(x) for x in weights)
                filter_arg = f"amix=inputs={n}:weights={w}:normalize=1"   # TODO: currently always normalised
            else:
                filter_arg = f"amix=inputs={n}:normalize=1"

            cmd += ["-filter_complex", filter_arg, mix["output"]]

        elif mode == "concat":   # TODO: think about loudness matching
            # build filter graph
            fade = mix.get("fade", 1.0)
            curve = mix.get("curve", "tri")
            
            filter_parts = []
            last = "0"
            
            for i in range(1, len(outputs)):
                out = f"x{i}"
                filter_parts.append(
                    f"[{last}][{i}]acrossfade=d={fade}:c1={curve}:c2={curve}[{out}]"
                )
                last = out
            
            filter_arg = ";".join(filter_parts)
            
            cmd += [
                "-filter_complex", filter_arg,
                "-map", f"[{last}]",
                mix["output"]
            ]
        
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

def run_job(sonifier, cfg, output):
    mode = cfg["mode"]
    temp = cfg.get("temp")

    if cfg.get("all_sites", False):
        return sonifier.sonify_all_sites(
            temperature=temp,
            output_path=output,
            mapping=cfg.get("mapping"),
            use_lfo=cfg.get("lfo", False),
            lfo_target=cfg.get("lfo_target", "pitch"),
            mode=mode
        )

    elif "sites" in cfg:
        site_configs = [{"site": s} for s in cfg["sites"]]
        return sonifier.sonify_multi_site(
            site_configs=site_configs,
            temperature=temp,
            output_path=output,
            mapping=cfg.get("mapping"),
            use_lfo=cfg.get("lfo", False),
            lfo_target=cfg.get("lfo_target", "pitch"),
            mode=mode
        )

    elif "site" in cfg:
        return sonifier.sonify_single_site(
            site_name=cfg["site"],
            temperature=temp,
            output_path=output,
            mapping=cfg.get("mapping"),
            use_lfo=cfg.get("lfo", False),
            lfo_target=cfg.get("lfo_target", "pitch"),
            mode=mode
        )

    else:
        raise ValueError("Job must specify site, sites, or all_sites")

def expand_sweeps(sweeps: dict):
    keys = sweeps.keys()
    values = sweeps.values()
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))

def merge(*dicts):
    out = {}
    for d in dicts:
        if d:
            out.update(d)
    return out
    
def sanitize(val):
    if val is None:
        return "none"
    return re.sub(r"[^a-zA-Z0-9._-]+", "", str(val))

def build_filename(cfg):
    parts = []

    for key in KEY_MAP:
        if key in cfg:
            v = cfg[key]
            if key == "temp" and v is not None:
                v = f"{v}K"
            parts.append(f"{key}-{sanitize(v)}")

    return "_".join(parts) + ".wav"

if __name__ == "__main__":
    import sys
    run_spec(sys.argv[1])
