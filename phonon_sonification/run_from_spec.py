import yaml
import itertools
import re
from copy import deepcopy
from pathlib import Path
from phonon_dos_sonifier import PhononDOSSonifier
from phonon_mixer import superposition, concatenation, composition

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

    outputs = {}  
    job_order = [] 

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
                
            job_name = job["name"]
            outputs[job_name] = output
            job_order.append(job_name)
             
    if "mix" in spec:
        run_mixer(outputs,job_order,spec)

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
    
def sanitise(val):
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
            parts.append(f"{key}-{sanitise(v)}")

    return "_".join(parts) + ".wav"

if __name__ == "__main__":
    import sys
    run_spec(sys.argv[1])
