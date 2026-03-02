import subprocess
import random
import tempfile
from pathlib import Path

def start_mixing(outputs,job_order,spec):

    cmd = ["ffmpeg"]

    mix = spec["mix"]
    
    quiet = mix.get("quiet", False)
    if quiet:
        cmd += ["-loglevel", "error"]

    overwrite = mix.get("overwrite", False)
    if overwrite:
        cmd.append("-y")
        
    mode = mix.get("mode", True)  
    if mode == "super":
        cmd += superposition(outputs,mix)
    elif mode == "concat":
        cmd += concatenation(outputs,job_order,mix,spec)
    else:
        RaiseError("mode not recognised")

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def superposition(outputs,mix):

    files = list(outputs.values())
    names = list(outputs.keys())
    n = len(files)

    inputs = []
    for wav in files:
        inputs += ["-i", wav]

    weights_cfg = mix.get("weights")
    normalise = mix.get("normalise", True)

    if weights_cfg:
        try:
            #get weights, if not specified set to one
            weights = [weights_cfg.get(name, 1.0) for name in names]
        except KeyError as e:
            raise ValueError(f"Missing weight for job: {e}")
        w = " ".join(str(x) for x in weights)
        filter_arg = f"amix=inputs={n}:weights={w}:normalize={1 if normalise else 0}"
    else:
        filter_arg = f"amix=inputs={n}:normalize={1 if normalise else 0}"

    return inputs + [
        "-filter_complex", filter_arg,
        mix["output"]
    ]
            
def concatenation(outputs,job_order,mix,spec):  
            
    """
    outputs: dict mapping job_name -> wav path
    mix: mix config
    """

    order = mix.get("order")

    if order.split()[0] == "random":
        jobs = spec["jobs"]
        job_names = []
        for job in jobs:
            job_names.append(job["name"])
        ordered_names = random.choices(job_names, k=int(order.split()[1]))
        ordered_files = [outputs[name] for name in ordered_names]
        
    elif order:
        try:
            ordered_names = order
            ordered_files = [outputs[name] for name in ordered_names]
        except KeyError as e:
            raise ValueError(f"Unknown job in mix.order: {e}")
    else:
        # Default: YAML job order
        ordered_names = job_order
        ordered_files = [outputs[name] for name in ordered_names]

    fade = mix.get("fade", 1.0)
    curve = mix.get("curve", "tri")
    target_lufs = mix.get("lufs", -16)   # sensible default

    # Add inputs
    inputs = []
    for wav in ordered_files:
        inputs += ["-i", wav]

    # Build acrossfade chain
    filter_parts = []

    # First loudness-normalise each input
    for i in range(len(ordered_files)):
        filter_parts.append(
            f"[{i}]loudnorm=I={target_lufs}:TP=-1.5:LRA=11[n{i}]"
        )
    last = "n0"

    # Then acrossfade them
    for i in range(1, len(ordered_files)):
        out = f"x{i}"
        filter_parts.append(
            f"[{last}][n{i}]acrossfade=d={fade}:c1={curve}:c2={curve}[{out}]"
        )
        last = out

    filter_arg = ";".join(filter_parts)

    return inputs + [
        "-filter_complex", filter_arg,
        "-map", f"[{last}]",
        mix["output"]
    ]

