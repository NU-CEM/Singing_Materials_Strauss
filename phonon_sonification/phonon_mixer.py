import subprocess
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
        cmd += concatenation(outputs,job_order,mix)
    elif mode == "compose":
        cmd += composition(outputs, job_order, mix)
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
            
def concatenation(outputs,job_order,mix):  
            
    """
    outputs: dict mapping job_name -> wav path
    mix: mix config
    """

    order = mix.get("order")

    if order:
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

def composition(outputs,mix):
    """
    Executes a sequence of concat/super stages defined in mix["composition"]
    """

    stages = mix.get("composition")
    if not stages:
        raise ValueError("compose mode requires mix.composition list")

    current_outputs = dict(outputs)
    current_order = list(job_order)

    temp_files = []

    for i, stage in enumerate(stages):
        stage_type = stage["type"]

        # each stage writes to a temp wav
        tmp = Path(tempfile.mkstemp(suffix=".wav")[1])
        temp_files.append(tmp)

        stage_mix = dict(stage)
        stage_mix["output"] = str(tmp)

        if stage_type == "concat":
            cmd = ["ffmpeg"] + concatenation(current_outputs, current_order, stage_mix)

        elif stage_type == "super":
            cmd = ["ffmpeg"] + superposition(current_outputs, stage_mix)

        else:
            raise ValueError(f"Unknown composition stage: {stage_type}")

        subprocess.run(cmd, check=True)

        # next stage sees only this result (named "result")
        current_outputs = {"result": str(tmp)}
        current_order = ["result"]

    # Final move to requested output
    final_out = mix["output"]
    Path(current_outputs["result"]).rename(final_out)

    return []