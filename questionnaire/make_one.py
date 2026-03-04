import subprocess

def make_one(wav, out):
    # 1️⃣ Create 2 seconds silence
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", "anullsrc",
        "-t", "2",
        "silence.wav"
    ], check=True)

    # 2️⃣ Create concat list
    with open("files.txt", "w") as f:
        f.write(f"file '{wav}'\n")
        f.write("file 'silence.wav'\n")

    # 3️⃣ Concatenate
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", "files.txt",
        "-c", "copy",
        "combined.wav"
    ], check=True)

    # 4️⃣ Loudness normalise + encode
    subprocess.run([
        "ffmpeg", "-y",
        "-i", "combined.wav",
        "-af", "loudnorm=I=-23:TP=-1.5:LRA=7",
        "-c:a", "aac",
        "-b:a", "256k",
        out
    ], check=True)

STO_choral = "./SrTiO3/mode-choral_sites-all_temp-none_lfo-True_lfo_target-volume_duration-5.0.wav"
STO_synth = "./SrTiO3/mode-synth_sites-all_temp-none_lfo-True_lfo_target-volume_duration-5.0.wav"
STO_spectral = "./SrTiO3/mode-spectral_sites-all_temp-none_lfo-True_lfo_target-volume_duration-5.0.wav"

make_one(STO_choral, "example_STO_choral.mp4")
make_one(STO_synth, "example_STO_synth.mp4")
make_one(STO_spectral, "example_STO_spectral.mp4")