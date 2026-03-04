import subprocess

def make_three(wav1, wav2, wav3, out):
    # 1️⃣ Create 2 seconds silence
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", "anullsrc",
        "-t", "2",
        "silence.wav"
    ], check=True)

    # 2️⃣ Create concat list file
    with open("files.txt", "w") as f:
        f.write(f"file '{wav1}'\n")
        f.write("file 'silence.wav'\n")
        f.write(f"file '{wav2}'\n")
        f.write("file 'silence.wav'\n")
        f.write(f"file '{wav3}'\n")
        f.write("file 'silence.wav'\n")

    # 3️⃣ Concatenate to combined.wav
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

BAs_choral = "./BAs/mode-choral_sites-all_temp-none_lfo-True_lfo_target-volume_duration-5.0_mp_id-mp-10044.wav"
BAs_synth = "./BAs/mode-synth_sites-all_temp-none_lfo-True_lfo_target-volume_duration-5.0_mp_id-mp-10044.wav"
BAs_spectral = "./BAs/mode-spectral_sites-all_temp-none_lfo-True_lfo_target-volume_duration-5.0_mp_id-mp-10044.wav"

make_three(BAs_spectral, BAs_choral, BAs_synth, "example_sonification.mp4")