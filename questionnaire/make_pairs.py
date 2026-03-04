import subprocess

def make_pair(wav1, wav2, out):
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

C_choral = "./C/mode-choral_sites-all_temp-none_lfo-True_lfo_target-volume_duration-5.0_mp_id-mp-66.wav"
C_synth = "./C/mode-synth_sites-all_temp-none_lfo-True_lfo_target-volume_duration-5.0_mp_id-mp-66.wav"
C_spectral = "./C/mode-spectral_sites-all_temp-none_lfo-True_lfo_target-volume_duration-5.0_mp_id-mp-66.wav"

STO_choral = "./SrTiO3/mode-choral_sites-all_temp-none_lfo-True_lfo_target-volume_duration-5.0.wav"
STO_synth = "./SrTiO3/mode-synth_sites-all_temp-none_lfo-True_lfo_target-volume_duration-5.0.wav"
STO_spectral = "./SrTiO3/mode-spectral_sites-all_temp-none_lfo-True_lfo_target-volume_duration-5.0.wav"

MgO_choral = "./MgO/mode-choral_sites-all_temp-none_lfo-True_lfo_target-volume_duration-5.0_mp_id-mp-1265.wav"
MgO_synth = "./MgO/mode-synth_sites-all_temp-none_lfo-True_lfo_target-volume_duration-5.0_mp_id-mp-1265.wav"
MgO_spectral = "./MgO/mode-spectral_sites-all_temp-none_lfo-True_lfo_target-volume_duration-5.0_mp_id-mp-1265.wav"

PbTe_choral = "./PbTe/mode-choral_sites-all_temp-none_lfo-True_lfo_target-volume_duration-5.0_mp_id-mp-19717.wav"
PbTe_synth = "./PbTe/mode-synth_sites-all_temp-none_lfo-True_lfo_target-volume_duration-5.0_mp_id-mp-19717.wav"
PbTe_spectral = "./PbTe/mode-spectral_sites-all_temp-none_lfo-True_lfo_target-volume_duration-5.0_mp_id-mp-19717.wav"

Si_choral = "./Si/mode-choral_sites-all_temp-none_lfo-True_lfo_target-volume_duration-5.0_mp_id-mp-149.wav"
Si_synth = "./Si/mode-synth_sites-all_temp-none_lfo-True_lfo_target-volume_duration-5.0_mp_id-mp-149.wav"
Si_spectral = "./Si/mode-spectral_sites-all_temp-none_lfo-True_lfo_target-volume_duration-5.0_mp_id-mp-149.wav"

make_pair(STO_choral,C_choral,"STO-C-choral.mp4")

make_pair(PbTe_choral,C_choral,"PbTe-C-choral.mp4")

make_pair(STO_choral,Si_choral,"STO-Si-choral.mp4")

make_pair(C_choral,BAs_choral,"C-BAs-choral.mp4")

make_pair(STO_choral,BAs_choral,"STO-BAs-choral.mp4")

make_pair(PbTe_choral,STO_choral,"PbTe-STO-choral.mp4")

make_pair(PbTe_choral,BAs_choral,"PbTe-BaAs-choral.mp4")

make_pair(STO_synth,C_synth,"STO-C-synth.mp4")

make_pair(PbTe_synth,C_synth,"PbTe-C-synth.mp4")

make_pair(STO_synth,Si_synth,"STO-Si-synth.mp4")

make_pair(C_synth,BAs_synth,"C-BAs-synth.mp4")

make_pair(STO_synth,BAs_synth,"STO-BAs-synth.mp4")

make_pair(PbTe_synth,STO_synth,"PbTe-STO-synth.mp4")

make_pair(PbTe_synth,BAs_synth,"PbTe-BaAs-synth.mp4")

make_pair(STO_spectral,C_spectral,"STO-C-spectral.mp4")

make_pair(PbTe_spectral,C_spectral,"PbTe-C-spectral.mp4")

make_pair(STO_spectral,Si_spectral,"STO-Si-spectral.mp4")

make_pair(C_spectral,BAs_spectral,"C-BAs-spectral.mp4")

make_pair(BAs_spectral,STO_spectral,"STO-BAs-spectral.mp4")

make_pair(PbTe_spectral,STO_spectral,"PbTe-STO-spectral.mp4")

make_pair(PbTe_spectral,BAs_spectral,"PbTe-BaAs-spectral.mp4")

make_pair(PbTe_spectral,MgO_spectral,"PbTe-MgO-spectral.mp4")

make_pair(PbTe_synth,MgO_synth,"PbTe-MgO-synth.mp4")

make_pair(PbTe_choral,MgO_choral,"PbTe-MgO-choral.mp4")

make_pair(STO_spectral,MgO_spectral,"STO-MgO-spectral.mp4")

make_pair(STO_synth,MgO_synth,"STO-MgO-synth.mp4")

make_pair(STO_choral,MgO_choral,"STO-MgO-choral.mp4")

make_pair(MgO_spectral,BAs_spectral,"MgO-BAs-pectral.mp4")

make_pair(MgO_synth,BAs_synth,"MgO-BAs-synth.mp4")

make_pair(MgO_choral,BAs_choral,"MgO-BAs-choral.mp4")