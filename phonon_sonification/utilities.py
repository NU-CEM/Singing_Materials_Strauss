def format_duration_for_strauss(duration_seconds: float) -> str:
    """Convert duration in seconds to STRAUSS Score format "Xm Ys" """
    minutes = int(duration_seconds // 60)
    seconds = int(duration_seconds % 60)
    return f"{minutes}m {seconds}s"

def process_imaginary(phonon_frequencies):
    # remove any imaginary modes
    phonon_frequencies = list(phonon_frequencies)
    phonon_cleaned_frequencies = [frequency for frequency in phonon_frequencies if frequency > 0]
    if len(phonon_frequencies) != len(phonon_cleaned_frequencies):
        print("There are {} imaginary frequencies which have not been processed".format(len(phonon_frequencies)-len(phonon_cleaned_frequencies)))
    return np.array(phonon_cleaned_frequencies)

def process_imaginary_dos(dos,phonon_frequencies) :
    # remove dos which correspond to imaginary modes
    phonon_frequencies = list(phonon_frequencies)
    dos_cleaned_frequencies = [dos[i] for i in range(len(phonon_frequencies)) if phonon_frequencies[i] > 0]

    return np.array(dos_cleaned_frequencies)
