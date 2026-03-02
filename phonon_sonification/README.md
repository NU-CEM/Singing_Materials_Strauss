## What is this?

First, bear with me, several parts of this package are (probably) currently broken. Things are changing fast and its not yet ready for public consumption.

- This is a Python-based, [Strauss](https://www.audiouniverse.org/research/strauss)-powered package which sonifies first-principles phonon data. 
- At the moment the focus is on phonon density of states data, which is read in from the [Materials Project](https://next-gen.materialsproject.org/) database.
- Currently three sonification types are supported: spectral, synth and choral. These can be mixed together in different ways (superimpose, concatenate or a mixture of the two). Mixing is implemented through `ffmpeg`.
- Users are encouraged to use `.yml` files rather than the command line interface to create and mix sonifications. The command line interface does not allow mixing.
- The package is designed to by extendable, so that new types of phonon data, new data interfaces, and new sonifications can all be considered.
- This package will power a Singing Materials web app which is currently in development.

### Package structure:

`cli.py` : command-line-interface for the `phonon_dos_sonifier` module. Allows the user to create `.wav` files for the spectral, synth or choral sonifications, but does not allow user to mix these together.

`mods.py` : dictionaries specifying mods to the Strauss synthesiser parameters.

`mp_interface.py` : interface to the Materials Project database. This is how we access the density of states data. This module puts the data into a nice dictionary format which is then read into `phonon_dos_sonifier.py`.

`phonon_dos_sonifier.py`: takes the density of states data and sonifies it using Strauss. Three sonification strategies are implemented: spectral (using the Strauss spectraliser), synth (using the Strauss synth) and choral (using the strauss sampler and choral samples).

`phonon_frequency_mapping.py` : functions for mapping from phonon frequencies to audible frequencies.

`phonon_mixer.py` : takes the `.wav` files created by `phonon_dos_sonifier.py` and mixes them together with `ffmpeg`.

`phonopy_interface.py` : reads in data from the phonopy output file format.

`run_from_spec.py` : reads in the `sonification_batch.yml` and uses these to create and mix sonifications. 

`sonification_batch.yml` : example yml file which specifies how to create the sonifications and mix them together.

`utilities.py` : small utility functions.

`visualisation.py` : functions for plotting the density of states data. because seeing things is still useful.



