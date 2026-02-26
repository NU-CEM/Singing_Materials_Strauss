"""
Phonon DOS Sonification using STRAUSS.

This script sonifies phonon DOS data from Materials Project.

Modes:
- Spectraliser → Phonon density of states transformed to sound via ifft.
- Synth →  All mappings below applied to create synthesised sound.
- Choral → Pitch shift and volume mapping applied to a voice sample.

Mappings:
- Phonon band centre → pitch_shift
- Projected DOS → volume 
- DOS inter quartile range → LFO modulation rate and depth

Features:
- Single site sonification 
- Multi-site chord sonification
- Temperature-dependent DOS weighting
- Spectraliser feature extraction

Requires: phonon_sonification package and MP_API_KEY in .env
"""

import numpy as np
from strauss.sonification import Sonification
from strauss.sources import Events, Objects
from strauss.score import Score
try:
    from strauss.generator import Synthesizer, Sampler, Spectralizer
except ImportError:
    from strauss.generator import Synthesizer, Sampler, Spectralizer

from typing import Dict, List, Tuple, Optional, Union
import warnings

# Import your existing modules

import sys
sys.path.append('../')
from phonon_sonification import MP_interface, utilities, phonon_frequency_mapping, mods
from phonon_sonification.MP_interface import dos_stats_analysis, scale_by_occupation
from phonon_sonification.utilities import format_duration_for_strauss
from phonon_sonification.phonon_frequency_mapping import phonon_to_audible_log, phonon_to_note

# STRAUSS Score requires a chord or note sequence. I don't really understand this.
STRAUSS_BASE_NOTE = [["G3"]]

# specify audio system.
AUDIO_SYSTEM = "stereo"

class PhononDOSSonifier:
    """
    Sonifies phonon DOS data from Materials Project using STRAUSS
    """
    
    def __init__(self, 
                 mp_id: str,
                 temperatures: Optional[List[float]] = None,
                 duration: float = 10.0,
                 fmin_audible: float = 196.0,
                 fmax_audible: float = 1500.0,
                 fmin_phonon: Optional[float] = None,
                 fmax_phonon: Optional[float] = None):
        """Initialize the phonon DOS sonifier.

        Args:
            mp_id: Materials Project ID
            temperatures: List of temperatures (K) for analysis
            duration: Duration of sonification in seconds
            min_audible: Minimum audible frequency (Hz)
            max_audible: Maximum audible frequency (Hz)
            fmin_phonon: Minimum phonon frequency (Hz) for mapping
                         If None, auto-calculated from data
            fmax_phonon: Maximum phonon frequency (Hz) for mapping
                         If None, auto-calculated from data

        Note: If you want to ensure consistent frequency mappings across sonifications,
        then fmin_phonon and fmax_phonon must be specified.
        """
        self.mp_id = mp_id
        self.duration = duration
        self.fmin_audible = fmin_audible
        self.fmax_audible = fmax_audible
        
        # Get DOS data
        print(f"Fetching phonon DOS data for {mp_id}...")
        self.dos_dict = dos_stats_analysis(mp_id, temp=temperatures)
        
        # Extract frequency range across all sites
        all_fmin = []
        all_fmax = []
        for site in self.dos_dict['projection'].keys():
            freqs = self.dos_dict['projection'][site]['frequencies']
            valid_freqs = freqs[freqs > 0]
            if len(valid_freqs) > 0:
                all_fmin.append(np.min(valid_freqs))
                all_fmax.append(np.max(valid_freqs))
        
        if fmin_phonon is None:
            self.fmin_phonon = min(all_fmin)
        else:
            self.fmin_phonon = fmin_phonon
            print(f"Phonon frequency minimum  (user specified): {self.fmin_phonon:.2e} Hz")
        print(f"Phonon frequency minimum  (data extracted): {min(all_fmin):.2e} Hz")
        
        if fmax_phonon is None:
            self.fmax_phonon = max(all_fmax)
        else:
            self.fmax_phonon = fmax_phonon
            print(f"Phonon frequency maximum  (user specified): {self.fmax_phonon:.2e} Hz")
        print(f"Phonon frequency maximum  (data extracted): {min(all_fmax):.2e} Hz")

        if min(all_fmin) < self.fmin_phonon or max(all_fmax) > self.fmax_phonon:
            print("warning: all frequencies outside of the user specified range will be ignored.")
        
    def map_phonon_to_audible(self, phonon_freq_hz: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Map phonon frequency (Hz) to audible frequency (Hz)"""
        is_scalar = isinstance(phonon_freq_hz, (int, float))
        freq_array = np.atleast_1d(phonon_freq_hz)
        
        result = phonon_to_audible_log(
            freq_array,
            self.fmin_phonon,
            self.fmax_phonon,
            self.fmin_audible,
            self.fmax_audible
        )
            
        return float(result[0]) if is_scalar else result
    
    def print_available_sites(self, temperature: Optional[float] = None):
        """Print information about available sites at an optionally specified temperature"""
        print(f"\nAvailable sites in {self.mp_id}:")
        print("="*80)
        
        for site in self.dos_dict['projection'].keys():
            site_data = self.dos_dict['projection'][site]
            
            if temperature is not None and 'thermal' in site_data['stats']:
                if str(temperature) in site_data['stats']['thermal']:
                    stats = site_data['stats']['thermal'][str(temperature)]
                    label = f"{int(temperature)}K"
                else:
                    stats = site_data['stats']['athermal']
                    label = "athermal"
            else:
                stats = site_data['stats']['athermal']
                label = "athermal"
            
            band_centre_hz = stats['band_centre']
            band_centre_note_info = phonon_to_note(band_centre_hz, self.fmin_phonon, self.fmax_phonon)
            q25_hz = stats['quantile_25']
            q25_note_info = phonon_to_note(q25_hz, self.fmin_phonon, self.fmax_phonon)
            q75_hz = stats['quantile_75']
            q75_note_info = phonon_to_note(q75_hz, self.fmin_phonon, self.fmax_phonon)
            
            print(f"\n{site:>15} ({label})")
            print(f"  Band centre:  {band_centre_hz:.2e} Hz → {band_centre_note_info['audible_frequency']:.1f} Hz ({band_centre_note_info['note-octave']})")        
            print(f"  Q25:          {band_centre_hz:.2e} Hz → {q25_note_info['audible_frequency']:.1f} Hz ({q25_note_info['note-octave']})")   
            print(f"  Q75:          {band_centre_hz:.2e} Hz → {q75_note_info['audible_frequency']:.1f} Hz ({q75_note_info['note-octave']})")   
            print(f"  IQR:          {stats['IQR']:.2e} Hz")
            print(f"  Integrated:   {stats['integrated_dos']:.2e}")
        
        print("\n" + "="*80)

    def get_site_stats(self,
                      site_name: str,
                      temperature: Optional[float] = None) -> Dict:

        # Get site data
        if site_name not in self.dos_dict['projection']:
            available = list(self.dos_dict['projection'].keys())
            raise ValueError(f"Site '{site_name}' not found. Available: {available}")
        
        site_data = self.dos_dict['projection'][site_name]
        frequencies_hz = site_data['frequencies']
        densities = site_data['densities']
        
        # Get stats
        if temperature is not None:
            if 'thermal' not in site_data['stats']:
                raise ValueError("No thermal data")
            if str(temperature) not in site_data['stats']['thermal']:
                raise ValueError(f"Temperature {temperature}K not found")
            stats = site_data['stats']['thermal'][str(temperature)]
            
        else:
            stats = site_data['stats']['athermal']

        return stats
        
    def spectraliser(self,
                     site_name: str,
                     temperature: Optional[float] = None,
                     mapping: Optional[Union[str, callable]] = None) -> Sonification:
        """
        Sonify the projected or full density of states using an inverse fast fourier transform.
        
        Args:
            site_name: Site to sonify (e.g., 'Fe_2', 'O_6')
            temperature: Temperature in K (defaults to None for athermal)
            mapping: Mapping to extract spectral features (defaults to None). Pre-defined functions can be specified, 
            or a lambda function can be passed as an argument (not supported by CLI).
        """
    
        score =  Score(STRAUSS_BASE_NOTE, self.duration)
        
        #set up spectralizer generator
        generator = Spectralizer()
        
        # Lets pick the mapping frequency range for the spectrum...
        generator.modify_preset({'min_freq':self.fmin_audible, 
                                 'max_freq':self.fmax_audible, 
                                 'interpolation_type': "preserve_power",
                                 'equal_loudness_normalisation': True })

        stats = self.get_site_stats(site_name, temperature)
        amplitudes = stats['densities']

        if mapping is None:
            pass
    
        elif mapping == "amplify_large_densities":
            amplitudes = amplitudes = [x**2 if x>(0.5*max(amplitudes)) else x for x in amplitudes] # anything in the upper 50% of dos is squared
    
        elif callable(mapping)==True:
            amplitudes = amplitudes * [mapping(x) for x in range(len(amplitudes))]  # apply a lambda mapping 

        else:
            print("Mapping not recognised. Skipping.")
            
        # set up spectrum and choose some envelope parameters for fade-in and fade-out
        data = {'spectrum':[amplitudes], 'pitch':[0], 
                'volume_envelope/D':[0.9], 
                'volume_envelope/S':[0.], 
                'volume_envelope/A':[0.1]}

        f = self.dos_dict['projection'][site_name]['frequencies']
        
        min_spectrum = ((min(f) - self.fmin_phonon)/(self.fmax_phonon-self.fmin_phonon))*100
        max_spectrum = ((max(f) - self.fmin_phonon)/(self.fmax_phonon-self.fmin_phonon))*100

        if min_spectrum < 0:
            min_spectrum = 0
            print("all frequencies below the specified minimum will be truncated")

        if max_spectrum <= 0:
            raise AssertionError ("all frequencies are below the specified minimum")
    
        # specify full range for the parameters
        mlims = {'spectrum': (str(min_spectrum)+"%",str(max_spectrum)+"%")}
        
        # set up source
        sources = Events(data.keys())
        sources.fromdict(data)
        sources.apply_mapping_functions(map_lims=mlims)
        
        # render and play sonification!
        soni = Sonification(score, sources, generator, AUDIO_SYSTEM)

        return soni

    def sonify_site_choral(self,
                           site_name: str,
                           sample_path: Optional[str] = "../data/samples/Solitary_Choir/Samples/",
                           temperature: Optional[float] = None) -> Sonification:
        """
        Sonify a single site using a choral sample.
        
        Args:
            site_name: Site to sonify (e.g., 'Fe_2', 'O_6')
            temperature: Temperature in K (None for athermal)
        """
        if temperature is not None:
            temp_label = f"{int(temperature)}K"

        else:
            temp_label = "athermal"
        
        stats = self.get_site_stats(site_name, temperature)
        
        band_centre_hz = stats['band_centre']
        q25_hz = stats['quantile_25']
        q75_hz = stats['quantile_75']
        iqr_hz = stats['IQR']
        
        print(f"\nSonifying: {site_name} ({temp_label})")
        print(f"  Band centre: {band_centre_hz:.2e} Hz")

        # Calculate volume using integrated dos
        if temperature is not None:
            all_integrated = [self.dos_dict['projection'][s]['stats']['thermal'][str(temperature)]['integrated_dos'] 
                            for s in self.dos_dict['projection'].keys()]
        else:
            all_integrated = [self.dos_dict['projection'][s]['stats']['athermal']['integrated_dos'] 
                            for s in self.dos_dict['projection'].keys()]
        rel_amplitude = stats['integrated_dos'] / max(all_integrated)
        volume = 0.1 + 0.8 * rel_amplitude
        print(f"  Volume: {volume:.2f} (from integrated dos)")

        # Create generator
        generator = Sampler("../data/samples/Solitary_Choir/Samples/")

        generator.modify_preset({'note_length':self.duration,
                                 'volume_envelope': {'use':'off',
                            # A,D,R values in seconds, S sustain fraction from 0-1 that note
                            # will 'decay' to (after time A+D)
                            'A':0.1,    # ✏️ Time to fade in note to maximum volume, using 10 ms
                            'D':0.0,    # ✏️ Time to fall from maximum volume to sustained level (s), irrelevant while S is 1 
                            'S':1.,      # ✏️ fraction of maximum volume to sustain note at while held, 1 implies 100% 
                            'R':.2}}) # ✏️ Time to fade out once note is released, using 100 ms
        
        notes = [["G2","A2","A#2","B2","C3","C#3","D3","D#3","E3","F3","F#3","G3","G#3","A3","A3","B3","C4","C#4","D4","D#4","E4","F4","F#4","G4","G#4","A4","A#4","B4","C5"]]
        
        # Create Score
        score =  Score(notes,self.duration,pitch_binning="uniform")

        # Set data and limits
        data = {'pitch': band_centre_hz,
                'time': [0],
                'volume': volume}
        
        mlims = {'time': [0,self.duration+0.3],
                'pitch': [self.fmin_phonon,self.fmax_phonon],
                'volume':[0,1]}
        
        # set up source
        sources = Objects(data.keys())
        sources.fromdict(data)
        sources.apply_mapping_functions(map_lims=mlims)

        soni = Sonification(score, sources, generator, AUDIO_SYSTEM)
        
        return soni
        
    def sonify_site_synth(self,
                   site_name: str,
                   temperature: Optional[float] = None,
                   use_lfo: bool = False,
                   lfo_target: str = 'pitch') -> Sonification:
        """
        Sonify a single site using a synthesiser.
        
        Args:
            site_name: Site to sonify (e.g., 'Fe_2', 'O_6')
            temperature: Temperature in K (None for athermal)
            use_lfo: Add LFO modulation (vibrato/tremolo from IQR)
            lfo_target: 'pitch' (vibrato) or 'volume' (tremolo)
        """
        if temperature is not None:
            temp_label = f"{int(temperature)}K"

        else:
            temp_label = "athermal"
        
        stats = self.get_site_stats(site_name, temperature)
        
        band_centre_hz = stats['band_centre']
        q25_hz = stats['quantile_25']
        q75_hz = stats['quantile_75']
        iqr_hz = stats['IQR']
        
        print(f"\nSonifying: {site_name} ({temp_label})")
        print(f"  Band centre: {band_centre_hz:.2e} Hz")
        print(f"  IQR: {iqr_hz:.2e} Hz")
        
        # Create Score
        score = Score(STRAUSS_BASE_NOTE, self.duration)
        
        # Create Generator - use complete default
        generator = Synthesizer()
        
        # testing here, but if implemented would have to have options for most elements (implemented as look-up table)
        #if site_name == 'Ca_1':
        #    generator.modify_preset(mods.brassy_mods)
        #elif site_name == 'O_1':
        #    generator.modify_preset(mods.organ_mods)
        #elif site_name == 'C_1':
        #    generator.modify_preset(mods.stringy_mods)
        #else:
        generator.load_preset('pitch_mapper')
        
        # Add LFO if requested
        if use_lfo:
            q25_audio = self.map_phonon_to_audible(q25_hz)
            q75_audio = self.map_phonon_to_audible(q75_hz)
            iqr_audio = self.map_phonon_to_audible(iqr_hz)
            
            # LFO rate from IQR (1-5 Hz)
            lfo_freq = 1.0 + 4.0 * (iqr_audio - self.fmin_audible) / (self.fmax_audible - self.fmin_audible)
            
            # LFO depth from IQR
            depth_factor = iqr_audio / (self.fmax_audible - self.fmin_audible)
            
            if lfo_target == 'pitch':
                lfo_amount = min(depth_factor * 12.0, 3.0)  # Max ±3 semitones
                preset_mods = {
                    'pitch_lfo': {
                        'use': True,
                        'freq': lfo_freq,
                        'amount': lfo_amount,
                        'wave': 'sine'
                    }
                }
                print(f"  Vibrato: {lfo_freq:.2f} Hz, ±{lfo_amount:.2f} semitones (from IQR)")
            else:
                lfo_amount = min(depth_factor * 3.0, 0.8)  # Max 60%
                preset_mods = {
                    'volume_lfo': {
                        'use': True,
                        'freq': lfo_freq,
                        'amount': lfo_amount,
                        'wave': 'sine'
                    }
                }
                print(f"  Tremolo: {lfo_freq:.2f} Hz, ±{lfo_amount:.2%} (from IQR)")
            
            generator.modify_preset(preset_mods)
        
        # Calculate parameters
        # TODO: might want to update this so it is temperature dependent.
        
        if temperature is not None:
            all_integrated = [self.dos_dict['projection'][s]['stats']['thermal'][str(temperature)]['integrated_dos'] 
                            for s in self.dos_dict['projection'].keys()]
        else:
            all_integrated = [self.dos_dict['projection'][s]['stats']['athermal']['integrated_dos'] 
                            for s in self.dos_dict['projection'].keys()]
        rel_amplitude = stats['integrated_dos'] / max(all_integrated)
        volume = 0.1 + 0.8 * rel_amplitude
        print(f"  Volume: {volume:.2f} (from integrated dos)")

        mlims = {
            'pitch_shift': [self.fmin_phonon,self.fmax_phonon],
            'volume': [0,1.0]
        }

        plims={
            'pitch_shift': [0,36],
            'volume': [0,1]
        }
        
        # Create data dict
        datadict = {
            'pitch': 0.,
            'pitch_shift': np.array([band_centre_hz,band_centre_hz]),
            'volume': np.array([volume, volume]),
            'time_evo': np.array([0.0, self.duration])
        }
        
        sources = Objects(datadict.keys())
        sources.fromdict(datadict)
        sources.apply_mapping_functions(map_lims=mlims, param_lims=plims)
        
        # Create Sonification
        soni = Sonification(score, sources, generator, AUDIO_SYSTEM)
        
        return soni


    def sonify_single_site(self,
                          site_name: str,
                          temperature: Optional[float] = None,
                          use_lfo: bool = False,
                          lfo_target: str = 'pitch',
                          output_path: str = None,
                          mapping: str = None,
                          mode: str = 'spectral') -> Sonification:
        """
        Sonify single site
        
        Args:
            site_name: Site name as a string
            temperature: Temperature in K
            output_path: Where to save
        """
        print(f"\n{'='*60}")
        print(f"Single site sonification: {self.mp_id}")
        print(f"Site: {site_name}")
        print(f"Duration: {self.duration}s")
        print(f"Temp: {temperature or 'athermal'}")
        print(f"Mode: {mode}")
        print(f"{'='*60}\n")

        if mode == 'spectral':
            soni = self.spectraliser(
                site_name,
                temperature=temperature,
                mapping=mapping
            )
            
        elif mode == 'synth':
            soni = self.sonify_site_synth(
                site_name,
                temperature=temperature,
                use_lfo=use_lfo,
                lfo_target=lfo_target
            )

        elif mode == 'choral':
            soni = self.sonify_site_choral(
                site_name,
                temperature=temperature
            )
        else: 
            raise AssertionError ("mode not recognised")
                    
        soni.render()

        # Save
        if output_path is None:
            temp_str = f"{int(temperature)}K" if temperature else "athermal"
            lfo_str = lfo_target if use_lfo else None
            mapping_str = mapping if mapping else None
            output = f"phonon_{self.mp_id}_{temperature or 'athermal'}_{mode}_{lfo_str or mapping_str or ''}_{site_name}.wav"
        soni.save(output_path)
        print(f"{'='*60}\n")

        return soni
    
    def sonify_multi_site(self,
                          site_configs: str,
                          temperature: Optional[float] = None,
                          use_lfo: bool = False,
                          lfo_target: str = 'pitch',
                          output_path: str = None,
                          mapping: str = None,
                          mode: str = 'spectral') -> Sonification:
        """
        Sonify multiple sites 
        
        Args:
            site_configs: List of dicts with 'site' key
            temperature: Temperature in K
            output_path: Where to save
        """
        print(f"\n{'='*60}")
        print(f"Multi-site sonification: {self.mp_id}")
        print(f"Duration: {self.duration}s")
        print(f"Sites: {len(site_configs)}")
        print(f"Temp: {temperature or 'athermal'}")
        print(f"Mode: {mode}")
        print(f"{'='*60}\n")

        soni_list = []
        for config in site_configs:
            site = config['site']
            if mode == 'synth':
                soni_list.append(self.sonify_site_synth(site, 
                                                        temperature= temperature,
                                                        use_lfo = use_lfo,
                                                        lfo_target = lfo_target))
            elif mode == 'choral':
                soni_list.append(self.sonify_site_choral(site, 
                                                         temperature= temperature))
            elif mode == 'spectral':
                soni_list.append(self.spectraliser(site, 
                                                   temperature= temperature,
                                                   mapping=mapping))
            else:
                print('mode is not recognised')
                
        for i in range(1,len(soni_list)):
            soni_list[i-1].render()
            soni_list[i].out_channels = soni_list[i-1].out_channels

        joint_soni = soni_list[-1]
        joint_soni.render()

        # Save
        if output_path is None:
            temp_str = f"{int(temperature)}K" if temperature else "athermal"
            lfo_str = lfo_target if use_lfo else None
            mapping_str = mapping if use_lfo else None
            output_path = f"phonon_{self.mp_id}_{temp_str}_{mode}_{lfo_str or mapping_str or ''}_chord.wav"
        
        joint_soni.save(output_path)
        print(f"{'='*60}\n")
        
        return joint_soni
    
    def sonify_all_sites(self,
                          temperature: Optional[float] = None,
                          use_lfo: bool = False,
                          lfo_target: str = 'pitch',
                          output_path: str = None,
                          mapping: str = None,
                          mode: bool = 'spectral') -> Sonification:
        """
        Convenience: sonify all sites
        """
        sites = list(self.dos_dict['projection'].keys())
        sites.remove('all')
        
        site_configs = []
        for site in sites:
            site_configs.append({'site': site})
        
        return self.sonify_multi_site(
            site_configs=site_configs,
            temperature=temperature,
            output_path=output_path,
            use_lfo = use_lfo,
            lfo_target = lfo_target,
            mapping = mapping,
            mode = mode
        )

def example_usage():
    """Example usage"""
    
    sonifier = PhononDOSSonifier(
        mp_id='mp-715',
        temperatures=[300, 600],
        duration=8.0
    )
    
    print("="*60)
    print("Example 1: Show site info")
    print("="*60)
    sonifier.print_available_sites(temperature=300)
    
    print("\n" + "="*60)
    print("Example 2: Single site with vibrato")
    print("="*60)
    soni = sonifier.sonify_site_synth(
        site_name='Fe2+',
        temperature=300,
        use_lfo=True,
        lfo_target='pitch'
    )
    soni.render()
    soni.save("example_fe_vibrato.wav")
    
    print("\n" + "="*60)
    print("Example 3: All sites chord")
    print("="*60)
    sonifier.sonify_all_sites(
        temperature=300,
        output_path="example_all_sites.wav"
    )
    
    print("\n" + "="*60)
    print("Complete!")
    print("  - example_fe_vibrato.wav")
    print("  - example_all_sites.wav")
    print("="*60)


if __name__ == "__main__":
    import argparse

    print("="*60)
    print("Phonon DOS Sonification with STRAUSS")
    print("="*60)
    
    parser = argparse.ArgumentParser(description="Sonify phonon DOS")
    parser.add_argument('mp_id', nargs='?', help='Materials Project ID')
    parser.add_argument('--temp', type=float, default=None, help='Temperature (K)')
    parser.add_argument('--duration', type=float, default=10.0, help='Duration (s)')
    parser.add_argument('--fmin_phonon', type=float, default=None, help='Minimum phonon frequency')
    parser.add_argument('--fmax_phonon', type=float, default=None, help='Maximum phonon frequency')
    parser.add_argument('--all-sites', action='store_true', help='Sonify all sites')
    parser.add_argument('--sites', nargs='+', help='Site names (e.g., Fe2+ O2-)')
    parser.add_argument('--site', type=str, help='Single site to sonify')
    parser.add_argument('--lfo', action='store_true', help='Enable LFO modulation from DOS quantiles')
    parser.add_argument('--lfo-target', choices=['pitch', 'volume'], default='pitch',
                       help='LFO target: pitch=vibrato, volume=tremolo')
    parser.add_argument('--info', action='store_true', help='Show available sites')
    parser.add_argument('--output', type=str, default=None, help='Output file')
    parser.add_argument('--mode',
                    default='spectral',
                    const='spectral',
                    nargs='?',
                    choices=['spectral', 'synth', 'choral'],
                    help='specify sonification mode')
    parser.add_argument('--mapping', type=str, default=None, help='Mapping to apply to DOS spectra')
    parser.add_argument('--examples', action='store_true', help='Run examples')
    
    args = parser.parse_args()

    if args.mapping is not None and args.mode != "spectral":
        parser.error("--mapping can only be used when --mode is 'spectral'")

    if args.sites and len(args.sites) == 1:
        args.site = args.sites[0]
        args.sites = None
    
    try:
        if args.examples:
            example_usage()
        
        elif args.mp_id:
            temps = [args.temp] if type(args.temp) is float else None
            
            sonifier = PhononDOSSonifier(
                mp_id=args.mp_id,
                temperatures=temps,
                duration=args.duration,
                fmin_phonon=args.fmin_phonon,
                fmax_phonon=args.fmax_phonon
            )
            
            if args.info:
                sonifier.print_available_sites(temperature=args.temp)
            
            elif args.all_sites:
                sonifier.sonify_all_sites(
                    temperature=args.temp,
                    output_path=args.output,
                    use_lfo=args.lfo,
                    lfo_target=args.lfo_target,
                    mapping=args.mapping,
                    mode=args.mode
                )                
            
            elif args.site:
                sonifier.sonify_single_site(
                    site_name=args.site,
                    temperature=args.temp,
                    output_path=args.output,
                    use_lfo=args.lfo,
                    lfo_target=args.lfo_target,
                    mapping=args.mapping,
                    mode=args.mode
                ) 

            elif args.sites:
                site_configs = []
                for site_spec in args.sites:
                    site_configs.append({'site': site_spec})
                
                sonifier.sonify_multi_site(
                    site_configs=site_configs,
                    temperature=args.temp,
                    output_path=args.output,
                    use_lfo=args.lfo,
                    lfo_target=args.lfo_target,
                    mapping=args.mapping,
                    mode=args.mode
                )
            
            else:
                print("\nError: Specify --info, --all-sites, --site, or --sites")
                print("\nQuick examples:")
                print(f"  python phonon_dos_sonifier.py mp-3953 --info --temp 300")
                print(f"  python phonon_dos_sonifier.py mp-3953 --temp 50 --all-sites")
                print(f"  python phonon_dos_sonifier.py mp-3953 --site O_6 --lfo")
                print(f"  python phonon_dos_sonifier.py mp-3953 --temp 300 --sites O_6 Ca_1")
        
        else:
            print("\nError: Provide mp_id or use --examples")
            parser.print_help()
    
    except ImportError as e:
        print(f"\nImportError: {e}")
        print("Install STRAUSS: pip install 'strauss[default]'")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


### find good test cases from "Rock music": soft, hard, complex, simple, bonding, "rattler", temperature --> create notebook to showcase, and generally test!

### specify material 
### specify temperature or athermal

### sounds (can select one or more):
### - spectral
### - synth
### - choral

### modes (select one only)
### - chord (play full dos data)
### - broken chord (play each projected dos with annotation of atom type)
### - composition (randomly create a three minute composition using chords and unannotated broken chords)