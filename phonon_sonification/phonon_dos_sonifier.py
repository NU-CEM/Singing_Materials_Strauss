"""
Phonon DOS Sonification using STRAUSS.

This script sonifies phonon DOS data from Materials Project.

Mappings:
- Phonon band centre → pitch_shift (semitones from C4)
- Projected DOS → volume (amplitude)
- DOS quantiles (Q25, Q75) → LFO rate and depth
  - Q25 position controls LFO frequency (1-5 Hz)
  - IQR (Q75-Q25) controls LFO modulation depth

Features:
- Single site sonification with optional vibrato/tremolo
- Multi-site chord sonification
- Temperature-dependent DOS weighting
- Uses STRAUSS default synthesizer (detuned sawtooth)

Requires: MP_interface.py, phonon_frequency_mapping.py, MP_API_KEY in .env
"""

import numpy as np
from strauss.sonification import Sonification
from strauss.sources import Events, Objects
from strauss.score import Score
try:
    from strauss.generator import Synthesizer, Sampler
except ImportError:
    from strauss.generator import Synthesizer, Sampler

from typing import Dict, List, Tuple, Optional, Union
import warnings

# Import your existing modules
from MP_interface import dos_stats_analysis, scale_by_occupation
from utilities import format_duration_for_strauss
from phonon_frequency_mapping import phonon_to_audible_log, phonon_to_note

# STRAUSS Score requires a chord or note sequence. I don't really understand this.
STRAUSS_BASE_NOTE = [["G3"]]


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
        if fmin_phonon is None or fmax_phonon is None:
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
            
            if fmax_phonon is None:
                self.fmax_phonon = max(all_fmax)
            else:
                self.fmax_phonon = fmax_phonon
        else:
            self.fmin_phonon = fmin_phonon
            self.fmax_phonon = fmax_phonon

        print(f"  Phonon range (all sites): {self.fmin_phonon:.2e} - {self.fmax_phonon:.2e} Hz")
    
    def phonon_to_pitch_shift(self, phonon_freq_hz: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert phonon frequency to pitch_shift in semitones from base note, always positive.

        TODO: update this so it does it for any base note."""
        is_scalar = isinstance(phonon_freq_hz, (int, float))
        freq_array = np.atleast_1d(phonon_freq_hz)
        
        # Map phonon to audible
        audible = phonon_to_audible_log(
            freq_array,
            self.fmin_phonon,
            self.fmax_phonon,
            self.fmin_audible,
            self.fmax_audible
        )
        
        # Convert to semitones from G3_freq
        G3_freq = 196.0
        semitones = 12 * np.log2(audible / G3_freq)
        
        # Calculate offset to make all values positive
        # Find what the minimum audible frequency maps to
        min_semitones = 12 * np.log2(self.fmin_audible / G3_freq)
        
        # Add offset so minimum is at 0
        semitones_positive = semitones - min_semitones
        
        return float(semitones_positive[0]) if is_scalar else semitones_positive
        
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
        """Print information about available sites"""
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
            band_centre_audio = self.map_phonon_to_audible(band_centre_hz)
            note_info = phonon_to_note(band_centre_hz, self.fmin_phonon, self.fmax_phonon)
            
            print(f"\n{site:>15} ({label})")
            print(f"  Band centre:  {band_centre_hz:.2e} Hz → {band_centre_audio:.1f} Hz ({note_info['note-octave']})")
            print(f"  Integrated:   {stats['integrated_dos']:.2e}")
            print(f"  Q25:         {stats['quantile_25']:.2e} Hz")
            print(f"  Q75:         {stats['quantile_75']:.2e} Hz")
            print(f"  IQR:         {stats['IQR']:.2e} Hz")
        
        print("\n" + "="*80)
    
    def sonify_site(self,
                   site_name: str,
                   temperature: Optional[float] = None,
                   use_lfo: bool = False,
                   lfo_target: str = 'pitch') -> Sonification:
        """
        Sonify a single site
        
        Args:
            site_name: Site to sonify (e.g., 'Fe_2', 'O_6')
            temperature: Temperature in K (None for athermal)
            use_lfo: Add LFO modulation (vibrato/tremolo from IQR)
            lfo_target: 'pitch' (vibrato) or 'volume' (tremolo)
        """
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
            temp_label = f"{int(temperature)}K"
        else:
            stats = site_data['stats']['athermal']
            temp_label = "athermal"
        
        band_centre_hz = stats['band_centre']
        q25_hz = stats['quantile_25']
        q75_hz = stats['quantile_75']
        iqr_hz = stats['IQR']

        pitch_shift_semitones = self.phonon_to_pitch_shift(band_centre_hz)
        
        print(f"\nSonifying: {site_name} ({temp_label})")
        print(f"  Band centre: {band_centre_hz:.2e} Hz → {pitch_shift_semitones:.1f} semitones")
        print(f"  IQR: {iqr_hz:.2e} Hz")
        
        # Create Score
        score = Score(STRAUSS_BASE_NOTE, format_duration_for_strauss(self.duration))
        
        # Create Generator - use complete default
        generator = Synthesizer()
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
        
        all_integrated = [self.dos_dict['projection'][s]['stats']['athermal']['integrated_dos'] 
                         for s in self.dos_dict['projection'].keys()]
        rel_amplitude = stats['integrated_dos'] / max(all_integrated)
        volume = 0.4 + 0.5 * rel_amplitude

        mlims = {
            'pitch_shift': [self.fmin_phonon,self.fmax_phonon]
        }

        plims={
            'pitch_shift': [0,36]
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
        audio_system = "stereo"
        soni = Sonification(score, sources, generator, audio_system)
        
        return soni
    
    def sonify_multi_site(self,
                         site_configs: List[Dict],
                         temperature: Optional[float] = None,
                         output_path: Optional[str] = None) -> Sonification:
        """
        Sonify multiple sites as a chord
        
        Args:
            site_configs: List of dicts with 'site' key
            temperature: Temperature in K
            output_path: Where to save
        """
        print(f"\n{'='*60}")
        print(f"Multi-site chord: {self.mp_id}")
        print(f"Duration: {self.duration}s, Temp: {temperature or 'athermal'}")
        print(f"Sites: {len(site_configs)}")
        print(f"{'='*60}\n")
        
        # Create Score
        score = Score(STRAUSS_BASE_NOTE, format_duration_for_strauss(self.duration))
        
        # Create generator with default preset
        generator = Synthesizer()
        generator.load_preset('pitch_mapper')
        
        # Create Objects source for all sites
        sources = Objects(['pitch', 'pitch_shift', 'volume', 'time_evo'])
        
        # Collect data for all sites
        pitches = []
        pitch_shifts = []
        volumes = []
        time_evos = []
        
        for config in site_configs:
            site = config['site']
            print(f"Adding: {site}")
            
            if site not in self.dos_dict['projection']:
                print(f"  Warning: {site} not found, skipping")
                continue
            
            site_data = self.dos_dict['projection'][site]
            
            # Get stats
            if temperature is not None:
                if 'thermal' in site_data['stats'] and str(temperature) in site_data['stats']['thermal']:
                    stats = site_data['stats']['thermal'][str(temperature)]
                else:
                    stats = site_data['stats']['athermal']
            else:
                stats = site_data['stats']['athermal']
            
            band_centre_hz = stats['band_centre']
            
            all_integrated = [self.dos_dict['projection'][s]['stats']['athermal']['integrated_dos'] 
                             for s in self.dos_dict['projection'].keys()]
            rel_amp = stats['integrated_dos'] / max(all_integrated)
            volume = 0.3 + 0.6 * rel_amp
            
            # Add to lists
            pitches.append(0.)
            pitch_shifts.append(band_centre_hz)
            volumes.append(volume)
            time_evos.append(np.array([0.0, self.duration]))
        
        # Create datadict
        datadict = {
            'pitch': pitches,
            'pitch_shift': pitch_shifts,
            'volume': volumes,
            'time_evo': time_evos
        }
        
        mlims = {
            'pitch_shift': [self.fmin_phonon,self.fmax_phonon]
        }

        plims={
            'pitch_shift': [0,36]
        }

        sources.fromdict(datadict)
        sources.apply_mapping_functions(map_lims=mlims, param_lims=plims)
        
        # Create Sonification
        print(f"\n{'='*60}")
        print("Rendering...")
        audio_system = "stereo"
        soni = Sonification(score, sources, generator, audio_system)
        soni.render()
        
        # Save
        if output_path is None:
            temp_str = f"{int(temperature)}K" if temperature else "athermal"
            output_path = f"phonon_{self.mp_id}_{temp_str}_chord.wav"
        
        soni.save(output_path)
        print(f"Saved: {output_path}")
        print(f"{'='*60}\n")
        
        return soni
    
    def sonify_all_sites(self,
                        temperature: Optional[float] = None,
                        output_path: Optional[str] = None) -> Sonification:
        """
        Convenience: sonify all sites
        """
        sites = list(self.dos_dict['projection'].keys())
        
        site_configs = []
        for site in sites:
            site_configs.append({'site': site})
        
        return self.sonify_multi_site(
            site_configs=site_configs,
            temperature=temperature,
            output_path=output_path
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
    soni = sonifier.sonify_site(
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
    
    print("Phonon DOS Sonification with STRAUSS")
    print("="*60)
    
    parser = argparse.ArgumentParser(description="Sonify phonon DOS")
    parser.add_argument('mp_id', nargs='?', help='Materials Project ID')
    parser.add_argument('--temp', type=float, default=None, help='Temperature (K)')
    parser.add_argument('--temps', type=float, nargs='+', help='Multiple temperatures')
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
    parser.add_argument('--examples', action='store_true', help='Run examples')
    
    args = parser.parse_args()
    
    try:
        if args.examples:
            example_usage()
        
        elif args.mp_id:
            temps = args.temps if args.temps else ([args.temp] if args.temp else None)
            
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
                    output_path=args.output
                )
            
            elif args.site:
                soni = sonifier.sonify_site(
                    site_name=args.site,
                    temperature=args.temp,
                    use_lfo=args.lfo,
                    lfo_target=args.lfo_target
                )
                
                output = args.output or f"phonon_{args.mp_id}_{args.site}_{args.temp or 'athermal'}.wav"
                soni.render()
                soni.save(output)
                print(f"\nSaved: {output}")
            
            elif args.sites:
                site_configs = []
                for site_spec in args.sites:
                    site_configs.append({'site': site_spec})
                
                sonifier.sonify_multi_site(
                    site_configs=site_configs,
                    temperature=args.temp,
                    output_path=args.output
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
