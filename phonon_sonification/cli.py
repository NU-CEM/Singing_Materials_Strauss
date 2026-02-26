
import argparse
from phonon_dos_sonifier import PhononDOSSonifier

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
