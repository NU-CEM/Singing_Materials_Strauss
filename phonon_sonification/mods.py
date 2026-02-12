stringy_mods = { 
          'oscillators': {
                'osc1': {
                    'form': 'saw',
                    'level': 1.,
                    'detune': 0.,
                    'phase': 'random',
                }
            },

        'volume_envelope': {
           'A': 0.03,    # Quick attack
           'D': 0.2,     # Moderate decay
           'S': 0.8,     # High sustain
           'R': 0.4,     # Gradual release
           'Ac': 0,      # Linear attack
           'Dc': 0.3,    # Slightly curved decay
           'Rc': 0.2,    # Slightly curved release
           'level': 1.0 }
        }

brassy_mods = {
        'oscillators': {
            'osc1': {
                'form': 'square',
                'level': 1.,
                'detune': 0.,
                'phase': 'random',
            }
        },
        'volume_envelope': {
            'A': 0.08,    # Slower attack (breath pressure)
            'D': 0.15,    # Quick decay
            'S': 0.9,     # Very high sustain
            'R': 0.2,     # Quick release
            'Ac': -0.3,   # Convex attack (accelerating)
            'Dc': 0,      # Linear decay
            'Rc': 0,      # Linear release
            'level': 1.0
}}

windy_mods = {
        'oscillators': {
            'osc1': {
                'form': 'tri',
                'level': 1.,
                'detune': 0.,
                'phase': 'random',
                            }
                        },

        'volume_envelope': {
            'A': 0.05,    # Moderate attack
            'D': 0.1,     # Quick decay
            'S': 0.7,     # Medium sustain
            'R': 0.3,     # Moderate release
            'Ac': 0.2,    # Concave attack (slowing)
            'Dc': 0.3,    # Curved decay
            'Rc': 0.3,    # Curved release
            'level': 1.0
        },
                }


organ_mods = {
'filter': 'off',  # Pure tone, no filtering
'volume_envelope': {
        'A': 0.01,    # Instant attack
        'D': 0.01,    # Minimal decay
        'S': 1.0,     # Full sustain
        'R': 0.08,    # Quick cutoff
        'Ac': 0,      # Linear
        'Dc': 0,      # Linear
        'Rc': 0,      # Linear
        'level': 1.0
        },
'oscillators': {
    'osc1': {
        'form': 'sine',
                'level': 1.,
                'detune': 0.,
                'phase': 'random',
                            }
                        },

        }