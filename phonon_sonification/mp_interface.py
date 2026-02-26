import numpy as np
import pandas as pd
import math
import scipy
from scipy import constants
import os
from dotenv import load_dotenv

load_dotenv () # use python-dotenv library for storing secrets in a .env file in project route (or at another path that is specified here)

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

def gamma_frequencies_from_mp_id(mp_id):
    """return phonon frequencies (in Hz) at gamma point from for a material hosted on the Materials Project.
    Material is identified using unique ID number. Note that to use this feature you need a Materials
    Project API key (https://materialsproject.org/api)."""
    import mp_api
    from mp_api.client import MPRester

    with MPRester(os.getenv('MP_API_KEY')) as mpr:
        try:
            bs = mpr.get_phonon_bandstructure_by_material_id(mp_id)
        except:
            print("this materials project entry does not appear to have phonon data")
            pass
    print("extracting frequencies for qpoint {}".format(bs.qpoints[0].cart_coords))

    phonon_frequencies = list(bs.to_pmg.bands[:,0*1E12])   # convert from THz to Hz
    phonon_frequencies = process_imaginary(phonon_frequencies)
    print("phonon frequencies are (Hz):", phonon_frequencies)

    return phonon_frequencies

def dos_data_from_mp_id(mp_id):
    """return dos data obect. This is for a material hosted on the Materials Project.
    Material is identified using unique ID number. Note that to use this feature you need a Materials
    Project API key (https://materialsproject.org/api)."""
    import mp_api
    from mp_api.client import MPRester
    with MPRester(os.getenv('MP_API_KEY')) as mpr:

        try: 
            dos = mpr.get_phonon_dos_by_material_id(mp_id)
        except:
            print("this materials project entry does not appear to have phonon data")
            pass

    return dos

def get_dos_raw(mp_id):
    """get the full and projected densities. Return as a nested dictionary. Arg is the materials project ID. """
    dos = dos_data_from_mp_id(mp_id)

    dos_dict = {}
    dos_dict['metadata'] = {'mp_id' : mp_id}
    dos_dict['projection'] = {}

    frequencies = np.array(dos.frequencies)*1E12 # convert from THz to Hz
    frequencies = process_imaginary(frequencies)  
    
    densities = process_imaginary_dos(dos.densities,frequencies) 
    dos_dict['projection']['all'] = {'densities': densities,
                                     'frequencies': frequencies}
            
    for i,site in enumerate(dos.structure.relabel_sites().sites):
        densities = process_imaginary_dos(dos.projected_densities[i],frequencies) 
        dos_dict['projection'][site.label] = {'densities': densities,
                                                        'frequencies': frequencies} 
    return dos_dict
    
def dos_stats_analysis(mp_id,temp=None):
    """for each entry in a dos_dict, calculate the integrated dos, the phonon band centre, the quantiles and the IQR and of the dos distribution in Hz (discounting any negative frequencies) and add these to the nested dicts. The dos is optionally weighted by Bose Einstein occupation at a specified temp."""
    if temp and 0 in temp:
        print ("cannot calculate stats for 0K")
        temp.remove(0)
    dos_dict = get_dos_raw(mp_id)
    for site in dos_dict['projection']:
        site_dict = dos_dict['projection'][site]
        site_dict['stats'] = {}
        site_dict['stats']['athermal'] = {}
        densities = site_dict['densities']
        f = site_dict['frequencies']
        site_dict['stats']['athermal']['band_centre'] = phonon_band_centre(f,densities)
        site_dict['stats']['athermal']['integrated_dos'] = integrated_dos(f,densities)
        site_dict['stats']['athermal']['quantile_25'] = weighted_quantile(f, densities, 0.25)
        site_dict['stats']['athermal']['quantile_75'] = weighted_quantile(f, densities, 0.75)
        site_dict['stats']['athermal']['IQR'] = phonon_dos_IQR(f,densities)
        site_dict['stats']['athermal']['densities'] = densities
        
        if temp:
            site_dict['stats']['thermal'] = {}
            if not hasattr(temp, '__iter__'):
                temp = [temp]    # if single temp provided as scalar, convert it to iterable list
            for t in temp:
                densities_scaled = scale_by_occupation(densities, f, t)
                site_dict['stats']['thermal'][str(t)] = {}
                site_dict['stats']['thermal'][str(t)]['band_centre'] = phonon_band_centre(f,densities_scaled)
                site_dict['stats']['thermal'][str(t)]['integrated_dos'] = integrated_dos(f,densities_scaled)
                site_dict['stats']['thermal'][str(t)]['quantile_25'] = weighted_quantile(f, densities_scaled, 0.25)
                site_dict['stats']['thermal'][str(t)]['quantile_75'] = weighted_quantile(f, densities_scaled, 0.75)
                site_dict['stats']['thermal'][str(t)]['IQR'] = phonon_dos_IQR(f,densities_scaled)
                site_dict['stats']['thermal'][str(t)]['densities'] = densities_scaled

    return dos_dict

def dos_dict_to_dataframe(dos_dict):
    """convert the nested dictionary to a pandas dataframe"""
    rows = []

    mp_id = dos_dict["metadata"]["mp_id"]

    for site, site_data in dos_dict["projection"].items():
        stats = site_data["stats"]

        # athermal row
        row_base = {
            "mp_id": mp_id,
            "site": site,
            "temperature": None
        }

        for key, value in stats["athermal"].items():
            row_base[key] = value

        rows.append(row_base)

        # thermal rows
        if "thermal" in stats:
            for T, values in stats["thermal"].items():
                row = {
                    "mp_id": mp_id,
                    "site": site,
                    "temperature": float(T)
                }
                row.update(values)
                rows.append(row)

    return pd.DataFrame(rows)

def animate_dos_vs_temperature(dos_dict, site_order=None, interval=500):
    from matplotlib.animation import FuncAnimation
    """
    Create matplotlib animation showing DOS projections and band centres vs temperature.
    
    interval : ms between frames
    """

    projections = dos_dict["projection"]
    mp_id = dos_dict["metadata"]["mp_id"]

    # Extract temperatures (as sorted floats)
    temps = list(next(iter(projections.values()))["stats"]["thermal"].keys())
    temps = sorted(int(t) for t in temps)

    # Decide plotting order
    if site_order is None:
        site_order = list(projections.keys())

    # Consistent colours
    cmap = plt.get_cmap("tab10")
    colours = {site: cmap(i % 10) for i, site in enumerate(site_order)}

    fig, ax = plt.subplots(figsize=(7,4))
    ax.set_title(f"Phonon DOS vs Temperature ({mp_id})")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("DOS")

    ax.set_autoscale_on(False)

    lines = {}
    centre_lines = {}
    q25_lines = {}
    q75_lines = {}


    # Initialise empty lines
    for site in site_order:
        f = projections[site]["frequencies"]

        (line,) = ax.plot(f, np.zeros_like(f), 
                          color=colours[site], 
                          label=site)
        lines[site] = line

        centre = ax.axvline(0, 
                            color=colours[site], 
                            linestyle="--", 
                            alpha=0.7)
        centre_lines[site] = centre

        q25 = ax.axvline(0,
                         color=colours[site],
                         linestyle=":",
                         alpha=0.3,    
                         linewidth=1.0)
        q75 = ax.axvline(0,
                         color=colours[site],
                         linestyle=":",
                         alpha=0.3,
                         linewidth=1.0)
    
        q25_lines[site] = q25
        q75_lines[site] = q75

    ax.legend()

    # Fix axis limits for stability
    all_f = projections[site_order[0]]["frequencies"]
    ax.set_xlim(all_f.min(), all_f.max())

    ax.set_ylim(0, 1.1 * max(
    (projections[s]["densities"] / projections[s]["densities"].sum()).max()
    for s in site_order))

    def update(frame):
        T = temps[frame]

        ax.set_title(f"Phonon DOS at T = {int(T)} K ({mp_id})")

        for site in site_order:
            f = projections[site]["frequencies"]

            # reconstruct weighted DOS visually
            weighted = scale_by_occupation(
                projections[site]["densities"], f, T
            )
            weighted = weighted / weighted.sum()
            
            lines[site].set_ydata(weighted)

            centre = projections[site]["stats"]["thermal"][str(T)]["band_centre"]
            centre_lines[site].set_xdata([centre, centre])

            q25 = projections[site]["stats"]["thermal"][str(T)]["quantile_25"]
            q75 = projections[site]["stats"]["thermal"][str(T)]["quantile_75"]
            
            q25_lines[site].set_xdata([q25, q25])
            q75_lines[site].set_xdata([q75, q75])
                    
        return (
                list(lines.values())
                + list(centre_lines.values())
                + list(q25_lines.values())
                + list(q75_lines.values())
            )

    anim = FuncAnimation(
        fig,
        update,
        frames=len(temps),
        interval=interval,
        blit=True
    )

    return anim

def plot_dos_at_temperature(dos_dict, T=None, site_order=None):
    """
    Plot projected DOS and band statistics at a single temperature.
    
    T=None → athermal DOS
    """

    projections = dos_dict["projection"]
    mp_id = dos_dict["metadata"]["mp_id"]

    if site_order is None:
        site_order = list(projections.keys())

    cmap = plt.get_cmap("tab10")
    colours = {site: cmap(i % 10) for i, site in enumerate(site_order)}

    fig, ax = plt.subplots(figsize=(7,4))
    ax.set_title(
        f"Phonon DOS {'(athermal)' if T is None else f'at T = {int(T)} K'} — {mp_id}"
    )
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("DOS")

    ax.set_autoscale_on(False)

    lines = {}

    for site in site_order:
        f = projections[site]["frequencies"]

        if T is None:
            weighted = projections[site]["densities"]
        else:
            weighted = scale_by_occupation(
                projections[site]["densities"], f, T
            )

        weighted = weighted / weighted.sum()

        ax.plot(
            f,
            weighted,
            color=colours[site],
            label=site
        )

        # Band centre
        if T is None:
            centre = projections[site]["stats"]["athermal"]["band_centre"]
            q25 = projections[site]["stats"]["athermal"]["quantile_25"]
            q75 = projections[site]["stats"]["athermal"]["quantile_75"]
        else:
            centre = projections[site]["stats"]["thermal"][str(T)]["band_centre"]
            q25 = projections[site]["stats"]["thermal"][str(T)]["quantile_25"]
            q75 = projections[site]["stats"]["thermal"][str(T)]["quantile_75"]

        ax.axvline(
            centre,
            color=colours[site],
            linestyle="--",
            linewidth=2,
            alpha=0.9
        )

        ax.axvline(
            q25,
            color=colours[site],
            linestyle=":",
            alpha=0.3
        )
        ax.axvline(
            q75,
            color=colours[site],
            linestyle=":",
            alpha=0.3
        )

    # Fixed y-scale
    ymax = max(
        (projections[s]["densities"] / projections[s]["densities"].sum()).max()
        for s in site_order
    )
    ax.set_ylim(0, ymax * 1.2)

    ax.set_xlim(
        projections[site_order[0]]["frequencies"].min(),
        projections[site_order[0]]["frequencies"].max()
    )

    ax.legend()
    plt.show()
    
def phonon_band_centre(f,dos):
    """for each particular chemical species, get the phonon band centre in Hz (discounting any negative frequencies). Return as a dictionary with species string as key."""
    return scipy.integrate.simpson(dos*f,f)/scipy.integrate.simpson(dos,f)
    
def integrated_dos(f,dos):
    """for each particular chemical species, get the integrated dos (discounting any negative frequencies). Return as a dictionary with species string as key."""
    return scipy.integrate.simpson(dos,f)

def phonon_dos_IQR(f,dos):
    """return the inter-quartile range of the dos distribution"""    
    return weighted_quantile(f, dos, 0.75) - weighted_quantile(f, dos, 0.25)
    
def bose_einstien_distribution(energy,temperature):
    return 1 / (math.exp(energy/(constants.Boltzmann*temperature)) - 1)

def frequency_to_energy(frequency):
    """convert frequency in Hz to energy in joules"""
    energy = constants.h*frequency
    return energy

def scale_by_occupation(densities, phonon_frequencies, temperature):
    """Re-scale the density by the Bose Einstein occupation factor for a state with energy h*frequency at specified temperature."""
    BE_occupations = np.array([bose_einstien_distribution(frequency_to_energy(frequency),temperature) 
                           for frequency in phonon_frequencies])

    # scale by occupation expressed as a fraction of the total occupation
    scaled_densities = densities*BE_occupations

    return scaled_densities

def weighted_quantile(values, weights, quantile):
    """
    Compute a weighted quantile of discrete data.
    
    values   : array-like data points
    weights  : array-like weights (same length as values)
    quantile : float in [0,1] (e.g. 0.25, 0.75)
    """
    values = np.asarray(values)
    weights = np.asarray(weights)

    # sort by values
    sorter = np.argsort(values)
    values = values[sorter]
    weights = weights[sorter]

    # cumulative distribution
    cum_weights = np.cumsum(weights)
    cum_weights /= cum_weights[-1]

    return np.interp(quantile, cum_weights, values)

