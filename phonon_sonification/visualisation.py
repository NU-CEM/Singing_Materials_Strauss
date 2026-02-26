import matplotlib
import matplotlib.pyplot as plt

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
