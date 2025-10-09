import seaborn as sns
import matplotlib.pyplot as plt
import rasterio as rs
import hickle as hkl
import numpy as np
import pickle
import pandas as pd

def hist_compare_s2_byband(location: str, 
                           tile_idx_a: tuple, 
                           tile_idx_b: tuple,  
                           tile_idx_c: tuple,
                           title:str,
                           color_dict: dict,
                           output_file: str = None
                           ):
    '''
    Each s2 band is plotted as it's own hist.
    Parameters:
    - location (str): The country.
    - tile_idx_a (tuple): The coordinates (x, y) of the first tile.
    - tile_idx_b (tuple): The coordinates (x, y) of the second tile.
    - tile_idx_c (tuple): The coordinates (x, y) of the third tile.
    - title (str): The base title for the histograms, describing the area.
    
    '''
    def load_ard(tile_idx):
        x, y = tile_idx
        ard = hkl.load(f'../../tmp/{location}/{str(x)}/{str(y)}/ard/{str(x)}X{str(y)}Y_ard.hkl')[..., 0:10]
        return ard

    # Load data for each tile
    s2_a = load_ard(tile_idx_a).flatten()
    s2_b = load_ard(tile_idx_b).flatten()
    s2_c = load_ard(tile_idx_c).flatten()

    # Determine common axis limits
    binwidth = 0.01
    global_min = min(s2_a.min(), s2_b.min(), s2_c.min())
    global_max = max(s2_a.max(), s2_b.max(), s2_c.max())
    bins = np.arange(global_min, global_max + binwidth, binwidth)

    plt.figure(figsize=(20,20))
    band_counter = 0
    
    for i in range(1, 11):
        plt.subplot(4,3,i)
        plt.hist(s2_a[..., band_counter].flatten(), alpha=0.5, label=str(tile_idx_a), edgecolor="black", bins=bins)
        plt.hist(s2_b[..., band_counter].flatten(), alpha=0.3, label=str(tile_idx_b), edgecolor="black", bins=bins)
        plt.hist(s2_c[..., band_counter].flatten(), alpha=0.3, label=str(tile_idx_c), edgecolor="black", bins=bins)
        
        plt.xlim(0.0, 0.5)
        plt.xticks(np.arange(0.0, 0.5, 0.1))
        plt.title(title + f' Band {str(band_counter)}')
        #plt.legend();

        band_counter += 1

    return None

def hist_individual_tile(
    location: str,
    tile_idx_a: tuple,
    tile_idx_b: tuple,
    tile_idx_c: tuple,
    tile_idx_d: tuple,
    title: str,
    color_dict: dict = {'monoculture': '#fe8266',
                  "agroforestry (cocoa)" : '#4dc348',
                  'agroforestry (shea)' : "#83e87a",
                  'natural': '#1c5718'},
    output_file: str = None,
    font=14
):
    '''
    Create histograms of Sentinel-2 data for four tiles.
    Imports the ARD and looks at 10 Sentinel-2 indices.
    Flattens the entire array.

    Parameters:
    - location (str): The country.
    - tile_idx_[a-d] (tuple): The (x, y) coordinates of each tile.
    - title (str): Title for the figure.
    - color_dict (dict): Mapping of land use class to color.
    - output_file (str): Optional path to save the figure.
    '''
    def load_ard(tile_idx):
        x, y = tile_idx
        ard = hkl.load(f'../../tmp/{location}/{x}/{y}/ard/{x}X{y}Y_ard.hkl')[..., 0:10]
        return ard

    # Load Sentinel-2 data for each tile
    s2_tiles = [load_ard(idx).flatten() for idx in [tile_idx_a, tile_idx_b, tile_idx_c, tile_idx_d]]
    systems = ['monoculture', 'agroforestry (cocoa)', 'agroforestry (shea)', 'natural']
    tile_indices = [tile_idx_a, tile_idx_b, tile_idx_c, tile_idx_d]

    # Calculate global bin range
    binwidth = 0.01
    global_min = min(data.min() for data in s2_tiles)
    global_max = max(data.max() for data in s2_tiles)
    bins = np.arange(global_min, global_max + binwidth, binwidth)

    xlim = (0.0, 0.6)
    xticks = np.arange(0.0, 0.6, 0.1)

    # Create 1x4 subplot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)

    # then flatten or index axes in the order you want:
    axes = axes.flatten()

    # define the order explicitly: first monoculture, then natural, then the two agroforestry
    order = [tile_idx_a, tile_idx_d, tile_idx_b, tile_idx_c]
    order_data = [load_ard(idx).flatten() for idx in order]
    order_systems = ['monoculture', 'natural', 'agroforestry (cocoa)', 'agroforestry (shea)']

    for ax, tile_idx, data, sys in zip(axes, order, order_data, order_systems):
        ax.hist(
            data,
            alpha=0.6,
            label=str(tile_idx),
            edgecolor="black",
            bins=bins,
            color=color_dict.get(sys, "#cccccc")
        )
        ax.set_xlim(xlim)
        ax.set_xticks(xticks)
        ax.set_title(f"{sys.capitalize()} System", fontsize=font)
        ax.set_xlabel("Reflectance Value", fontsize=font)
        ax.set_ylabel("Pixel Frequency", fontsize=font)

    #fig.suptitle(title)
    plt.tight_layout()

    # Add legend to the figure
    # handles = [plt.Line2D([0], [0], color=color, lw=4) for color in colors]
    # labels = [str(tile_idx) for tile_idx in tile_indices]
    # fig.legend(handles, 
    #            labels, 
    #            loc='upper right', 
    #            title="Tiles")
    # plt.legend(title="System", loc="upper left", bbox_to_anchor=(1.05, 1))

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')

    plt.show()

    return None

def hist_compare_s2(location: str, tile_idx_a: tuple, tile_idx_b: tuple, title:str, tile_idx_c: tuple = None):
    '''
    Type: Matplotlib histogram
    Purpose: Compare and visualize histograms of Sentinel-2 data for two specific tiles in a given location.

    Parameters:
    - location (str): The country.
    - tile_idx_a (tuple): The coordinates (x, y) of the first tile to compare.
    - tile_idx_b (tuple): The coordinates (x, y) of the second tile to compare.
    - title (str): The title for the histogram comparison plot, describing the area.

    This function loads analysis ready data for two tiles specified by their coordinates (tile_idx_a and tile_idx_b). 
    It then indexes the array to creates histograms of Sentinel-2 data and displays 
    them in a single plot for visual comparison.

    Returns:
    None
    
    '''
    x_a, y_a = tile_idx_a[0], tile_idx_a[1]
    x_b, y_b = tile_idx_b[0], tile_idx_b[1]
    ard_a = hkl.load(f'../../tmp/{location}/{str(x_a)}/{str(y_a)}/ard/{str(x_a)}X{str(y_a)}Y_ard.hkl')
    ard_b = hkl.load(f'../../tmp/{location}/{str(x_b)}/{str(y_b)}/ard/{str(x_b)}X{str(y_b)}Y_ard.hkl')
    
    s2_a = ard_a[..., 0:10]
    s2_b = ard_b[..., 0:10]
    
    plt.figure(figsize=(6,4))
    binwidth = .01
    min = s2_a.min()
    max = s2_a.max()

    # this asks for 33 binds between .01 and .6 -- np.arange(min, max + binwidth, binwidth)
    plt.hist(s2_a.flatten(), alpha=0.5, label=str(tile_idx_a), edgecolor="black", bins=np.arange(min, max + binwidth, binwidth))
    plt.hist(s2_b.flatten(), alpha=0.3, label=str(tile_idx_b), edgecolor="black", bins=np.arange(min, max + binwidth, binwidth))
    plt.xlim(0.0, 0.6)
    plt.xticks(np.arange(0.0, 0.6, 0.1))
    plt.title(title)
    plt.legend();

    if tile_idx_c is not None:
        x_c, y_c = tile_idx_c[0], tile_idx_c[1]
        ard_c = hkl.load(f'../../tmp/{location}/{str(x_c)}/{str(y_c)}/ard/{str(x_c)}X{str(y_c)}Y_ard.hkl')
        s2_c = ard_c[..., 0:10]
        plt.hist(s2_c.flatten(), alpha=0.3, label=str(tile_idx_c), edgecolor="black", bins=np.arange(min, max + binwidth, binwidth))
        plt.legend();

    return None

def spectral_signature(
    location: str,
    tile_idx_a: tuple,
    tile_idx_b: tuple,
    tile_idx_c: tuple,
    tile_idx_d: tuple,
    systems: list = ['monoculture', 'agroforestry (cocoa)', 'agroforestry (shea)', 'natural'],
    color_dict: dict = {
        'monoculture': '#fe8266',
        'agroforestry (cocoa)' : '#4dc348',
        'agroforestry (shea)' : "#83e87a",
        'natural': '#1c5718'
    },
    show_iqr: bool = False,
    font: int = 13,
    output_file=None
):
    """
    Plot spectral signatures (wavelength vs. reflectance) for four tiles.
    Uses mean reflectance over 14×14 pixels per Sentinel-2 band.
    The x-axis is continuous (400–2200 nm) with points at each band center.

    Parameters
    ----------
    location : str
        Country/area folder name used in your ARD path.
    tile_idx_[a-d] : tuple
        (x, y) tile indices.
    title : str
        Figure title.
    systems : list[str]
        Labels for the four curves, same order as tile_idx_[a-d].
    color_dict : dict
        Color map per system label.
    show_iqr : bool
        If True, shade the 25–75% reflectance range per band for each tile.
    font : int
        Base fontsize.
    """

    # Sentinel-2 band order (first 10 bands in your stack) and central wavelengths (nm)
    band_names = ["B02","B03","B04","B05","B06","B07","B08","B8A", "B11","B12"]
    wavelengths = np.array([490, 560, 665, 705, 740, 783, 842, 865, 1610, 2190], dtype=float)

    def load_s2_10bands(tile_idx):
        x, y = tile_idx
        ard = hkl.load(f'../../tmp/{location}/{x}/{y}/ard/{x}X{y}Y_ard.hkl')[..., 0:10]
        # If not already float in [0,1], convert from uint16
        if not isinstance(ard.flat[0], np.floating):
            ard = ard.astype(np.float32) / 65535.0
        return ard.astype(np.float32)

    tile_list = [tile_idx_a, tile_idx_b, tile_idx_c, tile_idx_d]
    curves_mean, curves_q1, curves_q3 = [], [], []

    for idx in tile_list:
        s2 = load_s2_10bands(idx)
        flat = s2.reshape(-1, s2.shape[-1])
        mean = np.mean(flat, axis=0)
        curves_mean.append(mean)
        if show_iqr:
            q1 = np.percentile(flat, 25, axis=0)
            q3 = np.percentile(flat, 75, axis=0)
            curves_q1.append(q1)
            curves_q3.append(q3)

    # --- Plot setup ---
    fig, ax = plt.subplots(figsize=(14, 5))
    for sys, mean_vals, *iqr in zip(systems, curves_mean, zip(curves_q1, curves_q3) if show_iqr else [()]*4):
        color = color_dict.get(sys, "#666666")
        plt.plot(wavelengths, mean_vals, marker="o", lw=2, label=sys, color=color)
        if show_iqr and iqr:
            q1, q3 = iqr[0]
            plt.fill_between(wavelengths, q1, q3, alpha=0.15, color=color, linewidth=0)

    # --- Continuous wavelength axis ---
    plt.xlim(470, 2230)
    plt.ylim(0, 0.6)


    # show vertical markers for band centers
    for wl, band in zip(wavelengths, band_names):
        plt.axvline(x=wl, color='gray', linestyle=':', alpha=0.3)
        plt.text(wl, -0.03, band, rotation=90, va='top', ha='center', fontsize=font-2)

    #plt.title(title, fontsize=font+1)
    plt.xlabel("Wavelength (nm)", fontsize=font)
    plt.xscale("log")
    plt.ylabel("Reflectance", fontsize=font)
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    plt.grid(True, alpha=0.25, axis='y')
    plt.legend(title="System", fontsize=font-1, title_fontsize=font-1, frameon=False, loc='upper right')
    plt.tight_layout()
    if output_file:
        fig.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    
    plt.show()

    

def heat_compare_arrays(arr_a, arr_b, vmin, vmax, title_a, title_b):
    '''
    Type: Seaborn heatmap
    Purpose: Compare and visualize two files (could be s2 data, ARD, feats, etc.)
    Requires 2D input arrays with shape (618, 614, 1)
    
    '''

    plt.figure(figsize=(11,4))
    plt.subplot(1,2,1)
    sns.heatmap(arr_a, 
                xticklabels=False, 
                yticklabels=False,
                cbar_kws = {'ticks' : [arr_a.min(), arr_a.max()]},
                vmin=vmin,
                vmax=vmax,
                ).set_title(str(title_a))
        
    plt.subplot(1,2,2)
    sns.heatmap(arr_b, 
                xticklabels=False, 
                yticklabels=False,
                vmin=vmin,
                vmax=vmax,
                cbar_kws = {'ticks' : [arr_b.min(), arr_b.max()]}).set_title(str(title_b));

    return None