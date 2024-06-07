import numpy as np
from scipy.interpolate import griddata

def eeg2map(data, eletrode_location_map):
    """Interpolate and normalize EEG topography, ignoring nan values

    Args:
        data: numpy.array, size = number of EEG channels
        n_grid: interger, interpolate to n_grid x n_grid array, default=64
    Returns:
        top_norm: normalized topography, n_grid x n_grid
    """
    n_grid = 64
    top = topo(data, eletrode_location_map, n_grid)
    mn = np.nanmin(top)
    mx = np.nanmax(top)
    top_norm = (top-mn)/(mx-mn)
    return top_norm



def topo(data, eletrode_location_map, n_grid=64):
    """Interpolate EEG topography onto a regularly spaced grid

    Args:
        data: numpy.array, size = number of EEG channels
        n_grid: integer, interpolate to n_grid x n_grid array, default=64
    Returns:
        data_interpol: cubic interpolation of EEG topography, n_grid x n_grid
                       contains nan values
    """
    n_channels = len(eletrode_location_map)
    locations = [electrode_location_info[1] for electrode_location_info in eletrode_location_map]
    locations /= np.linalg.norm(locations, 2, axis = 1, keepdims=True)
    c = [i for i, l in enumerate(eletrode_location_map) if (l[0] == 'Fz')][0]
    w = np.linalg.norm(locations - locations[c], 2, axis=1)
    arclen = np.arcsin(w / 2. * np.sqrt(4. - w * w))

    phi_re = locations[:,0] - locations[c][0]
    phi_im = locations[:,1] - locations[c][1]
    #print(type(phi_re), phi_re)
    #print(type(phi_im), phi_im)
    tmp = phi_re + 1j*phi_im
    #tmp = map(complex, locs[:,0]-locs[c][0], locs[:,1]-locs[c][1])
    #print(type(tmp), tmp)
    phi = np.angle(tmp)

    X = arclen * np.real(np.exp(1j * phi))
    Y = arclen * np.imag(np.exp(1j * phi))
    r = max([max(X), max(Y)])
    Xi = np.linspace(-r, r, n_grid)
    Yi = np.linspace(-r, r, n_grid)
    data_ip = griddata((X, Y), data, (Xi[None, :], Yi[:, None]), method = 'cubic')
    return data_ip


def findstr(s, L):
    """Find string in list of strings, returns indices.

    Args:
        s: query string
        L: list of strings to search
    Returns:
        x: list of indices where s is found in L
    """

    x = [i for i, l in enumerate(L) if (l==s)]
    return x
