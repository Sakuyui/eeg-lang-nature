import numpy as np
from scipy.interpolate import griddata
from sklearn.manifold import MDS


def eeg_to_topological_graph_mds(eeg_data_t, electrode_location_map, dedimension_method='simple', grid_size=64):
    cnt_channels = len(electrode_location_map)
    # Convert electrode location map to array
    ch_names = [electrode_location_map[channel_id][0] for channel_id in range(cnt_channels)]
    locations_3d = np.array([electrode_location_map[channel_id][1] for channel_id in range(cnt_channels)])
    # Projection to 2D
    if 'simple' == dedimension_method:
        locations_2d = locations_3d[:, :2]
    elif 'mds' == dedimension_method: 
        mds = MDS(n_components=2, dissimilarity='euclidean')
        locations_2d = mds.fit_transform(locations_3d)
    else:
        raise ValueError
        
    
    data_t = eeg_data_t
    n_grid = grid_size  # Adjust grid size as needed
    x = np.linspace(np.min(locations_2d), np.max(locations_2d), n_grid)
    y = np.linspace(np.min(locations_2d), np.max(locations_2d), n_grid)
    grid_x, grid_y = np.meshgrid(x, y)
    
    # Interpolate EEG data onto the scalp map
    
    grid_z = griddata((locations_2d[:,0], locations_2d[:,1]), data_t, (grid_x, grid_y), method='cubic')
    
    return np.nan_to_num(grid_z)



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
