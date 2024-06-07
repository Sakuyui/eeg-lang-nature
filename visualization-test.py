import numpy as np
import seaborn as sns
import os, sys
import pathlib
from mne_bids import (
    BIDSPath,
    read_raw_bids,
    print_dir_tree,
    make_report,
    find_matching_paths,
    get_entity_vals,
)
import scipy
import matplotlib.pyplot as plt
import mne
from utils.util_filesys import *

import matplotlib.pyplot as plt
import numpy as np; np.random.seed(1)

def plot_segmentation(time_points=[]):
    for time_point in time_points:
        plt.axvline(x=time_point, ymin=0, ymax = 1.0, linewidth=2, color='k')

mat_base_path = "./data/swez-ethz/"
info_file_name = "ID01_info.mat"
analysis_file_name = "ID01_1h.mat"

analysis_file_path = mat_base_path + analysis_file_name
info_file_path = mat_base_path + info_file_name

info_data =  scipy.io.loadmat(info_file_path)
data = scipy.io.loadmat(analysis_file_path)['EEG']
sampling_rate_hz = info_data['fs'][0]
print("sampling_rate = %lf" % sampling_rate_hz)

sel_times_from_sec = 0
sel_times_to_sec = 20
if sel_times_from_sec == -1 or sel_times_to_sec == -1:
    sel_times_from_sec = 0
    sel_times_to_sec = data.shape[1]
print(data.shape)
cnt_channels = data.shape[0]
data = data[:, (int)(sel_times_from_sec * sampling_rate_hz): (int)(sel_times_to_sec * sampling_rate_hz)]
print("Plot from %lf sec to %lf sec" % (sel_times_from_sec / sampling_rate_hz, sel_times_to_sec / sampling_rate_hz))

# info = mne.create_info(['ch' + str(ch_id) for ch_id in range(0, data.shape[0])] , sfreq=sampling_rate_hz, ch_types='misc', verbose=None)
# raw = mne.io.RawArray(np.array(data),info)

# raw.plot()




s=3
x = np.linspace(sel_times_from_sec, sel_times_to_sec, 2000)

channels = data.shape
fig, ax = plt.subplots()
for i in range(cnt_channels):
    ax.plot(x, np.take(data, x)[:, i] + s * i)

labels = ["PG{}".format(i) for i in range(a.shape[1])]
ax.set_yticks(np.arange(0, a.shape[1])*s)
ax.set_yticklabels(labels)
for t, l in zip(ax.get_yticklabels(), ax.lines):
    t.set_color(l.get_color())


plot_segmentation([1 * 512])

plt.show()

