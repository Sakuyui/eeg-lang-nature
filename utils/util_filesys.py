EEG_DATA_SUFFIX = set(['eeg', 'edf', 'bdf', 'gdf', 'cnt', 'egi', 'mff', 'set', 'fdt', 'data', 'nxe', 'lay', 'dat', 'xdf', 'xdfz'])
def get_all_directory_contains_eeg_data(base_path = "./", results = []):
    current_path_has_eeg_data = False
    file_and_folders = (os.listdir(base_path))
    #print(file_and_folders)
    for f_d in file_and_folders:
        path = os.path.join(base_path, f_d)
        if(pathlib.Path(path).suffix.replace(".","").lower() in EEG_DATA_SUFFIX):
            current_path_has_eeg_data = True
        else:
            if os.path.isdir(path):
                get_all_directory_contains_eeg_data(path, results)
    if current_path_has_eeg_data:
        results.append(base_path)
    return results