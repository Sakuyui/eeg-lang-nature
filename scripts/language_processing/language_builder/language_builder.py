import sys
import scipy.signal
sys.path.append("..")

from entities.word import *
from entities.eeg_state import EletrodeValuesEEGState
import numpy as np
from scipy import signal
from utils.microstates_algorithms import locmax


class AbstractWordListBuilder(object):
    def build_dictionary(self, eeg_matrix, eeg_info, electrode_locations, dictionary_building_config: Dict) -> AbstractEEGLanguageDictionary:
        raise NotImplementedError
    def deserialize_from_file(self, file_path):
        raise NotImplementedError
           
class AbstracrGrammarExtractor(object):
    def extract_grammar(self, eeg_matrix, eeg_info, dictionary):
        raise NotImplementedError
    
class DummyEEGLanguageDictionaryBuilder(AbstractWordListBuilder):
    def __init__(self):
        super().__init__()
    def deserialize_from_file(self, file_path):
        return DictionaryImplementedEEGLanguageDictionary(word_list=None).deserialize_from(file_path)
    def build_dictionary(self, eeg_matrix, eeg_info, electrode_locations, dictionary_building_config):
        cnt_words = dictionary_building_config['cnt_words']
        cnt_channels = eeg_matrix.shape[0]
        if cnt_channels > cnt_words:
            raise ValueError
        return DictionaryImplementedEEGLanguageDictionary([EletrodeValuesEEGState(np.array([[0]] * cnt_channels)) for i in range(0, cnt_words)])
    
class RandomEEGLanguageDictionaryBuilder(AbstractWordListBuilder):
    def __init__(self):
        super().__init__()
    def deserialize_from_file(self, file_path):
        return DictionaryImplementedEEGLanguageDictionary(word_list=None).deserialize_from(file_path)
    def build_dictionary(self, eeg_matrix, eeg_info, electrode_locations, dictionary_building_config):
        cnt_words = dictionary_building_config['cnt_words']
        cnt_channels = eeg_matrix.shape[0]
        cnt_samples = eeg_matrix.shape[1]
        if cnt_channels > cnt_words:
            raise ValueError
        return DictionaryImplementedEEGLanguageDictionary([EletrodeValuesEEGState(eeg_matrix[np.random.randint(0, cnt_samples)]) for _ in range(0, cnt_words)])
    
class GFPKmeansEEGLanguageDictionaryBuilder(AbstractWordListBuilder):
    def __init__(self, eletrode_location_map):
        super().__init__()
        self.eletrode_location_map = eletrode_location_map
        
    def _calculate_gfp(self, eeg_matrx):
        cnt_channels = eeg_matrx.shape[0]
        print("channels = %d" % cnt_channels)
        electrode_averages = np.average(eeg_matrx, axis=0)
        gfps = np.sqrt(np.sum((eeg_matrx - electrode_averages) ** 2 / cnt_channels, axis=0))
        return gfps        
    
    def deserialize_from_file(self, file_path):
        return DictionaryImplementedEEGLanguageDictionary(word_list=None).deserialize_from(file_path)
       
    def build_dictionary(self, eeg_matrix, eeg_info, electrode_locations, dictionary_building_config) -> AbstractEEGLanguageDictionary:
        from utils.microstates_algorithms import kmeans
        n_clusters = dictionary_building_config['cnt_words']
        n_runs = dictionary_building_config['cnt_runs']
        doplot = dictionary_building_config['doplot'] if 'doplot' in dictionary_building_config else False
        
         # --- normalized data ---
        data_norm = eeg_matrix - eeg_matrix.mean(axis=0, keepdims=True)
        data_norm /= data_norm.std(axis=0, keepdims=True)

        # calculate GFP
        gfps = self._calculate_gfp(eeg_matrix)

        gfp_peaks = locmax(gfps)
        data_cluster = eeg_matrix[:, gfp_peaks]
        data_cluster_norm = data_cluster - data_cluster.mean(axis=0, keepdims=True)
        data_cluster_norm /= data_cluster_norm.std(axis=0, keepdims=True)
        print("\t[+] Data format for clustering [channels, GFP Peeks]: {:d} x {:d}"\
            .format(data_cluster.shape[0], data_cluster.shape[1]))
        
        print("\n\t[+] Clustering algorithm: mod. K-MEANS.")
        maps = kmeans(eeg_matrix, n_maps=n_clusters, n_runs=5, doplot=False)
        
        # --- microstate sequence ---
        C = np.dot(data_norm.T, maps) / eeg_matrix.shape[1]
        L = np.argmax(C ** 2, axis = 1)
        del C

        # --- GEV ---
        maps_norm = maps - maps.mean(axis=0, keepdims=True)
        maps_norm /= maps_norm.std(axis=0, keepdims=True)

        # --- correlation data, maps ---
        cnt_channels = eeg_matrix.shape[0]
        C = np.dot(data_norm.T, maps_norm) / cnt_channels
        
        normal_gfp_2 = np.sum(gfps ** 2)
        # --- GEV_k & GEV ---
        gev = np.zeros(n_clusters)
        for k in range(n_clusters):
            r = L == k
            gev[k] = np.sum(gfps[r] ** 2 * C[r, k] ** 2) / normal_gfp_2
        
        print("\n\t[+] Global explained variance GEV = {:.3f}".format(gev.sum()))
        for k in range(n_clusters): print("\t\tGEV_{:d}: {:.3f}".format(k, gev[k]))

        if doplot:
            import matplotlib.pyplot as plt
            from utils.topological_graph import eeg2map
            plt.ion()
            # matplotlib's perceptually uniform sequential colormaps:
            # magma, inferno, plasma, viridis
            cm = plt.cm.magma
            fig, axarr = plt.subplots(1, n_clusters, figsize=(20,5))
            fig.patch.set_facecolor('white')
            for imap in range(n_clusters):
                axarr[imap].imshow(eeg2map(maps[:, imap], self.eletrode_location_map), cmap=cm, origin='lower')
                axarr[imap].set_xticks([])
                axarr[imap].set_xticklabels([])
                axarr[imap].set_yticks([])
                axarr[imap].set_yticklabels([])
            title = "Microstate maps ({:s})".format("GFP-Kmeans")
            axarr[0].set_title(title, fontsize=16, fontweight="bold")
            plt.show()

            # --- assign map labels manually ---
            
            order_str = ['w_' + str(map_id) for map_id in np.arange(0, n_clusters)]
            order = np.zeros(n_clusters, dtype=int)
            for i, s in enumerate(order_str):
                order[i] = int(i)
            
            print("\t\tRe-ordered labels: {:s}".format(", ".join(order_str)))
            # re-order return variables
            maps = maps[:,order]
            for i in range(len(L)):
                L[i] = order[L[i]]
            gev = gev[order]
            # Figure
            fig, axarr = plt.subplots(1, n_clusters, figsize=(20,5))
            fig.patch.set_facecolor('white')

            for imap in range(n_clusters):
                axarr[imap].imshow(eeg2map(maps[:, imap], self.eletrode_location_map), cmap=cm, origin='lower')
                axarr[imap].set_xticks([])
                axarr[imap].set_xticklabels([])
                axarr[imap].set_yticks([])
                axarr[imap].set_yticklabels([])
            title = "re-ordered microstate maps"
            axarr[0].set_title(title, fontsize=16, fontweight="bold")
            plt.show()
            plt.ioff()
        word_list = maps.T
        print("Build Word List Finished, in eletrode value representaion. Shape = %s" % str(word_list.shape))
        return DictionaryImplementedEEGLanguageDictionary([StateRepresentedEEGWord(EletrodeValuesEEGState(word_list[word_id]), word_id) for word_id in range(0, word_list.shape[0])])
