import numpy as np
from .microstate_stat_helper import *
from sequence_analysis.sequence_statisticians import calculate_entropy, calculate_max_entropy,\
    calculate_transition_matrix, calculate_empirical_symbol_distribution,\
        calculate_empirical_entropy

class SegmentationStatisticians(object):
    
    def __init__(self, language_configuration):
        self.language_configuration = language_configuration
    
    def calculate_transition_matrix(self, solution):
        word_count_in_language_configuration = self.language_configuration.configuration_word_count()
        print("There are %d words in the word list of the language configuration." % word_count_in_language_configuration)
        return calculate_transition_matrix(solution.get_word_sequence(), word_count_in_language_configuration)

    def calculate_empirical_symbol_distribution(self, solution):
        word_count_in_language_configuration = self.language_configuration.configuration_word_count()
        return calculate_empirical_symbol_distribution(solution.get_word_sequence(), word_count_in_language_configuration)
    
    def calculate_global_explained_variance(self):
        raise NotImplementedError

    def calculate_total_global_explained_variance(self):
        raise NotImplementedError

    def calculate_empirical_entropy(self, solution):
        return calculate_empirical_entropy(solution.get_word_sequence())

    def calculate_max_entropy(self):
        return calculate_max_entropy(self.language_configuration.configuration_word_count())

    def calculate_excess_entropy_rate(self):
        raise NotImplementedError

    def calculate_mc_entropy_rate(self):
        raise NotImplementedError

    def markov_tests(self, solution, order=0, alpha=0.01):
        word_count_in_language_configuration = self.language_configuration.configuration_word_count()
        word_id_sequence = solution.get_word_sequence()
        if order == 0:
            testMarkov0(word_id_sequence, word_count_in_language_configuration, alpha)
        elif order == 1:
            testMarkov1(word_id_sequence, word_count_in_language_configuration, alpha)
        elif order == 2:
            testMarkov2(word_id_sequence, word_count_in_language_configuration, alpha)
        else:
            raise NotImplementedError

    def stationarity_test(self, solution, block_size, alpha=0.01):
        word_id_sequence = solution.get_word_sequence()
        word_count_in_language_configuration = self.language_configuration.configuration_word_count()
        conditionalHomogeneityTest(word_id_sequence, word_count_in_language_configuration, block_size, alpha)

    def symmetry_test(self, solution, alpha=0.01):
        word_id_sequence = solution.get_word_sequence()
        word_count_in_language_configuration = self.language_configuration.configuration_word_count()
        symmetryTest(word_id_sequence, word_count_in_language_configuration, alpha)

    def markov_surrogate(self, solution):
        word_count_in_language_configuration = self.language_configuration.configuration_word_count()
        word_id_sequence = solution.get_word_sequence()
        p_hat = self.calculate_empirical_symbol_distribution(solution)
        T_hat = self.calculate_transition_matrix(solution)
        x_mc = surrogate_mc(p_hat, T_hat, word_count_in_language_configuration, len(word_id_sequence))
        p_surr = p_empirical(x_mc, word_count_in_language_configuration)
        T_surr = T_empirical(x_mc, word_count_in_language_configuration)
        return T_surr
    
    def time_lagged_mutual_information(self, solution, plot=True, n_mc = 10, lag_max = 100, alpha=0.01):
        word_id_sequence = solution.get_word_sequence()
        word_count_in_language_configuration = self.language_configuration.configuration_word_count()
        aif = mutinf(word_id_sequence, word_count_in_language_configuration, lag_max)
        print(f"\nComputing n = {n_mc:d} Markov surrogates...\n")
        p_hat = self.calculate_empirical_symbol_distribution(solution)
        T_hat = self.calculate_transition_matrix(solution)
        aif_array = mutinf_CI(p_hat, T_hat, len(word_id_sequence), alpha, n_mc, lag_max)
        pct = np.percentile(aif_array, [100.*alpha/2.,100.*(1-alpha/2.)], axis=0)
        print("\nMarkov surrogates and AIF confidence interval computed")
        if plot:
            plt.ioff()
            fig = plt.figure(1, figsize=(20,5))
            fig.patch.set_facecolor('white')
            t = np.arange(lag_max)*1000./512
            plt.semilogy(t, aif, '-sk', linewidth=3, label='AIF (EEG)')
            plt.semilogy(t, pct[0,:], '-k', linewidth=1)
            plt.semilogy(t, pct[1,:], '-k', linewidth=1)
            plt.fill_between(t, pct[0,:], pct[1,:], facecolor='gray', alpha=0.5, label='AIF (Markov)')
            plt.xlabel("time lag [ms]", fontsize=20, fontweight="bold")
            plt.ylabel("AIF [nats]", fontsize=20, fontweight="bold")
            plt.legend(fontsize=22)
            ax = plt.gca()
            ax.tick_params(axis='both', labelsize=18)
            plt.tight_layout()
            plt.show()