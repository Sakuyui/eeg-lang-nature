import torch
import torch.nn as nn
import heapq

class NN_CYK_Model(nn.Module):
    def __init__(self, args):
        super(NN_CYK_Model, self).__init__()
        self.args = args
        self.device = 'cpu'
        self.NT = args['NT']
        self.T = args['T']
        self.V = args['cnt_words']
        self.s_dim = args['s_dim']
        self.r = args['r_dim']
        self.word_emb_size = args['word_emb_size']
        self.enable_slide_windows = False if 'enable_slide_windows' not in args else args['enable_slide_windows']
        self.window_size = False if 'window_size' not in args else args['window_size']
        self.word_embeddings = args['word_embeddings']
        self.infuse_structure_only = True
        self.grammar_preterminates = args['grammar_preterminates']
        self.grammar_double_nonterminates = args['grammar_double_nonterminates']
        self.grammar_starts = args['grammar_starts']

        self.beam_search_strategy = args['beam_search_strategy']

        self.merge_model = None # A -> BC
        self.terminate_feature_generation_model = None # D -> E
        self.preterminate_feature_generation_model = None # E -> w
        # self.word_unary_grammar_id_mapping = args['word_unary_grammar_id_mapping']
        # self.unary_preterminate_grammar_id_mapping = args['unary_preterminate_grammar_id_mapping']
        # self.double_nonterminates_grammar_id_mapping = args['double_nonterminates_grammar_id_mapping']
        
        # check parameters
        if (self.enable_slide_windows and self.window_size <= 0):
            return None
        self.reset_global_context()
        
    def reset_global_context(self):
        self.records = []
        self.windows_beginning_offset = 0
        self.t = 0

    # each cell with form (possibility, feature_in_tensor, root_symbol_index)
    def forward(self, word): # input a word at time t.
        # generate the analysis of span [t, t]
        w = self.get_terminate_parses(word)

        # create cells for current time t.
        current_record = [[]] if len(self.records) == 0 else [[]] * (len(self.records[-1]) + 1)

        # fill the analysis of span [t, t] to the last cell of current time t.
        current_record[-1] = w

        # Sliding window control the maximum size of a record at time t.
        # When the windows size is large, we may utilize array to store records, rather than list. The array allow us adopting 
        # sliding windows more effectively when the window size is large. However, we currently utilize list to store records.
        # To do sliding window, we need first pop the first record (oldest record inside the window), and for each of rest records,
        # we need pop its first element, as the first element of each record relate to the "word" we'd like to forget.
        if self.enable_slide_windows:
            while len(self.records) >= self.window_size:
                self.records.pop(0)
                for i in range(0, len(self.records)):
                    self.records[i].pop(0)
                self.windows_beginning_offset += 1

        # Add record at current time to the global record set.
        self.records.append(current_record)
    
        for inner_record_index in range(0, len(current_record) - 1):
            current_record_length = len(current_record) 
            current_record[current_record_length - inner_record_index - 2] = []
            cell_data = []
            for split_index in range(current_record_length - inner_record_index - 2, self.t - self.windows_beginning_offset):
                # coordination = (index, t)
                cell1_coordination = (current_record_length - inner_record_index - 2, split_index)
                cell2_coordination = (split_index + 1, self.t - self.windows_beginning_offset)
                cell_data += self.merge_parses_in_two_cells(self.records[cell1_coordination[1]][cell1_coordination[0]],\
                                                           self.records[cell2_coordination[1]][cell2_coordination[0]],\
                                                           cell1_coordination, cell2_coordination)
                current_record[current_record_length - inner_record_index - 2] = self.beam_search_strategy(cell_data) if self.beam_search_strategy is not None else cell_data
        result = heapq.nlargest(1, self.records[-1][0])[0][1]
        self.t += 1
        return result
        
 
        
    def get_grammar_for_generation(self, parse_i, parse_j):
            return self.double_nonterminates_grammar_id_mapping["%s#%s" % (str(parse_i[0]), str(parse_j[0]))]
    
    def get_terminate_parses(self, w):
            feature = self.preterminate_feature_generation_model(self.word_embeddings[w], self.grammar_preterminates[:, w])
            # [possibility, feature, grammar id, symbol_id]
            w_terminate_grammars = self.grammar_preterminates[:, w]
            terminate_grammar_id_offset = self.NT + self.NT * (self.NT + self.T) * (self.NT + self.T)
            return [[w_terminate_grammars[index], feature, terminate_grammar_id_offset + self.V * index + w, index] for index in len(w_terminate_grammars)]
        
    def merge_parses_in_two_cells(self, cell1, cell2, cell1_id = None, cell2_id = None):
            result = []
            # parse format is a triple : [possibility, feature, grammar id]
            # A -> BC
            for parse_i in range(cell1):
                for parse_j in range(cell2):
                    p_1 = parse_i[0]
                    p_2 = parse_j[1]
                    symbol_id_1 = p_1[-1]
                    symbol_id_2 = p_2[-1]
                    grammars = self.grammar_double_nonterminates[:,symbol_id_1, symbol_id_2]
                    if self.infuse_structure_only:
                        feature = self.merge_model(parse_i[1], parse_j[2])
                    
                    for k in len(grammars):
                        p = p_1 * p_2 * grammars[k]
                        grammar_id = self.NT + (self.NT + self.T) * (self.NT + self.T) * k + \
                                symbol_id_1 * (self.NT + self.T) + symbol_id_2
                        if not self.infuse_structure_only:
                            
                            feature = self.merge_model(parse_i[1], parse_j[2], self.grammar_embeddings[grammar_id])
                            
                        result.append([p, feature, grammar_id, k])
            # S -> A
            for _, parse_i in enumerate(cell1 + cell2):
                p_1 = parse_i[0] 
                symbol_id = p_1[-1]
                grammars = self.grammar_starts[symbol_id, :]
                if self.infuse_structure_only:
                    feature = self.merge_model(parse_i[1], None)
                for k in len(grammars):
                    p = p_1 * grammars[k]
                    grammar_id = k
                    if not self.infuse_structure_only:
                            feature = self.merge_model(parse_i[1], parse_j[2], self.grammar_embeddings[grammar_id])
                            
                    result.append([p, feature, grammar_id, k])
            return result
        
def select_tops(cell, beam_size = 5):
    return heapq.nlargest(beam_size, cell)

