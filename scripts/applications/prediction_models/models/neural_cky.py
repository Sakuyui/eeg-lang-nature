import torch
import torch.nn as nn
import heapq

INDEX_PARSE_POSSIBILITY = 0
INDEX_PARSE_FEATURE = 1
INDEX_PARSE_GRAMMAR_ID = 2
INDEX_PARSE_SYMBOL_ID = 3

GRAMMAR_TYPE_START = 0
GRAMMAR_TYPE_NONTERMINATE = 1
GRAMMAR_TYPE_PRETERMINATE = 2

class NN_CYK_FeatureCombingModel_Preterminate(nn.Module):
    def __init__(self, word_embedding_length = 200, grammar_embedding_length = 0, output_feature_length = 128):
        super(NN_CYK_FeatureCombingModel_Preterminate, self).__init__()
        self.preterminate_grammar_feature_size = word_embedding_length
        self.grammar_embedding_size = grammar_embedding_length
        self.hidden_size = output_feature_length
        self.fc_with_grammar_embdedding = nn.Linear(word_embedding_length + grammar_embedding_length, output_feature_length)
        self.fc_no_grammar_embdedding = nn.Linear(word_embedding_length, output_feature_length)

    def forward(self, word_embedding, grammar_embedding = None):
        if self.preterminate_grammar_feature_size == 0 or self.grammar_embedding_size == 0:
            return self.fc_no_grammar_embdedding(torch.FloatTensor(word_embedding))
        return self.fc_with_grammar_embdedding(torch.cat([word_embedding, grammar_embedding]))
        
class NN_CYK_FeatureCombingModel_NonPreterminate(nn.Module):
    def __init__(self, feature_length = 128):
        super(NN_CYK_FeatureCombingModel_NonPreterminate, self).__init__()
        self.feature_length = feature_length
        self.fc_grammar_produce_single_symbol = nn.Linear(feature_length, feature_length)
        self.fc_grammar_produce_double_symbol = nn.Linear(feature_length * 2, feature_length)
        
    def forward(self, feature1 = None, feature2 = None):
        if feature1 is None and feature2 is None:
            return torch.zeros(self.feature_length)
        if feature1 is None:
            return self.fc_grammar_produce_single_symbol(feature2)
        if feature2 is None:
            return self.fc_grammar_produce_single_symbol(feature1)
        
        # Make symmetric hold in processing. i.e. the output irrelavant to the orders of two input features.
        concated_feature1 = torch.cat([feature1, feature2])
        concated_feature2 = torch.cat([feature2, feature1])
        output_feature1 = self.fc_grammar_produce_double_symbol(concated_feature1)
        output_feature2 = self.fc_grammar_produce_double_symbol(concated_feature2)
        return (output_feature1 + output_feature2) / 2
        
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

        self.merge_model = self.args['merge_model'] # A -> BC
        self.preterminate_feature_generation_model = self.args['preterminate_feature_generation_model']  # E -> w
        
        self.grammar_embeddings = None
        
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
        print("append record.")
        self.records.append(current_record)
    
        for inner_record_index in range(0, len(current_record) - 1):
            print("fill record in t=%d" % self.t)
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
        result = heapq.nlargest(1, self.records[-1][0], lambda x: x[0])[0][INDEX_PARSE_FEATURE]
        self.t += 1
        return result
    
    def get_terminate_parses(self, w):
        # [possibility, feature, grammar id, symbol_id]
        w_terminate_grammars = list(self.grammar_preterminates[:, w])
        terminate_grammar_id_offset = self.NT + self.NT * (self.NT + self.T) * (self.NT + self.T)
        if self.infuse_structure_only:
            feature = self.preterminate_feature_generation_model(self.word_embeddings[w])
            return [[w_terminate_grammars[index], feature, terminate_grammar_id_offset + self.V * index + w, index] for index in range(len(w_terminate_grammars))]
        else:   
            return [[w_terminate_grammars[index], self.preterminate_feature_generation_model(self.word_embeddings[w], self.grammar_embeddings[terminate_grammar_id_offset + self.V * index + w]),\
                terminate_grammar_id_offset + self.V * index + w, index] for index in range(len(w_terminate_grammars))]

    def merge_parses_in_two_cells(self, cell1, cell2, cell1_id = None, cell2_id = None):
            result = []
            # parse format is a triple : [possibility, feature, grammar id]
            # A -> BC
            for parse_i in cell1:
                for parse_j in cell2:
                    p_1 = parse_i[0]
                    p_2 = parse_j[0]
                    symbol_id_1 = parse_i[-1]
                    symbol_id_2 = parse_j[-1]

                    grammars = self.grammar_double_nonterminates[:,symbol_id_1, symbol_id_2]
                    
                    if self.infuse_structure_only:
                        feature = self.merge_model(torch.FloatTensor(parse_i[1]), torch.FloatTensor(parse_j[1]))
                    
                    for k in range(len(grammars)):
                        p = p_1 * p_2 * grammars[k]
                        grammar_id = self.NT + (self.NT + self.T) * (self.NT + self.T) * k + \
                                symbol_id_1 * (self.NT + self.T) + symbol_id_2
                        if not self.infuse_structure_only:
                            feature = self.merge_model(torch.FloatTensor(parse_i[INDEX_PARSE_FEATURE]), \
                                torch.FloatTensor(parse_j[INDEX_PARSE_FEATURE]), torch.FloatTensor(self.grammar_embeddings[grammar_id]))
                        result.append([p, feature, grammar_id, k])

            # S -> A
            for _, parse_i in enumerate(cell1 + cell2):
                p_1 = parse_i[INDEX_PARSE_POSSIBILITY]
                grammar_id = parse_i[INDEX_PARSE_GRAMMAR_ID]
                symbol_id = parse_i[INDEX_PARSE_SYMBOL_ID]
                # Skip for those parses that belong to start grammars or pre-terminate grammars.
                if grammar_id >= self.NT and grammar_id <= self.NT + (self.NT * (self.NT + self.T) * (self.NT + self.T)):
                    continue
                
                if self.infuse_structure_only:
                    feature = self.merge4_model(torch.FloatTensor(parse_i[INDEX_PARSE_FEATURE]), None)
                
                p = p_1 * self.grammar_starts[symbol_id]
                
                reduce_grammar_id = symbol_id
                if not self.infuse_structure_only:
                    feature = self.merge_model(parse_i[INDEX_PARSE_FEATURE], parse_j[INDEX_PARSE_FEATURE],\
                        self.grammar_embeddings[grammar_id])
                            
                result.append([p, feature, reduce_grammar_id, -1]) # symbol id = -1 denote it is parse reduce 
                                                                   # to a start symbol
                                                                   # symbol id 
                                                                   #   - -1: Start Symbol
                                                                   #   - [0, |P| - 1]: terminate symbols
                                                                   #   - [|P|, |P| + |NT| - 1]: non-terminate symbols
                                                                   #   - [|P| + |NT|, |P| + |NT| + |V| - 1]: word

            return result
    
        
    def is_start_symbol(self, symbol_id):
        return symbol_id == -1
        
    def is_preterminate_symbol(self, symbol_id):
        return symbol_id >= 0 and symbol_id < self.T
    
    def is_nonterminate_symbol(self, symbol_id):
        return symbol_id >= self.T + self.NT and symbol_id < self.T + self.NT 
    
    def is_word_symbol(self, symbol_id):
        return symbol_id >= self.T + self.NT and symbol_id < self.T + self.NT + self.V
        
    def get_grammar_id(self, grammar_type, reduce_symbol_id, produce_symbol_id_1 = None, produce_symbol_id_2 = None):
        if GRAMMAR_TYPE_START == grammar_type:
            if not (self.is_start_symbol(reduce_symbol_id) and \
                (self.is_preterminate_symbol(produce_symbol_id_1) or self.is_nonterminate_symbol(produce_symbol_id_1))):
                raise ValueError
            return produce_symbol_id_1 - self.T
            
        if GRAMMAR_TYPE_NONTERMINATE == grammar_type:
            if not (self.is_nonterminate_symbol(reduce_symbol_id) and \
                (self.is_nonterminate_symbol(produce_symbol_id_1) or self.is_preterminate_symbol(produce_symbol_id_1)) and \
                    (self.is_nonterminate_symbol(produce_symbol_id_2) or self.is_preterminate_symbol(produce_symbol_id_2))):
                raise ValueError
            sum_NT_T = self.NT + self.T
            return self.V + reduce_symbol_id * sum_NT_T * sum_NT_T + produce_symbol_id_1 * sum_NT_T + produce_symbol_id_2
        
        if GRAMMAR_TYPE_PRETERMINATE == grammar_type:
            if not (self.is_preterminate_symbol(reduce_symbol_id) and \
                (self.is_word_symbol(produce_symbol_id_1))):
                raise ValueError
            sum_NT_T = self.NT + self.T
            return self.V + self.NT * sum_NT_T * sum_NT_T + (reduce_symbol_id) * self.V + produce_symbol_id_1 - self.T - self.NT

def select_tops(cell, beam_size = 5):
    return heapq.nlargest(beam_size, cell, key= lambda x: x[0])
