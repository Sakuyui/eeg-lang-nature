import torch
import torch.nn as nn
from typing import Dict, List
from .td_pcfg import TDPCFG
from .res import ResLayer

class TNPCFG(nn.Module):
    def __init__(self, args):
        super(TNPCFG, self).__init__()
        print("build td pcfg")
        self.pcfg = TDPCFG()
        self.device = 'cpu'
        self.args = args
        self.NT = args['NT']
        self.T = args['T']
        self.V = args['cnt_words']
        self.s_dim = args['s_dim']
        self.r = args['r_dim']
        self.word_emb_size = args['word_emb_size']
        if 'summary_parameters' in args and args['summary_parameters']:
            print("device = %s, NT = %d, T = %d, V = %d, s_dim = %d, r = %d, word_emb_size = %d" % \
                (self.device, self.NT, self.T, self.V, self.s_dim, self.r, self.word_emb_size))

        ## root
        self.root_emb = nn.Parameter(torch.randn(1, self.s_dim))
        self.root_mlp = nn.Sequential(nn.Linear(self.s_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      nn.Linear(self.s_dim, self.NT))

        #terms
        self.term_emb = nn.Parameter(torch.randn(self.T, self.s_dim))
        self.term_mlp = nn.Sequential(nn.Linear(self.s_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      ResLayer(self.s_dim, self.s_dim),
                                      nn.Linear(self.s_dim, self.V))

        self.rule_state_emb = nn.Parameter(torch.randn(self.NT+self.T, self.s_dim))
        rule_dim = self.s_dim
        self.parent_mlp = nn.Sequential(nn.Linear(rule_dim,rule_dim),nn.ReLU(),nn.Linear(rule_dim,self.r))
        self.left_mlp = nn.Sequential(nn.Linear(rule_dim,rule_dim), nn.ReLU(),nn.Linear(rule_dim,self.r))
        self.right_mlp = nn.Sequential(nn.Linear(rule_dim,rule_dim),nn.ReLU(),nn.Linear(rule_dim,self.r))


    def forward(self, input, **kwargs):
        print(f"begin forward>>>>>>>>>>>>>, input shape = {input.shape}")
        
        x = input

        b, n = x.shape[:2]
        print("b, n = %d, %d" % (b, n)) # batch and number?

        def roots():
            roots = self.root_mlp(self.root_emb).log_softmax(-1)
            return roots.expand(b, roots.shape[-1]).contiguous()

        def terms():
            term_prob = self.term_mlp(self.term_emb).log_softmax(-1)
            print(term_prob.shape)
            return term_prob[torch.arange(self.T)[None,None].long(), x[:, :, None].long()]

        def rules():
            rule_state_emb = self.rule_state_emb
            nonterm_emb = rule_state_emb[:self.NT]
            head = self.parent_mlp(nonterm_emb).log_softmax(-1)
            left = self.left_mlp(rule_state_emb).log_softmax(-2)
            right = self.right_mlp(rule_state_emb).log_softmax(-2)
            head = head.unsqueeze(0).expand(b,*head.shape)
            left = left.unsqueeze(0).expand(b,*left.shape)
            right = right.unsqueeze(0).expand(b,*right.shape)
            return (head, left, right)

        root, unary, (head, left, right) = roots(), terms(), rules()
        print(unary.shape, root.shape, head.shape, left.shape, right.shape)
        return {'unary': unary,
                'root': root,
                'head': head,
                'left': left,
                'right': right,
                'kl': 0}

    def loss(self, input):
        rules = self.forward(input)
        seq_len = len(input)
        print(f"seq_len = {seq_len}")
        result = self.pcfg._inside(rules=rules, lens=seq_len)
        logZ = -result['partition'].mean()
        return logZ

    def evaluate(self, input, decode_type, **kwargs):
        rules = self.forward(input)
        if decode_type == 'viterbi':
            assert NotImplementedError
        elif decode_type == 'mbr':
            return self.pcfg.decode(rules=rules, lens=input['seq_len'], viterbi=False, mbr=True)
        else:
            raise NotImplementedError
