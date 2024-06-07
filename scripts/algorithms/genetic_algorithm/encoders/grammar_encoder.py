class AbstractEEGGrammarEncoder(object):
    def __init__(self):
        pass

    def encode(self):
        raise NotImplementedError
    
class EEGPCFGEncoder(AbstractEEGGrammarEncoder):
    def __init__(self):
        super().__init__()

    def encode(self):
        raise NotImplementedError
    
    