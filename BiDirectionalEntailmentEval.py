import typing

class BiDirectionalEntailmentEval:
    '''
    Score for Bi-Directional Entailment
    '''

    def __init__(self, responses: list[str]):
        self.responses = responses
        self.N = len(responses)
    
    def aggregate(self, verbose: bool=False):
        raise NotImplementedError