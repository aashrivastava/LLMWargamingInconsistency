from utils.promptopenai import OpenAIPrompting
from utils.game import GameSimulator
from metrics.BERTScoreEval import BERTScoreEval
from metrics.BiDirectionalEntailmentEval import BiDirectionalEntailmentEval
from metrics.MQAGEval import MQAGEval
from metrics.RankEval import RankEval

import typing
from tqdm import tqdm

class ConsistencyEval:
    '''
    DOCSTRING
    '''
    def __init__(self, prompting_model, metric, **kwargs):
        '''
        **kwargs is keyword arguments pertaining to particular consistency metric
        '''
        assert metric in ['bert', 'bidirection', 'mqag', 'rank']
        # assertion that given model is valid
        self.simulator = OpenAIPrompting(model=prompting_model)
        self.metric = metric
        if self.metric == 'bert':
            self.aggregator = BERTScoreEval(**kwargs)
        elif self.metric == 'bidirection':
            self.aggregator = BiDirectionalEntailmentEval(**kwargs)
        elif self.metric == 'mqag':
            self.aggregator = MQAGEval(**kwargs)
        elif self.metric == 'rank':
            self.aggregator = RankEval(**kwargs)
        else:
            raise MyException(f'{self.metric} is not valid')

    def aggregate(self, responses, verbose: bool=False):
        return self.aggregator.aggregate(responses, verbose=verbose)
    
    def main(self, prompt, N_responses=8, temperature=1.0):
        return self.aggregator.aggregate(self.get_responses(prompt, N_responses=N_responses, temperature=temperature))


class MyException(Exception):
    pass


        