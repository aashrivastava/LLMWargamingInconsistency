from utils.promptopenai import OpenAIPrompting
from utils.game import GameSimulator
from metrics.BERTScoreEval import BERTScoreEval
from metrics.BiDirectionalEntailmentEval import BiDirectionalEntailmentEval
from metrics.MQAGEval import MQAGEval
from metrics.RankEval import RankEval

import typing
from tqdm.auto import tqdm

class ConsistencyEval:
    '''
    DOCSTRING
    '''
    def __init__(self, metric: typing.Optional[str]=None, **kwargs):
        '''
        **kwargs is keyword arguments pertaining to particular consistency metric
        '''
        # assert metric in ['bert', 'bidirection', 'mqag', 'rank']
        # assertion that given model is valid
        # self.simulator = OpenAIPrompting(model=prompting_model)
        self.metric = metric
        if self.metric == 'bert':
            self.aggregator = BERTScoreEval(**kwargs)
        elif self.metric == 'bidirection':
            self.aggregator = BiDirectionalEntailmentEval(**kwargs)
        elif self.metric == 'mqag':
            self.aggregator = MQAGEval(**kwargs)
        elif self.metric == 'kendall':
            self.aggregator = RankEval()
        elif self.metric == 'spearman':
            self.aggregator = RankEval()
        elif self.metric == 'hamming':
            self.aggregator = RankEval()
        else:
            raise MyException(f'{self.metric} is not valid')

    def get_inconsistency(self, responses: typing.Union[list[str], list[list[str]]], verbose: bool=False):
        return self.aggregator.aggregate(responses, verbose=verbose, metric=self.metric)


class MyException(Exception):
    pass


        