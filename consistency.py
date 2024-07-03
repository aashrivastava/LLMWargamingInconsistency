from utils.promptopenai import OpenAIPrompting
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
        self.prompter = OpenAIPrompting(model=prompting_model)
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
        
    def get_responses(self, prompt, N_responses=8, temperature=1.0):
        '''
        IMPLEMENT DOCSTRING
        '''
        completion = self.prompter.get_ChatCompletions(prompt, 
                            N_responses=N_responses)
        # parse for semantic
        if self.metric in ['bert', 'bidirection', 'mqag', 'rank']:
            parsed_outputs = self.prompter.parse_outputs(completion)
            return parsed_outputs
        else:
            parsed_outputs = self.prompter.parse_outputs(completion)
            rankings = self.prompter.get_rankings(parsed_outputs)
            return rankings
        
    
    def aggregate(self, responses, verbose: bool=False):
        return self.aggregator.aggregate(responses, verbose=verbose)
    
    def main(self, prompt, N_responses=8, temperature=1.0):
        return self.aggregator.aggregate(self.get_responses(prompt, N_responses=N_responses, temperature=temperature))


class MyException(Exception):
    pass


        