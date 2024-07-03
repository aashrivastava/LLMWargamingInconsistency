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
    def __init__(self, prompting_model, method, **kwargs):
        '''
        **kwargs is keyword arguments pertaining to particular consistency method
        '''
        assert method in ['bert', 'bidirection', 'mqag', 'rank']
        # assertion that given model is valid
        self.prompter = OpenAIPrompting(model=prompting_model)
        self.method = method
        if method == 'bert':
            self.aggregator = BERTScoreEval(**kwargs)
        elif method == 'bidirection':
            self.aggregator = BiDirectionalEntailmentEval(**kwargs)
        elif method == 'mqag':
            self.aggregator = MQAGEval(**kwargs)
        elif method == 'rank':
            self.aggregator = RankEval()
        else:
            raise MyException(f'{method} is not valid')
        
    def get_responses(self, prompt, N_responses=20, temperature=1.0):
        '''
        IMPLEMENT DOCSTRING
        '''
        completion = self.prompter.get_ChatCompletions(prompt, 
                            N_responses=N_responses)
        # parse for semantic
        if self.method in ['bert', 'bidirection', 'mqag', 'rank']:
            parsed_outputs = self.prompter.parse_outputs(completion)
        else:
            parsed_outputs = self.prompter.parse_outputs(completion)
            rankings = self.prompter.get_rankings(parsed_outputs)
        
        return parsed_outputs
    
    def aggregate(self, responses, verbose: bool=False):
        return self.aggregator.aggregate(responses, verbose=verbose)


class MyException(Exception):
    pass


        