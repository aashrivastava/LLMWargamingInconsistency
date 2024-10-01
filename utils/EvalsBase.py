import typing
import numpy as np
import math
from tqdm import tqdm

## WHAT TO DO FOR LATER/TOMORROW
## ___PRESSING___
# connect promptopenai to this so I can just generate the responses list in proper format
## ___NICE TO HAVE___


class EvaluatorBasics:
    '''
    basic functions that each evaluator can use
    '''
    def __init__(self):
        pass
    
    def create_pairs(self, responses: list[str], verbose: bool=False) -> list[tuple[str, str]]:
        '''
        Given a list of N responses, generate collection of possible pairs to use. There are N^2 pairs given I am including all permutations of pairs.

        Inputs:
            responses: list[str]
                list of responses that LLM generates to given query
            verbose: bool
                Indicates whether you want progress bar to show
        
        Outputs:
            List[Tuple[str, str]]: List (length N^2) of pairs
        '''
        pairs = [(response_i, response_j) for response_i in tqdm(responses, desc='Creating Pairs...', disable=not verbose) for response_j in responses]
        return pairs
    
    def create_unique_pairs(self, responses: list[str] or list[dict[str, int]], verbose: bool=False) -> list[tuple[str, str]]:
        '''
        DOCSTRING
        '''
        unique_pairs = [(response_i, responses[j]) for i, response_i in tqdm(enumerate(responses), desc='Creating Pairs...', disable=not verbose) for j in range(i+1, len(responses))]

        return unique_pairs
