from bert_score import BERTScorer
import numpy as np
import torch
# from transformers import AutoTokenizer (probably)
import typing
# import promptopenai so that each metric can inherit the prompting so that you don't have to call
from tqdm import tqdm # progress bar

class BERTScoreEval:
    '''
    Evaluates similarity of LLM generated texts using BERTScore. Takes N sampled responses for a given query and calculates BERTScore between all combinations
    of pairs. Also aggregates these scores into one metric within range [0, 1]. 
    '''
    def __init__(self, lang: str = 'en', rescale_with_baseline: bool = True) -> None:
        self.lang = lang # uses roberta-large model if lang='en'
        self.rescale_with_baseline = rescale_with_baseline
        # self.scorer = BERTScorer(lang=self.lang, rescale_with_baseline=self.rescale_with_baseline)
        print('BERTScore Evaluator Initialized') # probably replace with logging
    
    def create_pairs(self, responses: list[str], verbose: bool = False) -> list[tuple[str, str]]:
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

    def aggregate(self, responses: list[str], verbose: bool = False) -> int:
        '''
        Given a list of N responses, generate the "unalikeness" metric using BERTScore

        Inputs:
            responses: list[str]
                list of responses that LLM generates to given query
        
        Outputs:
            int: "unalikeness" metric using BERTScore
        '''
        for pair in tqdm(pairs, desc='Calculating BERTEval...', disable=not verbose):
            raise NotImplementedError
    
# for my purposes 
if __name__ == '__main__':
    evaluator = BERTScoreEval()
    print(evaluator.create_pairs(['hi i am aryan', 'hi i am ha', 'james is my friend'] * 1000, verbose=True))