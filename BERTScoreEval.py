from bert_score import BERTScorer
import torch # to work with outputted tensors given by BERTScore
# from transformers import AutoTokenizer (probably)
import typing
# import promptopenai so that each metric can inherit the prompting so that you don't have to call
from tqdm import tqdm # progress bar

class BERTScoreEval:
    '''
    Evaluates similarity of LLM generated texts using BERTScore. Takes N sampled responses for a given query and calculates BERTScore between all combinations 
    of pairs. Also aggregates these scores into one metric within range [0, 1]. 
    '''
    def __init__(self, responses: list[str], lang: str = 'en', rescale_with_baseline: bool = True):
        self.responses = responses
        self.N = len(responses)
        self.lang = lang # uses roberta-large model if lang='en'
        self.rescale_with_baseline = rescale_with_baseline
        self.scorer = BERTScorer(lang=self.lang, rescale_with_baseline=self.rescale_with_baseline)
        print('BERTScore Evaluator Initialized') # probably replace with logging
    
    def create_pairs(self, verbose: bool = False) -> list[tuple[str, str]]:
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
        pairs = [(response_i, response_j) for response_i in tqdm(self.responses, desc='Creating Pairs...', disable=not verbose) for response_j in self.responses]
        return pairs

    def aggregate(self, verbose: bool = False) -> int:
        '''
        Given a list of N responses, generate the "unalikeness" metric using BERTScore.

        1 represents full unalikeness
        0 represents 100% alike

        Inputs:
            responses: list[str]
                list of responses that LLM generates to given query
        
        Outputs:
            int: "unalikeness" metric using BERTScore
        '''
        pairs = self.create_pairs(verbose=verbose)
        tot = 0
        for t1, t2 in tqdm(pairs, desc='Calculating BERTEval...', disable=not verbose):
            #print(t1, t2)
            P, R, F1 = self.scorer.score([t1], [t2])
            tot += (1 - F1.item()) 
        
        return tot / (self.N**2 - self.N)
    
# for my purposes 
if __name__ == '__main__':
    evaluator = BERTScoreEval(['test1', 'test2', 'test3'])
    #print(evaluator.create_pairs(verbose=True))
    print(evaluator.aggregate(verbose=False))