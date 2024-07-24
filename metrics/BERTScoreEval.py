from bert_score import BERTScorer
import torch # to work with outputted tensors given by BERTScore
# from transformers import AutoTokenizer (probably)
import typing
# import promptopenai so that each metric can inherit the prompting so that you don't have to call
from tqdm.auto import tqdm # progress bar
from utils.EvalsBase import EvaluatorBasics
from utils.promptopenai import OpenAIPrompting
import math
import numpy as np

## WHAT TO DO FOR LATER/TOMORROW
## ___PRESSING___
# connect promptopenai to these
# confirm that my aggregation is reasonable
## ___NICE TO HAVE___
# get a baseline for texts

class BERTScoreEval(EvaluatorBasics):
    '''
    Evaluates similarity of LLM generated texts using BERTScore. Takes N sampled responses for a given query and calculates BERTScore between all combinations 
    of pairs. Also aggregates these scores into one metric within range [0, 1]. 
    '''
    def __init__(self, lang: str = 'en', rescale_with_baseline: bool = True, device: str='cuda'):
        print('Initalizing BERTScore Evaluator...')
        self.lang = lang # uses roberta-large model if lang='en'
        self.rescale_with_baseline = rescale_with_baseline
        self.scorer = BERTScorer(lang=self.lang, rescale_with_baseline=self.rescale_with_baseline)
        if device == 'cuda':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        super().__init__()
        print(f'BERTScore Evaluator Initialized to {self.device}') # probably replace print with logging
    
    def get_single_score(self, r1: str, r2: str) -> float:
        '''
        DOCSTRING
        '''
        P, R, F1 = self.scorer.score([r1], [r2])

        return 1 - F1.item()

    def get_berts_within(self, responses: list[str], verbose: bool = False) -> list[float]:
        '''
        DOCSTRING
        '''
        result = []

        pairs = self.create_unique_pairs(responses, verbose=verbose)
        for t1, t2 in tqdm(pairs, desc='Getting BERTScores', disable=not verbose):
            P, R, F1 = self.scorer.score([t1], [t2])

            result.append(1 - F1.item())
        
        return result
    
    def get_berts_across(self, responses1: list[str], responses2: list[str], verbose: bool=False):
        '''
        DOCSTRING
        '''
        grid1, grid2 = np.meshgrid(responses1, responses2, indexing='ij')

        v_func = np.vectorize(get_single_score)

        return v_func(grid2, grid1)



    def aggregate(self, responses: list[str], verbose: bool = False, **kwargs) -> float:
        '''
        Given a list of N responses, generate the "unalikeness" metric using BERTScore.

        1 represents full unalikeness
        0 represents 100% alike

        Inputs:
            responses: list[str]
                list of responses that LLM generates to given query
            verbose: bool
                represents whether you want to visualize progress
        
        Outputs:
            int: "unalikeness" metric using BERTScore
        '''
        N = len(responses)
        # pairs = self.create_pairs(responses, verbose=verbose)
        unique_pairs = self.create_unique_pairs(responses, verbose=verbose)
        tot = 0
        for t1, t2 in tqdm(unique_pairs, desc='Calculating BERTEval...', disable=not verbose):
            P, R, F1 = self.scorer.score([t1], [t2])
            # if verbose:
            #     print(f'BERTScore between {t1} AND {t2}: {F1.item()}')

            tot += (1 - F1.item()) 
        
        # return 0 if we get a negative 1 - bert score for whatever reason
        if tot < 0:
            return 0
         # taking the minimum because it might be that bertscore gives something slightly larger than 1
        else:
            return min(tot / math.comb(N, 2), 1.0)
    
    
# for my purposes 
if __name__ == '__main__':
    evaluator = BERTScoreEval(rescale_with_baseline=True)
    # to test
    # 
    # ref = 'I want to drive a Mercedes Benz. I also think we should go take down their communications'
    a1 = 'we should nuke china'
    a2 = 'we should invade china'
    a3 = 'taiwan needs help'
    exps = [a1, a2, a3]
    e1 = 'we should nuke gray'
    e2 = 'we should invade gray'
    e3 = 'pink needs help'
    anons = [e1, e2, e3]

    print(evaluator.get_berts_across(anons, exps))

    