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
import time

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
    def __init__(self, 
        lang: str = 'en', 
        rescale_with_baseline: bool = True,
        model: str = 'microsoft/deberta-xlarge-mnli'):


        print('Initalizing BERTScore Evaluator...')
        self.lang = lang
        self.rescale_with_baseline = rescale_with_baseline
        self.model_type = model

        self.scorer = BERTScorer(lang=self.lang, 
            rescale_with_baseline=self.rescale_with_baseline,
            model_type = self.model_type)

        super().__init__()
        print(f'BERTScore Evaluator Initialized')
    
    def regular_score(self, cands, refs):
        P, R, F1 = self.scorer.score(cands, refs)

        return 1 - F1
    
    def get_berts_within(self, responses: list[str]) -> torch.Tensor:
        '''
        Given a list of responses, compute the individual BERTScores between each pair.
        Subtract from 1 in order to get INconsistency.

        Inputs:
            responses: list[str]
                List of responses
        
        Outputs:
            torch.Tensor: tensor with each computed BERTScore within each pair
        '''
        pairs = self.create_unique_pairs(responses)

        responses1 = []
        responses2 = []
        for r1, r2 in pairs:
            responses1.append(r1)
            responses2.append(r2)
        
        P, R, F1 = self.scorer.score(responses1, responses2)

        # just in case calculated bertscore is outside [0, 1]
        # very rare, really only happens when texts are the exact same
        return F1.apply_(lambda x: 0 if 1 - x < 0 else 1 if 1 - x > 1 else 1 - x)

    
    def get_single_score(self, r1: str, r2: str) -> float:
        '''
        Return the BERTScore dis-similarity between 2 strings

        Inputs:
            r1: str
                One string
            r2: str
                Other string
        
        Output: float
            Dis-similarity score based on BERTScore
        '''
        P, R, F1 = self.scorer.score([r1], [r2])

        return 1 - F1.item()
    
    def get_berts_across(self, responses1: list[str], responses2: list[str], verbose: bool=False):
        '''
        DOCSTRING
        '''
        grid1, grid2 = np.meshgrid(responses1, responses2, indexing='ij')

        v_func = np.vectorize(self.get_single_score)

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
    evaluator = BERTScoreEval()
    # to test
    # 
    # ref = 'I want to drive a Mercedes Benz. I also think we should go take down their communications'

    ref = "Eating watermelon seeds is generally safe, as they can pass through your digestive system without causing harm, although some people believe they may cause stomach discomfort or sprout in your stomach, which is a myth."

    cand = "Eating watermelon seeds is typically safe, as they can move through your digestive system without causing harm, although some individuals believe they may cause stomach discomfort or germinate in your stomach, which is a myth."

    
    F1_1 = evaluator.regular_score([ref], [cand])
    print(F1_1)
    # print(torch.mean(F1_1))

    # print(f'{F1_1}: Completed in {end-start:.2f} seconds')

    