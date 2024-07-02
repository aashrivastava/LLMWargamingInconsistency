from bert_score import BERTScorer
import torch # to work with outputted tensors given by BERTScore
# from transformers import AutoTokenizer (probably)
import typing
# import promptopenai so that each metric can inherit the prompting so that you don't have to call
from tqdm import tqdm # progress bar
from utils.EvalsBase import EvaluatorBasics
from utils.promptopenai import OpenAIPrompting

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

    def aggregate(self, responses: list[str], verbose: bool = False) -> int:
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
        N = len(responses)
        pairs = self.create_pairs(responses, verbose=verbose)
        tot = 0
        for t1, t2 in tqdm(pairs, desc='Calculating BERTEval...', disable=not verbose):
            #print(t1, t2)
            P, R, F1 = self.scorer.score([t1], [t2])
            tot += (1 - F1.item()) 
        
        return tot / (N**2 - N)
    
# for my purposes 
if __name__ == '__main__':
    evaluator = BERTScoreEval()
    ref = 'I think we should go to the store'
    contradict = 'I do not think we should go to the store'
    neutral = 'The mercedes is a good car'
    entails = 'I believe going to the store is a good idea'
    responses = [ref, entails, contradict, neutral]
    #print(evaluator.create_pairs(verbose=True))
    print(f'Unalikeness score: {evaluator.aggregate(responses, verbose=True)}')