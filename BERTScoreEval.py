from bert_score import BERTScorer
import torch # to work with outputted tensors given by BERTScore
# from transformers import AutoTokenizer (probably)
import typing
# import promptopenai so that each metric can inherit the prompting so that you don't have to call
from tqdm import tqdm # progress bar
from EvalsBase import EvaluatorBasics

class BERTScoreEval(EvaluatorBasics):
    '''
    Evaluates similarity of LLM generated texts using BERTScore. Takes N sampled responses for a given query and calculates BERTScore between all combinations 
    of pairs. Also aggregates these scores into one metric within range [0, 1]. 
    '''
    def __init__(self, lang: str = 'en', rescale_with_baseline: bool = True):
        print('Initalizing BERTScore Evaluator...')
        self.lang = lang # uses roberta-large model if lang='en'
        self.rescale_with_baseline = rescale_with_baseline
        self.scorer = BERTScorer(lang=self.lang, rescale_with_baseline=self.rescale_with_baseline)
        print('BERTScore Evaluator Initialized') # probably replace with logging
        super().__init__()

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
    #print(evaluator.create_pairs(verbose=True))
    print(f'Unalikeness score: {evaluator.aggregate(['Initiate a diplomatic and military coalition with our allies to pressure Country Red to withdraw. Deploy defensive forces to support Country Blue while seeking an immediate ceasefire through international mediation to prevent further escalation.',
                               'Deploy military forces to assist Country Blue in repelling the invasion, demonstrating our commitment to defending allies and preserving regional stability.', 
                               'Mobilize our military forces to the border of Country Red to deter further aggression, while simultaneously engaging in diplomatic efforts with international allies to condemn the invasion and seek a peaceful resolution.',
                           'Deploy immediate military aid to Country Blue to help repel the invasion, showing our commitment to our ally and deterring further aggression from Country Red.',
                           'Invoke our mutual defense treaty with Country Blue and deploy our military forces to support them in repelling the invasion, thereby defending our honor, preserving our territory by preventing the conflict from spreading, and maintaining our relationship with our ally.',
                           'Issue an ultimatum to Country Red demanding an immediate withdrawal from Country Blue, backed by a show of military force and readiness to deploy troops if the demand is not met, thus defending our honor, preserving territory, and supporting our ally.'], verbose=True)}')