import typing
import torch
from tqdm import tqdm
from selfcheckgpt.modeling_mqag import MQAG as mqag
from utils.EvalsBase import EvaluatorBasics 
import math
# if getting module not found error for above line, set an environment variable called PYTHONPATH to the ABSOLUTE path to LLMWargamingConfidence
# ie $ export PYTHONPATH='PathToLLMWargamingConfidence'

## WHAT TO DO FOR LATER/TOMORROW
## ___PRESSING___
# connect promptopenai to these
# confirm that my aggregation is reasonable
# figure out how I can run this with compute
## ___NICE TO HAVE___
# get a baseline for texts

class MQAGEval(EvaluatorBasics):
    '''
    Evaluates similarity of LLM generated texts using the automatic multiple-choice question answering framework. 
    Takes N sampled responses for a given query and calculates BERTScore between all combinations of pairs. Also aggregates these scores into one metric within 
    range [0, 1]. 
    '''
    def __init__(self, model: str='race', device: str='cuda', num_questions: int=3, scoring_method: str='counting'):
        print('Initializing MQAG Evaluator...')
        self.num_questions = num_questions
        self.scoring_method = scoring_method
        if device == 'cuda':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        self.model = mqag(g1_model_type=model, device=self.device)
        super().__init__()

        print(f'MQAGScoreEval initialized to {self.device}')
    

    def score_questions(self, cand: str, ref: str, verbose: bool=False) -> float:
        '''
        Given two texts (candidate and reference), score the questions based on answers supported
        by the two texts. The score depends on the scoring method described by MQAG paper (Manakul et al. (2023)).
        On a high level, the scores represent various statistical distances between the probabilities assigned to answers supported by text 1
        and probabilities assigned to answers supported by text 2

        Inputs:
            NOTE: cand and ref are arbitrary. e.g. metric should be same if text1 is cand or text1 is ref as long as text2 is the same
            cand: str
                One of the texts to base the answering on
            ref: str
                Other text to base the answering on
        Output:
            float: statistical distance between answers based on the cand and ref text
        '''
        score = self.model.score(candidate=cand, reference=ref, num_questions=self.num_questions, verbose=verbose)
        score = score[self.scoring_method] # 0 is most alike, 1 is most unalike

        return score
    
    def aggregate(self, responses: list[str], verbose: bool=False) -> float:
        '''
        Generates "unalikeness" metric across a list of N responses when you use MQAG to generate consistency scores across
        two texts.

        Inputs:
            responses: list[str]
                List of responses that LLM outputs given a particular query
            verbose: bool
                represents whether you want to visualize progress
        Outputs:
            float: "unalikeness" metric using MQAG
        '''
        N = len(responses)
        pairs = self.create_unique_pairs(responses, verbose=verbose)
        tot = 0
        for t1, t2 in tqdm(pairs, desc='Calculating MQAGEval...', disable=not verbose):
            tot += self.score_questions(t1, t2, num_questions=self.num_questions, scoring_method=scoring_method, verbose=verbose)
        
        return tot / math.comb(N, 2)
    
if __name__ == '__main__':
    evaluator = MQAGScoreEval()
    ref = 'I think we should go to the store'
    contradict = 'I do not think we should go to the store'
    neutral = 'The mercedes is a good car'
    entails = 'I believe going to the store is a good idea'
    responses = [ref, entails, contradict, neutral]
    print(evaluator.aggregate(verbose=True))

