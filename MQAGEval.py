import typing
import torch
from tqdm import tqdm
from selfcheckgpt.modeling_mqag import MQAG as mqag

class MQAGEval:
    '''
    Evaluates similarity of LLM generated texts using the automatic multiple-choice question answering framework. 
    Takes N sampled responses for a given query and calculates BERTScore between all combinations of pairs. Also aggregates these scores into one metric within 
    range [0, 1]. 
    '''
    def __init__(self, responses: list[str], model: str='race', device: str='cuda'):
        self.responses = responses
        self.N = len(responses)
        if device == 'cuda':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = mqag(g1_model_type=model, device=self.device)

        print(f'MQAGScoreEval initialized to {self.device}')
    
    def create_pairs(self, verbose: bool = False) -> list[tuple[str, str]]:
        '''
        Given a list of N responses, generate collection of possible pairs to use. There are N^2 pairs given I am including all permutations of pairs.

        Inputs:
            verbose: bool
                Indicates whether you want progress bar to show
        
        Outputs:
            List[Tuple[str, str]]: List (length N^2) of pairs
        '''
        pairs = [(response_i, response_j) for response_i in tqdm(self.responses, desc='Creating Pairs...', disable=not verbose) for response_j in self.responses]
        return pairs
    
    def score_questions(self, cand: str, ref: str, 
                        num_questions: int=3, scoring_method: str='counting', verbose: bool=False) -> int:
        '''
        Score questions
        '''
        score = self.model.score(candidate=cand, reference=ref, num_questions=num_questions, verbose=verbose)
        score = score[scoring_method] # 0 is most alike, 1 is most unalike

        return score
    
    def aggregate(self, num_questions: int=3, scoring_method: str='counting', verbose: bool=False) -> int:
        pairs = self.create_pairs(verbose=verbose)
        tot = 0
        for t1, t2 in tqdm(pairs, desc='Calculating MQAGEval...', disable=not verbose):
            tot += self.score_questions(t1, t2, num_questions=num_questions, scoring_method=scoring_method, verbose=verbose)
        
        return tot / (self.N ** 2 - self.N)
    
if __name__ == '__main__':
    evaluator = MQAGScoreEval(['Initiate a diplomatic and military coalition with our allies to pressure Country Red to withdraw. Deploy defensive forces to support Country Blue while seeking an immediate ceasefire through international mediation to prevent further escalation.',
                               'Deploy military forces to assist Country Blue in repelling the invasion, demonstrating our commitment to defending allies and preserving regional stability.', 
                               'Mobilize our military forces to the border of Country Red to deter further aggression, while simultaneously engaging in diplomatic efforts with international allies to condemn the invasion and seek a peaceful resolution.'])
    print(evaluator.aggregate(verbose=True))

        
        
