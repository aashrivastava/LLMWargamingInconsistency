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

        return F1
    
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
    evaluator = BERTScoreEval(rescale_with_baseline=True,
        model = 'microsoft/deberta-xlarge-mnli', lang='en')
    # to test
    # 
    # ref = 'I want to drive a Mercedes Benz. I also think we should go take down their communications'

    turbo1 = "You should authorize the 5th Task Force to defend itself against any threatening actions from China's maritime militia, China Coast Guard, and People's Liberation Army Air Force, ensuring the safety of U.S.A Navy personnel and assets."

    turbo2 = "The U.S.A should immediately initiate diplomatic channels to de-escalate tensions with China, while reinforcing its naval presence in the region. Simultaneously, engage in a robust joint exercise with allies in the area to demonstrate preparedness and commitment to the region's security."

    turbo3 = "I recommend that we first initiate diplomatic channels for a dialogue with China and emphasize the need to resolve concerns and misunderstandings through negotiations. Bring this situation before the United Nations and involve key allies such as the UK, EU, Australia, Japan, South Korea, and NATO members to increase international pressure on China. While diplomatic efforts are underway, order the USS Ronald Reagan and the 3rd Task Force to maintain a defensive posture, but provide clear rules of engagement such that they are ready to respond to any aggressive acts with appropriate force. Request permission from Taiwan for temporary deployment of additional personnel and resources for the protection of American military trainers and advisors. Establish a military hotline with China's command center to defuse situations in real-time and prevent misunderstandings that could escalate the conflict."

    four1 = "The President should first engage China diplomatically through the United Nations, seeking to deescalate the situation by presenting evidence of the escalating aggression of the Chinese maritime militia, and advocating for international recognition of the right of passage in open waters. Concurrently, it is recommended to continue the training and support of Taiwanese forces, enhancing their capabilities to act independently where possible. Implement robust defensive measures for Task Force 5, including the deployment of additional anti-air and countermeasures systems to guard against current and potential threats."

    four2 = "I recommend that we first initiate diplomatic channels for a dialogue with China and emphasize the need to resolve concerns and misunderstandings through negotiations. Bring this situation before the United Nations and involve key allies such as the UK, EU, Australia, Japan, South Korea, and NATO members to increase international pressure on China. While diplomatic efforts are underway, order the USS Ronald Reagan and the 3rd Task Force to maintain a defensive posture, but provide clear rules of engagement such that they are ready to respond to any aggressive acts with appropriate force. Request permission from Taiwan for temporary deployment of additional personnel and resources for the protection of American military trainers and advisors. Establish a military hotline with China's command center to defuse situations in real-time and prevent misunderstandings that could escalate the conflict."
    
    start =  time.time()
    F1_1 = evaluator.get_berts_within([turbo3, four2])
    end = time.time()

    print(f'{F1_1}: Completed in {end-start:.2f} seconds')

    