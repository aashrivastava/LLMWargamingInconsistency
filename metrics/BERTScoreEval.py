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
    # evaluator = BERTScoreEval(rescale_with_baseline=True,
    #     model = 'microsoft/deberta-xlarge-mnli', lang='en')
    # to test
    # 
    # ref = 'I want to drive a Mercedes Benz. I also think we should go take down their communications'

    sents = [
        '''
Veins appear blue due to the way light interacts with your skin and the blood within your veins. When light hits your skin, it penetrates to different depths and interacts with tissues and blood vessels. Higher energy blue light is scattered more than other colors by the skin and is more likely to return to the surface, making the veins look blue despite the blood being red.''',
        '''Veins appear blue primarily due to the way light penetrates the skin and is absorbed and scattered back to the viewer's eyes.''',
        '''Veins appear blue primarily due to the way light interacts with the skin and the blood. Light from the environment penetrates the skin and is absorbed differently by various wavelengths. Veins contain oxygen-depleted blood, which is darker and absorbs more light. When the blue light wavelengths are scattered back to the observer more than other colors, the veins look blue from the surface of the skin.''',
        '''Veins appear blue primarily due to the way light interacts with the skin and the blood. When light hits the skin, it penetrates to different depths and wavelengths of light are absorbed and scattered differently. Blue light, being of shorter wavelength, is scattered more than other colors and is reflected back to our eyes more strongly from deeper tissues where veins are located.''',
        '''Veins appear blue due to the way light interacts with the skin and blood. Light penetrating the skin scatters in all directions, but blue light scatters more than other colors, making it more visible when it comes back out of the skin.''',
        '''Veins appear blue primarily because of the way light interacts with the skin and blood. Deoxygenated blood in veins absorbs more wavelengths of light, reflecting mostly blue light back to our eyes through the skin.''',
        '''Veins appear blue because of the way light interacts with your skin and the blood inside your veins. Although the blood is always red, due to the oxygen-rich hemoglobin, the skin and tissues absorb more of the red light wavelengths and scatter the blue light back to your eyes, making the veins look blue from the surface.''',
        '''Veins appear blue not because the blood inside them is blue, but due to the way light penetrates the skin and is absorbed or scattered by the tissues and blood.''',
        '''Veins appear blue primarily due to the way light interacts with the skin and blood. Light that hits the skin can either be absorbed or scattered. Veins absorb more of the red wavelengths of light, reflecting back the blue wavelengths more prominently due to the scattering properties of skin. This effect makes the veins appear blue, especially through lighter skin.''',
        '''Veins appear blue due to the way light penetrates the skin and is absorbed and reflected back to the eye.''',
        '''Veins appear blue due to the way light interacts with our skin and the blood within our veins. Light that penetrates the skin gets absorbed and scattered, and the higher energy (shorter wavelength) blue light is scattered more than the other colors. When this scattered blue light reaches our eyes, it makes the veins look blue, even though the blood itself is red.'''
    ]

    for sent in sents:
        print(len(sent.split(' ')))
    
    # F1_1 = evaluator.get_berts_within(sents)
    # print(F1_1)
    # print(torch.mean(F1_1))

    # print(f'{F1_1}: Completed in {end-start:.2f} seconds')

    