import typing
from tqdm import tqdm
import math
from utils.EvalsBase import EvaluatorBasics

class RankEval(EvaluatorBasics):
    '''
    IMPLEMENT DOCSTRING
    '''
    def __init__(self, method: str='kendall'):
        self.method = method
        super().__init__()
    
    def kendalls_tau(self, rank1: dict[str, int], rank2: dict[str, int], verbose: bool=False):
        '''
        IMPLEMENT DOCSTRING
        '''
        assert len(rank1) == len(rank2)
        assert rank1.keys() == rank2.keys()
        
        # get the number of discordant pairs
        # discordance is defined as:
        #   Rank1_cat1 < Rank1_cat2 and Rank2_cat1 < Rank2_cat2 OR
        #   Rank1_cat1 > Rank1_cat2 and Rank2_cat1 > Rank2_cat2 OR
        categories = list(rank1.keys())
        num_discordant = 0

        for i in tqdm(range(len(rank1)), desc='Calculating Kendall\'s tau', disable=not verbose):
            for j in range(i+1, len(rank1)):
                rank1_i = rank1[categories[i]]
                rank1_j = rank1[categories[j]]

                rank2_i = rank2[categories[i]]
                rank2_j = rank2[categories[j]]

                direction_1 = rank1_i - rank1_j
                direction_2 = rank2_i - rank2_j

                if direction_1 * direction_2 < 0:
                    num_discordant += 1
        
        tau = 1 - ((2 * num_discordant)/math.comb(len(rank1), 2))

        # rescale to [0, 1]
        tau = (tau + 1)/2

        return tau
    
    def aggregate_tau(self, responses: list[dict[str, int]], verbose: bool=False):
        '''
        IMPLEMENT DOCSTRING
        '''
        pairs = self.create_unique_pairs(responses, verbose=verbose)
        N = len(responses)
        tot = 0
        # get pairs and think about how to aggregate
        
        

