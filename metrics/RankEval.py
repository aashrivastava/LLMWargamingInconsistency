import typing
from tqdm import tqdm
import math
from utils.EvalsBase import EvaluatorBasics
import random
import numpy as np
import scipy.stats as sc

## WHAT TO DO FOR LATER/TOMORROW
## ___PRESSING___
# nothing
## ___NICE TO HAVE___
# more ranking similarity metrics
# get a baseline for random rankings for different metrics

class RankEval(EvaluatorBasics):
    '''
    Used to calculate "unalikeness" metric given N rankings. There are 3 currently implemented metrics:
        kendall's tau
        spearman's rank coefficient
        hamming distance
    '''
    def __init__(self):
        super().__init__()
    
    def make_assertions(self, rank1: dict[str, int], rank2: dict[str, int]) -> None:
        '''
        Asserts that the rankings to compare have same size and rank the same categories

        Inputs:
            rank1, rank2: dict[str, int]
                Keys are the category and the value is its associated rank
        '''
        assert len(rank1) == len(rank2)
        assert rank1.keys() == rank2.keys()
    
    def _for_analysis(self, parsed_responses: list[dict[str, int]], method: str='kendall') -> tuple[list[float], list[float]]:
        '''
        just to do analysis on distribution of method
        '''
        if method == 'kendall':
            to_use = self._kendalls_tau
        elif method == 'spearman':
            to_use = self._spearmans_coef
        elif method == 'hamming':
            to_use = self._hamming_distance
        
        pairs = self.create_unique_pairs(parsed_responses)
        metrics = []
        one_minus = []
        for r1, r2 in pairs:
            metric = to_use(r1, r2)
            metrics.append(metric)
            one_minus.append(1 - metric)
        
        return metrics, one_minus


    def _kendalls_tau(self, rank1: dict[str, int], rank2: dict[str, int], verbose: bool=False) -> float:
        '''
        Calculates rescaled 1 -kendall's tau between two rankings. Rescaled from [-1, 1] to [0, 1]. 
        This is kendall inconsistency for one pair

        Inputs:
            rank1: dict[str, int]
                keys are category and the value is associated category's rank
            rank2: dict[str, int]
                keys are category and the value is associated category's rank
            verbose: bool   
                represents whether you want to visualize progress
        
        Output:
            float: Rescaled kendall's tau metric between given ranks
        '''
        self.make_assertions(rank1, rank2)
        
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

        tau_reversed = 1 - tau

        return tau_reversed
    
    def _aggregate_kendalls(self, responses: list[dict[str, int]], verbose: bool=False) -> float:
        '''
        Calculates "unalikness" metric for list of N responses based on kendall's tau. Basically just takes average of (1- kendall's tau)
        for each pair of responses

        Inputs:
            responses: list[list[str]]
                List of rankings that LLM gives
            verbose: bool
                represents whether you want to visualize progress
        
        Outputs:
            float: "unalikeness" metric using kendall's tau
        '''

        pairs = self.create_unique_pairs(responses, verbose=verbose)
        N = len(responses)
        
        # aggregator according to formula seen in paper
        tot = 0
        for r1, r2 in tqdm(pairs, desc='Calculating RankEval using Kendall\'s Tau...', disable=not verbose):
            curr_tau = self._kendalls_tau(r1, r2, verbose=verbose)
            # taus.append(curr_tau)
            # one_minus_taus.append(1 - curr_tau)
            tot += curr_tau
        
        
        return tot / math.comb(N, 2)
        # get pairs and think about how to aggregate
    
    def _spearmans_coef(self, rank1: dict[str, int], rank2: dict[str, int], verbose: bool=False) -> float:
        '''
        Calculates rescaled 1- spearman's rank coefficient between two rankings. Rescaled from [-1, 1] to [0, 1]

        Inputs:
            rank1: dict[str, int]
                keys are category and the value is associated category's rank
            rank2: dict[str, int]
                keys are category and the value is associated category's rank
            verbose: bool   
                represents whether you want to visualize progress
        
        Output:
            float: Rescaled spearman's coefficient between given ranks
        '''
        self.make_assertions(rank1, rank2)

        categories = list(rank1.keys())
        n = len(categories)

        sum_diffs = 0
        for cat in categories:
            sum_diffs += (rank1[cat] - rank2[cat])**2
        
        # linearly normalized into [0,1] instead of [-1, 1]
        rho = (2 - ((6 * sum_diffs)/(n*(n**2 - 1))))/2
        rho_reversed = 1 - rho

        return rho_reversed
    
    def _aggregate_spearmans(self, responses: list[dict[str, int]], verbose: bool=False) -> float:
        '''
        Calculates "unalikness" metric for list of N responses based on spearman's coefficient. Basically just takes average of 
        (1- spearman's coefficient) for each pair of responses

        Inputs:
            responses: list[str]
                List of responses that LLM outputs given a particular query
            verbose: bool
                represents whether you want to visualize progress
        
        Outputs:
            float: "unalikeness" metric using spearman's rank coefficient
        '''
        pairs = self.create_unique_pairs(responses, verbose=verbose)
        N = len(responses)
        
        # aggregator according to formula seen in paper
        tot = 0
        for r1, r2 in tqdm(pairs, desc='Calculating RankEval using Spearman\'s Rank Coefficient...', disable=not verbose):
            curr_rho = self._spearmans_coef(r1, r2, verbose=verbose)
            tot += curr_rho
        
        return tot / math.comb(N, 2)
    
    def _hamming_distance(self, rank1: dict[str, int], rank2: dict[str, int], verbose: bool=False) -> float:
        '''
        Calculates rescaled Hamming distance between two rankings. Simply counts how many differences in rankings there are.

        Inputs:
            rank1: dict[str, int]
                keys are category and the value is associated category's rank
            rank2: dict[str, int]
                keys are category and the value is associated category's rank
            verbose: bool   
                represents whether you want to visualize progress
        
        Output:
            float: Rescaled spearman's coefficient between given ranks
        '''
        self.make_assertions(rank1, rank2)

        categories = list(rank1.keys())
        tot_diffs = 0
        for cat in categories:
            if rank1[cat] != rank2[cat]:
                tot_diffs += 1
            
        return tot_diffs / len(rank1)
    
    def _aggregate_hamming(self, responses: list[dict[str, int]], verbose: bool=False) -> float:
        '''
        Calculates "unalikness" metric for list of N responses based on rescaled hamming distance. Basically just takes average of 
        hamming distance for each pair of responses

        Inputs:
            responses: list[str]
                List of responses that LLM outputs given a particular query
            verbose: bool
                represents whether you want to visualize progress
        
        Outputs:
            float: "unalikeness" metric using hamming distance
        '''
        pairs = self.create_unique_pairs(responses, verbose=verbose)
        N = len(responses)
        
        tot = 0
        for r1, r2 in tqdm(pairs, desc='Calculating RankEval using Spearman\'s Rank Coefficient...', disable=not verbose):
            tot += self._hamming_distance(r1, r2, verbose=verbose)
        
        return tot / math.comb(N, 2)
    
    def get_kendalls(self, responses: list[dict[str, int]], verbose: bool=False) -> list[float]:
        '''
        Get a list of all rescaled 1 - kendall's tau for each pair of rankings in responses

        Inputs:
            responses: list[dict[str, int]]
                list of rankings
            verbose: bool
                Indicates whether you want progress bar to show
        
        Output: list[float]
            List containing all taus
        '''
        result = []

        pairs = self.create_unique_pairs(responses, verbose=verbose)

        for r1, r2 in tqdm(pairs, desc='Getting Kendalls for Responses...', disable=not verbose):
            try:
                result.append(self._kendalls_tau(r1, r2, verbose=verbose))
            except:
                result.append(np.nan)
        
        return result
    
    def get_spearmans(self, responses: list[dict[str, int]], verbose: bool=False) -> list[float]:
        '''
        Get a list of all rescaled 1 - spearman's for each pair of rankings in responses

        Inputs:
            responses: list[dict[str, int]]
                list of rankings
            verbose: bool
                Indicates whether you want progress bar to show
        
        Output: list[float]
            List containing spearmans 
        '''
        result = []
        pairs = self.create_unique_pairs(responses, verbose=verbose)

        for r1, r2 in tqdm(pairs, desc='Getting Spearmans for Responses...', disable=not verbose):
            try:
                result.append(self._spearmans_coef(r1, r2, verbose=verbose))
            except:
                result.append(np.nan)
        
        return result
    
    def get_hamming(self, responses: list[dict[str, int]], verbose: bool=False) -> list[float]:
        '''
        Get a list of all rescaled hamming distance for each pair of rankings in responses

        Inputs:
            responses: list[dict[str, int]]
                list of rankings
            verbose: bool
                Indicates whether you want progress bar to show
        
        Output: list[float]
            List containing all hamming
        '''
        result = []
        pairs = self.create_unique_pairs(responses, verbose=verbose)

        for r1, r2 in tqdm(pairs, desc='Getting Hammings for Responses...', disable=not verbose):
            try:
                result.append(self._hamming_distance(r1, r2, verbose=verbose))
            except:
                result.append(np.nan)
        
        return result
    
    def get_metric_within(self, responses: list[dict[str, int]], metric: str='kendall', verbose: bool = False) -> list[float]:
        '''
        Given a set of responses, get a list of the individual dis-similarities between all pairs

        Inputs:
            responses: list[str]
                Set of responses
            metric: str
                Denotes which metric out of kendall, spearman, or hamming you want to use
            verbose: bool
                Denotes whether you want progress bar to show
        
        Output: list[float]
            list of dis-similarities based on BERTScore
        '''
        result = []

        pairs = self.create_unique_pairs(responses, verbose=verbose)
        for r1, r2 in tqdm(pairs, desc='Getting BERTScores', disable=not verbose):
            if metric == 'kendall':
                score = self._kendalls_tau(r1, r2)
            elif metric == 'spearman':
                score = self._spearmans_coef(r1, r2)
            else:
                score = self._hamming_distance(r1, r2)


            result.append(score)
        
        return result
    
    def get_metric_across(self, ranks1: list[dict[str, int]], ranks2: list[dict[str, int]], metric: str='kendall'):
        '''
        Given two sets of rankings, get a matrix that contains all the inconsistency scores between each pair.
        ranks1 is on y axis, rank2 on x axis

        Inputs:
            ranks1: list[dict[str, int]]
                set of rankings
            ranks2: list[dict[str, int]]
                set of rankings
            metric: str
                metric to use
        
        Output: np.ndarray
            matrix containing the inconsistency scores between each pair
        '''
        assert metric in ['kendall', 'spearman', 'hamming']

        grid1, grid2 = np.meshgrid(ranks1, ranks2, indexing='ij')

        if metric == 'kendall':
            v = np.vectorize(self._kendalls_tau)
        elif metric == 'spearman':
            v = np.vectorize(self._spearmans_coef)
        else:
            v = np.vectorize(self._hamming_distance)
        
        return v(grid2, grid1)

    def aggregate(self, responses: list[dict[str, int]], metric: str='kendall', verbose: bool=False) -> float:
        '''
        Get inconsistency metric based on specified metric

        Inputs:
            responses: list[str]
                List of responses that LLM outputs given a particular query
            verbose: bool
                represents whether you want to visualize progress
        
        Outputs:
            float: "unalikeness" metric using specified metric from constructor
        '''
        assert len(responses) >= 2

        if metric == 'kendall':
            return self._aggregate_kendalls(responses, verbose=verbose)
        elif metric == 'spearman':
            return self._aggregate_spearmans(responses, verbose=verbose)
        elif metric == 'hamming':
            return self._aggregate_hamming(responses, verbose=verbose)
        else: raise NameError('This is not a valid metric')
        