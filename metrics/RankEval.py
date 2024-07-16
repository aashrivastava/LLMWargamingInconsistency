import typing
from tqdm import tqdm
import math
from utils.EvalsBase import EvaluatorBasics
import random

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
        Calculates kendall's tau between two rankings. Rescaled from [-1, 1] to [0, 1]

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

        return tau
    
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
            tot += (1 - curr_tau)
        
        
        return tot / math.comb(N, 2)
        # get pairs and think about how to aggregate
    
    def _spearmans_coef(self, rank1: dict[str, int], rank2: dict[str, int], verbose: bool=False) -> float:
        '''
        Calculates spearman's rank coefficient between two rankings. Rescaled from [-1, 1] to [0, 1]

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
        return (2 - ((6 * sum_diffs)/(n*(n**2 - 1))))/2
    
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
            tot += (1 - self._spearmans_coef(r1, r2, verbose=verbose))
        
        return tot / math.comb(N, 2)
    
    def _hamming_distance(self, rank1: dict[str, int], rank2: dict[str, int], verbose: bool=False) -> float:
        '''
        Calculates Hamming distance between two rankings. Simply counts how many differences in rankings there are.
        I divide by the number of categories to rescale to a number between [0,1]

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
    
    def aggregate(self, responses: list[dict[str, int]], metric: str='kendall', verbose: bool=False) -> float:
        '''
        Depending on what metric was specified in the constructor, choose which aggregator to use

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
        
if __name__ == '__main__':
    rank1 = ['a', 'b', 'c', 'd']
    rank2 = ['a', 'b', 'c', 'd']
    rank3 = ['d', 'c', 'b','a']

    rankings = [rank1, rank3]
    print(rankings)

    evaluator = RankEval()
    print(f'Kendall: {evaluator.aggregate(rankings, metric='spearman')}')

