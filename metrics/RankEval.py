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
    IMPLEMENT DOCSTRING
    '''
    def __init__(self, method: str='kendall'):
        self.method = method
        super().__init__()
    
    def make_assertions(self, rank1: dict[str, int], rank2: dict[str, int]):
        assert len(rank1) == len(rank2)
        assert rank1.keys() == rank2.keys()
    
    def _kendalls_tau(self, rank1: dict[str, int], rank2: dict[str, int], verbose: bool=False):
        '''
        IMPLEMENT DOCSTRING
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
    
    def _aggregate_kendalls(self, responses: list[dict[str, int]], verbose: bool=False):
        '''
        IMPLEMENT DOCSTRING
        '''
        pairs = self.create_unique_pairs(responses, verbose=verbose)
        N = len(responses)
        
        # aggregator according to formula seen in paper
        tot = 0
        for r1, r2 in tqdm(pairs, desc='Calculating RankEval using Kendall\'s Tau...', disable=not verbose):
            tot += (1 - self._kendalls_tau(r1, r2, verbose=verbose))
        
        return tot / math.comb(N, 2)
        # get pairs and think about how to aggregate
    
    def _spearmans_coef(self, rank1: dict[str, int], rank2: dict[str, int], verbose: bool=False):
        '''
        IMPLEMENT DOCSTRING
        '''
        self.make_assertions(rank1, rank2)

        categories = list(rank1.keys())
        n = len(categories)

        sum_diffs = 0
        for cat in categories:
            sum_diffs += (rank1[cat] - rank2[cat])**2
        
        # linearly normalized into [0,1] instead of [-1, 1]
        return (2 - ((6 * sum_diffs)/(n*(n**2 - 1))))/2
    
    def _aggregate_spearmans(self, responses: list[dict[str, int]], verbose: bool=False):
        '''
        IMPLEMENT DOCSTRING
        '''
        pairs = self.create_unique_pairs(responses, verbose=verbose)
        N = len(responses)
        
        # aggregator according to formula seen in paper
        tot = 0
        for r1, r2 in tqdm(pairs, desc='Calculating RankEval using Spearman\'s Rank Coefficient...', disable=not verbose):
            tot += (1 - self._spearmans_coef(r1, r2, verbose=verbose))
        
        return tot / math.comb(N, 2)
    
    def _hamming_distance(self, rank1: dict[str, int], rank2: dict[str, int], verbose: bool=False):
        '''
        implement docstring
        '''
        self.make_assertions(rank1, rank2)

        categories = list(rank1.keys())
        tot_diffs = 0
        for cat in categories:
            if rank1[cat] != rank2[cat]:
                tot_diffs += 1
            
        return tot_diffs / len(rank1)
    
    def _aggregate_hamming(self, responses: list[dict[str, int]], verbose: bool=False):
        '''
        IMPLEMENT DOCSTRING
        '''
        pairs = self.create_unique_pairs(responses, verbose=verbose)
        N = len(responses)
        
        tot = 0
        for r1, r2 in tqdm(pairs, desc='Calculating RankEval using Spearman\'s Rank Coefficient...', disable=not verbose):
            tot += self._hamming_distance(r1, r2, verbose=verbose)
        
        return tot / math.comb(N, 2)
    
    def aggregate(self, responses, verbose: bool=False):
        assert len(responses) >= 2

        if self.method == 'kendall':
            return self._aggregate_kendalls(responses, verbose=verbose)
        elif self.method == 'spearman':
            return self._aggregate_spearmans(responses, verbose=verbose)
        elif self.method == 'hamming':
            return self._aggregate_hamming(responses, verbose=verbose)
        
if __name__ == '__main__':
    def generate_random_ranking():
        items = ['a', 'b', 'c', 'd']
        random.shuffle(items)
        return {item: rank + 1 for rank, item in enumerate(items)}
    
    random_rankings = [generate_random_ranking() for _ in range(10000)]
    for i, ranking in enumerate(random_rankings, start=1):
        print(f"Ranking {i}: {ranking}")

    evaluator = RankEval()
    print(f'Hamming: {evaluator.aggregate(random_rankings, method='hamming')}')
    print(f'Spearmans: {evaluator.aggregate(random_rankings, method='spearman')}')
    print(f'Kendall: {evaluator.aggregate(random_rankings, method='kendall')}')


