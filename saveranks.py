import os
from metrics.RankEval import RankEval
from utils.parse_csv import Parser
from tqdm.auto import tqdm
import numpy as np

parser = Parser()
reval = RankEval()

# def check_bi(t1, t2):
#     d1 = bieval.entails_neutral_contradict(t1, t2)
#     d2 = bieval.entails_neutral_contradict(t2, t1)

#     if (d1 == 2 and d2 >= 0) or (d2 == 2 and d1 >= 0):
#         return 1
#     else:
#         return 0

def get_all(responses: list[dict[str, int]]):

    k = reval.get_kendalls(responses)
    s = reval.get_spearmans(responses)
    h = reval.get_hamming(responses)
    
    return k, s, h


paths = [
    ('/Users/aryanshrivastava/Desktop/LLMWargamingConfidence/logging/outputs/v4/gpt3.5turbo/rank/status_quo/gpt3.5turbo-rank-False-20-1.0/main', ''),
    ('/Users/aryanshrivastava/Desktop/LLMWargamingConfidence/logging/outputs/v4/gpt3.5turbo/rank/status_quo/gpt3.5turbo-rank-True-20-1.0/main', ''),
]

for path in paths:
    for i in range(1, 21):
        try:
            m1, m2 = parser.parse_rankings(f'{path[0]}/run{i}/run{i}{path[1]}.csv')
        except:
            print('cant parse')
            print(f'{path[0]}/run{i}{path[1]}/run{i}{path[1]}.csv')
            break
        
        try:
            k1, s1, h1 = get_all(m1)
            k2, s2, h2 = get_all(m2)
        except:
            print(f'{path[0]}/run{i}/run{i}.csv')
            continue

        k1, k2 = np.array(k1), np.array(k2)
        s1, s2 = np.array(s1), np.array(s2)
        h1, h2 = np.array(h1), np.array(h2)

        np.savez(f'{path[0]}/run{i}/run{i}_ranks.npz',
            kendall_move1=k1,
            kendall_move2=k2,
            spearman_move1=s1,
            spearman_move2=s2,
            hamming_move1=h1,
            hamming_move2=h2)
