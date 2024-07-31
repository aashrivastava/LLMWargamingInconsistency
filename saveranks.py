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

def get_all(responses: list[dict[str, int]], verbose=False):

    k = reval.get_kendalls(responses, verbose=verbose)
    s = reval.get_spearmans(responses, verbose=verbose)
    h = reval.get_hamming(responses, verbose=verbose)
    
    return k, s, h


paths = [
    '/Users/aryanshrivastava/Desktop/LLMWargamingConfidence/logging/outputs/v4/gpt4-rank-False-20-1.0',
]

for path in paths:
    for i in range(1, 21):
        m1, m2 = parser.parse_free(f'{path[0]}')
