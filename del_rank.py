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
    if 'False' in path:
        file_end = '_fixed'
        start = 7
    else:
        file_end = ''
        start = 1
    for i in range(start,21):
        os.makedirs(f'{path}/run{i}_fixed', exist_ok=True)
        try:
            m1, m2 = parser.parse_rankings(f'{path}/run{i}{file_end}.csv')
        except Exception as e:
            print(e, i, path)
            break
        print('done parsing')

        # t0 = time.time()
        k1, s1, h1 = get_all(m1)
        k1 = np.array(k1)
        s1 = np.array(s1)
        h1 = np.array(h1)
        print('Done Move 1')
        # t1 = time.time()
        # print(f'Elapsed time: {t1 - t0:.2f} seconds')
        
        # t0 = time.time()
        k2, s2, h2 = get_all(m2)
        k2 = np.array(k2)
        s2 = np.array(s2)
        h2 = np.array(h2)
        print('Done Move 2')
        # t1 = time.time()
        # print(f'Elapsed time: {t1 - t0:.2f} seconds')

        np.save(f'{path}/run{i}_fixed/kendall_move1.npy', k1)
        np.save(f'{path}/run{i}_fixed/kendall_move2.npy', k2)
        np.save(f'{path}/run{i}_fixed/spearman_move1.npy', s1)
        np.save(f'{path}/run{i}_fixed/spearman_move2.npy', s2)
        np.save(f'{path}/run{i}_fixed/hamming_move1.npy', h1)
        np.save(f'{path}/run{i}_fixed/hamming_move2.npy', h2)
        print(f'Done: run{i}')

    print('--------------DONE ONE WHOLE MODEL--------------')
