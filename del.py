import os
from metrics.BERTScoreEval import BERTScoreEval
from metrics.BiDirectionalEntailmentEval import BiDirectionalEntailmentEval
from utils.parse_csv import Parser
from tqdm.auto import tqdm
import numpy as np
import time

parser = Parser()
beval = BERTScoreEval()
bieval = BiDirectionalEntailmentEval()

# def check_bi(t1, t2):
#     d1 = bieval.entails_neutral_contradict(t1, t2)
#     d2 = bieval.entails_neutral_contradict(t2, t1)

#     if (d1 == 2 and d2 >= 0) or (d2 == 2 and d1 >= 0):
#         return 1
#     else:
#         return 0

def get_both(responses: list[str], verbose=True):
    result = [[], []]
    pairs = beval.create_unique_pairs(responses)


    for t1, t2 in tqdm(pairs, desc='Getting arrays...', disable=not verbose):
        bert = beval.get_single_score(t1, t2)
        d1 = bieval.entails_neutral_contradict(t1, t2)
        d2 = bieval.entails_neutral_contradict(t2, t1)

        if (d1 == 2 and d2 >= 0) or (d2 == 2 and d1 >= 0):
            bi = 1
        else:
            bi = 0
        result[0].append(bert)
        result[1].append(bi)
    
    return result[0], result[1]


paths = ['/Users/aryanshrivastava/Desktop/LLMWargamingConfidence/logging/outputs/v4/gpt3.5turbo-free-False-20-1.0',
    '/Users/aryanshrivastava/Desktop/LLMWargamingConfidence/logging/outputs/v4/gpt3.5turbo-free-True-20-1.0',
    '/Users/aryanshrivastava/Desktop/LLMWargamingConfidence/logging/outputs/v4/gpt4-free-False-20-1.0',
    '/Users/aryanshrivastava/Desktop/LLMWargamingConfidence/logging/outputs/v4/gpt4-free-True-20-1.0'
]

for path in paths:
    if path == '/Users/aryanshrivastava/Desktop/LLMWargamingConfidence/logging/outputs/v4/gpt3.5turbo-free-False-20-1.0':
        start = 4
    else:
        start = 1

    if 'False' in path:
        file_end = '_fixed'
    else:
        file_end = ''
    for i in range(start,21):
        os.makedirs(f'{path}/run{i}_fixed', exist_ok=True)
        m1, m2 = parser.parse_free(f'{path}/run{i}{file_end}.csv')
        print('done parsing')

        # t0 = time.time()
        berts1, bis1 = get_both(m1)
        berts1 = np.array(berts1)
        bis1 = np.array(bis1)
        print('Done Move 1')
        # t1 = time.time()
        # print(f'Elapsed time: {t1 - t0:.2f} seconds')
        
        # t0 = time.time()
        berts2, bis2 = get_both(m2)
        berts2 = np.array(berts2)
        bis2 = np.array(bis2)
        print('Done Move 2')
        # t1 = time.time()
        # print(f'Elapsed time: {t1 - t0:.2f} seconds')

        np.save(f'{path}/run{i}_fixed/bert_move1.npy', berts1)
        np.save(f'{path}/run{i}_fixed/bert_move2.npy', berts2)
        np.save(f'{path}/run{i}_fixed/bidir_move1.npy', bis1)
        np.save(f'{path}/run{i}_fixed/bidir_move2.npy', bis2)
        print(f'Done: run{i}')

    print('--------------DONE ONE WHOLE MODEL--------------')
