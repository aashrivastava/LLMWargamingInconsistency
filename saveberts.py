import os
from metrics.BERTScoreEval import BERTScoreEval
from utils.parse_csv import Parser
from tqdm.auto import tqdm
import numpy as np
import time

parser = Parser()
beval = BERTScoreEval()


paths = [
    ('/Users/aryanshrivastava/Desktop/LLMWargamingConfidence/logging/outputs/v4/gpt3.5turbo/free/status_quo/gpt3.5turbo-free-False-20-1.0/main', ''),
    ('/Users/aryanshrivastava/Desktop/LLMWargamingConfidence/logging/outputs/v4/gpt3.5turbo/free/status_quo/gpt3.5turbo-free-True-20-1.0/main', ''),
]


for path in tqdm(paths, desc='Running through models...'):
    for i in tqdm(range(1, 21)):
        m1, m2 = parser.parse_free(f'{path[0]}/run{i}/run{i}{path[1]}.csv')
        
        t0 = time.time()
        berts1, berts2 = beval.get_berts_within(m1), beval.get_berts_within(m2)
        t1 = time.time()
        print(f'RUN {i}: Got both moves in {t1 - t0:.2f} seconds')

        berts1, berts2 = berts1.numpy(), berts2.numpy()

        np.savez(f'{path[0]}/run{i}/run{i}_berts.npz',
            move1=berts1,
            move2=berts2)

        
        



# for path in paths:
#     if path == '/Users/aryanshrivastava/Desktop/LLMWargamingConfidence/logging/outputs/v4/gpt3.5turbo-free-False-20-1.0':
#         start = 4
#     else:
#         start = 1

#     if 'False' in path:
#         file_end = '_fixed'
#     else:
#         file_end = ''
#     for i in range(start,21):
#         os.makedirs(f'{path}/run{i}_fixed', exist_ok=True)
#         m1, m2 = parser.parse_free(f'{path}/run{i}{file_end}.csv')
#         print('done parsing')

#         # t0 = time.time()
#         berts1, bis1 = get_both(m1)
#         berts1 = np.array(berts1)
#         bis1 = np.array(bis1)
#         print('Done Move 1')
#         # t1 = time.time()
#         # print(f'Elapsed time: {t1 - t0:.2f} seconds')
        
#         # t0 = time.time()
#         berts2, bis2 = get_both(m2)
#         berts2 = np.array(berts2)
#         bis2 = np.array(bis2)
#         print('Done Move 2')
#         # t1 = time.time()
#         # print(f'Elapsed time: {t1 - t0:.2f} seconds')

#         np.save(f'{path}/run{i}_fixed/bert_move1.npy', berts1)
#         np.save(f'{path}/run{i}_fixed/bert_move2.npy', berts2)
#         print(f'Done: run{i}')

#     print('--------------DONE ONE WHOLE MODEL--------------')
