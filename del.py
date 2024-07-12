from main import Pipeline
from tqdm.auto import tqdm
import re
import csv

perms = [
    ('gpt-3.5-turbo', True),
    ('gpt-4-turbo', False),
    ('gpt-4-turbo', True)
]

def run_20_simuls(model, explicit_country):
    model_dir_name = re.sub(r'-', '', model)
    dir_name = f'{model_dir_name}-free-{explicit_country}-20-1.0-bert'
    pipe = Pipeline(model, 'free', explicit_country, 20, 1.0, 'bert')
    consistencies = []
    for i in tqdm(range(20), desc='Run simulations...'):
        response_saver = f'logging/responses/{dir_name}/run{i}'
        chat_saver = f'logging/chats/{dir_name}/run{i}'
        weird_saver = f'logging/weird/{dir_name}/run{i}'
        consistency_move1, consistency_move2 = pipe.main(response_saver, chat_saver, weird_saver)
        consistencies.append((consistency_move1, consistency_move1))
        print(f'Consistency move 1: {consistency_move1}')
        print(f'Consistency move 2: {consistency_move2}')
    with open(f'logging/consistency/{dir_name}/consistency.csv', 'w') as f:
        writer = csv.writer(f)
        header = ['Move 1', 'Move 2']
        writer.writerow(header)
        for consistency in consistencies:
            writer.writerow(list(consistency))

for perm in perms:
    run_20_simuls(perm[0], perm[1])
