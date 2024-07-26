from main import Pipeline
from tqdm.auto import tqdm
import re
import csv
from utils.game import GameSimulator

perms = [
    ('gpt-3.5-turbo', False),
    ('gpt-4', False),
    ('claude-3-5-sonnet-20240620', False)
]

def run_20_simuls(model, explicit_country):
    if model != 'dummy' and 'claude' in model:
        model_dir_name = re.sub(r'-', '', model)[:-8]
        dir_name = f'{model_dir_name}-rank-{explicit_country}-20-1.0'
    elif model != 'dummy' and 'gpt' in model:
        model_dir_name = re.sub(r'-', '', model)
        dir_name = f'{model_dir_name}-rank-{explicit_country}-20-1.0'
    else:
        model_dir_name = 'dummy'
        dir_name = 'dummy'
    simulator = GameSimulator(model, 'rank', explicit_country, 20, 1.0)
    for i in tqdm(range(2,20), desc='Run simulations...'):
        o_directory = f'logging/outputs/v4/{dir_name}'
        f_name = f'run{i+1}_fixed'

        if 'claude' in model:
            outputs, chats = simulator.run_basic_anthropic()
            simulator.write_outputs(outputs, o_directory, f_name=f_name)
        elif 'gpt' in model:
            outputs, chats = simulator.run_basic_oai()
            simulator.write_outputs(outputs, o_directory, f_name=f_name)
    
    simulator.write_chat(chats, f'logging/chats/v4/{dir_name}', 'chat_fixed')

for perm in perms:
    run_20_simuls(perm[0], perm[1])
