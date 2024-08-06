from main import Pipeline
from tqdm.auto import tqdm
import re
import csv
from utils.game import GameSimulator

perms = [
    ('gpt-3.5-turbo', False),
    ('gpt-4', False),
]

def run_20_simuls_rank(model, explicit_country, start, end):
    if model != 'dummy' and 'claude' in model:
        model_dir_name = re.sub(r'-', '', model)[:-8]
        dir_name = f'{model_dir_name}-rank-{explicit_country}-20-1.0'
    elif model != 'dummy' and ('gpt' in model or 'lama' in model):
        model_dir_name = re.sub(r'-', '', model)
        dir_name = f'{model_dir_name}-rank-{explicit_country}-20-1.0'
    else:
        model_dir_name = 'dummy'
        dir_name = 'dummy'
    simulator = GameSimulator(model, 'rank', explicit_country, 20, 1.0)
    for i in tqdm(range(start, end), desc='Run simulations...'):
        o_directory = f'logging/outputs/v4/{dir_name}'
        f_name = f'run{i+1}'

        if 'claude' in model:
            outputs, chats = simulator.run_basic_anthropic()
            simulator.write_outputs(outputs, o_directory, f_name=f_name)
        elif 'gpt' in model or 'lama' in model:
            outputs, chats = simulator.run_basic_oai()
            simulator.write_outputs(outputs, o_directory, f_name=f_name)
    
    simulator.write_chat(chats, f'logging/chats/v4/{dir_name}', 'chat')

def run_20_simuls_free(model, explicit_country, start, end):
    if model != 'dummy' and 'claude' in model:
        model_dir_name = re.sub(r'-', '', model)[:-8]
        dir_name = f'{model_dir_name}-free-{explicit_country}-20-1.0'
    elif model != 'dummy' and ('gpt' in model or 'lama' in model):
        model_dir_name = re.sub(r'-', '', model)
        dir_name = f'{model_dir_name}-free-{explicit_country}-20-1.0'
    else:
        model_dir_name = 'dummy'
        dir_name = 'dummy'
    if not explicit_country:
        fixed = ''
    else:
        fixed = ''
    simulator = GameSimulator(model, 'free', explicit_country, 20, 1.0)
    for i in tqdm(range(start, end), desc='Run simulations...'):
        o_directory = f'logging/outputs/v4/{dir_name}'
        f_name = f'run{i+1}{fixed}'

        if 'claude' in model:
            outputs, chats = simulator.run_basic_anthropic()
            simulator.write_outputs(outputs, o_directory, f_name=f_name)
        elif 'gpt' in model or 'lama' in model:
            outputs, chats = simulator.run_basic_oai()
            simulator.write_outputs(outputs, o_directory, f_name=f_name)
    
    simulator.write_chat(chats, f'logging/chats/v4/{dir_name}', 'chat_fixed')

perms = [('gpt-4o', False), ('gpt-4o', True)]

for perm in perms:
    run_20_simuls_free(perm[0], perm[1], 0, 20)
