# from main import Pipeline
from tqdm.auto import tqdm
import re
import csv
import os
from utils.game import GameSimulator

perms = [
    ('gpt-3.5-turbo', False),
    ('gpt-4', False),
]

def run_main(model, explicit_country, response_env, temperature=1.0, N_responses=20, start=1, end=20):
    if model != 'dummy' and 'claude' in model:
        model_dir_name = re.sub(r'-', '', model)[:-8]
        dir_name = f'{model_dir_name}-{response_env}-{explicit_country}-{N_responses}-{temperature}'
    elif model != 'dummy' and 'gpt' in model:
        model_dir_name = re.sub(r'-', '', model) 
        dir_name = f'{model_dir_name}-{response_env}-{explicit_country}-{N_responses}-{temperature}'
    elif model != 'dummy' and 'lama' in model:
        model_dir_name = re.sub(r'-', '', model)
        dir_name = f'{model_dir_name}-{response_env}-{explicit_country}-{N_responses}-{temperature}'
    elif model != 'dummy':
        print('Model not supported')
        return
    else:
        model_dir_name = 'dummy'
        dir_name = 'dummy'
    
    # make output directory
    os.makedirs(f'logging/outputs/v4/{dir_name}/main', exist_ok=True)
    # make chat directory
    os.makedirs(f'logging/chats/v4/{dir_name}', exist_ok=True)

    o_directory = os.path.abspath(f'logging/outputs/v4/{dir_name}/main')
    chat_directory = os.path.abspath(f'logging/chats/v4/{dir_name}')
    
    simulator = GameSimulator(model, f'{response_env}', explicit_country, temperature, N_responses)

    for i in tqdm(range(start, end+1), desc='Getting Completions...'):
        o_file = f'run{i}'

        if 'claude' in model:
            outputs, chats = simulator.run_basic_anthropic()
        elif 'gpt' in model:
            outputs, chats = simulator.run_basic_oai()
        elif 'lama' in model:
            outputs, chats = simulator.run_basic_llama()
        simulator.write_outputs(outputs, o_directory, f_name=o_file)
    
    simulator.write_chat(chats, chat_directory, 'chat')
        
        
perms = [
    ['gpt-4o-mini', False, 'free', 0.8, 20],
    ['gpt-4o-mini', True, 'rank', 0.8, 20],
    ['gpt-4o-mini', False, 'rank', 0.8, 20]

]
if __name__ == '__main__':
    for perm in perms:
        run_main(perm[0], perm[1], perm[2], temperature=perm[3], N_responses=perm[4])


# def run_20_simuls_rank(model, explicit_country, start, end):
#     if model != 'dummy' and 'claude' in model:
#         model_dir_name = re.sub(r'-', '', model)[:-8]
#         dir_name = f'{model_dir_name}-rank-{explicit_country}-20-1.0'
#     elif model != 'dummy' and ('gpt' in model):
#         model_dir_name = re.sub(r'-', '', model)
#         dir_name = f'{model_dir_name}-rank-{explicit_country}-20-1.0'
#     elif model!= 'dummy' and 'lama' in model:
#         print('here')
#         dir_name = f'llama3.170b-rank-{explicit_country}-20-1.0'
#     else:
#         model_dir_name = 'dummy'
#         dir_name = 'dummy'
#     if 'lama' not in model:
#         simulator = GameSimulator(model, 'rank', explicit_country, 20, 1.0)
#     else:
#         simulator = GameSimulator(model, 'rank', explicit_country, 20, 0.7)
#     for i in tqdm(range(start, end), desc='Run simulations...'):
#         o_directory = f'logging/outputs/v4/{dir_name}'
#         f_name = f'run{i+1}'

#         if 'claude' in model:
#             outputs, chats = simulator.run_basic_anthropic()
#             simulator.write_outputs(outputs, o_directory, f_name=f_name)
#         elif 'gpt' in model:
#             outputs, chats = simulator.run_basic_oai()
#             simulator.write_outputs(outputs, o_directory, f_name=f_name)
#         elif 'lama' in model:
#             outputs, chats = simulator.run_basic_llama()
#             simulator.write_outputs(outputs, o_directory, f_name=f_name)
    
#     simulator.write_chat(chats, f'logging/chats/v4/{dir_name}', 'chat')

# def run_20_simuls_free(model, explicit_country, start, end):
#     if model != 'dummy' and 'claude' in model:
#         model_dir_name = re.sub(r'-', '', model)[:-8]
#         dir_name = f'{model_dir_name}-free-{explicit_country}-20-1.0'
#     elif model != 'dummy' and ('gpt' in model):
#         model_dir_name = re.sub(r'-', '', model)
#         dir_name = f'{model_dir_name}-free-{explicit_country}-20-1.0'
#     elif model != 'dummy' and 'lama' in model:
#         # print('here')
#         dir_name = f'llama3.170b-free-{explicit_country}-20-1.0'
#     else:
#         model_dir_name = 'dummy'
#         dir_name = 'dummy'
#     if not explicit_country:
#         fixed = ''
#     else:
#         fixed = ''
#     if 'lama' not in model:
#         simulator = GameSimulator(model, 'free', explicit_country, 20, 1.0)
#     else:
#         simulator = GameSimulator(model, 'free', explicit_country, 20, 0.7)
#     for i in tqdm(range(start, end), desc='Run simulations...'):
#         o_directory = f'logging/outputs/v4/{dir_name}'
#         f_name = f'run{i+1}{fixed}'

#         if 'claude' in model:
#             outputs, chats = simulator.run_basic_anthropic()
#             simulator.write_outputs(outputs, o_directory, f_name=f_name)
#         elif 'gpt' in model:
#             outputs, chats = simulator.run_basic_oai()
#             simulator.write_outputs(outputs, o_directory, f_name=f_name)
#         elif 'lama' in model:
#             outputs, chats = simulator.run_basic_llama()
#             simulator.write_outputs(outputs, o_directory, f_name=f_name)
    
#     simulator.write_chat(chats, f'logging/chats/v4/{dir_name}', 'chat_fixed')

# perms = [('meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo', False), ('meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo', True)]

# run_20_simuls_rank(perms[0][0], perms[0][1], 0, 1)
