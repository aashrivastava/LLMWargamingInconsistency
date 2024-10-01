# from main import Pipeline
from tqdm.auto import tqdm
import re
import csv
import os
from utils.game import GameSimulator

def run_main(model, explicit_country, response_env, adversary_response, temperature=1.0, N_responses=20, start=1, end=20, ablated_ranks=False):
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
    if adversary_response == 'revisionist':
        adversary_response_dir_name = 'revisionist'
    elif adversary_response == 'status quo':
        adversary_response_dir_name = 'status_quo'
    
    if not ablated_ranks:
        ablated_ranks = ''
    
    output_dir = f'logging/outputs/v4/{model_dir_name}/{response_env}/{adversary_response_dir_name}/{ablated_ranks}/{dir_name}/main'
    for i in range(start, end+1):
        os.makedirs(f'{output_dir}/run{i}', exist_ok=True)

    o_directory = os.path.abspath(output_dir)
    
    simulator = GameSimulator(model, response_env, explicit_country, adversary_response, temperature, N_responses, ablated_ranks=ablated_ranks)

    for i in tqdm(range(start, end+1), desc='Getting Completions...'):
        o_file = f'run{i}'

        if 'claude' in model:
            outputs, chats = simulator.run_basic_anthropic()
        elif 'gpt' in model:
            outputs, chats = simulator.run_basic_oai()
        elif 'lama' in model:
            outputs, chats = simulator.run_basic_llama()
        simulator.write_outputs(outputs, f'{o_directory}/run{i}', f_name=o_file)
    
    simulator.write_chat(chats, o_directory, 'chat')


def run_initial_setting(model, o_dir, f_name, explicit_country=True, response_env='free', temperature=1.0, N_responses=20, identifiable_country='Taiwan', role='president', decision_country='ally', ablated_free=False, ablated_ranks=False, get_chats=True):
    '''
    Runs the "INITIAL SETTING" experiment. This will do everythign from prompting the model to writing the response(s) to a specified directory `o_dir`
    '''
    # makes sure model is supported
    assert 'claude' in model or 'gpt' in model

    o_dir = os.path.abspath(o_dir)

    simulator = GameSimulator(model, control_level=response_env, explicit_country=explicit_country, identifiable_country=identifiable_country, role=role, decision_country=decision_country,
                              temperature=temperature, N_responses=N_responses, ablated_free=ablated_free, ablated_ranks=ablated_ranks)
    
    if 'claude' in model:
        outputs, chat = simulator.run_basic_anthropic_initial_setting()
        # print(outputs)
        # print('')
        # print(chat[-1]['content'])
    elif 'gpt' in model:
        outputs, chat = simulator.run_basic_oai_inital_setting()
        # print(outputs)
        # print('')
        # print(chat[-1]['content'])
    
    simulator.write_outputs(outputs, o_dir, f_name=f_name)
    if get_chats:
        try:
            simulator.write_chat(chat, o_dir, f_name=f'chats/CHAT-{f_name}')
        except FileNotFoundError:
            os.makedirs(f'{o_dir}/chats')
            simulator.write_chat(chat, o_dir, f_name=f'chats/CHAT-{f_name}')
    else:
        pass
