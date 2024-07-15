from main import Pipeline
from tqdm.auto import tqdm
import re
import csv
from utils.game import GameSimulator

perms = [
    ('gpt-3.5-turbo', False),
    ('gpt-3.5-turbo', True)
]

def run_20_simuls(model, explicit_country):
    model_dir_name = re.sub(r'-', '', model)
    dir_name = f'{model_dir_name}-rank-{explicit_country}-20-1.0'
    simulator = GameSimulator(model, 'rank', explicit_country, 20, 1.0)
    for i in tqdm(range(20), desc='Run simulations...'):
        response_dir = f'logging/responses/v2/{dir_name}'
        chat_dir = f'logging/chats/v2/{dir_name}'
        weird_dir = f'logging/weird/v2/{dir_name}'
        reasoning_dir = f'logging/reasoning/v2/{dir_name}'

        f_name = f'run{i}'

        chat, responses, weird, reasons = simulator.run()

        simulator.write_responses(responses, response_dir, f_name)
        simulator.write_chat(chat, chat_dir, f_name)
        simulator.write_weird(weird, weird_dir, f_name)
        simulator.write_reasoning(reasons, reasoning_dir, f_name)

run_20_simuls(perms[1][0], perms[1][1])

# for perm in perms:
#     run_20_simuls(perm[0], perm[1])
