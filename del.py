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
    if model != 'dummy':
        model_dir_name = re.sub(r'-', '', model)
        dir_name = f'{model_dir_name}-rank-{explicit_country}-20-1.0'
    else:
        model_dir_name = 'dummy'
        dir_name = 'dummy'
    simulator = GameSimulator(model, 'rank', explicit_country, 20, 1.0)
    for i in tqdm(range(20), desc='Run simulations...'):
        directory = f'logging/outputs/v3/{dir_name}'
        f_name = f'run{i+1}'

        outupts = simulator.run_basic()
        simulator.write_outputs(outupts, directory, f_name=f_name)

for perm in perms:
    run_20_simuls(perm[0], perm[1])
