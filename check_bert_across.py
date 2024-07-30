from metrics.BERTScoreEval import BERTScoreEval as B
from utils.parse_csv import Parser
import numpy as np

parser = Parser()
b = B()
def check_bert_across(direxp, diranon, save_path):
    m1exp, m2exp = parser.parse_free(direxp)
    m1anon, m2anon = parser.parse_free(diranon)

    np.savez(
        f'{save_path}/bert_matrices.npz',
        move1 = b.get_berts_across(m1anon, m1exp),
        move2 = b.get_berts_across(m2anon, m2exp)
    )

go_through = [
    ('/Users/aryanshrivastava/Desktop/LLMWargamingConfidence/logging/outputs/v4/gpt4-free-True-20-1.0',
     '/Users/aryanshrivastava/Desktop/LLMWargamingConfidence/logging/outputs/v4/gpt4-free-False-20-1.0',
     '/Users/aryanshrivastava/Desktop/LLMWargamingConfidence/logging/across/4free')
]

for g in go_through:
    for i in range(4, 21):
        check_bert_across(
            f'{g[0]}/run{i}_fixed/run{i}.csv',
            f'{g[1]}/run{i}_fixed/run{i}_fixed.csv',
            f'{g[2]}/run{i}'
        )