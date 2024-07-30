from metrics.RankEval import RankEval
from utils.parse_csv import Parser
import numpy as np

parser = Parser()
rankeval = RankEval()

def check_rank_across(direxp, diranon, save_path, metric):
    m1exp, m2exp = parser.parse_rankings(direxp)
    m1anon, m2anon = parser.parse_rankings(diranon)

    np.savez(
        f'{save_path}/{metric}_matrices.npz',
        move1 = rankeval.get_metric_across(m1anon, m1exp, metric=metric),
        move2 = rankeval.get_metric_across(m2anon, m2exp, metric=metric)
    )

go_through = [
    ('/Users/aryanshrivastava/Desktop/LLMWargamingConfidence/logging/outputs/v4/gpt4-rank-True-20-1.0',
     '/Users/aryanshrivastava/Desktop/LLMWargamingConfidence/logging/outputs/v4/gpt4-rank-False-20-1.0',
     '/Users/aryanshrivastava/Desktop/LLMWargamingConfidence/logging/across/4rank')
]

for g in go_through:
    for i in range(7, 21):
        check_rank_across(
            f'{g[0]}/run{i}_fixed/run{i}.csv',
            f'{g[1]}/run{i}_fixed/run{i}_fixed.csv',
            f'{g[2]}/run{i}',
            metric='kendall'
        )
        check_rank_across(
            f'{g[0]}/run{i}_fixed/run{i}.csv',
            f'{g[1]}/run{i}_fixed/run{i}_fixed.csv',
            f'{g[2]}/run{i}',
            metric='spearman'
        )
        check_rank_across(
            f'{g[0]}/run{i}_fixed/run{i}.csv',
            f'{g[1]}/run{i}_fixed/run{i}_fixed.csv',
            f'{g[2]}/run{i}',
            metric='hamming'
        )
