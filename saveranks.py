import os
from metrics.RankEval import RankEval
from utils.parse_csv import Parser
from tqdm.auto import tqdm
import numpy as np

parser = Parser()
reval = RankEval()

def get_all(responses: list[dict[str, int]]):

    k = reval.get_kendalls(responses)
    s = reval.get_spearmans(responses)
    h = reval.get_hamming(responses)
    
    return k, s, h