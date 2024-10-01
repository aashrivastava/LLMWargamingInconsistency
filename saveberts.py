import os
from metrics.BERTScoreEval import BERTScoreEval
from utils.parse_csv import Parser
from tqdm.auto import tqdm
import numpy as np
import time

parser = Parser()
beval = BERTScoreEval()

def save_berts(text_dir, two_moves, o_dir):
    for i in range(1, 21):
        if not two_moves:
            m1 = parser.parse_free(text_dir)
        else:
            m1, m2 = parser.parse_free(text_dir)
        
        berts1 = beval.get_berts_within(m1)
        berts1 = berts1.numpy()
        if m2:
            berts2 = beval.get_berts_within(m2)
            berts2 = berts2.numpy()
            np.savez(o_dir, move1=berts1, move2=berts2)
        np.savez(o_dir, move1=berts1)
        
