import typing
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.function

class BiDirectionalEntailmentEval:
    '''
    Evaluates similarity of LLM generated outputs using Bi-Directional Entailment. Takes N sampled responses for a given query and calculates
    a Bi-Directional Entailment Unalikeness that aggregates individual scores into a metric that can take on values from [0,1]

    This is initially just using Deberta model
    '''

    def __init__(self, responses: list[str]):
        raise NotImplementedError