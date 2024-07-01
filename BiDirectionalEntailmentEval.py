import typing
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional
from EvalsBase import EvaluatorBasics

# _____NOTES_____
# microsoft/deberta-v2-xlarge-mnli 
#   - idx 0 is contradict, idx 1 is neutral, idx 2 is entails
# microsoft/deberta-large-mnli
#   - idx 0 is contradict, idx 1 is neutral, idx 2 is entails
# microsoft/deberta-xlarge-mnli
#   - idx 0 is contradict, idx 1 is neutral, idx 2 is entails
# potsawee/deberta-v3-large-mnli
#   - idx 0 is entails, idx 1 is contradict
# MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c
#   - idx 0 is entails, idx 1 is contradict
# FacebookAI/roberta-large-mnli
#   - idx 0 is contradict, idx 1 is neutral, idx 2 is entails
# there are more, just search "deberta mnli" on huggingface

class BiDirectionalEntailmentEval(EvaluatorBasics):
    '''
    Evaluates similarity of LLM generated outputs using Bi-Directional Entailment. Takes N sampled responses for a given query and calculates
    a Bi-Directional Entailment Unalikeness that aggregates individual scores into a metric that can take on values from [0,1]

    This is initially just using Deberta model
    '''

    def __init__(self, model: str='MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c'):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model_name = model
        if self.model_name in ['microsoft/deberta-v2-xlarge-mnli', 'microsoft/deberta-large-mnli', 'microsoft/deberta-xlarge-mnli',\
                               'FacebookAI/roberta-large-mnli']:
            self.output_type = 'triple'
        else:
            self.output_type = 'binary'
        self.model = AutoModelForSequenceClassification.from_pretrained(model)
        super().__init__()

    def _get_probs(self, text1: str, text2: str):
        '''
        DOCSTRING
        '''
        inputs = self.tokenizer(text1, text2, return_tensors='pt')
        outputs = self.model(**inputs)
        logits = outputs.logits

        probs = torch.nn.functional.softmax(logits, dim=1).data[0]
        
        return probs
    
    def if_entails_neutral_contradict(self, text1: str, text2: str) -> bool:
        '''
        DOCSTRING
        '''
        softmax_probs = self._get_probs(text1, text2)

        return True if torch.argmax(softmax_probs) == 2 else False
    
    def if_entails_or_not(self, text1: str, text2: str) -> bool:
        '''
        DOCSTRING
        '''
        softmax_probs = self._get_probs(text1, text2)

        return True if torch.argmax(softmax_probs) == 0 else False
        
    def aggregate(self, response: list[str], verbose: bool=False):
        '''
        DOCSTRING
        '''
        # implement algorithm 1 from semantic uncertainty paper

if __name__ == '__main__':
    evaluator = BiDirectionalEntailmentEval()
    ref = 'I think we should go to the store'
    contradict = 'I do not think we should go to the store'
    neutral = 'The mercedes is a good car'
    entails = 'I believe going to the store is a good idea'
    print(evaluator.if_entails(ref, contradict))
    print(evaluator.if_entails(ref, neutral))
    print(evaluator.if_entails(ref, entails))