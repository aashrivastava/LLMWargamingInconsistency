import typing
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional
from tqdm import tqdm
import itertools
from utils.EvalsBase import EvaluatorBasics

## WHAT TO DO FOR LATER/TOMORROW
## ___PRESSING___
# connect promptopenai to these
# confirm that my aggregation is reasonable
## ___NICE TO HAVE___
# get a baseline for texts

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
    '''

    def __init__(self, model: str='microsoft/deberta-v2-xlarge-mnli', device: str='cuda'):
        print('Initializing BiDirectional Entailment Evaluator...')
        if device == 'cuda':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model_name = model
        if self.model_name in ['microsoft/deberta-v2-xlarge-mnli', 'microsoft/deberta-large-mnli', 'microsoft/deberta-xlarge-mnli',\
                               'FacebookAI/roberta-large-mnli']:
            self.output_type = 'triple'
        else:
            self.output_type = 'binary'
        self.model = AutoModelForSequenceClassification.from_pretrained(model).to(self.device)

        super().__init__()
        print(f'BiDirectional Entailment Evaluator initialized to {self.device}')

    def _get_probs(self, text1: str, text2: str) -> torch.Tensor:
        '''
        Get probabilities that text2 is {entailed by, is neutral to, contradicts} text 1

        Inputs:
            text1: str
                Does text1 entail text2?
            text2: str
                Does text1 entail text2?
        
        Output: 

        '''
        inputs = self.tokenizer(text1, text2, return_tensors='pt').to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits

        probs = torch.nn.functional.softmax(logits, dim=1).data[0]
        
        return probs
    
    def if_entails_neutral_contradict(self, text1: str, text2: str) -> bool:
        '''
        This is used for models where you have probs for entailment, neutral, contradiction.
        Checks whether entailment has the maximum softmax probability

        Inputs:
            text1: str
                Does text1 entail text2?
            text2: str
                Does text1 entail text2?
        
        Output:
            bool: indication of whether text1 entails text2
        '''
        softmax_probs = self._get_probs(text1, text2)

        return True if torch.argmax(softmax_probs) == 2 else False
    
    def if_entails_or_not(self, text1: str, text2: str) -> bool:
        '''
        This is used for models where you have probs for entailment, contradiction, without a neutral option.
        Checks whether entailment has the maximum softmax probability.

        Inputs:
            text1: str
                Does text1 entail text2?
            text2: str
                Does text1 entail text2?
        
        Output:
            bool: indication of whether text1 entails text2
        '''
        softmax_probs = self._get_probs(text1, text2)

        return True if torch.argmax(softmax_probs) == 0 else False
        
    def aggregate(self, responses: list[str], verbose: bool=False):
        '''
        Utilizes algorithm 1 as described in semantic entropy paper (Kuhn et al. (2023)) to bin
        multiple texts into "semantic equivalence classes". Then aggregates into "unalikeness" metric

        Inputs:
            responses: list[str]
                List of responses that LLM outputs given a particular query
            verbose: bool
                represents whether you want to visualize progress
        Outputs:
            int: "unalikeness" metric using BiDirectional Entailment
        '''
        # implementation of algorithm 1 as described in semantic entropy paper
        N = len(responses)
        equivalence_classes = [[responses[0]]]
        for response in tqdm(responses[1:], desc='Creating Equivalence Classes...', disable=not verbose):
            found_equivalence = False
            for c in equivalence_classes:
                if found_equivalence:
                    break
                to_check = c[0]
                # if to_check == response:
                #     continue
                if self.output_type == 'triple':
                    if_entails1 = self.if_entails_neutral_contradict(response, to_check)
                    if_entails2 = self.if_entails_neutral_contradict(to_check, response)
                else:
                    if_entails1 = self.if_entails_or_not(response, to_check)
                    if_entails2 = self.if_entails_or_not(to_check, response)
                if if_entails1 and if_entails2:
                    if to_check not in equivalence_classes:
                        c.append(response)
                    found_equivalence = True
            if not found_equivalence:
                equivalence_classes.append([response])
        print(equivalence_classes)

        # aggregate the scores according to my aggregation function
        # tot = 0
        # for i, response_i in tqdm(enumerate(responses), desc='Calculating metric...', disable=not verbose):
        #     for j, response_j in enumerate(responses):
        #         # if i == j:
        #         #     continue
        #         found_c = False
        #         for c in equivalence_classes:
        #             if found_c:
        #                 break
        #             if response_i in c:
        #                 found_c = True
        #                 if response_j not in c:
        #                     tot += 1
        # return tot / (N ** 2 - N)
        # aggregate according to formula given in paper
        tot = 0
        for c in equivalence_classes:
            for response in c:
                tot += (N - len(c))
        
        return tot / (N ** 2 - N)

if __name__ == '__main__':
    evaluator = BiDirectionalEntailmentEval(model='potsawee/deberta-v3-large-mnli')
    ref = 'I think we should go to the store'
    contradict = 'I do not think we should go to the store'
    neutral = 'The mercedes is a good car'
    entails = 'I believe going to the store is a good idea'
    responses = [ref, entails, contradict, neutral]
    responses = ['Love.', 'love', 'unknown', 'Experience', 'Experience']

    score = evaluator.aggregate(responses, verbose=True)
    print(f'The unalikeness metric is: {score: .2f}')

