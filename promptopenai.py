import openai as oai
import os
import typing
from tqdm import tqdm

key = os.environ.get("OPENAI_API_KEY")

class OpenAIPrompting:
    '''
    Prompt OpenAI and get responses
    '''
    def __init__(self, model: str):
        self.model = model
        self.n_responses = n_responses
        self.client = OpenAI()
    
    def get_completions(self, prompt, n_responses: int=20):
        '''
        Given a prompt, get n_responses chat completions 
        '''
        raise NotImplementedError

    

    