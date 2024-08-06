import openai as oai
import os
import typing
import json
# from tqdm import tqdm

## WHAT TO DO FOR LATER/TOMORROW
## ___PRESSING___
# WRITE PROMPTS TO TXT FILE
# THINK ABOUT PROMPT FORMAT AND OUTPUT FORMAT (AND HOW TO ELICIT IT)
## ___NICE TO HAVE___

# ___MODELS TO USE___
# GPT-4
'''
gpt-4-turbo: The latest GPT-4 Turbo model with vision capabilities. Vision requests can now use JSON mode and function calling. Currently points to gpt-4-turbo-2024-04-09.	128,000 tokens	Up to Dec 2023
gpt-4-turbo-2024-04-09:	GPT-4 Turbo with Vision model. Vision requests can now use JSON mode and function calling. gpt-4-turbo currently points to this version.	128,000 tokens	Up to Dec 2023
gpt-4-turbo-preview: GPT-4 Turbo preview model. Currently points to gpt-4-0125-preview.	128,000 tokens	Up to Dec 2023
gpt-4-0125-preview:	GPT-4 Turbo preview model intended to reduce cases of “laziness” where the model doesn’t complete a task. Returns a maximum of 4,096 output tokens. Learn more.	128,000 tokens	Up to Dec 2023
gpt-4-1106-preview:	GPT-4 Turbo preview model featuring improved instruction following, JSON mode, reproducible outputs, parallel function calling, and more. Returns a maximum of 4,096 output tokens. This is a preview model. Learn more.	128,000 tokens	Up to Apr 2023
gpt-4: Currently points to gpt-4-0613. See continuous model upgrades.	8,192 tokens	Up to Sep 2021
gpt-4-0613: Snapshot of gpt-4 from June 13th 2023 with improved function calling support.	8,192 tokens	Up to Sep 2021
gpt-4-0314:	Legacy Snapshot of gpt-4 from March 14th 2023.	8,192 tokens	Up to Sep 2021
'''
# GPT-3.5
'''
gpt-3.5-turbo-0125: The latest GPT-3.5 Turbo model with higher accuracy at responding in requested formats and a fix for a bug which caused a text encoding issue for non-English language function calls. Returns a maximum of 4,096 output tokens. Learn more.	16,385 tokens	Up to Sep 2021
gpt-3.5-turbo: Currently points to gpt-3.5-turbo-0125.	16,385 tokens	Up to Sep 2021
gpt-3.5-turbo-1106: GPT-3.5 Turbo model with improved instruction following, JSON mode, reproducible outputs, parallel function calling, and more. Returns a maximum of 4,096 output tokens. Learn more.	16,385 tokens	Up to Sep 2021
gpt-3.5-turbo-instruct: Similar capabilities as GPT-3 era models. Compatible with legacy Completions endpoint and not Chat Completions.	4,096 tokens	Up to Sep 2021
'''
# GPT BASE MODELS
'''
babbage-002: Replacement for the GPT-3 ada and babbage base models.	16,384 tokens	Up to Sep 2021
davinci-002: Replacement for the GPT-3 curie and davinci base models.	16,384 tokens	Up to Sep 2021
'''

class OpenAIPrompting:
    '''
    Prompt OpenAI and get responses. Also write to .txt files (TO IMPLEMENT) and parse into workable data structures
    '''
    def __init__(self, model: str):
        self.model = model
        # self.n_responses = n_responses
        if 'lama' in model:
            print('here')
            self.client = oai.OpenAI(
                api_key = os.environ.get('TOGETHER_API_KEY'),
                base_url = 'https://api.together.xyz/v1'
            )
        else:
            self.client = oai.OpenAI(
                organization=os.environ.get('OPENAI_ORG_KEY'),
                project=os.environ.get('OPENAI_PROJECT_ID')
            )
        
        
    def get_completions(self, curr_chat: list[dict[str, str]], N_responses: int=20, temperature: float=1.0, json_mode: bool=False):
        '''
        ## TODO:
        ##  Implement **kwargs so that user can pass other things if they want beyond these explicit ones
        Given a prompt, get n_responses chat completions 

        Inputs:
            prompt (json):
                This is a json file with system and user messages. # TBD
            N_responses (int):
                This specifies how many completions I want for a given query
        Output:
            Chat completion given by openai api based on prompt and hyperparameters
        '''
        # gpt-4 does not accept response_format parameter
        if json_mode:
            completions = self.client.chat.completions.create(
                model = self.model,
                messages = curr_chat,
                n = N_responses,
                temperature = temperature,
                response_format = {'type': 'json_object'}
            )
            greedy_decode = self.client.chat.completions.create(
                model = self.model,
                messages = curr_chat,
                temperature = 0.0,
                response_format = {'type': 'json_object'}
            )
        else:
            completions = self.client.chat.completions.create(
                model = self.model,
                messages = curr_chat,
                n = N_responses,
                temperature = temperature,
            )
            greedy_decode = self.client.chat.completions.create(
                model = self.model,
                messages = curr_chat,
                temperature = 0.0,
            )

        # uses the response generated by model using greedy decoding as example response to go into move 2
        curr_chat.append({'role': greedy_decode.choices[0].message.role, 'content': greedy_decode.choices[0].message.content})
        return completions
    
    def parse_outputs(self, completions) -> list[str]:
        '''
        Get list of strings which is just the text outputs of the chat completions given by openai api

        Inputs:
            completions:
                completions given by openai api
        
        Output:
            list[str]: list of strings where each string is one text output of openai
        '''
        # use response.choices[i].message.content
        responses = [completion_message.message.content for completion_message in completions.choices]
        # print(responses)


        return responses


if __name__ == '__main__':
    x = OpenAIPrompting('meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo')
    y = x.parse_outputs(x.get_completions([{'role': 'user', 'content': 'What happens to you if you eat watermelon seeds? Respond in 1 sentence'}], N_responses=20, temperature=1))
    for _ in y:
        print(_)
    