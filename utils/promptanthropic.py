import anthropic
import os
import typing

class AnthropicPrompting:
    '''
    DOCSTRING
    '''
    def __init__(self, model: str='claude-3-5-sonnet-20240620'):
        self.model = model
        self.client = anthropic.Anthropic()
    
    def get_completions(self, system: str, curr_chat: list[dict[str, str]], N_responses: int=20, temperature: float=1.0):
        '''
        DOCSTRING
        '''
        completions = []
        for _ in range(N_responses):
            completion = self.client.messages.create(
                model = self.model,
                system = system,
                messages = curr_chat,
                temperature = temperature,
                max_tokens = 4096
            )
            completions.append(completion)
        greedy_decode = self.client.messages.create(
            model = self.model,
            system = system,
            messages = curr_chat,
            temperature = 0.0,
            max_tokens = 4096
        )
        
        curr_chat.append({'role': greedy_decode.role, 'content': greedy_decode.content[0].text})

        return completions
    
    def parse_outputs(self, response) -> list[str]:
        '''
        DOCSTRING
        '''
        responses = [completion.content[0].text for completion in response]

        return responses

if __name__ == '__main__':
    a = AnthropicPrompting()
    system = 'Respond as a teenager'
    curr_chat = [{'role': 'user', 'content': 'hello'}]
    comps = a.get_completions(system, curr_chat, N_responses=2)
    responses = a.parse_outputs(comps)
    for r in responses:
        print(r)
