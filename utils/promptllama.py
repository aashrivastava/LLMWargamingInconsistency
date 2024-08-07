from together import Together
import typing
import os

class LlamaPrompting:
    def __init__(self, model: str):
        self.model = model
        if 'lama' in model:
            self.client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
        
    def get_completions(self, curr_chat: list[dict[str, str]], N_responses: int=20, temperature: float=0.7):
        if 'lama' in self.model:
            completions = self.client.chat.completions.create(
                model=self.model,
                messages = curr_chat,
                temperature=0.7,
                repetition_penalty=1.3,
                stop=['<|eot_id|>'],
                n=N_responses
            )
            greedy_decode = self.client.chat.completions.create(
                model=self.model,
                messages = curr_chat,
                temperature=0.0,
                repetition_penalty=1.3,
                stop=['<|eot_id|>'],
            )

        curr_chat.append({'role': 'assistant', 'content': greedy_decode.choices[0].message.content})
        return completions

if __name__ == '__main__':
    x = LlamaPrompting('meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo')
    messages = [
        {'role': 'system',
         'content': 'say hello in 5 words!'},
        {'role': 'user',
         'content': 'Hello!'}
    ]
    y = x.get_completions(messages, N_responses=2)
    print(y.choices[0].message.role)
    print(y.choices[0].message.content)