from utils.promptopenai import OpenAIPrompting
from utils.createchats import ChatCreation
from metrics.BERTScoreEval import BERTScoreEval
from metrics.BiDirectionalEntailmentEval import BiDirectionalEntailmentEval
from metrics.MQAGEval import MQAGEval
from metrics.RankEval import RankEval

import typing
import json
from tqdm import tqdm

class GameSimulator:
    '''
    IMPLEMENT DOCSTRING
    '''
    def __init__(self, model, control_level, explicit_country, N_responses, temperature):
        assert control_level.lower() in ['free', 'rank', 'nudge']
        self.model = model
        self.control_level = control_level
        self.explicit_country = explicit_country
        self.N_responses = N_responses
        self.temperature = temperature
        self.chatcreator = ChatCreation(self.control_level, self.explicit_country)
        self.prompter = OpenAIPrompting(model=self.model)

    
    def run(self):
        responses = []
        
        first_prompt = self.chatcreator.move_1()
        curr_chat = first_prompt
        # curr_chat should be updated with first response
        move1_completions = self.prompter.get_completions(curr_chat, N_responses=self.N_responses, temperature=self.temperature)
        responses.append(self.prompter.parse_outputs(move1_completions, self.control_level))

        self.chatcreator.move_2(curr_chat)
        move2_completions = self.prompter.get_completions(curr_chat,N_responses=self.N_responses, temperature=self.temperature)
        responses.append(self.prompter.parse_outputs(move2_completions, self.control_level))

        return curr_chat, responses

if __name__ == '__main__':
    play = GameSimulator('gpt-3.5-turbo', 'free', False, N_responses=2, temperature=1.0)
    run = play.run()
    responses = run[1]
    chat_hist = run[0]
    with open('responses.txt', 'w') as file:
        for i, sublist in enumerate(responses):
            file.write('## MOVE {i+1}' + '\n\n')
            for response in sublist:
                file.write(response + '\n')

    with open('chat.json', 'w') as json_file:
        json.dump(chat_hist, json_file, indent=4)