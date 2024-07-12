from utils.promptopenai import OpenAIPrompting
from utils.createchats import ChatCreation
from metrics.BERTScoreEval import BERTScoreEval
from metrics.BiDirectionalEntailmentEval import BiDirectionalEntailmentEval
from metrics.MQAGEval import MQAGEval
from metrics.RankEval import RankEval

import typing
import json
import csv
from tqdm.auto import tqdm

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
        if self.model != 'dummy':
            self.prompter = OpenAIPrompting(model=self.model)
        else:
            self.prompter = None

    
    def run(self):
        responses = []
        weird_outputs = []
        
        first_prompt = self.chatcreator.move_1()
        curr_chat = first_prompt
        # curr_chat should be updated with first response
        if self.prompter:
            print('Getting move 1 completions...')
            move1_completions = self.prompter.get_completions(curr_chat, N_responses=self.N_responses, temperature=self.temperature)
            json_outputs = self.prompter.parse_outputs(move1_completions, self.control_level)
            print('Got move 1 completions')
            orders_move1 = []
            weird_outputs_move1 = []
            for json_output in json_outputs:
                try:
                    order = json.loads(json_output)['orders']
                    orders_move1.append(order)
                except:
                    order = json_output
                    weird_outputs_move1.append(order)
            responses.append(orders_move1)
            weird_outputs.append(weird_outputs_move1)
        else:
            responses.append(['pass' for i in range(self.N_responses)])

        self.chatcreator.move_2(curr_chat)
        if self.prompter:
            print('Getting move 2 completions...')
            move2_completions = self.prompter.get_completions(curr_chat, N_responses=self.N_responses, temperature=self.temperature)
            json_outputs = self.prompter.parse_outputs(move2_completions, self.control_level)
            orders_move2 = []
            weird_outputs_move2 = []
            for json_output in json_outputs:
                try:
                    order = json.loads(json_output)['orders']
                    orders_move2.append(order)
                except:
                    order = json_output
                    weird_outputs_move2.append(order)
            responses.append(orders_move2)
            weird_outputs.append(weird_outputs_move2)
            print('Got move 2 completions')
        else:
            responses.append(['pass' for i in range(self.N_responses)])

        return curr_chat, responses, weird_outputs

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