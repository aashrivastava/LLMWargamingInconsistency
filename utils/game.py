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
        if self.control_level == 'rank':
            self.json_mode = False
        else:
            self.json_mode = True
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
        reasoning = []
        
        first_prompt = self.chatcreator.move_1()
        curr_chat = first_prompt
        # curr_chat should be updated with first response
        if self.prompter:
            print('Getting move 1 completions...')
            move1_completions = self.prompter.get_completions(curr_chat, N_responses=self.N_responses, temperature=self.temperature, json_mode=self.json_mode)
            outputs = self.prompter.parse_outputs(move1_completions, self.control_level)
            print('Got move 1 completions')
            orders_move1 = []
            reasoning_move1 = []
            weird_outputs_move1 = []
            for output in outputs:
                # make sure json parseable
                found_weird = False
                if self.control_level == 'rank':
                    ranking = output.strip().split('\n')
                    if len(ranking) == 19:
                        orders_move1.append(output.strip().split('\n'))
                    else:
                        weird_outputs_move1.append(output)
                        print('Found weird')
                        print(ranking)
                else:
                    try:
                        output_json = json.loads(output)
                    except:
                        weird_outputs_move1.append(output)
                        found_weird = True
                        print('Found weird')
                    # make sure orders property exists and is in fine format
                    if not found_weird:
                        try:
                            orders = output_json['orders']
                            if self.control_level == 'rank':
                                assert len(orders) == 19
                            orders_move1.append(orders)
                        except:
                            weird_outputs_move1.append(output)
                            found_weird = True
                            print('Found weird')
                    if not found_weird:
                        try:
                            reason = output_json['reasoning']
                            reasoning_move1.append(reason)
                        except:
                            weird_outputs_move1.append(output)
                            found_weird = True
                            print('Found weird')
            responses.append(orders_move1)
            weird_outputs.append(weird_outputs_move1)
            reasoning.append(reasoning_move1)

        else:
            responses.append(['pass' for i in range(self.N_responses)])

        self.chatcreator.move_2(curr_chat)
        if self.prompter:
            print('Getting move 2 completions...')
            move2_completions = self.prompter.get_completions(curr_chat, N_responses=self.N_responses, temperature=self.temperature, json_mode=self.json_mode)
            outputs = self.prompter.parse_outputs(move2_completions, self.control_level)
            print('Got move 2 completions')
            orders_move2 = []
            reasoning_move2 = []
            weird_outputs_move2 = []
            for output in outputs:
                # make sure json parseable
                found_weird = False
                if self.control_level == 'rank':
                    ranking = output.strip().split('\n')
                    if len(ranking) == 19:
                        orders_move2.append(output.strip().split('\n'))
                    else:
                        weird_outputs_move2.append(output)
                        print('Found weird')
                        print(output)
                else:
                    try:
                        output_json = json.loads(output)
                    except:
                        weird_outputs_move2.append(output)
                        found_weird = True
                        print('Found weird')
                    # make sure orders property exists and is in fine format
                    if not found_weird:
                        try:
                            orders = output_json['orders']
                            if self.control_level == 'rank':
                                assert len(orders) == 19
                            orders_move2.append(orders)
                        except:
                            weird_outputs_move2.append(output)
                            found_weird = True
                            print('Found weird')
                    if not found_weird:
                        try:
                            reason = output_json['reasoning']
                            reasoning_move2.append(reason)
                        except:
                            weird_outputs_move2.append(output)
                            found_weird = True
                            print('Found weird')
            responses.append(orders_move2)
            weird_outputs.append(weird_outputs_move2)
            reasoning.append(reasoning_move2)

        return curr_chat, responses, weird_outputs, reasoning
    
    def write_chat(self, chat: list[dict[str, str]], save_dir: str, f_name: str):
        '''
        docstring
        '''
        chat_saver = f'{save_dir}/{f_name}'
        with open(f'{chat_saver}.csv', 'w', newline='') as chat_file:
            writer = csv.writer(chat_file)
            header = ['Role', 'Content']
            writer.writerow(header)
            for message in chat:
                writer.writerow([message['role']] + [message['content']])
    
    def write_responses(self, responses: typing.Union[list[list[str]], list[list[list[str]]]], save_dir: str, f_name: str):
        '''
        docstring
        '''
        response_saver = f'{save_dir}/{f_name}'
        labels = ['Move 1 Responses', 'Move 2 Responses']

        with open(f'{response_saver}.csv', 'w', newline='') as responses_file:
            writer = csv.writer(responses_file)
            header = ['Move Number'] + [f'Response {i+1}' for i in range(self.N_responses)]
            writer.writerow(header)
            for label, move_i_responses in zip(labels, responses):
                writer.writerow([label] + move_i_responses)
    
    def write_weird(self, weirdos: list[list[str]], save_dir: str, f_name: str):
        '''
        DOCSTRING
        '''
        weird_saver = f'{save_dir}/{f_name}'
        labels = ['Move 1 Weird Responses', 'Move 2 Weird Responses']

        with open(f'{weird_saver}.csv', 'w', newline='') as weird_outputs_file:
            writer = csv.writer(weird_outputs_file)
            header = ['Move Number'] + [f'Response {i+1}' for i in range(max(len(weirdos[0]), len(weirdos[1])))]
            writer.writerow(header)
            for label, move_i_responses in zip(labels, weirdos):
                writer.writerow([label] + move_i_responses)
    
    def write_reasoning(self, reasoning: list[list[str]], save_dir: str, f_name: str):
        '''
        DOCSTRING
        '''
        reasoning_saver = f'{save_dir}/{f_name}'
        labels = ['Move 1 Reasoning', 'Move 2 Reasoning']

        with open(f'{reasoning_saver}.csv', 'w', newline='') as reasoning_file:
            writer = csv.writer(reasoning_file)
            header = ['Move Number'] + [f'Response {i+1}' for i in range(self.N_responses)]
            writer.writerow(header)
            for label, move_i_responses in zip(labels, reasoning):
                writer.writerow([label] + move_i_responses)

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