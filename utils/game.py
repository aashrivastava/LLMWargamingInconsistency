from utils.promptopenai import OpenAIPrompting
from utils.promptanthropic import AnthropicPrompting
from utils.promptllama import LlamaPrompting
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
    def __init__(self, model, control_level='free', explicit_country=True, adversary_response='revisionist', temperature=1.0, N_responses=20, identifiable_country='Taiwan', role='president', decision_country='ally', ablated_free=False, ablated_ranks=False):
        assert control_level.lower() in ['free', 'rank', 'nudge']
        self.model = model
        self.control_level = control_level
        self.explicit_country = explicit_country
        self.adversary_response = adversary_response
        self.N_responses = N_responses
        self.temperature = temperature
        self.ablated_ranks = ablated_ranks
        self.ablated_free = ablated_free
        self.identifiable_country = identifiable_country
        self.role = role
        self.decision_country = decision_country
        self.chatcreator = ChatCreation(control_level=self.control_level, 
                                        explicit_country=self.explicit_country, 
                                        adversary_response=self.adversary_response, 
                                        identifiable_country=self.identifiable_country,
                                        role=self.role,
                                        decision_country=self.decision_country,
                                        ablated_free=self.ablated_free,
                                        ablated_ranks=self.ablated_ranks)
        if self.model != 'dummy' and ('gpt' in self.model):
            self.prompter = OpenAIPrompting(model=self.model)
        elif self.model != 'dummy' and 'claude' in self.model:
            self.prompter = AnthropicPrompting(model=self.model)
        elif self.model != 'dummy' and 'lama' in self.model:
            self.prompter = LlamaPrompting(model=self.model)
        else:
            self.prompter = None
    
    def run_basic_oai_inital_setting(self):
        assert 'gpt' in self.model

        responses = []
        first_prompt = self.chatcreator.move_1()
        curr_chat = first_prompt

        if self.prompter:
            print('Getting completions...')
            move1_completions = self.prompter.get_completions(curr_chat, N_responses=self.N_responses, temperature=self.temperature)
            outputs_1 = self.prompter.parse_outputs(move1_completions)
            print('Got completions')
            responses.append(outputs_1)
        
        return responses, curr_chat
    
    def run_basic_anthropic_initial_setting(self):
        assert 'claude' in self.model

        responses = []
        first_prompt = self.chatcreator.move_1()
        system_prompt, curr_chat = first_prompt[0]['content'], first_prompt[1:]

        if self.prompter:
            print('Getting move 1 completions...')
            move1_completions = self.prompter.get_completions(system_prompt, curr_chat, N_responses=self.N_responses, temperature=self.temperature)
            outputs_1 = self.prompter.parse_outputs(move1_completions)
            print('Got completions')
            responses.append(outputs_1)
        
        return responses, curr_chat
    
    def run_basic_oai(self):
        assert 'gpt' in self.model

        responses = []
        first_prompt = self.chatcreator.move_1()
        curr_chat = first_prompt

        if self.prompter:
            print('Getting move 1 completions...')
            move1_completions = self.prompter.get_completions(curr_chat, N_responses=self.N_responses, temperature=self.temperature)
            outputs_1 = self.prompter.parse_outputs(move1_completions)
            print('Got move 1 completions!')
            responses.append(outputs_1)
            self.chatcreator.move_2(curr_chat)
            print('Getting move 2 completions...')
            move2_completions = self.prompter.get_completions(curr_chat, N_responses=self.N_responses, temperature=self.temperature)
            outputs_2 = self.prompter.parse_outputs(move2_completions)
            print('Got move 2 completions!')
            responses.append(outputs_2)
        # dummy is running
        else:
            o_1 = ['text move 1' for i in range(self.N_responses)]
            responses.append(o_1)
            o_2 = ['text move 2' for i in range(self.N_responses)]
            responses.append(o_2)
        
        return responses, curr_chat
    
    def run_basic_llama(self):
        assert 'lama' in self.model

        responses = []
        first_prompt = self.chatcreator.move_1()
        curr_chat = first_prompt

        if self.prompter:
            print('Getting move 1 completions...')
            move1_completions = self.prompter.get_completions(curr_chat, N_responses=self.N_responses, temperature=self.temperature)
            outputs_1 = self.prompter.parse_outputs(move1_completions)
            print('Got move 1 completions!')
            responses.append(outputs_1)
            self.chatcreator.move_2(curr_chat)
            print('Getting move 2 completions...')
            move2_completions = self.prompter.get_completions(curr_chat, N_responses=self.N_responses, temperature=self.temperature)
            outputs_2 = self.prompter.parse_outputs(move2_completions)
            print('Got move 2 completions!')
            responses.append(outputs_2)
        # dummy is running
        else:
            o_1 = ['text move 1' for i in range(self.N_responses)]
            responses.append(o_1)
            o_2 = ['text move 2' for i in range(self.N_responses)]
            responses.append(o_2)
        
        return responses, curr_chat
        
        return responses, curr_chat

    
    def run_basic_anthropic(self):
        assert 'claude' in self.model

        responses = []
        first_prompt = self.chatcreator.move_1()
        system_prompt, curr_chat = first_prompt[0]['content'], first_prompt[1:]

        if self.prompter:
            print('Getting move 1 completions...')
            move1_completions = self.prompter.get_completions(system_prompt, curr_chat, N_responses=self.N_responses, temperature=self.temperature)
            outputs_1 = self.prompter.parse_outputs(move1_completions)
            print('Got move 1 completions!')
            responses.append(outputs_1)
            self.chatcreator.move_2(curr_chat)
            print('Getting move 2 completions...')
            move2_completions = self.prompter.get_completions(system_prompt, curr_chat, N_responses=self.N_responses, temperature=self.temperature)
            outputs_2 = self.prompter.parse_outputs(move2_completions)
            print('Got move 2 completions!')
            responses.append(outputs_2)
        # dummy is running
        else:
            o_1 = ['text move 1' for i in range(self.N_responses)]
            responses.append(o_1)
            o_2 = ['text move 2' for i in range(self.N_responses)]
            responses.append(o_2)
        
        return responses, curr_chat
    
    def write_outputs(self, outputs: list[list[str], list[str]], save_dir: str, f_name: str='outputs') -> None:
        '''
        DOCSTRING
        '''
        response_f = f'{save_dir}/{f_name}'
        labels = ['Move 1 Responses', 'Move 2 Responses']

        with open(f'{response_f}.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['Move Number'] + [f'Response {i+1}' for i in range(self.N_responses)]
            writer.writerow(header)
            for label, move_i_outputs in zip(labels, outputs):
                writer.writerow([label] + move_i_outputs)
    
    def run(self):
        responses = []
        weird_outputs = []
        reasoning = []
        
        first_prompt = self.chatcreator.move_1()
        curr_chat = first_prompt
        # curr_chat should be updated with first response
        if self.prompter:
            print('Getting move 1 completions...')
            move1_completions = self.prompter.get_completions(curr_chat, N_responses=self.N_responses, temperature=self.temperature)
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
                        orders_move1.append(ranking)
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
            move2_completions = self.prompter.get_completions(curr_chat, N_responses=self.N_responses, temperature=self.temperature)
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
                        orders_move2.append(ranking)
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
    
    def write_chat(self, chat: list[dict[str, str]], save_dir: str, f_name: str='chat'):
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