from utils.game import GameSimulator
from utils.consistency import ConsistencyEval
import typing
import csv
import json


class Pipeline:
    '''
    IMPLEMENT DOCSTRING
    '''
    def __init__(self, model, control_level, explicit_country, N_responses, temperature, metric):
        self.model = model
        self.control_level = control_level
        self.explicit_country = explicit_country
        self.N_responses = N_responses
        self.temperature = temperature
        self.metric = metric
        self.evaluator = ConsistencyEval(self.metric)
        self.simulator = GameSimulator(self.model, self.control_level, self.explicit_country, self.N_responses, self.temperature)
    
    def main(self, response_saver: typing.Optional[str]=None, chat_saver: typing.Optional[str]=None, weird_saver: typing.Optional[str]=None):
        chat_hist, responses, weird_outputs = self.simulator.run()
        # save prompting and response history to csvs if files are specified
        if response_saver:
            labels = ['Move 1 Responses', 'Move 2 Responses']
            with open(f'{response_saver}.csv', 'w', newline='') as responses_file:
                writer = csv.writer(responses_file)
                header = ['Move Number'] + [f'Response {i+1}' for i in range(self.N_responses)]
                writer.writerow(header)
                for label, move_i_responses in zip(labels, responses):
                    writer.writerow([label] + move_i_responses)
        if chat_saver:
            with open(f'{chat_saver}.csv', 'w', newline='') as chat_file:
                writer = csv.writer(chat_file)
                header = ['Role', 'Content']
                writer.writerow(header)
                for message in chat_hist:
                    writer.writerow([message['role']] + [message['content']])
        
        if weird_saver:
            labels = ['Move 1 Weird Responses', 'Move 2 Weird Responses']
            with open(f'{weird_saver}.csv', 'w', newline='') as weird_outputs_file:
                writer = csv.writer(weird_outputs_file)
                header = ['Move Number'] + [f'Response {i+1}' for i in range(max(len(weird_outputs[0]), len(weird_outputs[1])))]
                writer.writerow(header)
                for label, move_i_responses in zip(labels, weird_outputs):
                    writer.writerow([label] + move_i_responses)
        
        # get consistency metrics
        consistency_move1 = self.evaluator.get_inconsistency(responses[0], verbose=True)
        consistency_move2 = self.evaluator.get_inconsistency(responses[1], verbose=True)
        return consistency_move1, consistency_move2

if __name__ == '__main__':
    pipe = Pipeline('gpt-3.5-turbo', 'free', False, 2, 1.0, 'bert')
    consistency_move1, consistency_move2 = pipe.main()
    print(f'Consistency move 1: {consistency_move1}')
    print(f'Consistency move 2: {consistency_move2}')