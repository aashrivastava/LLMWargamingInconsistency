import os
import typing
import json

class ChatCreation:
    '''
    IMPLEMENT DOCSTRING
    '''
    def __init__(self, control_level, explicit_country):
        assert control_level in ['free', 'rank', 'nudge']

        self.control_level = control_level
        self.explicit_country = explicit_country
    
    def get_text_path(self, file_to_use: str) -> str:
        curr_path = os.getcwd()
        parent_directory = os.path.dirname(curr_path)
        wargame_folder_path = os.path.join(parent_directory, 'LLMWargamingConfidence/wargame')
        file_to_use_path = os.path.join(wargame_folder_path, file_to_use)

        return file_to_use_path
    
    def get_replacement_path(self, replacement_to_use: str) -> str:
        curr_path = os.getcwd()
        parent_directory = os.path.dirname(curr_path)
        wargame_folder_path = os.path.join(parent_directory, 'LLMWargamingConfidence/wargame')
        replacements_path = os.path.join(wargame_folder_path, replacement_to_use)

        return replacements_path

    def create_system_prompt(self):
        if self.control_level == 'free':
            file_to_use = 'system_free_v4.txt'
        elif self.control_level == 'rank':
            file_to_use = 'system_options_v4.txt'
        elif self.control_level == 'nudge':
            file_to_use = 'system_nudge.txt'
        else:
            raise FileNotFoundError('Invalid control_level')
        
        if self.explicit_country:
            replacement_file = 'replacement_explicit.json'
        else:
            replacement_file = 'replacement_anonymous.json'
        
        file_to_use_path = self.get_text_path(file_to_use)
        replacement_to_use_path = self.get_replacement_path(replacement_file)

        with open(replacement_to_use_path, 'r') as f:
            replacements = json.load(f)
        
        with open(file_to_use_path, 'r') as f:
            system_prompt = f.read()
        
        system_prompt = system_prompt.format(**replacements)

        return system_prompt
    
    def create_context(self) -> str:
        scenario = 'scenario.txt'
        # avail_forces = 'available_forces.txt'

        if self.explicit_country:
            replacement_file = 'replacement_explicit.json'
            nation_description = None
        else:
            replacement_file = 'replacement_anonymous.json'
            nation_description = 'nation_descriptions.txt'
        
        # go through directory to find path for file
        scenario_path = self.get_text_path(scenario)
        # avail_forces_path = self.get_text_path(avail_forces)
        if nation_description:
            nation_desc_path = self.get_text_path(nation_description)
        else:
            nation_desc_path = None
        replacement_to_use_path = self.get_replacement_path(replacement_file)


        with open(replacement_to_use_path, 'r') as f:
            replacements = json.load(f)
        
        try:
            with open(scenario_path, 'r') as f1, open(nation_desc_path, 'r') as f3: #open(avail_forces_path, 'r') as f2, 
                context = f3.read() + '\n\n' + f1.read() + '\n\n'
        except TypeError:
            with open(scenario_path, 'r') as f1: # open(avail_forces_path, 'r') as f2
                context = f1.read() + '\n\n'
        
        context = context.format(**replacements)

        return context
    
    def create_incident1_and_question(self):
        incident = 'incident_move1.txt'

        if self.control_level == 'free':
            question = 'question_free_v4.txt'
        elif self.control_level == 'nudge':
            question = 'question_nudge.txt'
        else:
            question = 'question_options_v4.txt'

        if self.explicit_country:
            replacement_file = 'replacement_explicit.json'
        else:
            replacement_file = 'replacement_anonymous.json'

        incident_path = self.get_text_path(incident)
        question_path = self.get_text_path(question)
        replacement_to_use_path = self.get_replacement_path(replacement_file)

        with open(replacement_to_use_path, 'r') as f:
            replacements = json.load(f)
        
        with open(incident_path, 'r') as f1, open(question_path, 'r') as f2:
            
            move_1 = f1.read() + '\n\n' + f2.read()
        
        move_1 = move_1.format(**replacements)

        return move_1
    
    def move_1(self) -> list[dict[str, str]]:
        system_prompt = self.create_system_prompt()
        first_message = self.create_context() + self.create_incident1_and_question()

        return [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': first_message}
        ]
    
    def create_incident2_and_question(self):
        incident = 'incident_move2.txt'
        response = 'adversary_response.txt'

        if self.control_level == 'free':
            question = 'question_free_v4.txt'
        elif self.control_level == 'nudge':
            question = 'question_nudge.txt'
        else:
            question = 'question_options_v4.txt'

        if self.explicit_country:
            replacement_file = 'replacement_explicit.json'
        else:
            replacement_file = 'replacement_anonymous.json'

        incident_path = self.get_text_path(incident)
        question_path = self.get_text_path(question)
        response_path = self.get_text_path(response)
        replacement_to_use_path = self.get_replacement_path(replacement_file)

        with open(replacement_to_use_path, 'r') as f:
            replacements = json.load(f)
        
        with open(incident_path, 'r') as f1, open(response_path, 'r') as f2, open(question_path, 'r') as f3:
            move_2 = f1.read() + '\n\n' + f2.read() + '\n\n' + f3.read()
        
        move_2 = move_2.format(**replacements)

        return move_2
    
    def move_2(self, chat_hist):
        chat_hist.append({'role': 'user', 'content': self.create_incident2_and_question()})


if __name__ == '__main__':
    x = ChatCreation('free', False)
    y = x.move_1()
    # print(y[0]['content'], y[1]['content'])
    # print('---------------------')
    x.move_2(y)
    for chat in y:
        print(chat['content'])
    


