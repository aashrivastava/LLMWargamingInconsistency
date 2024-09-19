import os
import typing
import json
import numpy as np

class ChatCreation:
    '''
    IMPLEMENT DOCSTRING
    '''
    def __init__(self, control_level='free', explicit_country=True, adversary_response='revisionist', identifiable_country='Taiwan', role='president', decision_country='ally', ablated_ranks=False):
        assert control_level in ['free', 'rank', 'nudge']
        assert explicit_country in [True, False]
        assert adversary_response in ['revisionist', 'status quo']
        assert identifiable_country.lower() in ['taiwan', 'cyprus', 'norway', 'india', 'ukraine']
        assert role.lower() in ['president', 'automated', 'recommender']

        self.control_level = control_level
        self.explicit_country = explicit_country
        self.adversary_response = adversary_response
        self.ablated_ranks = ablated_ranks
        self.identifiable_country = identifiable_country 
        self.role = role
        self.decision_country = decision_country
    
    def get_text_path(self, file_to_use: str) -> str:
        curr_path = os.getcwd()
        parent_directory = os.path.dirname(curr_path)
        wargame_folder_path = os.path.join(parent_directory, 'LLMWargamingConfidence/wargame')
        file_to_use_path = os.path.join(wargame_folder_path, file_to_use)

        return file_to_use_path
    
    def _get_replacement_path(self, replacement_to_use: str) -> str:
        curr_path = os.getcwd()
        parent_directory = os.path.dirname(curr_path)
        wargame_folder_path = os.path.join(parent_directory, 'LLMWargamingConfidence/wargame')
        replacements_path = os.path.join(wargame_folder_path, replacement_to_use)

        return replacements_path
    
    def _pick_replacement(self):
        if self.explicit_country and self.identifiable_country.lower() == 'taiwan':
            replacement_file = 'replacement_explicit.json'
        elif self.explicit_country and self.identifiable_country.lower() == 'ukraine':
            replacement_file = 'replacement_explicit_ukraine.json'
        elif self.explicit_country and self.identifiable_country.lower() == 'cyprus':
            replacement_file = 'replacement_explicit_cyprus.json'
        elif self.explicit_country and self.identifiable_country.lower() == 'india':
            replacement_file = 'replacement_explicit_sircreek.json'
        elif self.explicit_country and self.identifiable_country.lower() == 'norway':
            replacement_file = 'replacement_explicit_norway.json'
        elif not self.explicit_country:
            replacement_file = 'replacement_anonymous.json'
        
        replacement_to_use_path = self._get_replacement_path(replacement_file)
        
        with open(replacement_to_use_path, 'r') as f:
            replacements = json.load(f)
        
        if self.decision_country.lower() == 'ally':
            replacements["DECISION_COUNTRY"] = replacements["ALLY"]
        elif self.decision_country.lower() == 'adversary':
            replacements["DECISION_COUNTRY"] = replacements["ADVERSARY"]
        elif self.decision_country.lower() == 'aggrieved':
            replacements["DECISION_COUNTRY"] = replacements["AGGRIEVED"]
        else:
            raise Exception('Specified Decision Country Invalid')

        
        return replacements

    def create_system_prompt(self):
        if self.control_level == 'free' and self.role == 'president':
            file_to_use = 'system_free_v4.txt'
        elif self.control_level == 'free' and self.role == 'automated':
            file_to_use = 'system_free_automated.txt'
        elif self.control_level == 'free' and self.role == 'recommender':
            file_to_use = 'system_free_recommender.txt'
        elif self.control_level == 'rank':
            file_to_use = 'system_options_v4.txt'
        else:
            raise FileNotFoundError('Invalid control_level')
        
        replacements = self._pick_replacement()
        
        file_to_use_path = self.get_text_path(file_to_use)
        # print(replacement_to_use_path)
        
        with open(file_to_use_path, 'r') as f:
            system_prompt = f.read()
        
        system_prompt = system_prompt.format(**replacements)

        return system_prompt
    
    def create_context(self) -> str:
        scenario = 'scenario.txt'
        # avail_forces = 'available_forces.txt'

        if self.explicit_country:
            nation_description = None
        else:
            nation_description = 'nation_descriptions.txt'
        
        replacements = self._pick_replacement()
        
        # go through directory to find path for file
        scenario_path = self.get_text_path(scenario)
        # avail_forces_path = self.get_text_path(avail_forces)
        if nation_description:
            nation_desc_path = self.get_text_path(nation_description)
        else:
            nation_desc_path = None
        
        try:
            with open(scenario_path, 'r') as f1, open(nation_desc_path, 'r') as f3: #open(avail_forces_path, 'r') as f2, 
                context = f3.read() + '\n\n' + f1.read() + '\n\n'
        except TypeError:
            with open(scenario_path, 'r') as f1: # open(avail_forces_path, 'r') as f2
                context = f1.read() + '\n\n'
        
        context = context.format(**replacements)

        return context
    
    def create_context_ablated(self) -> str:
        '''
        This creates the context of our wargame if we want to do level one ablations.

        This performs these ablations RANDOMLY. Running the full code to generate responses will save the generated prompt in a file.s
        '''
        scenario_shell = 'scenario_shell.txt'
        ablation_options = 'bullet_replacements.json'


        if self.explicit_country:
            nation_description = None
        else:
            nation_description = 'nation_descriptions.txt'
        
        replacements = self._pick_replacement()
        
        # go through directory to find path for files
        scenario_path = self.get_text_path(scenario_shell)
        ablation_options_path = self._get_replacement_path(ablation_options)

        with open(ablation_options_path, 'r') as f:
            ablations = json.load(f)

        if nation_description:
            nation_desc_path = self.get_text_path(nation_description)
        else:
            nation_desc_path = None
        
        try:
            with open(scenario_path, 'r') as f1, open(nation_desc_path, 'r') as f3: #open(avail_forces_path, 'r') as f2, 
                context = f3.read() + '\n\n' + f1.read() + '\n\n'
        except TypeError:
            with open(scenario_path, 'r') as f1: # open(avail_forces_path, 'r') as f2
                context = f1.read() + '\n\n'
        
        for i in range(1, 20):
            bullet_choice = np.random.choice(ablations[f'BULLET_{i}'])
            # choice_index = np.where(np.array(ablations[f'BULLET_{i}']) == bullet_choice)[0][0]
            context = context.replace(f'{{{f'BULLET_{i}'}}}', bullet_choice)
        
        context = context.format(**replacements)

        return context
    
    def create_incident1_and_question(self):
        incident = 'incident_move1.txt'

        if self.control_level == 'free':
            if self.role == 'president':
                question = 'question_free_v4.txt'
            elif self.role == 'automated':
                question = 'question_free_automated.txt'
            elif self.role == 'recommender':
                question = 'question_free_recommender.txt'
        elif self.control_level == 'rank':
            if not self.ablated_ranks:
                question = 'question_options_v4.txt'
            elif self.ablated_ranks == 'reversed':
                question = 'question_options_v4_reversed.txt'

        replacements = self._pick_replacement()

        incident_path = self.get_text_path(incident)
        question_path = self.get_text_path(question)
        
        with open(incident_path, 'r') as f1, open(question_path, 'r') as f2:
            
            move_1 = f1.read() + '\n\n' + f2.read()
        
        move_1 = move_1.format(**replacements)

        return move_1
    
    def create_incident1_and_question_ablated(self):
        incident = 'incident_move1_shell.txt'
        ablation_options = 'bullet_incidentmove1.json'

        if self.control_level == 'free':
            if self.role == 'president':
                question = 'question_free_v4.txt'
            elif self.role == 'automated':
                question = 'question_free_automated.txt'
            elif self.role == 'recommender':
                question = 'question_free_recommender.txt'
        elif self.control_level == 'rank':
            if not self.ablated_ranks:
                question = 'question_options_v4.txt'
            elif self.ablated_ranks == 'reversed':
                question = 'question_options_v4_reversed.txt'

        ablation_options_path = self._get_replacement_path(ablation_options)
        with open(ablation_options_path, 'r') as f:
            ablations = json.load(f)


        replacements = self._pick_replacement()

        incident_path = self.get_text_path(incident)
        question_path = self.get_text_path(question)
        
        with open(incident_path, 'r') as f1, open(question_path, 'r') as f2:
            
            move_1 = f1.read() + '\n\n' + f2.read()
        
        for i in range(1, 12):
            bullet_choice = np.random.choice(ablations[f'BULLET_{i}'])
            # choice_index = np.where(np.array(ablations[f'BULLET_{i}']) == bullet_choice)[0][0]
            move_1 = move_1.replace(f'{{{f'BULLET_{i}'}}}', bullet_choice)
        
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

        if self.adversary_response == 'revisionist':
            response = 'revisionist_response.txt'
        elif self.adversary_response == 'status quo':
            response = 'statusquo_response.txt'

        if self.control_level == 'free':
            question = 'question_free_v4.txt'
        elif self.control_level == 'nudge':
            question = 'question_nudge.txt'
        else:
            if not self.ablated_ranks:
                question = 'question_options_v4.txt'
            elif self.ablated_ranks == 'reversed':
                question = 'question_options_v4_reversed.txt'

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
    x = ChatCreation(identifiable_country='India', role='automated', decision_country='adversary')
    y = x.create_context_ablated()
    print(y)
    # print(y[0]['content'])
    # print('----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
    # print(y[1]['content'])
    # print('---------------------')
    # x.move_2(y)
    # for chat in y:
    #     print(chat['content'])
    


