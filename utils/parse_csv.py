import csv
import re
import pandas as pd

class Parser:
    '''
    DOCSTRING
    '''
    def __init__(self):
        pass

    def remove_reasoning(self, response: str):
        '''
        DOCSTRING
        '''
        pattern = r'(.|\n)+(?=REASONING)'
        rank_str = re.match(pattern, response).group(0).strip()

        rank_list = rank_str.split('\n')
        
        return {rank[0]: int(re.findall(r'(\d+)', rank)[0]) for rank in rank_list}



    def parse_rankings(self, file: str) -> tuple(list[dict[str, int]]):
        '''
        DOCSTRING
        '''
        df = pd.read_csv(file)
        df = df.set_index('Move Number')
        df = df.map(self.remove_reasoning)


        return df.iloc[0].tolist(), df.iloc[1].tolist()
        
        



if __name__ == '__main__':
    parser = Parser()
    parsed = parser.parse_rankings('/Users/aryanshrivastava/Desktop/LLMWargamingConfidence/logging/outputs/v4/gpt4-rank-False-20-1.0/run20.csv')
    for p in parsed:
        print(p)
    