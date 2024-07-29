import csv
import re
import pandas as pd
import typing

class Parser:
    '''
    DOCSTRING
    '''
    def __init__(self):
        pass

    def get_rank(self, response: str) -> dict[str, int]:
        '''
        DOCSTRING
        '''
        pattern = r'(?s).+(?=REASONING)'
        rank_str = re.match(pattern, response).group(0).strip()

        rank_list = re.split(r'\n+', rank_str)

        # figure out what to do in duplicate case
        try:
            return {rank[0]: int(re.findall(r'(\d+)', rank)[0]) for rank in rank_list}
        except:
            print('GOT ERROR!')
    
    def get_free(self, response: str) -> str:
        '''
        DOCSTRING
        '''
        pattern = r'(?s)(?<=RECOMMENDATIONS:).+(?=REASONING)'
        rec_str = re.findall(pattern, response)[0].strip()

        # deal with case it gives list of recommendations
        rec_str = ' '.join(rec_str.split('\n')).strip()

        return rec_str
        
    def parse_rankings(self, file: str) -> tuple[list[dict[str, int]]]:
        '''
        DOCSTRING
        '''
        df = pd.read_csv(file)
        df = df.set_index('Move Number')
        df = df.map(self.get_rank)


        return df.iloc[0].tolist(), df.iloc[1].tolist()
    
    def parse_free(self, file: str) -> tuple[list[str]]:
        '''
        DOCSTRING
        '''
        df = pd.read_csv(file)
        df = df.set_index('Move Number')
        df = df.map(self.get_free)

        return df.iloc[0].tolist(), df.iloc[1].tolist()

if __name__ == '__main__':
    parser = Parser()
    parsed = parser.parse_rankings('/Users/aryanshrivastava/Desktop/LLMWargamingConfidence/logging/outputs/v4/gpt4-rank-False-20-1.0/run4_fixed.csv')
    print(parsed[0][5])
    