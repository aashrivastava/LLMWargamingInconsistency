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
        try:
            pattern = r'(?s)(?<=RECOMMENDATIONS:).+(?=REASONING)'
            pattern = r'(?s)RECOMMENDATIONS?:(.+?)(?=REASONING)'
            rec_str = re.findall(pattern, response)[0].strip()
        except:
            print(response)
            raise Exception

        # deal with case it gives list of recommendations
        rec_str = ' '.join(rec_str.split('\n')).strip()

        return rec_str
    
    def get_free_reasoning(self, response: str) -> str:
        '''
        Parse the reasoning from a free-form response
        '''
        try:
            pattern = r'REASONING:\s*(.*)$'
            reas_str = re.findall(pattern, response)[0].strip()
        except:
            print(response)
            raise Exception
        
        reas_str = ' '.join(reas_str.split('\n')).strip()
        return reas_str
        
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

        try:
            return df.iloc[0].tolist(), df.iloc[1].tolist()
        except:
            # in the case that only one move is present
            return df.iloc[0].tolist()
