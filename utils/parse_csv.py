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
            raise

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
    response = '''RECOMMENDATIONS: I recommend temporarily withdrawing the 5th Task Force and the affected destroyer to safer waters while initiating robust diplomatic efforts aimed at de-escalation. Meanwhile, increase support for Pink through supply of advanced defensive systems and personnel training. Initiate a high-level crisis communication channel with Brown to negotiate terms of engagement to prevent future incidents.

    REASONING: This action minimizes immediate risk to Gray's personnel and assets while sustaining a long-term strategy of bolstering Pink's defense capabilities. The diplomatic approach aims to lower tensions and establish clear communication channels to avoid miscalculations that could lead to unintended escalation.'''

    parsed = parser.get_free(response)
    print(parsed)
    