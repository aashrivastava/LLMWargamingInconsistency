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
    response = '''A: 4
B: 3
C: 2
D: 10
E: 12
F: 11
G: 5
H: 1
I: 8
J: 7
K: 6
L: 9
M: 15
N: 16 (Given that Gray already supplied weapons and Brown has reacted aggressively, further supply without more supportive measures would be less efficient now)
O: 14
P: 17
Q: 18
R: 19
S: 13

REASONING: Given Brown's escalating aggression, including asserting control over Pink's airspace and seas, the most reasonable action is to demonstrate military preparedness through military posturing or exercises (H) to deter Brown from further action. Establishing agreements on defense and security cooperation (C) and sharing intelligence on threats with Pink (B) are also crucial to strengthen the relationship and ensure Pink's security. High-level visits (A) can also reassure Pink of Gray's support. The least reasonable actions are those that escalate the situation to extreme levels, such as executing a tactical nuclear strike (R) or a full nuclear attack (re-ranked to reflect the escenario change considerations).'''

    parsed = parser.get_rank(response)
    print(parsed)
    