from utils.parse_csv import Parser
from transformers import AutoTokenizer
import numpy as np

def compute_length(tokenizer, response, parser):
    '''
    Get token lengths of LLM responses
    '''

    return len(tokenizer.tokenize(response))

def compute_length_wrapped(response):
    return compute_length(tokenizer, response, parser)

def apply_to_responses(parser, file, func_vectorized):
    
    responses_m1, responses_m2 = parser.parse_free(file)

    responses_m1 = np.array(responses_m1)
    responses_m2 = np.array(responses_m2)

    return func_vectorized(responses_m1), func_vectorized(responses_m2)


if __name__ == '__main__':
    parser = Parser()
    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-xlarge-mnli', use_fast=False)

    v = np.vectorize(compute_length_wrapped)

    def do_all(file):
        
        m1_lens, m2_lens = apply_to_responses(parser, file, v)
        x = (m1_lens.mean(), m2_lens.mean())
        return x
    v_all = np.vectorize(do_all)

    # claude
    claude = []

    for i in range(1, 21):
        f_dir = f'/Users/aryanshrivastava/Desktop/LLMWargamingConfidence/logging/outputs/v4/claude35sonnet-free-False-20-1.0/main/run{i}_fixed/run{i}_fixed.csv'
        claude.append(f_dir)
    for i in range(1, 21):
        f_dir = f'/Users/aryanshrivastava/Desktop/LLMWargamingConfidence/logging/outputs/v4/claude35sonnet-free-True-20-1.0/main/run{i}/run{i}.csv'
        claude.append(f_dir)
    claude = np.array(claude)
    mean_lengths = v_all(claude)

    print('CLAUDE AVERAGE OUTPUT LENGTH')
    print(mean_lengths[0].mean())
    print(mean_lengths[1].mean())

    # gpt 3,5
    gpt_35 = []

    for i in range(1, 21):
        f_dir = f'/Users/aryanshrivastava/Desktop/LLMWargamingConfidence/logging/outputs/v4/gpt3.5turbo-free-False-20-1.0/main/run{i}_fixed/run{i}_fixed.csv'
        gpt_35.append(f_dir)
    for i in range(1, 21):
        f_dir = f'/Users/aryanshrivastava/Desktop/LLMWargamingConfidence/logging/outputs/v4/gpt3.5turbo-free-True-20-1.0/main/run{i}/run{i}.csv'
        gpt_35.append(f_dir)
    gpt_35 = np.array(gpt_35)
    mean_lengths = v_all(gpt_35)

    print('GPT 3.5 AVERAGE OUTPUT LENGTH')
    print(mean_lengths[0].mean())
    print(mean_lengths[1].mean())
    
    # gpt 4
    gpt_4 = []

    for i in range(1, 21):
        f_dir = f'/Users/aryanshrivastava/Desktop/LLMWargamingConfidence/logging/outputs/v4/gpt4-free-False-20-1.0/main/run{i}_fixed/run{i}_fixed.csv'
        gpt_4.append(f_dir)
    for i in range(1, 21):
        f_dir = f'/Users/aryanshrivastava/Desktop/LLMWargamingConfidence/logging/outputs/v4/gpt4-free-True-20-1.0/main/run{i}/run{i}.csv'
        gpt_4.append(f_dir)
    gpt_4 = np.array(gpt_4)
    mean_lengths = v_all(gpt_4)

    print('GPT 4 AVERAGE OUTPUT LENGTH')
    print(mean_lengths[0].mean())
    print(mean_lengths[1].mean())

    # gpt 4o
    gpt_4o = []

    for i in range(1, 21):
        f_dir = f'/Users/aryanshrivastava/Desktop/LLMWargamingConfidence/logging/outputs/v4/gpt4o-free-False-20-1.0/main/run{i}/run{i}.csv'
        gpt_4o.append(f_dir)
    for i in range(1, 21):
        f_dir = f'/Users/aryanshrivastava/Desktop/LLMWargamingConfidence/logging/outputs/v4/gpt4o-free-True-20-1.0/main/run{i}/run{i}.csv'
        gpt_4o.append(f_dir)
    gpt_4o = np.array(gpt_4o)
    mean_lengths = v_all(gpt_4o)

    print('GPT 4o AVERAGE OUTPUT LENGTH')
    print(mean_lengths[0].mean())
    print(mean_lengths[1].mean())

    # gpt 4o mini
    gpt_4o_mini = []

    for i in range(1, 21):
        f_dir = f'/Users/aryanshrivastava/Desktop/LLMWargamingConfidence/logging/outputs/v4/gpt4omini-free-False-20-1.0/main/run{i}/run{i}.csv'
        gpt_4o_mini.append(f_dir)
    for i in range(1, 21):
        f_dir = f'/Users/aryanshrivastava/Desktop/LLMWargamingConfidence/logging/outputs/v4/gpt4omini-free-True-20-1.0/main/run{i}/run{i}.csv'
        gpt_4o_mini.append(f_dir)
    gpt_4o_mini = np.array(gpt_4o_mini)
    mean_lengths = v_all(gpt_4o_mini)

    print('GPT 4o MINI AVERAGE OUTPUT LENGTH')
    print(mean_lengths[0].mean())
    print(mean_lengths[1].mean())


    
