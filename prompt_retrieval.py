import os
import json
import numpy as np
from torch import nn
from tqdm import tqdm
from transformers import AutoTokenizer
import shutil
import random


def get_instance_length(input_text,output_text,tokenizer):
    return len(tokenizer(input_text)['input_ids']),len(tokenizer(output_text)['input_ids'])

def prompt_retrieval(train_embs,test_embs,train_examples,eval_examples,return_string,format_example,
                     maximum_input_len,args, label_map,prompt_identifier='prompts',single_context_example_len=None):
    
    """
    Given the test examples (eval_examples) and a pool of annotated ICL examples (train_examples),
    it retrieves top-k similar ICL examples (see args.few_shot for defining k).
    Results are saved as json files.

    """
    
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    eval_example_num = len(eval_examples)
    bar = tqdm(range(eval_example_num), desc="Retrieve examples from annotated pool")

    #Model selection
    if "llama" in args.model_name:
        if '7B' in args.model_name:
            tokenizer_name = "/home/ubuntu/llama_models/7B_hf"
        elif '13B' in args.model_name:
            tokenizer_name = "/home/ubuntu/llama_models/13B_hf"
        elif '65B' in args.model_name:
            tokenizer_name = "/home/ubuntu/llama_models/65B_hf"
        from transformers import LlamaTokenizer
        #tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False, legacy = False)

    elif 'falcon' in args.model_name:
        if '7B' in args.model_name:
            tokenizer_name = "tiiuae/falcon-7b"
        if '40B' in args.model_name:
            tokenizer_name = "tiiuae/falcon-40b"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    elif 'mosaic' in args.model_name:
        if '7B' in args.model_name:
            tokenizer_name = 'EleutherAI/gpt-neox-20b'
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    elif 'gpt' in args.model_name:
        tokenizer_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)



    #Create folders to save the retrieved prompts
    prompt_cache_dir = os.path.join(args.output_dir,prompt_identifier)
    if not os.path.isdir(prompt_cache_dir):
        os.makedirs(prompt_cache_dir, exist_ok=True)
    else:
        shutil.rmtree(prompt_cache_dir)  
        os.makedirs(prompt_cache_dir, exist_ok=True)


    total_num_selected = 0

    #Retrieve ICL examples for each test instance
    for test_id, one_test_instance in enumerate(eval_examples):

        one_test_instance_input_text,one_test_instance_output_text = format_example(example=one_test_instance,args=args,
                                                                                        label_map=label_map)
        cur_prompt_string_len = get_instance_length(one_test_instance_input_text,one_test_instance_output_text,tokenizer)[0]
        if args.prompt_retrieval_method in ['similar', 'noknn']:
            test_e_reshape = test_embs[test_id].reshape(1, -1)
            scores = cos(test_e_reshape, train_embs).numpy()
            sorted_indices = np.argsort(scores)
        elif args.prompt_retrieval_method=='random':
            sorted_indices = np.random.permutation(range(len(train_examples)))
        # elif args.prompt_retrieval_method=='noknn':
        #     #sorted_indices = np.asarray(list(range(len(train_examples)))[::-1])
        #     sorted_indices = np.asarray(list(range(len(train_examples))))
        else:
            raise ValueError(f"The prompt retrieval method {args.prompt_retrieval_method} is not supported")
        selected_indices = []
        num_indices = len(sorted_indices)
        num_selected = 0

        #Top-k selection
        for idx in range(num_indices - 1, -1, -1):
            if (args.prompt_retrieval_method in ['similar', 'knn']) and scores[sorted_indices[idx]]==1:
               continue
            cur_example_input_text,cur_example_output_text = format_example(example=train_examples[sorted_indices[idx]],
                                                                            args=args,label_map=label_map)
            cur_len = sum(get_instance_length(cur_example_input_text, cur_example_output_text,tokenizer=tokenizer))
            if single_context_example_len is not None and cur_len>single_context_example_len:
                continue
            cur_prompt_string_len += cur_len
            
            if cur_prompt_string_len > maximum_input_len:
                break
            
            selected_indices.append(idx)
            num_selected +=1
            total_num_selected += 1
            if num_selected == args.few_shot:
                break

        if args.prompt_retrieval_method in ['similar']:
            one_test_emb = test_embs[test_id]
            indices_scores = []
            for idx in selected_indices:
                indices_scores.append(
                    [idx, cos(train_embs[sorted_indices[idx]].reshape(1, -1), one_test_emb.reshape(1, -1)).item()])
            indices_scores = sorted(indices_scores, key=lambda x: x[1], reverse=True)
            new_selected_indices = [x[0] for x in indices_scores]
            if args.prompt_retrieval_method in ['similar']:
                assert new_selected_indices == selected_indices, f"new_selected_indices={new_selected_indices}, " \
                                                                f"selected_indices={selected_indices}"
            selected_indices = new_selected_indices
        elif args.prompt_retrieval_method in ['noknn']:
            random.shuffle(selected_indices)


        select_num = len(selected_indices)
        second_phase_selected_indices = []
        if return_string:
            cur_train_data = ''
        else:
            cur_train_data = []

        #Create json file and save
        for idx in range(select_num - 1, -1, -1):
            cur_input_text, cur_output_text = format_example(
                example=train_examples[sorted_indices[selected_indices[idx]]],
                args=args, label_map=label_map)
            if return_string:
                cur_train_data += f'{cur_input_text}{cur_output_text}\n\n'
            else:
                if args.task_name=='hellaswag':
                    cur_train_data.append({
                        'input': cur_input_text,
                        'output': cur_output_text,
                        'options': train_examples[sorted_indices[selected_indices[idx]]]['endings']
                    })
                else:
                    cur_train_data.append({
                        'input': cur_input_text,
                        'output': cur_output_text
                    })
            second_phase_selected_indices.append([sorted_indices[selected_indices[idx]].item()])
        if return_string:
            cur_train_data += format_example(
                example=one_test_instance,
                args=args, label_map=label_map)[0]
        # print(f'{len(second_phase_selected_indices)} examples in context')

        with open(os.path.join(prompt_cache_dir,f"{one_test_instance['id']}.json"),'w') as f:
            json.dump([[test_id, second_phase_selected_indices, one_test_instance['label']],
                    cur_train_data,
                    one_test_instance
                    ], f, indent=4)

        bar.update(1)

    print("Avg. prompt number", total_num_selected / eval_example_num)
