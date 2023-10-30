import argparse
import random
import os
import torch
import numpy as np
import json
import nltk
from tqdm import tqdm
import copy
from sklearn.metrics import f1_score, confusion_matrix

from MetaICL.metaicl.data import MetaICLData
from MetaICL.metaicl.model import MetaICLModel

from get_task import get_task
from utils import calculate_sentence_transformer_embedding

from prompt_retrieval import prompt_retrieval
from annotation_methods import  selective_annotation_adaptive_phases



parser = argparse.ArgumentParser()
parser.add_argument('--task_name', required=True,type=str)
parser.add_argument('--selective_annotation_method', required=True,type=str)
parser.add_argument('--model_cache_dir', required=True,type=str)
parser.add_argument('--data_cache_dir', required=True,type=str)
parser.add_argument('--output_dir', required=True,type=str)
parser.add_argument('--model_key', type=str)
parser.add_argument('--prompt_retrieval_method', default='similar',type=str)
parser.add_argument('--model_name', default='EleutherAI/gpt-j-6B',type=str)
parser.add_argument('--embedding_model', default='sentence-transformers/all-mpnet-base-v2',type=str)
parser.add_argument('--annotation_size', default=100,type=int)
parser.add_argument('--seed', default=0,type=int)
parser.add_argument('--batch_size', default=10,type=int)
parser.add_argument('--min_choose', default=10,type=int)
parser.add_argument('--max_choose', default=50,type=int)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--annotation_path',default='first_phase_selected_indices',type=str)
parser.add_argument('--priority',default='diversity',type=str, choices=['diversity', 'difficulty', 'random'])
parser.add_argument('--fig_path',default='output',type=str)
parser.add_argument('--few_shot',default=5,type=int) #0 means we concat as much as possible
parser.add_argument('--steps',default=1,type=int) 
parser.add_argument('--init',default='cluster',type=str, choices=['random', 'cluster']) 
parser.add_argument('--trust',action='store_true')
parser.add_argument('--init_size',default=10,type=int) 
parser.add_argument('--sample_k',action='store_true')
parser.add_argument('--evaluate_calibration',action='store_true')
parser.add_argument('--phases',default=2,type=int) 

##Method
parser.add_argument('--ada_icl_plus',action='store_true')

###Graph 
parser.add_argument('--k_graph',default=15,type=int) 
parser.add_argument('--hard_limit',default=0.5,type=float) 
parser.add_argument('--two_hop',action='store_true')
parser.add_argument('--thres_graph',action='store_true')
parser.add_argument('--mc_selection',default='hard',type=str, choices=['hard', 'hard_easy', 'easy']) 


parser.add_argument('--do_inference',action='store_true')



args = parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    return preds, labels


if __name__=='__main__':
    set_seed(args.seed)

    args.output_dir += f"/adaptive_phases_few_shot-{args.few_shot}-{args.phases}/{args.task_name}_lm-{args.model_name}_annotation-{args.selective_annotation_method}_budget-{args.annotation_size}_init-{args.init}-{args.sample_k}_seed{args.seed}"

    

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir,exist_ok=True)

    with open(os.path.join(args.output_dir,'result_summary.txt'), 'w') as f:
        f.write(f"{args.output_dir}\n")

    print("\n")
    print("=====================")
    print("DATASET: ", args.task_name)
    print("Seed and method", args.seed, args.selective_annotation_method, args.init, args.annotation_size)
    print("=====================")
    print("\n")
    train_examples,eval_examples,train_text_to_encode,eval_text_to_encode,format_example,label_map = get_task(args=args)
    print("Embedding model: ", args.embedding_model)
    # train_examples = eval_examples
    # train_text_to_encode = eval_text_to_encode
    total_train_embeds = calculate_sentence_transformer_embedding(text_to_encode=train_text_to_encode,
                                                                  args=args)

    
    total_eval_embeds = calculate_sentence_transformer_embedding(text_to_encode=eval_text_to_encode,
                                                                  args=args)
    

    output_dir_examples = os.path.join(args.output_dir,'examples')
    if not os.path.isdir(output_dir_examples):
        os.makedirs(output_dir_examples, exist_ok=True)

    path = os.path.join(output_dir_examples, 'all_train_examples.json')
    with open(path, 'w') as fout:
        json.dump(train_examples, fout, indent=4)

    path = os.path.join(output_dir_examples, 'all_eval_examples.json')
    with open(path, 'w') as fout:
        json.dump(eval_examples, fout, indent=4)

    if args.task_name in ['mnli','rte','sst5','mrpc','dbpedia_14','hellaswag','ag_news', 'trec', 'amazon', 'ethos', 'sst2']:
        
        if 'gpt' in args.model_name:
            tokenizer_name = 'gpt2'
        else:
            tokenizer_name = args.model_name

        data_module = MetaICLData(method="direct", max_length=1024, max_length_per_example=300, tokenizer_name=tokenizer_name)
        
        print("Model using", args.model_name)
        inference_model = MetaICLModel(args=args)
        inference_model.load()
        #inference_model.cuda()
        inference_model.eval()
        tokenizer_gpt = None
        return_string = False
        single_input_len = 250
        maximum_input_len = 1000


        

        predicted_eval_examples = []
        path = os.path.join(output_dir_examples, args.selective_annotation_method+"_"+"B"+str(args.annotation_size)+'selected_train_examples.json')

        
        if args.do_inference and os.path.isfile(os.path.join(args.output_dir, f"selected_indices_final.json")):
            with open(os.path.join(args.output_dir, f"selected_indices_final.json")) as f:
                all_phases_selected_indices = json.load(f)

            print("reusing annotated data")
        else:
            print("new annotated data")
            all_phases_selected_indices  = selective_annotation_adaptive_phases(embeddings=total_train_embeds,
                                                                train_examples=train_examples,
                                                                return_string=return_string,
                                                                format_example=format_example,
                                                                maximum_input_len=maximum_input_len,
                                                                label_map=label_map,
                                                                single_context_example_len=single_input_len,
                                                                inference_model=inference_model,
                                                                inference_data_module=data_module,
                                                                tokenizer_gpt=tokenizer_gpt,
                                                                args=args)
            with open(os.path.join(args.output_dir, f"selected_indices_final.json"),'w') as f:
                json.dump(all_phases_selected_indices,f,indent=4)
            processed_train_examples = [train_examples[idx] for idx in all_phases_selected_indices]
            with open(path, 'w') as fout:
                json.dump(processed_train_examples, fout, indent=4)

        #print(le)
        
        processed_train_examples = [train_examples[idx] for idx in all_phases_selected_indices]

        processed_eval_examples = eval_examples

        prompt_retrieval(train_embs=total_train_embeds[all_phases_selected_indices],test_embs=total_eval_embeds,train_examples=processed_train_examples,
                         eval_examples=processed_eval_examples,return_string=return_string,format_example=format_example,
                         maximum_input_len=maximum_input_len,single_context_example_len=single_input_len,label_map=label_map,args=args)

        prompt_cache_dir = os.path.join(args.output_dir, 'prompts')
        candidate_prompt_files = os.listdir(prompt_cache_dir)
        prompt_files = [f for f in candidate_prompt_files if f.endswith('.json')]
        assert len(prompt_files) == len(processed_eval_examples), f"len(prompt_files)={len(prompt_files)}," \
                                                                  f"len(processed_eval_examples)={len(processed_eval_examples)}"
        output_dir = os.path.join(args.output_dir,'results_final_test')
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        count = 0
        running_flag = True
        golds = []
        preds = []
        scores = []
        if not args.task_name in ['hellaswag','xsum','nq']:
            all_labels = []
            label_to_digit = {}
            for k, v in label_map.items():
                all_labels.append(v)
                label_to_digit[v] = k
        execution_count = 0


        ###Single evaluation
        while running_flag:
            running_flag = False
        count += 1
        bar = tqdm(range(len(prompt_files)), desc=f"  LLM inference")
        for file in prompt_files:
            bar.update(1)

            if args.task_name != 'hellaswag':
                with open(os.path.join(prompt_cache_dir, file)) as f:
                    one_test_example = json.load(f)
                cur_train_data = one_test_example[1]
                for idx in range(len(cur_train_data)):
                    cur_train_data[idx]['options'] = all_labels
                for idx in range(len(cur_train_data)):
                    cur_train_data[idx]['options'] = all_labels
                
                
                cur_input = format_example(one_test_example[2], label_map=label_map, args=args)[0]
                data_module.k = len(cur_train_data)
                data_module.tensorize(cur_train_data, [cur_input], options=all_labels)
                prediction = inference_model.do_predict(data_module, require_loss=True, do_probs=False)[0]
                with open(os.path.join(output_dir, file), 'w') as f:
                    json.dump(prediction, f)
                preds.append(label_to_digit[prediction[0]])
                scores.append(prediction[1])
                golds.append(one_test_example[2]['label'])

                new_predicted_example = copy.deepcopy(one_test_example[2])
                new_predicted_example["prediction"] = label_to_digit[prediction[0]]
                new_predicted_example["score"] = prediction[1]
                new_predicted_example["retrieved"] = one_test_example[1]

                predicted_eval_examples.append(new_predicted_example)
    

        path = os.path.join(output_dir_examples, args.model_name+'_'+args.selective_annotation_method+"_"+"B"+str(args.annotation_size)+'FS'+str(args.few_shot)+'_predictions_eval_examples.json')
        with open(path, 'w') as fout:
            json.dump(predicted_eval_examples, fout, indent=4)


        results = []
        assert len(golds) == len(preds), f"len(golds)={len(golds)}, len(preds)={len(preds)}"
        total = len(golds)
        correct = 0
        for p, g in zip(golds, preds):
            if p == g:
                correct += 1
                results.append(1)
            else:
                results.append(0)
        
        with open(os.path.join(args.output_dir,'result_summary.txt'), 'a') as f:
            f.write(f"{len(golds)} examples, accuracy is: {correct / total}\n")

        if args.task_name == 'mrpc':
            
            f1 = f1_score(golds,preds)
            acc = correct / total
            tn, fp, fn, tp = confusion_matrix(golds, preds).ravel()
            specificity = tn / (tn+fp)
            print(f'The f1 score, acc,  and specificity are {f1} {acc} {specificity}\n')
        else:
            print(f'The accuracy score is {correct / total}\n')
