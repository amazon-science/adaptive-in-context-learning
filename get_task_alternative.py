import os
import json
import random
from tqdm import tqdm
from datasets import load_dataset
from utils import calculate_sentence_transformer_embedding
from algorithms import cluster

def format_dataset(sample):
    question = sample['question']['text']
    context = sample['document']['tokens']['token']
    is_html = sample['document']['tokens']['is_html']
    long_answers = sample['annotations']['long_answer']
    short_answers = sample['annotations']['short_answers']

    context_string =  " ".join([context[i] for i in range(len(context)) if not is_html[i]])

    # 0 - No ; 1 - Yes
    for answer in sample['annotations']['yes_no_answer']:
        if answer == 0 or answer == 1:
            return {"question": question, "short": ["no" if answer == 0 else "yes"], "long": [], "category": "no" if answer == 0 else "yes"}

    short_targets = []
    for s in short_answers:
        short_targets.extend(s['text'])
    short_targets = list(set(short_targets))

    long_targets = []
    for s in long_answers:
        if s['start_token'] == -1:
            continue
        answer = context[s['start_token']: s['end_token']]
        html = is_html[s['start_token']: s['end_token']]
        new_answer = " ".join([answer[i] for i in range(len(answer)) if not html[i]])
        if new_answer not in long_targets:
            long_targets.append(new_answer)

    category = "other" if len(short_targets) > 0 else "null"

    return {"question": question, "short": short_targets, "long": long_targets, "category": category}

def process_mnli_examples(examples):
    processed_examples = []
    idx = 0
    for raw_data in tqdm(examples,desc='process mnli examples'):
        processed_examples.append({
            'id': idx,
            'label': raw_data['label'],
            'premise': raw_data['premise'],
            'hypothesis': raw_data['hypothesis'],
        })
        idx += 1
    return processed_examples

def process_rte_examples(examples):
    processed_examples = []
    idx = 0
    for raw_data in tqdm(examples,desc='process rte examples'):
        processed_examples.append({
            'id': idx,
            'label': raw_data['label'],
            'sentence1': raw_data['sentence1'],
            'sentence2': raw_data['sentence2'],
        })
        idx += 1
    return processed_examples

def process_sst5_examples(examples):
    processed_examples = []
    idx = 0
    for raw_data in tqdm(examples,desc='process sst5 examples'):
        processed_examples.append({
            'id': idx,
            'label': raw_data['label'],
            'text': raw_data['text'],
        })
        idx += 1
    return processed_examples

def process_sst2_examples(examples):
    processed_examples = []
    idx = 0
    for raw_data in tqdm(examples,desc='process sst2 examples'):
        processed_examples.append({
            'id': idx,
            'label': raw_data['label'],
            'sentence': raw_data['sentence'],
        })
        idx += 1
    return processed_examples



def process_amazon_examples(examples):
    processed_examples = []
    idx = 0
    for raw_data in tqdm(examples,desc='process amazon polarity examples'):
        processed_examples.append({
            'id': idx,
            'label': raw_data['label'],
            'title': raw_data['title'],
            'content': raw_data['content'],
        })
        idx += 1
    return processed_examples

def process_mrpc_examples(examples):
    processed_examples = []
    idx = 0
    for raw_data in tqdm(examples,desc='process mrpc examples'):
        processed_examples.append({
            'id': idx,
            'label': raw_data['label'],
            'sentence1': raw_data['sentence1'],
            'sentence2': raw_data['sentence2'],
        })
        idx += 1
    return processed_examples

def process_dbpedia_examples(examples):
    processed_examples = []
    idx = 0
    for raw_data in tqdm(examples,desc='process dbpedia_14 examples'):
        processed_examples.append({
            'id': idx,
            'label': raw_data['label'],
            'title': raw_data['title'],
            'content': raw_data['content'],
        })
        idx += 1
    return processed_examples

def process_agnews_examples(examples):
    processed_examples = []
    idx = 0
    for raw_data in tqdm(examples,desc='process ag_news examples'):
        processed_examples.append({
            'id': idx,
            'label': raw_data['label'],
            'text': raw_data['text'],
        })
        idx += 1
    return processed_examples

def process_ethos_examples(examples):
    processed_examples = []
    idx = 0
    for raw_data in tqdm(examples,desc='process ethos examples'):
        processed_examples.append({
            'id': idx,
            'label': raw_data['label'],
            'text': raw_data['text'],
        })
        idx += 1
    return processed_examples

def process_trec_examples(examples):
    processed_examples = []
    idx = 0
    for raw_data in tqdm(examples,desc='process trec examples'):
        if 'label-coarse' in raw_data:
            processed_examples.append({
                'id': idx,
                'label': raw_data['label-coarse'],
                'text': raw_data['text'],
            })
        elif 'label' in raw_data:
            processed_examples.append({
                'id': idx,
                'label': raw_data['label'],
                'text': raw_data['text'],
            })
        idx += 1
    return processed_examples

def process_hellaswag_examples(examples):
    processed_examples = []
    idx = 0
    for raw_data in tqdm(examples,desc='process hellaswag examples'):
        processed_examples.append({
            'id': idx,
            'ctx_a': raw_data['ctx_a'],
            'ctx_b': raw_data['ctx_b'],
            'ctx':raw_data['ctx'],
            'endings':raw_data['endings'],
            'label':int(raw_data['label']),
            'activity_label':raw_data['activity_label']
        })
        idx += 1
    return processed_examples

def process_xsum_examples(examples):
    processed_examples = []
    for i,e in enumerate(examples):
        processed_examples.append({
            'id':i,
            'document':e["document"],
            'summary':e["summary"],
            'label':e["summary"],
        })
    return processed_examples

def process_nq_examples(examples):
    processed_examples = []
    for idx,e in enumerate(examples):
        processed_examples.append({
            'id':idx,
            'question':e['question'],
            'short_targets':e['short'],
            'category':e['category'],
            'long': e['long'],
            'label':e['short'],
        })
    return processed_examples

def process_gsm_examples(examples):
    processed_examples = []
    for idx,e in enumerate(examples):
         processed_examples.append({
            'id':idx,
            'question':e['question'],
            'rationale':e['answer'].split("####")[0].strip(),
            'label':e["answer"].split("####")[-1].strip()
        })
    return processed_examples

def get_task(args):
    task_name = args.task_name
    data_cache_dir = args.data_cache_dir
    #print("New prompt")
    if task_name=='mnli':
        if False: #os.path.isfile(os.path.join(args.output_dir,f'train_examples_seed_{args.seed}.json')) and \
            #os.path.isfile(os.path.join(args.output_dir,f'eval_examples_seed_{args.seed}.json')):
            print('use cached examples')
            with open(os.path.join(args.output_dir,f'train_examples_seed_{args.seed}.json')) as f:
                total_train_examples = json.load(f)
            with open(os.path.join(args.output_dir,f'eval_examples_seed_{args.seed}.json')) as f:
                total_eval_examples = json.load(f)
        else:
            mnli_datasets = load_dataset('glue', 'mnli', cache_dir=data_cache_dir)
            total_train_examples = [e for e in mnli_datasets['train']]
            

            if not args.sample_k:
                total_train_examples = random.sample(total_train_examples, 310)
            else:
                total_train_examples = random.sample(total_train_examples, 6000)
                total_train_examples = process_mnli_examples(total_train_examples)
                # all_train_text_to_encode = ["{}. Based on that information, is the claim {} \"True\", \"False\", or \"Inconclusive\"?"
                #                         .format(raw_item["premise"], raw_item["hypothesis"]) for raw_item in total_train_examples]
                all_train_text_to_encode = ["{}? [MASK], {}"
                                        .format(raw_item["premise"], raw_item["hypothesis"]) for raw_item in total_train_examples]
                total_train_embeds = calculate_sentence_transformer_embedding(text_to_encode=all_train_text_to_encode,
                                                                  args=args)
                init_train, _ = cluster(embeddings=total_train_embeds, examples = total_train_examples,select_num=310, thres=False, seed=args.seed)
                total_train_embeds = total_train_embeds[init_train]
                train_examples_old = total_train_examples.copy()
                total_train_examples = [train_examples_old[idx] for idx in init_train]

            total_train_examples = process_mnli_examples(total_train_examples)
            total_eval_examples = [e for e in mnli_datasets['validation_matched']]
            total_eval_examples = random.sample(total_eval_examples, 256)
            total_eval_examples = process_mnli_examples(total_eval_examples)
            with open(os.path.join(args.output_dir,f'train_examples_seed_{args.seed}.json'),'w') as f:
                json.dump(total_train_examples,f,indent=4)
            with open(os.path.join(args.output_dir,f'eval_examples_seed_{args.seed}.json'),'w') as f:
                json.dump(total_eval_examples,f,indent=4)
        if args.debug:
            args.annotation_size = 10
            args.batch_size = 1
            total_train_examples = total_train_examples[:50]
            total_eval_examples = total_eval_examples[:5]
        def format_example(example,label_map,**kwargs):
            #return f"{example['premise']}? [MASK], {example['hypothesis']}", f"{label_map[example['label']]}"
            return f"{example['premise']}. Based on that information, is the claim {example['hypothesis']} \"True\", " \
               f"\"False\", or \"Inconclusive\"?\nanswer:", f"{label_map[example['label']]}"

        # all_train_text_to_encode = ["{}. Based on that information, is the claim {} \"True\", \"False\", or \"Inconclusive\"?"
        #                                 .format(raw_item["premise"], raw_item["hypothesis"]) for raw_item in total_train_examples]
        all_train_text_to_encode = ["{}? [MASK], {}"
                                        .format(raw_item["premise"], raw_item["hypothesis"]) for raw_item in total_train_examples]
        # all_eval_text_to_encode = ["{}. Based on that information, is the claim {} \"True\", \"False\", or \"Inconclusive\"?"
        #                                 .format(raw_item["premise"], raw_item["hypothesis"]) for raw_item in total_eval_examples]
        all_eval_text_to_encode = ["{}? [MASK], {}"
                                        .format(raw_item["premise"], raw_item["hypothesis"]) for raw_item in total_eval_examples]
        label_map = {0:"True",1:"Inconclusive",2:"False"}
        #label_map = {0:"[MASK]: Yes",1:"[MASK]: Maybe",2:"[MASK]: No"}
    elif task_name=='rte':
        if False: #os.path.isfile(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json')) and \
        #         os.path.isfile(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json')):
            print('use cached examples')
            with open(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json')) as f:
                total_train_examples = json.load(f)
            with open(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json')) as f:
                total_eval_examples = json.load(f)
        else:
            rte_datasets = load_dataset('glue', 'rte', cache_dir=data_cache_dir)
            total_train_examples = [e for e in rte_datasets['train']]
            if not args.sample_k:
                total_train_examples = random.sample(total_train_examples, 310)
            else:
                total_train_examples = process_rte_examples(total_train_examples)
                # all_train_text_to_encode = ["{}.\nquestion: {}".format(raw_item["sentence1"], raw_item["sentence2"])
                #                     for raw_item in total_train_examples]
                all_train_text_to_encode = ["{}? [MASK], {}"
                                        .format(raw_item["sentence1"], raw_item["sentence2"]) for raw_item in total_train_examples]
                total_train_embeds = calculate_sentence_transformer_embedding(text_to_encode=all_train_text_to_encode,
                                                                  args=args)
                init_train, _ = cluster(embeddings=total_train_embeds, examples = total_train_examples,select_num=310, thres=False, seed=args.seed)
                total_train_embeds = total_train_embeds[init_train]
                train_examples_old = total_train_examples.copy()
                total_train_examples = [train_examples_old[idx] for idx in init_train]

                
            total_train_examples = process_rte_examples(total_train_examples)
            total_eval_examples = [e for e in rte_datasets['validation']]
            total_eval_examples = random.sample(total_eval_examples, 256)
            total_eval_examples = process_rte_examples(total_eval_examples)
            #if not args.sample_k:
            with open(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json'), 'w') as f:
                json.dump(total_train_examples, f, indent=4)
            with open(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json'), 'w') as f:
                json.dump(total_eval_examples, f, indent=4)
        if args.debug:
            args.annotation_size = 10
            args.batch_size = 1
            total_train_examples = total_train_examples[:50]
            total_eval_examples = total_eval_examples[:5]
        def format_example(example,label_map,**kwargs):
            return f"{example['sentence1']}.\nquestion: {example['sentence2']}. True or False?\nanswer:",\
                   f"{label_map[example['label']]}"
            #return f"{example['sentence1']}? [MASK], {example['sentence2']}", f"{label_map[example['label']]}"

        # all_train_text_to_encode = ["{}.\nquestion: {}".format(raw_item["sentence1"], raw_item["sentence2"])
        #                             for raw_item in total_train_examples]
        # all_eval_text_to_encode = ["{}.\nquestion: {}".format(raw_item["sentence1"], raw_item["sentence2"])
        #                             for raw_item in total_eval_examples]
        
        all_train_text_to_encode = ["{}? [MASK], {}".format(raw_item["sentence1"], raw_item["sentence2"]) 
                                    for raw_item in total_train_examples]
        all_eval_text_to_encode = ["{}? [MASK], {}".format(raw_item["sentence1"], raw_item["sentence2"])
                                    for raw_item in total_eval_examples]
        # all_train_text_to_encode1 = [raw_item["sentence1"] for raw_item in total_train_examples]
        # all_train_text_to_encode2 = [raw_item["sentence2"] for raw_item in total_train_examples]

        # all_eval_text_to_encode1 = [raw_item["sentence1"] for raw_item in total_eval_examples]
        # all_eval_text_to_encode2 = [raw_item["sentence2"] for raw_item in total_eval_examples]

        label_map = {0:"True",1:"False"}
        #label_map = {0:"[MASK]: Yes",1:"[MASK]: No"}
    elif task_name=='sst5':
        if False: # os.path.isfile(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json')) and \
                #os.path.isfile(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json')):
            print('use cached examples')
            with open(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json')) as f:
                total_train_examples = json.load(f)
            with open(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json')) as f:
                total_eval_examples = json.load(f)
        else:
            sst5_datasets = load_dataset('SetFit/sst5',cache_dir=data_cache_dir)
            total_train_examples = [e for e in sst5_datasets['train']]
            if not args.sample_k:
                total_train_examples = random.sample(total_train_examples, 310)
            else:
                total_train_examples = random.sample(total_train_examples, 6000)
                total_train_examples = process_sst5_examples(total_train_examples)
                all_train_text_to_encode = [raw_item["text"] for raw_item in total_train_examples]
                total_train_embeds = calculate_sentence_transformer_embedding(text_to_encode=all_train_text_to_encode,
                                                                  args=args)
                init_train, _ = cluster(embeddings=total_train_embeds, examples = total_train_examples,select_num=310, thres=False, seed=args.seed)
                total_train_embeds = total_train_embeds[init_train]
                train_examples_old = total_train_examples.copy()
                total_train_examples = [train_examples_old[idx] for idx in init_train]

            total_train_examples = process_sst5_examples(total_train_examples)
            total_eval_examples = [e for e in sst5_datasets['test']]
            total_eval_examples = random.sample(total_eval_examples, 256)
            total_eval_examples = process_sst5_examples(total_eval_examples)
            with open(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json'), 'w') as f:
                json.dump(total_train_examples, f, indent=4)
            with open(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json'), 'w') as f:
                json.dump(total_eval_examples, f, indent=4)
        if args.debug:
            args.annotation_size = 10
            args.batch_size = 1
            total_train_examples = total_train_examples[:50]
            total_eval_examples = total_eval_examples[:5]
        def format_example(example,label_map,**kwargs):
            return f"Content: {example['text']} Sentiment: ",\
                   f"{label_map[example['label']]}"

        all_train_text_to_encode = [raw_item["text"] for raw_item in total_train_examples]
        all_eval_text_to_encode = [raw_item["text"] for raw_item in total_eval_examples]
        label_map = {0:"terrible",1:"bad",2:"okay",3:"good",4:"great"}
    elif task_name=='sst2':
        if False: # os.path.isfile(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json')) and \
                #os.path.isfile(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json')):
            print('use cached examples')
            with open(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json')) as f:
                total_train_examples = json.load(f)
            with open(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json')) as f:
                total_eval_examples = json.load(f)
        else:
            sst2_datasets = load_dataset('glue', 'sst2',cache_dir=data_cache_dir)
            total_train_examples = [e for e in sst2_datasets['train']]
            if not args.sample_k:
                total_train_examples = random.sample(total_train_examples, 310)
            else:
                total_train_examples = random.sample(total_train_examples, 6000)
                total_train_examples = process_sst2_examples(total_train_examples)
                all_train_text_to_encode = [raw_item["sentence"] for raw_item in total_train_examples]
                total_train_embeds = calculate_sentence_transformer_embedding(text_to_encode=all_train_text_to_encode,
                                                                  args=args)
                init_train, _ = cluster(embeddings=total_train_embeds, examples = total_train_examples,select_num=310, thres=False, seed=args.seed)
                total_train_embeds = total_train_embeds[init_train]
                train_examples_old = total_train_examples.copy()
                total_train_examples = [train_examples_old[idx] for idx in init_train]

            total_train_examples = process_sst2_examples(total_train_examples)
            total_eval_examples = [e for e in sst2_datasets['validation']]
            total_eval_examples = random.sample(total_eval_examples, 256)
            total_eval_examples = process_sst2_examples(total_eval_examples)
            with open(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json'), 'w') as f:
                json.dump(total_train_examples, f, indent=4)
            with open(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json'), 'w') as f:
                json.dump(total_eval_examples, f, indent=4)
        if args.debug:
            args.annotation_size = 10
            args.batch_size = 1
            total_train_examples = total_train_examples[:50]
            total_eval_examples = total_eval_examples[:5]
        def format_example(example,label_map,**kwargs):
            # return f"{example['sentence']} It was ",\
            #        f"{label_map[example['label']]}"
            return f"Content: {example['sentence']} Sentiment: ",\
                    f"{label_map[example['label']]}"

        all_train_text_to_encode = [raw_item["sentence"] for raw_item in total_train_examples]
        all_eval_text_to_encode = [raw_item["sentence"] for raw_item in total_eval_examples]
        label_map = {0:"Negative",1:"Positive"}
    elif task_name=='amazon':
        if False: #os.path.isfile(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json')) and \
                #os.path.isfile(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json')):
            print('use cached examples')
            with open(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json')) as f:
                total_train_examples = json.load(f)
            with open(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json')) as f:
                total_eval_examples = json.load(f)
        else:
            amazon_datasets = load_dataset('amazon_polarity',cache_dir=data_cache_dir)
            total_train_examples = [e for e in amazon_datasets['train']]
            if not args.sample_k:
                total_train_examples = random.sample(total_train_examples, 310)
            else:
                total_train_examples = random.sample(total_train_examples, 6000)
                total_train_examples = process_amazon_examples(total_train_examples)
                all_train_text_to_encode = [raw_item["content"] for raw_item in total_train_examples]
                total_train_embeds = calculate_sentence_transformer_embedding(text_to_encode=all_train_text_to_encode,
                                                                  args=args)
                init_train, _ = cluster(embeddings=total_train_embeds, examples = total_train_examples,select_num=310, thres=False, seed=args.seed)
                total_train_embeds = total_train_embeds[init_train]
                train_examples_old = total_train_examples.copy()
                total_train_examples = [train_examples_old[idx] for idx in init_train]

            total_train_examples = process_amazon_examples(total_train_examples)
            total_eval_examples = [e for e in amazon_datasets['test']]
            total_eval_examples = random.sample(total_eval_examples, 256)
            total_eval_examples = process_amazon_examples(total_eval_examples)
            with open(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json'), 'w') as f:
                json.dump(total_train_examples, f, indent=4)
            with open(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json'), 'w') as f:
                json.dump(total_eval_examples, f, indent=4)
        if args.debug:
            args.annotation_size = 10
            args.batch_size = 1
            total_train_examples = total_train_examples[:50]
            total_eval_examples = total_eval_examples[:5]
        def format_example(example,label_map,**kwargs):
            # return f"Title: {example['title']}\n Review: {example['content']}\n Sentiment:",\
            #        f"{label_map[example['label']]}"
            return f"Title: {example['title']}\n Review: {example['content']}\n Sentiment: ",\
                   f"{label_map[example['label']]}"

        all_train_text_to_encode = [raw_item["title"]+" : "+ raw_item['content'] for raw_item in total_train_examples]
        all_eval_text_to_encode = [raw_item["title"]+" : "+ raw_item['content'] for raw_item in total_eval_examples]
        label_map = {0:"Negative",1:"Positive"}
    elif task_name=='ethos':
        if os.path.isfile(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json')) and \
                os.path.isfile(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json')):
            print('use cached examples')
            with open(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json')) as f:
                total_train_examples = json.load(f)
            with open(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json')) as f:
                total_eval_examples = json.load(f)
        else:
            amazon_datasets = load_dataset('ethos', 'binary', cache_dir=data_cache_dir)
            total_train_examples = [e for e in amazon_datasets['train']]
            all_examples = random.sample(total_train_examples, 310+256)
            total_train_examples = all_examples[:310]
            total_train_examples = process_ethos_examples(total_train_examples)
            total_eval_examples = all_examples[310:]
            total_eval_examples = process_ethos_examples(total_eval_examples)
            with open(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json'), 'w') as f:
                json.dump(total_train_examples, f, indent=4)
            with open(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json'), 'w') as f:
                json.dump(total_eval_examples, f, indent=4)
        if args.debug:
            args.annotation_size = 10
            args.batch_size = 1
            total_train_examples = total_train_examples[:50]
            total_eval_examples = total_eval_examples[:5]
        def format_example(example,label_map,**kwargs):
            return f"Does the following statement contain hate speech: {example['text']}\n Answer:",\
                   f"{label_map[example['label']]}"

        all_train_text_to_encode = [raw_item['text'] for raw_item in total_train_examples]
        all_eval_text_to_encode = [raw_item['text'] for raw_item in total_eval_examples]
        label_map = {0:"No, it does not contain hate speech.",1:"Yes, it contains hate speech."}
    elif task_name=='mrpc':
        if False: #os.path.isfile(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json')) and \
                #os.path.isfile(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json')):
            print('use cached examples')
            with open(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json')) as f:
                total_train_examples = json.load(f)
            with open(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json')) as f:
                total_eval_examples = json.load(f)
        else:
            mrpc_datasets = load_dataset('glue','mrpc',cache_dir=data_cache_dir)
            total_train_examples = [e for e in mrpc_datasets['train']]
            if not args.sample_k:
                total_train_examples = random.sample(total_train_examples, 310)
            else:
                total_train_examples = process_mrpc_examples(total_train_examples)
                all_train_text_to_encode = ["{} [MASK], {}"
                                        .format(raw_item["sentence1"], raw_item["sentence2"]) for raw_item in total_train_examples]
                
                total_train_embeds = calculate_sentence_transformer_embedding(text_to_encode=all_train_text_to_encode,
                                                                  args=args)
                init_train, _ = cluster(embeddings=total_train_embeds, examples = total_train_examples, select_num=310, thres=False, seed=args.seed)
                total_train_embeds = total_train_embeds[init_train]
                train_examples_old = total_train_examples.copy()
                total_train_examples = [train_examples_old[idx] for idx in init_train]
            
            total_train_examples = process_mrpc_examples(total_train_examples)
            total_eval_examples = [e for e in mrpc_datasets['validation']]
            total_eval_examples = random.sample(total_eval_examples, 256)
            total_eval_examples = process_mrpc_examples(total_eval_examples)
            with open(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json'), 'w') as f:
                json.dump(total_train_examples, f, indent=4)
            with open(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json'), 'w') as f:
                json.dump(total_eval_examples, f, indent=4)
        if args.debug:
            args.annotation_size = 10
            args.batch_size = 1
            total_train_examples = total_train_examples[:50]
            total_eval_examples = total_eval_examples[:5]
        def format_example(example,label_map,**kwargs):
            return f"Are the following two sentences 'equivalent' or 'not equivalent'?\n" \
                   f"{example['sentence1']}.\n{example['sentence2']}.\nanswer:",\
                   f"{label_map[example['label']]}"
            # return f"{example['sentence1']} [MASK], {example['sentence2']}",\
            #        f"{label_map[example['label']]}"

        # all_train_text_to_encode = ["{}.\n{}".format(raw_item["sentence1"], raw_item["sentence2"])
        #                             for raw_item in total_train_examples]
        all_train_text_to_encode = ["{} [MASK], {}".format(raw_item["sentence1"], raw_item["sentence2"]) 
                                        for raw_item in total_train_examples]
                
        # all_eval_text_to_encode = ["{}.\n{}".format(raw_item["sentence1"], raw_item["sentence2"])
        #                            for raw_item in total_eval_examples]
        all_eval_text_to_encode = ["{} [MASK], {}".format(raw_item["sentence1"], raw_item["sentence2"])
                                   for raw_item in total_eval_examples]
        #label_map = {0:"[MASK]: No",1:"[MASK]: Yes"}
        label_map = {0:"not equivalent",1:"equivalent"}
    elif task_name=='dbpedia_14':
        if os.path.isfile(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json')) and \
                os.path.isfile(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json')):
            print('use cached examples')
            with open(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json')) as f:
                total_train_examples = json.load(f)
            with open(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json')) as f:
                total_eval_examples = json.load(f)
        else:
            dbpedia_datasets = load_dataset('dbpedia_14', cache_dir=data_cache_dir)
            total_train_examples = [e for e in dbpedia_datasets['train']]
            total_train_examples = random.sample(total_train_examples, 310)
            total_train_examples = process_dbpedia_examples(total_train_examples)
            total_eval_examples = [e for e in dbpedia_datasets['test']]
            total_eval_examples = random.sample(total_eval_examples, 256)
            total_eval_examples = process_dbpedia_examples(total_eval_examples)
            with open(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json'), 'w') as f:
                json.dump(total_train_examples, f, indent=4)
            with open(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json'), 'w') as f:
                json.dump(total_eval_examples, f, indent=4)
        if args.debug:
            args.annotation_size = 10
            args.batch_size = 1
            total_train_examples = total_train_examples[:50]
            total_eval_examples = total_eval_examples[:5]
        def format_example(example,label_map,**kwargs):
            return f"title: {example['title']}; content: {example['content']}",\
                   f"{label_map[example['label']]}"

        all_train_text_to_encode = ["title: {} ; content: {}".format(raw_item["title"], raw_item["content"])
                                    for raw_item in total_train_examples]
        all_eval_text_to_encode = ["title: {} ; content: {}".format(raw_item["title"], raw_item["content"])
                                   for raw_item in total_eval_examples]
        label_map = {0: "company",1: "educational institution",2: "artist",3: "athlete",4: "office holder",
            5: "mean of transportation",6: "building",7: "natural place",8: "village",9: "animal",10: "plant",
            11: "album",12: "film",13: "written work"}
        
    elif task_name=='ag_news':
        if False: #os.path.isfile(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json')) and \
            #    os.path.isfile(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json')):
            print('use cached examples')
            with open(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json')) as f:
                total_train_examples = json.load(f)
            with open(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json')) as f:
                total_eval_examples = json.load(f)
        else:
            agnews_datasets = load_dataset('ag_news', cache_dir=data_cache_dir)
            total_train_examples = [e for e in agnews_datasets['train']]
            if not args.sample_k:
                total_train_examples = random.sample(total_train_examples, 310)
            else:
                total_train_examples = random.sample(total_train_examples, 6000)
                total_train_examples = process_agnews_examples(total_train_examples)
                all_train_text_to_encode = ["{}".format(raw_item["text"])
                                    for raw_item in total_train_examples]
                total_train_embeds = calculate_sentence_transformer_embedding(text_to_encode=all_train_text_to_encode,
                                                                  args=args)
                init_train, _ = cluster(embeddings=total_train_embeds, examples = total_train_examples,select_num=310, thres=False, seed=args.seed)
                total_train_embeds = total_train_embeds[init_train]
                train_examples_old = total_train_examples.copy()
                total_train_examples = [train_examples_old[idx] for idx in init_train]

            total_train_examples = process_agnews_examples(total_train_examples)
            total_eval_examples = [e for e in agnews_datasets['test']]
            total_eval_examples = random.sample(total_eval_examples, 256)
            total_eval_examples = process_agnews_examples(total_eval_examples)
            with open(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json'), 'w') as f:
                json.dump(total_train_examples, f, indent=4)
            with open(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json'), 'w') as f:
                json.dump(total_eval_examples, f, indent=4)
        if args.debug:
            args.annotation_size = 10
            args.batch_size = 1
            total_train_examples = total_train_examples[:50]
            total_eval_examples = total_eval_examples[:5]
        def format_example(example,label_map,**kwargs):
            return f"Content: {example['text']} Topic: ",\
                   f"{label_map[example['label']]}"

        all_train_text_to_encode = ["{}".format(raw_item["text"])
                                    for raw_item in total_train_examples]
        all_eval_text_to_encode = ["{}".format(raw_item["text"])
                                   for raw_item in total_eval_examples]
        label_map = {0: "World",1: "Sports",2: "Business",3: "Sci/Tech"}
        
    elif task_name=='trec':
        if False: #os.path.isfile(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json')) and \
            #     os.path.isfile(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json')):
            print('use cached examples')
            with open(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json')) as f:
                total_train_examples = json.load(f)
            with open(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json')) as f:
                total_eval_examples = json.load(f)
        else:
            trec_datasets = load_dataset('trec', cache_dir=data_cache_dir)
            total_train_examples = [e for e in trec_datasets['train']]
            if not args.sample_k:
                total_train_examples = random.sample(total_train_examples, 310)
            else:
                
                total_train_examples = process_trec_examples(total_train_examples)
                all_train_text_to_encode = ["{}".format(raw_item["text"])
                                    for raw_item in total_train_examples]
                total_train_embeds = calculate_sentence_transformer_embedding(text_to_encode=all_train_text_to_encode,
                                                                  args=args)
                
                init_train, _ = cluster(embeddings=total_train_embeds, examples = total_train_examples,select_num=310, thres=False, seed=args.seed)
                total_train_embeds = total_train_embeds[init_train]
                train_examples_old = total_train_examples.copy()
                total_train_examples = [train_examples_old[idx] for idx in init_train]
            
            total_train_examples = process_trec_examples(total_train_examples)
            total_eval_examples = [e for e in trec_datasets['test']]
            total_eval_examples = random.sample(total_eval_examples, 256)
            total_eval_examples = process_trec_examples(total_eval_examples)
            with open(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json'), 'w') as f:
                json.dump(total_train_examples, f, indent=4)
            with open(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json'), 'w') as f:
                json.dump(total_eval_examples, f, indent=4)
        if args.debug:
            args.annotation_size = 10
            args.batch_size = 1
            total_train_examples = total_train_examples[:50]
            total_eval_examples = total_eval_examples[:5]
        def format_example(example,label_map,**kwargs):
            return f"Content: {example['text']} Answer Type:",\
                   f"{label_map[example['label']]}"

        all_train_text_to_encode = ["{}".format(raw_item["text"])
                                    for raw_item in total_train_examples]
        all_eval_text_to_encode = ["{}".format(raw_item["text"])
                                   for raw_item in total_eval_examples]
        label_map = {0: "ABBR",1: "ENTY",2: "DESC",3: "HUM", 4:"LOC", 5:"NUM"}
        label_map = {0: "Abbreviation",1: "Entity",2: "Description",3: "Human", 4:"Location", 5:"Numeric"}
    elif task_name=='hellaswag':
        if os.path.isfile(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json')) and \
                os.path.isfile(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json')):
            print('use cached examples')
            with open(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json')) as f:
                total_train_examples = json.load(f)
            with open(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json')) as f:
                total_eval_examples = json.load(f)
        else:
            hellaswag_datasets = load_dataset('hellaswag',cache_dir=data_cache_dir)
            total_train_examples = [e for e in hellaswag_datasets['train']]
            total_train_examples = random.sample(total_train_examples, 310)
            total_train_examples = process_hellaswag_examples(total_train_examples)
            total_eval_examples = [e for e in hellaswag_datasets['validation']]
            total_eval_examples = random.sample(total_eval_examples, 256)
            total_eval_examples = process_hellaswag_examples(total_eval_examples)
            with open(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json'), 'w') as f:
                json.dump(total_train_examples, f, indent=4)
            with open(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json'), 'w') as f:
                json.dump(total_eval_examples, f, indent=4)
        if args.debug:
            args.annotation_size = 10
            args.batch_size = 1
            total_train_examples = total_train_examples[:50]
            total_eval_examples = total_eval_examples[:5]
        def format_example(example,label_map,**kwargs):
            return f"The topic is {example['activity_label']}. {example['ctx_a']} " \
                   f"{example['ctx_b']} ",f"{example['endings'][example['label']]}"

        all_train_text_to_encode = [f"The topic is {raw_item['activity_label']}. {raw_item['ctx_a']} {raw_item['ctx_b']} | " \
                                  f"{raw_item['endings'][0]} | " \
                                  f"{raw_item['endings'][1]} | " \
                                  f"{raw_item['endings'][2]} | " \
                                  f"{raw_item['endings'][3]}" for raw_item in total_train_examples]
        all_eval_text_to_encode = [f"The topic is {raw_item['activity_label']}. {raw_item['ctx_a']} {raw_item['ctx_b']} | " \
                                  f"{raw_item['endings'][0]} | " \
                                  f"{raw_item['endings'][1]} | " \
                                  f"{raw_item['endings'][2]} | " \
                                  f"{raw_item['endings'][3]}" for raw_item in total_eval_examples]
        label_map = None
    elif task_name == 'xsum':
        if os.path.isfile(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json')) and \
                os.path.isfile(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json')):
            print('use cached examples')
            with open(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json')) as f:
                total_train_examples = json.load(f)
            with open(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json')) as f:
                total_eval_examples = json.load(f)
        else:
            xsum_dataset = load_dataset('xsum',cache_dir=data_cache_dir)
            total_train_examples = [e for e in xsum_dataset['train']]
            total_train_examples = random.sample(total_train_examples, 310)
            total_train_examples = process_xsum_examples(total_train_examples)
            total_eval_examples = [e for e in xsum_dataset['test']]
            total_eval_examples = random.sample(total_eval_examples, 256)
            total_eval_examples = process_xsum_examples(total_eval_examples)
            with open(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json'), 'w') as f:
                json.dump(total_train_examples, f, indent=4)
            with open(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json'), 'w') as f:
                json.dump(total_eval_examples, f, indent=4)
        if args.debug:
            args.annotation_size = 10
            args.batch_size = 1
            total_train_examples = total_train_examples[:50]
            total_eval_examples = total_eval_examples[:5]
        def format_example(example,label_map,**kwargs):
            return f"write a short summary:\n{example['document']}\nTL;DR:",f"{example['summary']}"

        all_train_text_to_encode = [raw_item['document']
                                    for raw_item in total_train_examples]
        all_eval_text_to_encode = [raw_item['document']
                                   for raw_item in total_eval_examples]
        label_map = None
    elif task_name == 'nq':
        if os.path.isfile(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json')) and \
                os.path.isfile(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json')):
            print('use cached examples')
            with open(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json')) as f:
                total_train_examples = json.load(f)
            with open(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json')) as f:
                total_eval_examples = json.load(f)
        else:
            nq_dataset = load_dataset('natural_questions', name="dev", cache_dir=data_cache_dir)
            #first_sub_sample_indices = random.sample(range(len(nq_dataset['validation'])), 12000)
            train_data = nq_dataset['validation'].map(format_dataset)
            total_train_examples = train_data.remove_columns(["annotations", "document", "id"]).filter(
                lambda x: x['category'] != "null")
            total_train_examples = [e for e in total_train_examples]
            total_train_examples = random.sample(total_train_examples, 310)
            total_train_examples = process_nq_examples(total_train_examples)
            total_eval_examples = nq_dataset['validation'].map(format_dataset).remove_columns(
                ["annotations", "document", "id"]).filter(lambda x: x['category'] != "null")
            total_eval_examples = [e for e in total_eval_examples]
            total_eval_examples = random.sample(total_eval_examples, 256)
            total_eval_examples = process_nq_examples(total_eval_examples)
            with open(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json'), 'w') as f:
                json.dump(total_train_examples, f, indent=4)
            with open(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json'), 'w') as f:
                json.dump(total_eval_examples, f, indent=4)
        if args.debug:
            args.annotation_size = 10
            args.batch_size = 1
            total_train_examples = total_train_examples[:50]
            total_eval_examples = total_eval_examples[:5]

        def format_example(example, label_map, **kwargs):
            if example['category'] in ['yes', 'no']:
                return f"Write an answer: {example['question']}\nclass", f"{example['category']}"
            assert example['category'] == 'other', example['category']
            assert len(example['short_targets']) > 0, f"{example['short_targets']}"
            return f"Write an answer: {example['question']}\n{example['category']} ", f"{example['short_targets'][0]}"

        all_train_text_to_encode = [raw_item['question']
                                    for raw_item in total_train_examples]
        all_eval_text_to_encode = [raw_item['question']
                                   for raw_item in total_eval_examples]
        label_map = None

    elif task_name == 'gsm8k':
        if False: #os.path.isfile(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json')) and \
                #os.path.isfile(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json')):
            print('use cached examples')
            with open(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json')) as f:
                total_train_examples = json.load(f)
            with open(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json')) as f:
                total_eval_examples = json.load(f)
        else:
            gsm_dataset = load_dataset('gsm8k', 'main', cache_dir=data_cache_dir)
            
            total_train_examples = [e for e in gsm_dataset['train']]
            total_train_examples = random.sample(total_train_examples, 310)
            total_train_examples = process_gsm_examples(total_train_examples)
            total_eval_examples = [e for e in gsm_dataset['test']]
            total_eval_examples = random.sample(total_eval_examples, 256)
            total_eval_examples = process_gsm_examples(total_eval_examples)
            with open(os.path.join(args.output_dir, f'train_examples_seed_{args.seed}.json'), 'w') as f:
                json.dump(total_train_examples, f, indent=4)
            with open(os.path.join(args.output_dir, f'eval_examples_seed_{args.seed}.json'), 'w') as f:
                json.dump(total_eval_examples, f, indent=4)
        if args.debug:
            args.annotation_size = 10
            args.batch_size = 1
            total_train_examples = total_train_examples[:50]
            total_eval_examples = total_eval_examples[:5]

        def format_example(example, label_map, **kwargs):
            
            return f"Question: {example['question']}\n Let's think step by step. ", f"{example['rationale']}.\nTherefore, the answer is {example['label']}."

        all_train_text_to_encode = [raw_item['question']
                                    for raw_item in total_train_examples]
        all_eval_text_to_encode = [raw_item['question']
                                   for raw_item in total_eval_examples]
        label_map = None
    else:
        raise ValueError(f"{args.task_name} is not supported")
    return total_train_examples,total_eval_examples,all_train_text_to_encode, \
           all_eval_text_to_encode, format_example,label_map
