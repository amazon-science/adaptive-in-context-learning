import os
import random
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from prompt_retrieval import prompt_retrieval
from utils import reliability_plot, embedding_plot
from collections import Counter
import statistics

def fast_votek_mod(embeddings,selected_indices,select_num,k,vote_file=None):
    """
    Fast votek method -- similar to kmeans, but uses a graph.

    Args:
        embeddings
        selected_indices: already selected indices (to be excluded)
        select_num: new budget
        k: graph hyperparameter
        vote_file: for saving results. Defaults to None.

    Reference: https://arxiv.org/abs/2209.01975

    Returns:
        list: New selected indices
    """
    
    n = len(embeddings)
    bar = tqdm(range(n),desc=f'voting')
    vote_stat = defaultdict(list)
    for i in range(n):
        cur_emb = embeddings[i].reshape(1, -1)
        cur_scores = np.sum(cosine_similarity(embeddings, cur_emb), axis=1)
        sorted_indices = np.argsort(cur_scores).tolist()[-k-1:-1]
        for idx in sorted_indices:
            if idx!=i:
                vote_stat[idx].append(i)
        bar.update(1)
    if vote_file is not None:
        with open(vote_file,'w') as f:
            json.dump(vote_stat,f)
    votes = sorted(vote_stat.items(),key=lambda x:len(x[1]),reverse=True)
    new_selected_indices = []
    selected_times = defaultdict(int)
    while len(new_selected_indices)<select_num:
        cur_scores = defaultdict(int)
        for idx,candidates in votes:
            if idx in selected_indices+new_selected_indices:
                cur_scores[idx] = -100
                continue
            for one_support in candidates:
                if not one_support in selected_indices:
                    cur_scores[idx] += 10 ** (-selected_times[one_support])
        cur_selected_idx = max(cur_scores.items(),key=lambda x:x[1])[0]
        new_selected_indices.append(int(cur_selected_idx))
        for idx_support in vote_stat[cur_selected_idx]:
            selected_times[idx_support] += 1
    return new_selected_indices

def density_max_coverage(embeddings,hard_idx, easy_idx, selected_indices,select_num,k,vote_file=None, weighted=False, two_hop = True, thres_graph=False, mc_selection="hard"):
    """
    MaxCover porblem formulation and solution.

    Args:
        embeddings 
        hard_idx: indices the model is uncertain about
        easy_idx: indices the model is confident about
        selected_indices: already annotated indices
        select_num: new budget
        k: graph hyperparameter for k-NN
        vote_file (optional): for saving results. Defaults to None.
        weighted (bool, optional): AdaICL or AdaICL+. Defaults to False.
        two_hop (bool, optional): one-hop or two-hop graph. Defaults to True.
        thres_graph (bool, optional): kNN or threshold graph. Defaults to False.
        mc_selection (str, optional): selecting hard (vs. easy vs. both) examples. Defaults to "hard".

    Returns:
        list: New annotated data
    """
    
    if mc_selection=="hard":
        selected = easy_idx.copy() + selected_indices.copy()
    elif mc_selection=="hard_easy":
        selected = selected_indices.copy()
    elif mc_selection=="easy":
        selected = hard_idx.copy() + selected_indices.copy()
    #selected_indices = easy_idx.copy() + selected_indices.copy()
    n = len(embeddings)
    print("2hop graph: ", two_hop)
    
    bar = tqdm(range(n),desc=f'voting')
    vote_stat = defaultdict(list)
    if not thres_graph:
        for i in range(n):
            cur_emb = embeddings[i].reshape(1, -1)
            cur_scores = np.sum(cosine_similarity(embeddings, cur_emb), axis=1)
            sorted_indices = np.argsort(cur_scores).tolist()[-k-1:-1]
            for idx in sorted_indices:
                if idx!=i:
                    vote_stat[idx].append(i)
            bar.update(1)
        
    else:
        print("Threshold graph")
        thresholds = []
        for i in range(n):
            cur_emb = embeddings[i].reshape(1, -1)
            cur_scores = np.sum(cosine_similarity(embeddings, cur_emb), axis=1)
            thres_idx = np.argsort(cur_scores).tolist()[-k-1]
            thresholds.append(cur_scores[thres_idx])
        thresholds.sort()
        mean_thres = statistics.median(thresholds) #sum(thresholds) / len(thresholds)

        for i in range(n):
            cur_emb = embeddings[i].reshape(1, -1)
            cur_scores = np.sum(cosine_similarity(embeddings, cur_emb), axis=1)
            sorted_indices = np.argsort(cur_scores).tolist()
            for idx in sorted_indices:
                if idx!=i and cur_scores[idx] > mean_thres: # and idx in hard_idx:
                    vote_stat[idx].append(i)
            bar.update(1)

    if vote_file is not None:
        with open(vote_file,'w') as f:
            json.dump(vote_stat,f)

    votes = sorted(vote_stat.items(),key=lambda x:len(x[1]),reverse=True)
    new_selected_indices = []
    
    selected_times = defaultdict(int)
    egonet = defaultdict(list)

    #Create egonets
    for idx,candidates in votes:
        for idx_support in candidates:
            if (idx_support in hard_idx) and (idx_support not in egonet[idx]):
                egonet[idx].append(idx_support)
                selected_times[idx] += 1
                if two_hop:
                    neigh_2hop = vote_stat[idx_support]
                    for node in neigh_2hop:
                        if (node in hard_idx) and (node != idx) and (node not in egonet[idx]):
                            egonet[idx].append(node)
                            selected_times[idx] += 1

    

    print("Distribution of Sets: ", selected_times)
    print("Weighted sum:", weighted)

    egonet_greedy = sorted(egonet.items(),key=lambda x:len(x[1]),reverse=True)

    selected_weight = defaultdict(int)

    #print("Egonets:", egonet_greedy)
    while len(new_selected_indices)<select_num:
        cur_scores = defaultdict(int)
        for idx,candidates in egonet_greedy:
            if idx in selected+new_selected_indices:
                cur_scores[idx] = -100 #sanity check
                continue
            for idx_support in candidates:
                if idx_support in hard_idx: #sanity check
                    if weighted:
                        cur_scores[idx] += 10 ** (-selected_weight[idx_support])
                    else:
                        cur_scores[idx] += 1

        cur_selected_idx = max(cur_scores.items(),key=lambda x:x[1])[0]
        new_selected_indices.append(int(cur_selected_idx))

        for idx_support in egonet[cur_selected_idx]:
            selected_weight[idx_support] += 1
            if (not weighted) and (idx_support in hard_idx):
                hard_idx.remove(idx_support)
                
            
        if len(hard_idx) == 0: #only true for weighted=False
            print("All hard examples covered, annotation size:", len(new_selected_indices) )
            break

    return new_selected_indices


def cluster(embeddings,select_num, examples, flag_idx = None, thres=False, reverse=False, clustering_model=None,seed=0):

    """
    Clustering with K-Means utilities. 
    """
    if thres:
        len_list = []
        n = len(examples)

        for ex in examples:
            if "content" in ex:
                sent = ex["content"]
            elif "sentence1" in ex:
                sent = ex["sentence1"]
            elif "sentence" in ex:
                sent = ex["sentence"]
            elif "text" in ex:
                sent = ex["text"]
            elif "premise" in ex:
                sent = ex["premise"]
            elif "ctx" in ex:
                sent = ex["ctx"]
            elif "question" in ex:
                sent = ex["question"]
            sent_len = len(sent.strip().split())
            len_list.append(sent_len)
        assert len(len_list) == n

        len_list = sorted(len_list)

        
        thres_min = 0 
        thres_max = max(len_list[int(0.75*n)], 400)
    else:
        thres_min = 0 
        thres_max = 20000 


    corpus_embeddings = embeddings
    num_clusters = select_num

    # Perform kmean clustering if no model is given
    if clustering_model is  None:
        num_clusters = select_num
        clustering_model = KMeans(n_clusters=num_clusters, random_state=seed)
        clustering_model.fit(corpus_embeddings)
        cluster_assignment = clustering_model.labels_
    else:
        num_clusters = len(clustering_model.cluster_centers_.tolist())
        cluster_assignment = clustering_model.predict(corpus_embeddings)
        

    clustered_sentences = [[] for i in range(num_clusters)]


    #distance matrix for each datapoint and cluster centroid
    dist = clustering_model.transform(corpus_embeddings)
    clustered_dists = [[] for i in range(num_clusters)]
    clustered_idx = [[] for i in range(num_clusters)]

    for cluster_id in range(num_clusters):
        for sentence_id, _ in enumerate(cluster_assignment):
            clustered_dists[cluster_id].append(dist[sentence_id][cluster_id])
            clustered_idx[cluster_id].append(sentence_id)
   
    demos = []

    #Return closest points. Flag_idx flags the candidate points. Thres is a threshold on the length.
    for i in range(len(clustered_dists)):
        tmp = list(map(list, zip(range(len(clustered_dists[i])), clustered_dists[i])))
        top_min_dist = sorted(tmp, key=lambda x: x[1], reverse=reverse)

        ok = 0
        for element in top_min_dist:
            min_idx = element[0]
            idx = clustered_idx[i][min_idx]

            if idx in demos:
                continue
            if flag_idx is not None:
                if idx not in flag_idx:
                    continue

            if thres:
                if "content" in examples[idx]:
                    sent = examples[idx]["content"]
                elif "sentence1" in examples[idx]:
                    sent = examples[idx]["sentence1"]
                elif "sentence" in examples[idx]:
                    sent = examples[idx]["sentence"]
                elif "text" in examples[idx]:
                    sent = examples[idx]["text"]
                elif "premise" in examples[idx]:
                    sent = examples[idx]["premise"]
                elif "ctx" in examples[idx]:
                    sent = examples[idx]["ctx"]
                elif "question" in examples[idx]:
                    sent = examples[idx]["question"]
                if len(sent.strip().split()) >= thres_min and len(sent.strip().split()) <= thres_max:
                    demos.append(idx)
                    ok = 1
                    break
            else:
                demos.append(idx)
                ok = 1
                break
        if ok == 0: #recheck
            for element in top_min_dist:
                min_idx = element[0]
                idx = clustered_idx[i][min_idx]
                if idx in demos:
                    continue
                else:
                    demos.append(idx)
                    break
    return demos, clustering_model




def uncertainty_ranking(selected_indices_first, selected_indices_second, train_embs,test_embs,train_examples,test_examples,return_string,format_example,maximum_input_len,
                        label_map,single_context_example_len,inference_model,inference_data_module,tokenizer_gpt,args, step=0, return_sorted_dict=True):
    """
    Ranks points based on their uncertaintly (from highest to lowest)
    """
    if not args.task_name in ['hellaswag', 'xsum','nq']:
        all_labels = []
        label_to_digit = {}
        for k, v in label_map.items():
            all_labels.append(v)
            label_to_digit[v] = k
    batch_count = step
    
    cur_annotated_examples = [train_examples[idx] for idx in selected_indices_first]
    eval_examples = [test_examples[idx] for idx in selected_indices_second]

    #Retrieval
    prompt_retrieval(train_embs=train_embs[selected_indices_first],
                        test_embs=test_embs[selected_indices_second],
                        train_examples=cur_annotated_examples,
                        eval_examples=eval_examples,
                        return_string=return_string,
                        format_example=format_example,
                        maximum_input_len=maximum_input_len,
                        args=args,label_map=label_map,
                        prompt_identifier=f'prompts_{batch_count}',
                        single_context_example_len=single_context_example_len
                        )

    candidate_prompt_files = os.listdir(os.path.join(args.output_dir,f'prompts_{batch_count}'))
    prompt_files = [f for f in candidate_prompt_files if f.endswith('.json')]


    output_dir = os.path.join(args.output_dir,f'results_iteration_{batch_count}')
    prompt_dir = os.path.join(args.output_dir,f'prompts_{batch_count}')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    count = step
    
    count += 1
    bar = tqdm(range(len(prompt_files)), desc=f"  prediction iteration {batch_count}")

    #Ranking based on model's loss (see inference_model.do_predict)
    for file in prompt_files:
        bar.update(1)
        
        with open(os.path.join(prompt_dir, file)) as f:
            one_test_example = json.load(f)
        cur_train_data = one_test_example[1]
        for idx in range(len(cur_train_data)):
            cur_train_data[idx]['options'] = all_labels

        cur_input = format_example(one_test_example[2],label_map=label_map,args=args)[0]
        inference_data_module.k = len(cur_train_data)
        inference_data_module.tensorize(cur_train_data, [cur_input], options=all_labels)
        prediction = inference_model.do_predict(inference_data_module, require_loss=True)[0]
        with open(f"{output_dir}/{file}", 'w') as f:
            json.dump(prediction, f)


    #Save results and return sorted dictionary: "id": [label_prediction, uncertainty_score]
    idx_scores = {}
    idx_preds = {}
    n = len(test_examples)
    for idx in selected_indices_second:
        if idx in selected_indices_first:
            # if args.task_name in ['xsum','nq']:
            #     idx_scores[idx] = float('inf')
            # else:
            #     idx_scores[idx] = float('inf')
            continue
        
        with open(f"{output_dir}/{idx}.json") as f:
            one_pred = json.load(f)
            if args.task_name in ['nq']:
                idx_scores[idx] = sum(one_pred['choices'][0]["logprobs"]["token_logprobs"]) / len(
                    one_pred['choices'][0]["logprobs"]["token_logprobs"])
            else:
                idx_scores[idx] = (one_pred[0], one_pred[1])
    if args.task_name in ['xsum','nq']:
        sorted_scores = sorted(idx_scores.items(), key=lambda x: x[1][1])
    else:
        sorted_scores = sorted(idx_scores.items(), key=lambda x:x[1][1],reverse=True)


    sorted_scores_len = len(sorted_scores)

    sorted_scores_dict = {}
    selected_indices = []
    for (idx, score) in sorted_scores:
        if score[1] > -10000:
            selected_indices.append(idx)
            sorted_scores_dict[idx] = score

    if not return_sorted_dict:
        return selected_indices, sorted_scores

    return selected_indices, sorted_scores_dict



def votek_mod(selected_indices, pool_idx, train_embs,test_embs,train_examples,test_examples,return_string,format_example,maximum_input_len,
                        label_map,single_context_example_len,inference_model,inference_data_module,tokenizer_gpt,args, k=20, step=0):
    
    """
    Vote-k method, which uniformly (wrt uncertainty) samples diverse datapoints. 
    Reference: https://arxiv.org/abs/2209.01975

    """

    n = len(train_embs)
    bar = tqdm(range(n),desc=f'voting')
    vote_stat = defaultdict(list)
    for i in range(n):
        cur_emb = train_embs[i].reshape(1, -1)
        cur_scores = np.sum(cosine_similarity(train_embs, cur_emb), axis=1)
        sorted_indices = np.argsort(cur_scores).tolist()[-k-1:-1]
        for idx in sorted_indices:
            if idx!=i:
                vote_stat[idx].append(i)
        bar.update(1)
    
    votes = sorted(vote_stat.items(),key=lambda x:len(x[1]),reverse=True)

 
    uncertainty_indices, sorted_scores = uncertainty_ranking(selected_indices, 
                                                pool_idx, 
                                                train_embs=train_embs,
                                                test_embs=test_embs,
                                                train_examples=train_examples,
                                                test_examples=test_examples,
                                                return_string=return_string,
                                                format_example=format_example,
                                                maximum_input_len=maximum_input_len,
                                                label_map=label_map,
                                                single_context_example_len=single_context_example_len,
                                                inference_model=inference_model,
                                                inference_data_module=inference_data_module,
                                                tokenizer_gpt=tokenizer_gpt,
                                                args=args,
                                                step=step,
                                                return_sorted_dict=False)


    # if args.evaluate_calibration:
    #     ece_score, acc = reliability_plot(args, label_map, train_examples,phase=step)
    #     #embedding_plot(args,label_map,selected_indices,train_embs,phase=step)

    #     with open(os.path.join(args.output_dir,'result_summary.txt'), 'a') as f:
    #         f.write(f"{len(pool_idx)} train examples, accuracy is {acc}, ece is {ece_score}\n")    
            
    sorted_scores_len = len(sorted_scores)


    new_selected_indices = []
    selected_times = defaultdict(int)
    select_num_1 = args.annotation_size #+ init_size - len(selected_indices)
    inter = int(len(pool_idx) * 0.9 / select_num_1)
    for prev_idx in selected_indices:
        for idx_support in vote_stat[str(prev_idx)]:
            selected_times[idx_support] += 1
    count_t = 0
    while len(new_selected_indices) < args.annotation_size  and count_t * inter < sorted_scores_len:
        cur_scores = defaultdict(int)
        for idx, _ in sorted_scores[count_t * inter:(count_t + 1) * inter]:
            if not str(idx) in vote_stat:
                cur_scores[idx] = 0
                continue
            candidates = vote_stat[str(idx)]
            if idx in selected_indices or idx in new_selected_indices:
                cur_scores[idx] = -100
                continue
            for one_support in candidates:
                if not one_support in selected_indices:
                    cur_scores[idx] += 10 ** (-selected_times[one_support])
        cur_selected_idx = max(cur_scores.items(), key=lambda x: x[1])[0]
        new_selected_indices.append(cur_selected_idx)
        if cur_selected_idx in vote_stat:
            for idx_support in vote_stat[cur_selected_idx]:
                selected_times[idx_support] += 1
        count_t += 1
    if len(new_selected_indices) < args.annotation_size :
        unselected_indices = []
        for unselected_i in pool_idx:
            if not unselected_i in selected_indices and not not unselected_i in new_selected_indices:
                unselected_indices.append(unselected_i)
        new_selected_indices += random.sample(unselected_indices, args.annotation_size - len(new_selected_indices))
        print(f"{args.annotation_size  - len(new_selected_indices)} examples are randomly selected")
    return new_selected_indices
