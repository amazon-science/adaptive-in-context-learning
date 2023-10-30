import torch
import openai
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import GPTJForCausalLM
from collections import OrderedDict
#import sqlparse
from nltk import tokenize
import torch.nn.functional as F
import seaborn as sns
from scipy import stats
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import numpy as np

def calculate_sentence_transformer_embedding(text_to_encode,args):
    num = len(text_to_encode)
    emb_model = SentenceTransformer(args.embedding_model)
    embeddings = []
    bar = tqdm(range(0,num,20),desc='calculate embeddings')
    for i in range(0,num,20):
        embeddings += emb_model.encode(text_to_encode[i:i+20]).tolist()
        bar.update(1)
    embeddings = torch.tensor(embeddings)
    embeddings = F.normalize(embeddings, p=2, dim=-1) #embeddings / (embeddings.norm(dim=1)[:, None] + 1e-6)

    # mean_embeddings = torch.mean(embeddings, 0, True)
    # embeddings = embeddings #- mean_embeddings
    return embeddings

def get_sub_answers(answers, begin=0, end=None):
    return [" ".join(x.split(" ")[begin:end]) for x in answers if len(x.split(" ")) > 1]

PUNCTUATION_SET_TO_EXCLUDE = set(''.join(['‘', '’', '´', '`', '.', ',', '-', '"']))
def expand_to_aliases(given_answers, make_sub_answers=False):
    if make_sub_answers:
        given_answers = given_answers + get_sub_answers(given_answers, begin=1) + get_sub_answers(given_answers, end=-1)
    answers = []
    for answer in given_answers:
        alias = answer.replace('_', ' ').lower()
        alias = ''.join(c if c not in PUNCTUATION_SET_TO_EXCLUDE else ' ' for c in alias)
        answers.append(' '.join(alias.split()).strip())
    return set(answers)

def compute_acc(gold, pred, n_slot=30):

    if type(gold) == dict:
        gold = [f"{k}-{v}" for k, v in gold.items()]
    if type(pred) == dict:
        pred = [f"{k}-{v}" for k, v in pred.items()]

    miss_gold = 0
    miss_slot = []
    for g in gold:
        if g not in pred:
            miss_gold += 1
            miss_slot.append(g.rsplit("-", 1)[0])
    wrong_pred = 0
    for p in pred:
        if p not in gold and p.rsplit("-", 1)[0] not in miss_slot:
            wrong_pred += 1
    ACC_TOTAL = n_slot
    ACC = n_slot - miss_gold - wrong_pred
    ACC = ACC / float(ACC_TOTAL)
    return ACC


def compute_prf(gold, pred):

    if type(gold) == dict:
        gold = [f"{k}-{v}" for k, v in gold.items()]
    if type(pred) == dict:
        pred = [f"{k}-{v}" for k, v in pred.items()]

    TP, FP, FN = 0, 0, 0
    if len(gold) != 0:
        count = 1
        for g in gold:
            if g in pred:
                TP += 1
            else:
                FN += 1
        for p in pred:
            if p not in gold:
                FP += 1
        precision = TP / float(TP+FP) if (TP+FP) != 0 else 0
        recall = TP / float(TP+FN) if (TP+FN) != 0 else 0
        F1 = 2 * precision * recall / \
            float(precision + recall) if (precision+recall) != 0 else 0
    else:
        if len(pred) == 0:
            precision, recall, F1, count = 1, 1, 1, 1
        else:
            precision, recall, F1, count = 0, 0, 0, 1
    return F1, recall, precision, count


def evaluate(preds: dict, golds: dict):

    gold_slots = list(golds.keys())
    for k in gold_slots:
        if '|' in golds[k]:
            gold_values = golds[k].split('|')
            if k in preds and preds[k] in gold_values:
                golds[k] = preds[k]

    jga, acc, f1 = 0, 0, 0

    if preds == golds:
        jga = 1
    acc = compute_acc(golds, preds)
    f1 = compute_prf(golds, preds)[0]

    return jga, acc, f1


def embedding_plot(args, label_map, selected_indices,total_train_embeds, phase=0):
    """
    Visualization of PCA (2 components) of the data points in the embedding space (e.g., SBERT)

    Args:
        args
        label_map (dict): label mapping
        selected_indices (list): selected data for annotation
        total_train_embeds (npy): embedding space
        phase (int, optional): selection phase. Defaults to 0.
    """
    
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(total_train_embeds)

    output_dir = os.path.join(args.output_dir,'figs')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    label_to_digit = {}
    for k, v in label_map.items():
        label_to_digit[v] = k

    x1 = []
    x2 = []
    y_col = []

    if phase != -1:
        candidate_results_files = os.listdir(os.path.join(args.output_dir,f'results_iteration_{phase}'))
    else:
        candidate_results_files = os.listdir(os.path.join(args.output_dir,'results_final_test'))
    #print(candidate_prompt_files)
    result_files = [f for f in candidate_results_files if f.endswith('.json')]

    if phase != -1:
        output_dir = os.path.join(args.output_dir,f'results_iteration_{phase}')
    else:
        output_dir = os.path.join(args.output_dir,'results_final_test')

    for file in result_files:
        with open(f"{output_dir}/{file}", 'r') as f:
            example_pred = json.load(f)
        idx = int(file[:-5])
        y_col.append(-example_pred[1])
        x1.append(pca_result[idx, 0])
        x2.append(pca_result[idx, 1])

    ymax = max(y_col)
    ymin = min(y_col)
    
    y_scaled = [ (yi - ymin) / (ymax - ymin) for yi in y_col ]

    for idx in selected_indices:
        x1.append(pca_result[idx, 0])
        x2.append(pca_result[idx, 1])
        y_scaled.append(1)

    cmap = sns.diverging_palette(10, 133, as_cmap=True)
    plt.figure()
    sns_sctter = sns.scatterplot(
    x=x1, y=x2,
    hue=y_scaled,
    palette=cmap,
    legend=False
    )

    for idx in selected_indices:
        x1 = pca_result[idx, 0]
        x2 = pca_result[idx, 1]
        plt.text(x = x1, y = x2, s = "x", color = "blue", fontsize="large") # set colour of line

    output_dir = os.path.join(args.output_dir,'figs')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if phase != -1 :
        fig_path = os.path.join(output_dir,"embedding_plot_"+str(phase)+".png")
    else:
        fig_path = os.path.join(output_dir, "embedding_plot_final_test.png")

    fig = sns_sctter.get_figure()
    fig.savefig(fig_path)



def reliability_plot(args, label_map, train_examples, phase=0, bins=10, do_plot=False):
    """
    Generate a binned reliability plot.

    Parameters
    ----------
        bins (int): Number of bins to perform binned statistics in the adjusted score space.
        is_calibrated (bool): whether the score to compare is before or after calibration.
    """

    label_to_digit = {}
    for k, v in label_map.items():
        label_to_digit[v] = k

    results = []
    preds = []
    golds = []
    y_col = []


    # current_dir = os.getcwd() 
    # par_dir = os.path.dirname(current_dir) 
    output_dir_ori = args.output_dir #os.path.join(par_dir,output_dir)

    #output_dir_ori = output_dir.copy()

    if phase != -1:
        candidate_results_files = os.listdir(os.path.join(output_dir_ori,f'results_iteration_{phase}'))
    else:
        candidate_results_files = os.listdir(os.path.join(output_dir_ori,'results_final_test'))

    result_files = [f for f in candidate_results_files if f.endswith('.json')]

    if phase != -1:
        output_dir = os.path.join(output_dir_ori,f'results_iteration_{phase}')
    else:
        output_dir = os.path.join(output_dir_ori,'results_final_test')

    for file in result_files:
        with open(f"{output_dir}/{file}", 'r') as f:
            example_pred = json.load(f)
        idx = int(file[:-5])
        y_col.append(-example_pred[1])
        pred = label_to_digit[example_pred[0]]
        preds.append(pred)
        gold = train_examples[idx]["label"]
        golds.append(gold)
        if pred == gold: results.append(1)
        else: results.append(0)

    ymax = max(y_col)
    ymin = min(y_col)
    
    y_scaled = [ (yi - ymin) / (ymax - ymin) for yi in y_col ]

    ece_score = compute_ece(y_scaled, results)
    print("ECE error: ", ece_score)

    acc = sum(results) / len(results)
    print("Train acc: ", acc)

    if do_plot:
        scores_compare = np.array(y_scaled)
        scores_true = np.array(results)

        quantiles = np.linspace(0, 1, bins+1)
        bin_edges = np.quantile(scores_compare, quantiles)
        bin_assignment = np.digitize(scores_compare, bin_edges, right=True)
        # scores_compare_bin_means = [scores_compare[bin_assignment == i].mean() for i in range(1, len(bin_edges))]
        scores_compare_bin_means = bin_edges[:-1] + (bin_edges[1:] - bin_edges[:-1])/2
        scores_true_bin_means = [scores_true[bin_assignment == i].mean() for i in range(1, len(bin_edges))]

        plt.figure()
        #assert label in self.supported_metric_list
        s = sns.JointGrid(x=scores_compare, y=scores_true)
        sns.histplot(x=scores_compare, ax=s.ax_marg_x, color="limegreen", alpha=0.4, bins=60)
        sns.histplot(y=scores_true, ax=s.ax_marg_y, color="blueviolet", alpha=0.4, bins=60)
        
        scores_compare_bin_means = np.array(scores_compare_bin_means)
        scores_true_bin_means = np.array(scores_true_bin_means)

        ax = s.ax_joint
        ax.bar(scores_compare_bin_means, scores_true_bin_means, color='dodgerblue', alpha=0.6, width=0.05)
        ax.plot([min(scores_compare), max(scores_compare) ], [0, 1], 'deeppink', linestyle='--', linewidth=2, alpha=0.7)
        ax.grid(True)
        s.ax_marg_y.grid(False)

        ax.set_ylabel("Accuracy", fontsize=16)
        ax.set_xlabel("Confidence", fontsize=16)
        
        ax.set_ylim([0, 1])
        ax.set_xlim([0, 1])
        ax.tick_params(direction="in", labelsize=14)
        ax.set_yticklabels([])
        ax.grid(True)
        s.ax_marg_y.grid(False)

        output_dir = os.path.join(output_dir,'figs')
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        if phase != -1 :
            fig_path = os.path.join(output_dir,"reliability_plot_"+str(phase)+".png")
        else:
            fig_path = os.path.join(output_dir, "reliability_plot_final_test.png")
        fig = ax.get_figure()
        fig.set_size_inches(1.3,2)
        fig.savefig(fig_path)

    return ece_score, acc


def compute_ece(scores_compare, scores_true, bins=10):
    """
    Compute the expected calibration error (ECE).

    ECE is calculated by:
        ECE = \Sum^{M}_{m=1} |B_{m}|/n  *  |acc(B_{m}) - conf(B_{m})|
    where
        acc(B_{m}) = 1 / |B_{m}| * \Sum_{i \in B_{m}} \mathbf{1}(\hat{y}_i = y_i),
        conf(B_{m}) = 1 / |B_{m}| * \Sum_{i \in B_{m}} \hat{p}_i
    It partitions predictions into M equally-spaced bins (similar to the reliability diagrams) and
    taking a weighted average of the bins' accuracy/confidence difference.

    Reference - https://arxiv.org/pdf/1706.04599.pdf

    Parameters
    ----------
        bins (int): Number of bins to perform binned statistics in the adjusted score space.

    Returns:
        float: a score between 0 and 1
    """
    scores_compare = np.array(scores_compare)
    scores_true = np.array(scores_true)

    # Define quantiles for adaptive binning
    quantiles = np.linspace(0, 1, bins+1)
    bin_edges = np.quantile(scores_compare, quantiles)
    bin_assignment = np.digitize(scores_compare, bin_edges, right=True)
    
    ece = 0
    for bin_num in range(1, len(bin_edges)):
        compare_scores_bin = scores_compare[bin_assignment == bin_num]
        label_scores_bin = scores_true[bin_assignment == bin_num]

        ave_label_score_bin = label_scores_bin.mean()
        ave_compare_score_bin = compare_scores_bin.mean()

        num_samples = len(compare_scores_bin)
        ece += abs(ave_label_score_bin - ave_compare_score_bin) * num_samples / len(scores_compare)
    
    return ece