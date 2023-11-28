"""
This is the script for partitioning the SuperGLUE dataset into client splits using Dual-Dirichlet partition strategy and store them into separate directories as required by the dataloader into `superglue.py`.
Run it by providing your desired (relative/absolute) data root path, SuperGLUE dataset name, total number of clients, and maximum number of training elements for each client.
"""
import os
import json
import shutil
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset

"""
python superglue_partition.py --dataset boolq
python superglue_partition.py --dataset cb
python superglue_partition.py --dataset copa
python superglue_partition.py --dataset multirc
python superglue_partition.py --dataset rte
python superglue_partition.py --dataset wic
python superglue_partition.py --dataset wsc
"""

def superglue2alpaca(sample, dataset):
    def highlight_span(text, span, length, highlighter="**"):
        """Highlight a span of text, where span is the index of the word to highlight, and length if the number of words to highlight."""
        words = text.split()
        words[span] = f"{highlighter}{words[span]}"
        words[span + length - 1] = f"{words[span + length - 1]}{highlighter}"
        return " ".join(words)

    def count_words(sentence):
        """return the number of words in a sentence"""
        return len(sentence.split())
    if dataset == 'boolq':
        return {
            "instruction": f"The following reading comprehension question requires you to understand the following passage and answer a question related to the passage. Please answer with only ``True'' or ``False'' to the question: {sample['question']}?", 
            "input": sample['passage'],
            "output": "True" if sample['label'] else "False"
        }
    elif dataset == 'cb':
        return {
            "instruction": f"Please determine whether the hypothesis ``{sample['hypothesis']}'' entails, contradicts, or is unrelated to the following premise: ``{sample['premise']}''. Please respond with either ``Entailment'', ``Contradiction'', or ``Neutral''.",
            "output": ["Entailment", "Contradiction", "Neutral"][sample['label']]
        }
    elif dataset == 'copa':
        return {
            "instruction": f"Given the following premise, please determine whether Choice One, ``{sample['choice1']}'', or Choice Two, ``{sample['choice2']}'', is the {sample['question']} of the premise. Please respond with either ``One'' or ``Two''.",
            "input": sample["premise"],
            "output": ["One", "Two"][sample['label']]
        }
    elif dataset == 'multirc':
        return {
            "instruction": f"Given the following paragraph, please determine whether ``{sample['answer']}'' is a correct answer to the question ``{sample['question']}''. Please respond with either ``Yes'' or ``No''.",
            "input": sample["paragraph"],
            "output": ["No", "Yes"][sample['label']]
        }
    elif dataset == 'rte':
        return {
            "instruction": f"Please determine whether the sentence ``{sample['premise']}'' entails the hypothesis ``{sample['hypothesis']}'' or not. Please respond with either ``Yes'' or ``No''.",
            "output": ["Yes", "No"][sample['label']]
        }
    elif dataset == 'wic':
        return {
            "instruction": f"Please determine whether the word ``{sample['word']}'' has the same meaning in the following two sentences: ``{sample['sentence1']}'' and ``{sample['sentence2']}'' Please respond with either ``Yes'' or ``No''.",
            "output": ["No", "Yes"][sample['label']]
        }
    elif dataset == 'wsc':
        return {
            "instruction": f"Please determine whether the highlighted pronoun ``{sample['span2_text']}'' refers to the highlighted ``{sample['span1_text']}'' in the following sentence. Please respond with either ``Yes'' or ``No''.",
            "input": f"``{highlight_span(highlight_span(sample['text'], sample['span1_index'], count_words(sample['span1_text'])), sample['span2_index'], count_words(sample['span2_text']))}''",
            "output": ["No", "Yes"][sample['label']]
        }
    else:
        raise NotImplementedError

def plot_distribution(K, classes, res, filename):
    """Visualize the data distribution for different classes"""
    _, ax = plt.subplots(figsize=(20, K/2+3))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    colors = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', 
            '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5', 
            '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f', 
            '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']

    ax.barh(range(K), res[0], color=colors[0])
    for i in range(1, len(classes)):
        ax.barh(range(K), res[i], left=np.sum(res[:i], axis=0), color=colors[i])

    ax.set_ylabel("Client")
    ax.set_xlabel("Number of Elements")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(filename)

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--dataset', type=str, default='cb', choices=['boolq', 'cb', 'copa', 'multirc', 'record', 'rte', 'wic', 'wsc'])
parser.add_argument('--num_clients', type=int, default=4)
parser.add_argument('--data_root_path', type=str, default='/projects/bbvf/zl52/globus-compute-endpoint/superglue_partitioned_data', help="PLEASE replace this to your own root data path.")
parser.add_argument('--max_num_elements', type=int, default=10000, help="Maximum number of training elements per client, use -1 if no maximum requirements.")

args = parser.parse_args()

# Clean up the root path for the specific dataset if there already exists one
root_path = f'{args.data_root_path}/{args.dataset}'
if os.path.exists(root_path):
    shutil.rmtree(root_path)
os.makedirs(root_path)

dataset = load_dataset("super_glue", args.dataset)
train_set = dataset['train']
val_set = dataset['validation']

# Store the same validation dataset for all clients
val_set_json = [superglue2alpaca(sample, args.dataset) for sample in val_set]
for client_idx in range(args.num_clients):
    os.makedirs(f'{root_path}/{client_idx}')
    with open(f'{root_path}/{client_idx}/val.json', 'w') as f:
        json.dump(val_set_json, f, indent=4)

# Partition and store the training set
np.random.seed(args.seed)
num_clients = args.num_clients
labels = []
label_indices = {}
for i in range(len(train_set)):
    label = train_set[i]['label']
    if label not in label_indices:
        label_indices[label] = []
        labels.append(label)
    label_indices[label].append(i)
labels.sort()
for label in label_indices:
    np.random.shuffle(label_indices[label])

alpha1 = args.num_clients * 2
alpha2 = 2
p1 = [1 / args.num_clients for _ in range(args.num_clients)]
p2 = [len(label_indices[label]) for label in labels]
p2 = [p / sum(p2) for p in p2]
q1 = [alpha1 * i for i in p1]
q2 = [alpha2 * i for i in p2]
weights = np.random.dirichlet(q1) 
individuals = np.random.dirichlet(q2, args.num_clients)

classes = [len(label_indices[label]) for label in labels]

normalized_portions = np.zeros(individuals.shape)
for i in range(num_clients):
    for j in range(len(classes)):
        normalized_portions[i][j] = weights[i] * individuals[i][j] / np.dot(weights, individuals.transpose()[j])

res = np.multiply(np.array([classes] * num_clients), normalized_portions).transpose()

for i in range(len(classes)):
    total = 0
    for j in range(num_clients - 1):
        res[i][j] = int(res[i][j])
        total += res[i][j]
    res[i][num_clients - 1] = classes[i] - total

num_elements = np.array(res.transpose(), dtype=np.int32)
sum_elements = np.cumsum(num_elements, axis=0)

for i in range(num_clients):
    train_set_json = []
    train_set_label = []
    for j, label in enumerate(labels):
        start = 0 if i == 0 else sum_elements[i-1][j]
        end = sum_elements[i][j]
        for idx in label_indices[label][start:end]:
            train_set_json.append(superglue2alpaca(train_set[idx], args.dataset))
            train_set_label.append(train_set[idx]['label'])
    # Chunk the client training set if necessary
    if len(train_set_json) > args.max_num_elements:
        indices = np.arange(len(train_set_json))
        np.random.shuffle(indices)
        train_set_json_chunk = []
        train_set_label_counter = {}
        for idx in indices[:args.max_num_elements]:
            train_set_json_chunk.append(train_set_json[idx])
            if train_set_label[idx] not in train_set_label_counter:
                train_set_label_counter[train_set_label[idx]] = 1
            else:
                train_set_label_counter[train_set_label[idx]] += 1
        for label in train_set_label_counter:
            res[label][i] = train_set_label_counter[label]
        train_set_json = train_set_json_chunk
    np.random.shuffle(train_set_json)
    with open(f'{root_path}/{i}/train.json', 'w') as f:
        json.dump(train_set_json, f, indent=4)
        
plot_distribution(num_clients, classes, res, f'{root_path}/distribution.pdf')
