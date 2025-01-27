import collections
import json

from urllib.parse import urlencode

import numpy as np
import pandas
import torch
from torch import pdist, nn, mm

from tqdm import tqdm

from sentence_transformers import SentenceTransformer, util

model_name = 'paraphrase-multilingual-mpnet-base-v2'
threshold = 0.2

from redis import Redis

redis = Redis(decode_responses=True, port=6380)


def get(url: str, headers, timeout= None):
    page_text = redis.get(url)
    try:
        if not page_text:
            result = requests.get(url, headers=headers, timeout=timeout)
            if result.status_code < 400:
                page_text = result.text
                if page_text is not None:
                    redis.set(url, page_text)
            else:
                print(result, result.text)
                return None
    except requests.exceptions.ReadTimeout:
        page_text = None
    except requests.exceptions.MissingSchema:
        page_text = None
    except requests.exceptions.ConnectTimeout:
        page_text = None
    return page_text


def cluster_similar_expressions(ordered_dict, model, threshold=0.4):
    terms = list(ordered_dict.keys())
    counts = list(ordered_dict.values())
    # Embedding
    vectors = torch.Tensor(model.encode(terms))

    # Computing cosine distance matrix
    vectors = nn.functional.normalize(vectors, p=2, dim=-1)
    dist = 1 - mm(vectors, vectors.t())

    # Thresholding and keeping only upper triangle
    mask = dist < threshold
    dist = torch.triu(dist * mask.int().float())

    # Clustering
    already_in_cluster = set()
    merged_ordered_dict = {}
    for i in range(dist.shape[0]):
        if i not in already_in_cluster:
            indices = np.nonzero(dist[i])
            new_key = ", ".join([terms[index] for index in indices])
            total = 0
            for index in indices:
                index = index.item()
                total += counts[index]
                already_in_cluster.add(index)
            merged_ordered_dict[new_key] = total
    return merged_ordered_dict


sentences_df = pandas.read_csv(
    "extracted_knowlege/sentences_corrections.tsv",
    sep='\t', header=None)

corpus = sentences_df.iloc[:, 1].tolist()

import requests

matched_conditions = {}
conditions = {
    'verbs': (lambda x: x['vn'] == 'Verb', lambda x: x['text']),
    'manners': (lambda x: x['pb'] == 'AM-MNR', lambda x: x['text']),
    'directions': (
        lambda x: x['pb'] == 'AM-DIR' or 'direction' in x['description'] or 'path' in x['description'],
        lambda x: x['text'])
}
limit = 20
for sentence in tqdm(corpus):
    # print("Sentence: "+ sentence)
    url = f"http://localhost:8080/predict/semantics?{urlencode({'utterance': sentence})}"
    response = get(url, {})
    response = json.loads(response)

    if 'props' in response:
        for prop in response['props']:
            spans = prop['spans']
            for span in spans:
                for condition, value in conditions.items():
                    if value[0](span):
                        if condition not in matched_conditions:
                            matched_conditions[condition] = []
                        matched_conditions[condition].append((conditions[condition][1](span),prop['sense']))
    # limit -= 1
    # if limit == 0:
    #     break
model = SentenceTransformer(model_name)

for condition, cond_list in matched_conditions.items():
    print(f"****{condition}****")
    cond_histogram = dict(collections.Counter(cond_list))
    words_col = []
    counts_col = []
    for entry in cond_histogram.items():
        words_col.append(entry[0])
        counts_col.append(entry[1])
    clustered_histogram = cluster_similar_expressions(cond_histogram, model, threshold)
    clustered_histogram = sorted(clustered_histogram.items(), reverse=True, key=lambda t: t[1])
    print(clustered_histogram)
    with open(f"semantic_clusters_{condition}_{model_name}.tsv", "w") as f:
        f.write(f"terms\tcount\n")
        for item in clustered_histogram:
            f.write(f"{item[0]}\t{item[1]}\t{item[2]}\n")
