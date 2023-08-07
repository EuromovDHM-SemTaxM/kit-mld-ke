import collections
import json

from urllib.parse import urlencode

import numpy as np
import pandas
import torch
from torch import pdist, nn, mm

from tqdm import tqdm


model_name = "paraphrase-multilingual-mpnet-base-v2"
threshold = 0.2

from redis import Redis

redis = Redis(decode_responses=True, port=6379)


def get(url: str, headers, timeout=None):
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


# sentences_df = pandas.read_csv(
#     "extracted_knowlege/Gauss_303_.csv", sep=",", header=None
# )

# corpus = sentences_df.iloc[:, 0].tolist()

with open("extracted_knowlege/h3D_preds_refs.csv", "r") as f:
    lines = f.readlines()
refs = []
preds = []

for id, line in enumerate(lines):
    rp = line.replace("\n", "").split(",")
    local_refs = rp[1:]
    pred = rp[0]
    for i in range(len(local_refs)):
        refs.append(local_refs[i])
        preds.append(pred)

corpus = preds
import requests

matched_conditions = {}
span_conditions = {
    "agents": (lambda x: x["vn"] == "Agent", lambda x: x["text"]),
    "actions": (lambda x: x["vn"] == "Verb", lambda x: x["text"]),
    "manners": (lambda x: x["pb"] == "AM-MNR", lambda x: x["text"]),
    "directions": (
        lambda x: x["pb"] == "AM-DIR"
        or ("direction" in x["description"] and "AM-LVB" not in x["pb"])
        or "path" in x["description"],
        lambda x: x["text"],
    ),
}
limit = 20
output_document = []
for index, sentence in tqdm(list(enumerate(corpus))):
    sentence = sentence.replace("<eos>", ".")
    # print("Sentence: "+ sentence)
    url = (
        f"http://localhost:8080/predict/semantics?{urlencode({'utterance': sentence})}"
    )
    response = get(url, {})
    if response is not None:
        response = json.loads(response)
        annotation = {}
        senses = []
        if "props" in response:
            for prop in response["props"]:
                if "performance" not in prop["sense"]:
                    spans = prop["spans"]
                    for span in spans:
                        for condition, value in span_conditions.items():
                            if value[0](span):
                                if condition not in annotation:
                                    annotation[condition] = []

                                annotation[condition].append(
                                    (
                                        span_conditions[condition][1](span),
                                        span["start"],
                                        span["end"],
                                    )
                                )
                    senses.append(prop["sense"])
            output_document.append(
                {
                    "sentence": sentence,
                    "sentence_index": index,
                    "senses": prop["sense"],
                    "annotations": annotation,
                }
            )

with open("extracted_knowlege/semantic_segmentation_jit.json", "w") as f:
    json.dump(output_document, f, indent=2)
    # limit -= 1
    # if limit == 0:
    #     break
# model = SentenceTransformer(model_name)

# for condition, cond_list in matched_conditions.items():
#     print(f"****{condition}****")
#     cond_histogram = dict(collections.Counter(cond_list))
#     words_col = []
#     counts_col = []
#     for entry in cond_histogram.items():
#         words_col.append(entry[0])
#         counts_col.append(entry[1])
#     clustered_histogram = cluster_similar_expressions(cond_histogram, model, threshold)
#     clustered_histogram = sorted(
#         clustered_histogram.items(), reverse=True, key=lambda t: t[1]
#     )
#     print(clustered_histogram)
#     with open(f"semantic_clusters_{condition}_{model_name}.tsv", "w") as f:
#         f.write(f"terms\tcount\n")
#         for item in clustered_histogram:
#             f.write(f"{item[0]}\t{item[1]}\n")
