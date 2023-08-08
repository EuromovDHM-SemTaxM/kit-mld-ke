import collections
import json
import sys

from urllib.parse import urlencode
from anyio import Path

import numpy as np
import pandas
import torch
from torch import pdist, nn, mm

from tqdm import tqdm
import requests

model_name = "paraphrase-multilingual-mpnet-base-v2"
threshold = 0.2

from pyclinrec.dictionary import MgrepDictionaryLoader
from pyclinrec.recognizer import IntersStemConceptRecognizer

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


def parse_text(text):
    url = f"http://localhost:8080/predict/semantics?{urlencode({'utterance': text})}"
    response = get(url, {})
    return json.loads(response) if response is not None else ""


def is_compositional(propositions):
    return any(
        propositions[i]["start"] != propositions[i + 1]["start"]
        for i in range(len(propositions) - 1)
    )


def is_successive(propositions):
    overlapping_counter = sum(
        propositions[i]["start"] == propositions[i + 1]["start"]
        for i in range(len(propositions) - 1)
    )
    return overlapping_counter == 0


def create_part_annotator():
    dictionary_loader = MgrepDictionaryLoader("resources/body_part_dictionary.tsv")
    concept_recognizer = IntersStemConceptRecognizer(
        dictionary_loader,
        "resources/stopwords_en.txt",
        "resources/termination_terms_en.txt",
    )

    concept_recognizer.initialize()

    return concept_recognizer


def determine_composition_type(prop_annotations):
    if len(prop_annotations) == 1:
        return "single"
    elif is_compositional(prop_annotations):
        return "compositional"
    elif is_successive(prop_annotations):
        return "successive"


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


def extract_prop_text(tokens, prop_start, prop_end):
    prop_tokens = [token["text"] for token in tokens[prop_start : prop_end + 1]]
    return " ".join(prop_tokens)


def process_spans(spans):
    annotation = {}
    prop_start = sys.maxsize
    prop_end = -1
    for span in spans:
        prop_start = min(span["start"], prop_start)
        prop_end = max(span["end"], prop_end)
        for condition, value in span_conditions.items():
            if value[0](span):  # Check if span satisfies condition (call to lambda)
                if condition not in annotation:
                    annotation[condition] = []

                annotation[condition].append(
                    {
                        "text": span_conditions[condition][1](span),
                        "start": span["start"],
                        "end": span["end"],
                    }
                )

    return prop_start, prop_end, annotation


def process_corpus(corpus):
    annotator = create_part_annotator()

    output_document = []
    for index, sentence in tqdm(list(enumerate(corpus))):
        sentences = sentence if isinstance(sentence, list) else [sentence]
        index_output = []
        for sentence in sentences:
            response = parse_text(sentence.replace("<eos>", "."))
            if not response or len(response) == 0 or "props" not in response:
                output_document.append(
                    {"sentence": sentence, "sentence_index": index, "propositions": []}
                )
                continue

            tokens = response["tokens"]
            prop_annotations = [
                process_prop(annotator, tokens, prop)
                for prop in response["props"]
                if not excluded(prop)
            ]

            index_output.append(
                {
                    "sentence": sentence,
                    "sentence_index": index,
                    "propositions": prop_annotations,
                    "composition": determine_composition_type(prop_annotations),
                }
            )
        output_document.append(index_output)
    return output_document


def process_prop(annotator, tokens, prop):
    spans = prop["spans"]

    prop_start, prop_end, span_annotation = process_spans(spans)

    prop_text = extract_prop_text(tokens, prop_start, prop_end)

    _, _, annotations = annotator.match_mentions(prop_text)

    span_annotation["body_parts"] = set()
    for annotation in annotations:
        span_annotation["body_parts"].add(annotation.concept_id)

    span_annotation["body_parts"] = list(span_annotation["body_parts"])

    return {
        "start": prop_start,
        "end": prop_end,
        "text": prop_text,
        "sense": prop["sense"],
        "annotations": span_annotation,
    }


def excluded(prop):
    return (
        "performance" in prop["sense"]
        or "hold" in prop["sense"]
        or "seem" in prop["sense"]
    )


if __name__ == "__main__":
    input_file = sys.argv[1]
    stem = Path(input_file).stem
    with open(input_file, "r") as f:
        lines = f.readlines()
    refs = []
    preds = []

    for line in lines:
        rp = line.replace("\n", "").split(",")
        local_refs = rp[1:]
        pred = rp[0]
        refs.append(local_refs)
        preds.append(pred)

    preds_output = process_corpus(preds)
    with open(f"{stem}_preds.json", "w") as f:
        json.dump(preds_output, f, indent=2)

    refs_output = process_corpus(refs)
    with open(f"{stem}_refs.json", "w") as f:
        json.dump(refs_output, f, indent=2)
