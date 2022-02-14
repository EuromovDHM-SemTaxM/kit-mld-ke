import json

full_metadata = {}
with open("motion_data_full.json") as fp:
    full_metadata.update(json.load(fp))

cmu_taxonomy = full_metadata['cmu_description_taxonomy']
del full_metadata['cmu_description_taxonomy']

kit_taxonomy = full_metadata['kit_description_taxonomy']
del full_metadata['kit_description_taxonomy']

kit_corpus = []
cmu_corpus = []

corpus = ""
for key in full_metadata.keys():
    annotations = []
    with open(f"data/{str(key).zfill(5)}_annotations.json", "r") as fp_annot:
        annotations.extend(json.load(fp_annot))
    record = full_metadata[key]
    source = record['metadata']['source']['database']['identifier']
    additional_text = ""
    if source == "cmu":
        additional_text = record['description']
    elif source == "kit":
        additional_text += record['comment']
    for annotation in annotations:
        if annotation.endswith("."):
            annotation = annotation[:-1]
        corpus += annotation + ".\n"

    corpus += f"{additional_text}\n"


