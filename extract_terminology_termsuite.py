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
        # additional_text = record['description']
        pass
    elif source == "kit":
        # additional_text += record['comment']
        pass
    for annotation in annotations:
        if annotation.endswith("."):
            annotation = annotation[:-1]
        corpus += annotation + ".\n"

    corpus += f"{additional_text}\n"

with open("full_corpus_corrected", "w") as fc:
    fc.write(corpus)

import requests

# result = requests.post("http://localhost:8083/extract_terminology", json={'source_language': 'en', 'corpus': corpus,
#                                                                           'method': 'tbxtools'},
#                        headers={'Accept': 'application/json'}).json()


result = requests.post("http://localhost:8083/extract_terminology?language=en", data=corpus,
                      headers={'Accept': 'application/json', 'Content-Type': 'plain/text'}).json()

with open("kit_terminology_termsuite.json", "w") as fp:
    json.dump(result, fp, indent=4, sort_keys=True)


#

#
# result = requests.post("http://localhost:8083/extract_terminology?language=en", data=corpus,
#                        headers={'Accept': 'application/json', 'Content-Type': 'plain/text'}).json()

print(result)


# print("Post-processing")
#
# result = requests.post("http://localhost:8888/postprocess_terminology",
#                        json={'source_language': 'en', 'terms': result,
#                              'tasks': 'accents,plurals,numbers,patterns'},
#                        headers={'Accept': 'application/json'}).json()
# print(result)
