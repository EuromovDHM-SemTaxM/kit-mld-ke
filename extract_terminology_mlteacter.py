import json
# torch and tranformers for model and training
import torch
from transformers import XLMRobertaTokenizer, AutoTokenizer
from transformers import XLMRobertaForSequenceClassification

max_len = 512


def extract_terms(validation_df, xlmr_model, xlmr_tokenizer, device="cpu"):
    print(len(validation_df))
    term_list = []

    # put model in evaluation mode
    xlmr_model.eval()

    for index, row in validation_df.iterrows():
        sentence = row['n_gram'] + ". " + row["Context"]
        label = validation_df["Label"]

        encoded_dict = xlmr_tokenizer.encode_plus(sentence,
                                                  max_length=max_len,
                                                  padding='max_length',
                                                  truncation=True,
                                                  return_tensors='pt')
        input_id = encoded_dict['input_ids'].to(device)
        attn_mask = encoded_dict['attention_mask'].to(device)
        label = torch.tensor(0).to(device)

        with torch.no_grad():
            output = xlmr_model(input_id,
                                token_type_ids=None,
                                attention_mask=attn_mask,
                                labels=label)
            loss = output.loss
            logits = output.logits

        logits = logits.detach().cpu().numpy()
        pred = labels[logits[0].argmax(axis=0)]
        if pred == "Term":
            term_list.append(row['n_gram'])

    return set(term_list)


full_metadata = {}
with open("motion_data_full.json") as fp:
    full_metadata.update(json.load(fp))

cmu_taxonomy = full_metadata['cmu_description_taxonomy']
del full_metadata['cmu_description_taxonomy']

kit_taxonomy = full_metadata['kit_description_taxonomy']
del full_metadata['kit_description_taxonomy']

corpus = []

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
        corpus.append(annotation)

    # corpus += f"{additional_text}\n"

xlmr_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

labels_ = []
input_ids_ = []
attn_masks_ = []

# for each datasample:
for sentence in corpus:
    # create requiered input, i.e. ids and attention masks
    encoded_dict = xlmr_tokenizer.encode_plus(sentence,
                                              max_length=512,
                                              padding='max_length',
                                              truncation=True,
                                              return_tensors='pt')

    # add encoded sample to lists
    input_ids_.append(encoded_dict['input_ids'])
    attn_masks_.append(encoded_dict['attention_mask'])
    labels_.append(row['Label'])

# Convert each Python list of Tensors into a 2D Tensor matrix.
input_ids_ = torch.cat(input_ids_, dim=0)
attn_masks_ = torch.cat(attn_masks_, dim=0)

# labels to tensor
labels_ = torch.tensor(labels_)

checkpoint = torch.load('checkpoints/checkpoint_3.pth.tar')
xlmr_model = XLMRobertaForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=2)
xlmr_model.load_state_dict(checkpoint['model_state_dict'])

extracted_terms = extract_terms(train_data_lombalgie, xlmr_model, xlmr_tokenizer)
