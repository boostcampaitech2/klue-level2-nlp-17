import pickle as pickle
import os
import pandas as pd
import torch
import configparser


class RE_Dataset(torch.utils.data.Dataset):
    """Dataset 구성을 위한 class."""

    def __init__(self, pair_dataset, labels):
        self.pair_dataset = pair_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: val[idx].clone().detach() for key, val in self.pair_dataset.items()
        }
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def preprocessing_dataset(dataset):
    """처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
    out_dataset = pd.DataFrame(
        {
            "id": dataset["id"],
            "sentence": dataset["sentence"],
            "subject_entity": dataset["subject_entity"],
            "object_entity": dataset["object_entity"],
            "label": dataset["label"],
        }
    )

    return out_dataset


def load_data(dataset_dir):
    """csv 파일을 경로에 맡게 불러 옵니다."""
    pd_dataset = pd.read_csv(dataset_dir)
    dataset = preprocessing_dataset(pd_dataset)

    dataset.drop_duplicates(
    subset=['sentence','subject_entity','object_entity', "label"],
    inplace= True
    )

    return dataset

def tokenized_dataset(dataset, tokenizer):
    """tokenizer에 따라 sentence를 tokenizing 합니다."""
    concat_entity = []
    for e01, e02 in zip(dataset["subject_entity"], dataset["object_entity"]):
        temp = ""
        temp = e01 + "[SEP]" + e02
        concat_entity.append(temp)
        
    # read Config file
    config = configparser.ConfigParser()
    config.read("config.ini")
    cf = config["project_config"]
    Tokenizer_NAME = cf["model_name"]

    if "roberta" in Tokenizer_NAME:
        tokenized_sentences = tokenizer(
            concat_entity,
            list(dataset["sentence"]),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
            add_special_tokens=True,
            return_token_type_ids=False,  # roberta 일 때, 추가
        )
    else:
        tokenized_sentences = tokenizer(
            concat_entity,
            list(dataset["sentence"]),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
            add_special_tokens=True,
        )
    return tokenized_sentences
