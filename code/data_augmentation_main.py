import argparse
import json
from importlib import import_module
import pandas as pd
import os
from tqdm import tqdm
import torch
import random
from pororo import Pororo
from koeda import AEDA
from koeda import RS
from koeda import RI


def round_trip_translation(args, dataset):
    language_list = args["language_list"].split()
    not_augmentate_label_list = args["not_augmentate_label_list"].split()
    mt = Pororo(task="translation", lang="multi")
    for language in language_list:
        for index, data in tqdm(dataset.iterrows(), total=dataset.shape[0]):
            if data["label"] in not_augmentate_label_list:
                continue
            sentence = mt(data["sentence"], src="ko", tgt=language)
            sentence = mt(sentence, src=language, tgt="ko")
            sentence = sentence
            new_row = {
                "id": dataset.iloc[-1]["id"] + 1,
                "sentence": sentence,
                "subject_entity": data["subject_entity"],
                "object_entity": data["object_entity"],
                "label": data["label"],
            }
            dataset = dataset.append(new_row, ignore_index=True)

    save_file = os.path.join(
        args["train_dir"], "translate_" + "_".join(language_list) + "_train.csv"
    )
    dataset.to_csv(save_file)
    print("round_trip_translation Done")
    return dataset


def change_entity(args, dataset):
    change_entity_only = [
        "per:alternate_names",
        "per:spouse",
        "per:colleagues",
        "per:other_family",
        "per:siblings",
    ]
    added_index = len(dataset)
    for idx, row in tqdm(dataset.iterrows()):
        if row["label"] in change_entity_only:
            dataset.loc[added_index] = [
                added_index,
                row["sentence"],
                row["object_entity"],
                row["subject_entity"],
                row["label"],
            ]
        elif row["label"] == "org:member_of":
            dataset.loc[added_index] = [
                added_index,
                row["sentence"],
                row["object_entity"],
                row["subject_entity"],
                "org:members",
            ]
        elif row["label"] == "per:parents":
            dataset.loc[added_index] = [
                added_index,
                row["sentence"],
                row["object_entity"],
                row["subject_entity"],
                "per:children",
            ]
        elif row["label"] == "per:children":
            dataset.loc[added_index] = [
                added_index,
                row["sentence"],
                row["object_entity"],
                row["subject_entity"],
                "per:parents",
            ]
        added_index += 1

    save_file = os.path.join(args["train_dir"], "change_entity_train.csv")
    dataset.to_csv(save_file)
    print("change_entity Done")
    return dataset


def aeda(args, dataset):
    SPACE_TOKEN = "\u241F"

    def replace_space(text: str) -> str:
        return text.replace(" ", SPACE_TOKEN)

    def revert_space(text: list) -> str:
        clean = " ".join("".join(text).replace(SPACE_TOKEN, " ").split()).strip()
        return clean

    class AEDA_No_Space(AEDA):
        def _aeda(self, data: str, p: float) -> str:
            if p is None:
                p = self.ratio

            split_words = self.morpheme_analyzer.morphs(replace_space(data))
            words = self.morpheme_analyzer.morphs(data)

            new_words = []
            q = random.randint(1, int(p * len(words) + 1))
            qs_list = [
                index
                for index in range(len(split_words))
                if split_words[index] != SPACE_TOKEN
            ]
            qs = random.sample(qs_list, q)

            for j, word in enumerate(split_words):
                if j in qs:
                    new_words.append(SPACE_TOKEN)
                    new_words.append(
                        self.punctuations[random.randint(0, len(self.punctuations) - 1)]
                    )
                    new_words.append(SPACE_TOKEN)
                    new_words.append(word)
                else:
                    new_words.append(word)

            augmented_sentences = revert_space(new_words)

            return augmented_sentences

    aeda_no_space = AEDA_No_Space(
        morpheme_analyzer="Okt",
        punc_ratio=0.3,
        punctuations=[".", ",", "!", "?", ";", ":"],
    )

    added_index = dataset.id.to_numpy()[-1] + 1
    label_order = dataset["label"].value_counts()
    label_order_top = label_order.index[:9].tolist()

    for idx, row in tqdm(dataset.iterrows()):
        if row.label in label_order_top:
            continue
        sentence_aeda = aeda_no_space(row["sentence"])
        dataset.loc[added_index] = [
            added_index,
            sentence_aeda,
            row["subject_entity"],
            row["object_entity"],
            row["label"],
        ]
        added_index += 1

    save_file = os.path.join(args["train_dir"], "aeda_train.csv")
    dataset.to_csv(save_file)
    print("AEDA Done")
    return dataset


def eda_rs(train_dataset, n, criteria, p):
    """
        Param
        - dataset : eda-rs 대상
        - n : 증강 개수
        - criteria : 데이터 분포의 기준
        - p : 문장에 대하여 변화시킬 비율
        Usage
            dataset에 대하여, criteria 비율 이하에 대하여 RS 증강.
    """
    eda_lbl = (
        train_dataset["label"]
        .value_counts(normalize=True)
        .loc[lambda x: x < criteria]
        .index.tolist()
    )  # criteria 이하만 증강

    eda_trg_data = train_dataset[
        train_dataset["label"].isin(eda_lbl)
    ].copy()  # 증강시킬 label만 있는 dataframe 복제

    eda_data = pd.DataFrame(columns=train_dataset.keys())  # 빈 dataframe

    # sub -> @ , obj -> # 변경
    aug_sentence = [
        eda_trg_data["sentence"]
        .iloc[i]
        .replace(eda_trg_data["subject_entity"].iloc[i], "@")
        .replace(eda_trg_data["object_entity"].iloc[i], "#")
        for i in range(len(eda_trg_data))
    ]

    ## RS
    augmenter = RS(morpheme_analyzer="Mecab")
    result = augmenter(aug_sentence, p=p, repetition=n)

    for i in range(n):
        tmp = eda_trg_data.copy()
        tmp["sentence"] = result[i::n]
        tmp["sentence"] = [
            tmp["sentence"]
            .iloc[i]
            .replace("@", tmp["subject_entity"].iloc[i])
            .replace("#", tmp["object_entity"].iloc[i])
            for i in range(len(tmp))
        ]
        eda_data = eda_data.append(tmp)

    print("eda_rs Done")
    return eda_data


def eda_ri(train_dataset, n, criteria, p):
    """
        Param
        - dataset : eda-rs 대상
        - n : 증강 개수
        - criteria : 데이터 분포의 기준
        - p : 문장에 대하여 변화시킬 비율
        Usage
            dataset에 대하여, criteria 비율 이하에 대하여 RI 증강.
    """
    eda_lbl = (
        train_dataset["label"]
        .value_counts(normalize=True)
        .loc[lambda x: x < criteria]
        .index.tolist()
    )  # criteria 이하만 증강

    eda_trg_data = train_dataset[
        train_dataset["label"].isin(eda_lbl)
    ].copy()  # 증강시킬 label만 있는 dataframe 복제

    eda_data = pd.DataFrame(columns=train_dataset.keys())  # 빈 dataframe

    # sub -> @ , obj -> # 변경
    aug_sentence = [
        eda_trg_data["sentence"]
        .iloc[i]
        .replace(eda_trg_data["subject_entity"].iloc[i], "@")
        .replace(eda_trg_data["object_entity"].iloc[i], "#")
        for i in range(len(eda_trg_data))
    ]

    ## RI
    augmenter = RI(morpheme_analyzer="Mecab", stopword=False)
    result = augmenter(aug_sentence, p=p, repetition=n)

    for i in range(n):
        tmp = eda_trg_data.copy()
        tmp["sentence"] = result[i::n]
        tmp["sentence"] = [
            tmp["sentence"]
            .iloc[i]
            .replace("@", tmp["subject_entity"].iloc[i])
            .replace("#", tmp["object_entity"].iloc[i])
            for i in range(len(tmp))
        ]
        eda_data = eda_data.append(tmp)

    print("eda_ri Done")
    return eda_data


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args = argparse.ArgumentParser(description="how to use argparser")
    args.add_argument(
        "-c",
        "--config",
        default="./config.json",
        type=str,
        help="config.json file path",
    )
    args = args.parse_args()

    args = json.load(open(args.config, "rt"))

    dataset = getattr(import_module("load_data"), args["load_data"])(args["train_data"])
    print(dataset.head(5))

    augmentation_method_list = args["augmentation_method_list"].split()

    for augmentation_method in augmentation_method_list:
        dataset = getattr(import_module("data_augmentation_main"), augmentation_method)(
            args, dataset
        )

    save_file = os.path.join(
        args["train_dir"], "_".join(augmentation_method_list) + "_train.csv"
    )
    dataset.to_csv(save_file)

