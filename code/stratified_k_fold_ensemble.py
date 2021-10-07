import pickle as pickle
import os
import pandas as pd
from scipy.sparse import data
import torch
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    RobertaConfig,
    RobertaTokenizer,
    RobertaForSequenceClassification,
    BertTokenizer,
)
from load_data import *
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import configparser
import wandb
import datetime
from dateutil.tz import gettz
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pickle as pickle
import argparse
from tqdm import tqdm


def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = [
        "no_relation",
        "org:top_members/employees",
        "org:members",
        "org:product",
        "per:title",
        "org:alternate_names",
        "per:employee_of",
        "org:place_of_headquarters",
        "per:product",
        "org:number_of_employees/members",
        "per:children",
        "per:place_of_residence",
        "per:alternate_names",
        "per:other_family",
        "per:colleagues",
        "per:origin",
        "per:siblings",
        "per:spouse",
        "org:founded",
        "org:political/religious_affiliation",
        "org:member_of",
        "per:parents",
        "org:dissolved",
        "per:schools_attended",
        "per:date_of_death",
        "per:date_of_birth",
        "per:place_of_birth",
        "per:place_of_death",
        "org:founded_by",
        "per:religion",
    ]
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return (
        sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices)
        * 100.0
    )


def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(30)[labels]

    score = np.zeros((30,))
    for c in range(30):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(
            targets_c, preds_c
        )
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0


def compute_metrics(pred):
    """validation을 위한 metrics function"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = pred.predictions

    # calculate accuracy using sklearn's function
    f1 = klue_re_micro_f1(preds, labels)
    auprc = klue_re_auprc(probs, labels)
    acc = accuracy_score(labels, preds)  # 리더보드 평가에는 포함되지 않습니다.

    return {
        "micro f1 score": f1,
        "auprc": auprc,
        "accuracy": acc,
    }


def label_to_num(label):
    num_label = []
    with open("dict_label_to_num.pkl", "rb") as f:
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])

    return num_label


def inference(model, tokenized_sent, device):
    """
    test dataset을 DataLoader로 만들어 준 후,
    batch_size로 나눠 model이 예측 합니다.
    """
    dataloader = DataLoader(tokenized_sent, batch_size=16, shuffle=False)
    model.eval()
    output_pred = []
    output_prob = []
    for i, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            outputs = model(
                input_ids=data["input_ids"].to(device),
                attention_mask=data["attention_mask"].to(device),
                # token_type_ids=data['token_type_ids'].to(device)  # roberta-base 일 때 None
            )
        logits = outputs[0]
        prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        result = np.argmax(logits, axis=-1)

        output_pred.append(result)
        output_prob.append(prob)

    return (
        np.concatenate(output_pred).tolist(),
        np.concatenate(output_prob, axis=0).tolist(),
    )


def num_to_label(label):
    """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
    """
    origin_label = []
    with open("dict_num_to_label.pkl", "rb") as f:
        dict_num_to_label = pickle.load(f)
    for v in label:
        origin_label.append(dict_num_to_label[v])

    return origin_label


def load_test_dataset(dataset_dir, tokenizer):
    """
    test dataset을 불러온 후,
    tokenizing 합니다.
    """
    test_dataset = load_data(dataset_dir)
    test_label = list(map(int, test_dataset["label"].values))
    # tokenizing dataset
    tokenized_test = tokenized_dataset(test_dataset, tokenizer)
    return test_dataset["id"], tokenized_test, test_label


def main_inference(args):
    # read Config file
    config = configparser.ConfigParser()
    config.read("config.ini")

    cf = config["project_config"]

    """
    주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # load tokenizer
    # Tokenizer_NAME = "klue/bert-base"
    Tokenizer_NAME = cf["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)

    ## load my model
    MODEL_NAME = args.model_dir  # model dir.
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.parameters
    model.to(device)

    ## load test datset
    test_dataset_dir = "/opt/ml/klue-level2-nlp-17/dataset/test/test_data.csv"
    test_id, test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
    Re_test_dataset = RE_Dataset(test_dataset, test_label)

    ## predict answer
    pred_answer, output_prob = inference(
        model, Re_test_dataset, device
    )  # model에서 class 추론
    pred_answer = num_to_label(pred_answer)  # 숫자로 된 class를 원래 문자열 라벨로 변환.

    ## make csv file with predicted answer
    #########################################################
    # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
    output = pd.DataFrame(
        {
            "id": test_id,
            "pred_label": pred_answer,
            "probs": output_prob,
        }
    )
    return output


def train(args):
    # read Config file
    config = configparser.ConfigParser()
    config.read("config.ini")

    cf = config["project_config"]
    wcf = config["wandb_config"]

    # wandb
    if wcf.getboolean("wandb"):
        wandb.init(project=wcf["project_name"])
        run_name_params = [
            "epoch",
            "learning_rate",
            "train_batch_size",
            "eval_batch_size",
            "model_name",
        ]
        # 21-10-01 00:00 # Parameters...
        wandb.run.name = (
            datetime.datetime.now(gettz("Asia/Seoul")).strftime("%y-%m-%d %H:%M")
            + " | "
            + " | ".join([cf[s] for s in run_name_params])
        )
        wandb.run.save()

    # load model and tokenizer
    # MODEL_NAME = "klue/bert-base"
    MODEL_NAME = cf["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # load dataset
    dataset = load_data("/opt/ml/klue-level2-nlp-17/dataset/train/train_translate_entities_aeda_augmented.csv")
    label = np.array(label_to_num(dataset["label"].values))

    # train_dataset, dev_dataset, train_label, dev_label = train_test_split(
    #     dataset,
    #     label,
    #     test_size=0.2,
    #     random_state=42,
    #     shuffle=True,
    #     stratify=label,
    # )
    
    n_output = []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, dev_index in skf.split(dataset, label):
        train_dataset, dev_dataset = dataset.iloc[train_index, :], dataset.iloc[dev_index, :]
        train_label, dev_label = label[train_index].tolist(), label[dev_index].tolist()

        # dev_dataset = load_data("../dataset/train/dev.csv") # validation용 데이터는 따로 만드셔야 합니다.
        # dev_label = label_to_num(dev_dataset['label'].values)

        # tokenizing dataset
        tokenized_train = tokenized_dataset(train_dataset, tokenizer)
        tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

        # make dataset for pytorch.
        RE_train_dataset = RE_Dataset(tokenized_train, train_label)
        RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print(device)
        # setting model hyperparameter
        model_config = AutoConfig.from_pretrained(MODEL_NAME)
        model_config.num_labels = 30

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, config=model_config
        )
        print(model.config)
        model.parameters
        model.to(device)

        # 사용한 option 외에도 다양한 option들이 있습니다.
        # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
        training_args = TrainingArguments(
            output_dir="./results",  # output directory
            save_total_limit=5,  # number of total save model.
            save_steps=int(cf["save_steps"]),  # model saving step.
            num_train_epochs=int(cf["epoch"]),  # total number of training epochs
            learning_rate=float(cf["learning_rate"]),  # learning_rate
            per_device_train_batch_size=int(
                cf["train_batch_size"]
            ),  # batch size per device during training
            per_device_eval_batch_size=int(
                cf["eval_batch_size"]
            ),  # batch size for evaluation
            warmup_steps=500,  # number of warmup steps for learning rate scheduler
            weight_decay=0.01,  # strength of weight decay
            logging_dir="./logs",  # directory for storing logs
            logging_steps=100,  # log saving step.
            evaluation_strategy="steps",  # evaluation strategy to adopt during training
            # `no`: No evaluation during training.
            # `steps`: Evaluate every `eval_steps`.
            # `epoch`: Evaluate every end of epoch.
            eval_steps=int(cf["eval_steps"]),  # evaluation step.
            load_best_model_at_end=True,
        )
        trainer = Trainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=RE_train_dataset,  # training dataset
            eval_dataset=RE_dev_dataset,  # evaluation dataset
            compute_metrics=compute_metrics,  # define metrics function
        )

        # train model
        trainer.train()
        model.save_pretrained("./best_model")
        print(args)
        n_output.append(main_inference(args))
    
    ensemble_id = n_output[0].id
    ensemble_probs = np.sum([out.probs.tolist() for out in n_output], axis=0) / 5
    ensemble_probs = ensemble_probs.tolist()
    ensemble_pred_label = np.argmax(ensemble_probs, axis=1)
    ensemble_pred_label = num_to_label(ensemble_pred_label)
    output = pd.DataFrame(
        {
            "id": ensemble_id,
            "pred_label": ensemble_pred_label,
            "probs": ensemble_probs,
        }
    )
    output.to_csv(
        "./prediction/submission.csv", index=False
    )  # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
    #### 필수!! ##############################################
    print("---- Finish! ----")


def main(args):
    train(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model dir
    parser.add_argument("--model_dir", type=str, default="./best_model")
    args = parser.parse_args()
    main(args)
