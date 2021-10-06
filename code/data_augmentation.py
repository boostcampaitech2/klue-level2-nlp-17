import pandas as pd
import numpy as np
from tqdm import tqdm
from time import time
from koeda import AEDA


data_dir = "/opt/ml/klue-level2-nlp-17/dataset/train/train.csv"
df = pd.read_csv(data_dir)

label_order = df["label"].value_counts(normalize=True)
label_order_top = label_order.index[:9].tolist()

print("-----Change Entities-----")
st = time()

change_entity_only = ["per:alternate_names", "per:spouse", "per:colleagues", "per:other_family", "per:siblings"]
added_index = len(df)
for idx, row in df.iterrows():
    if row.label in change_entity_only:
        df.loc[added_index] = [added_index, row.sentence, row.object_entity, row.subject_entity, row.label, row.source]
    elif row.label == "org:member_of":
        df.loc[added_index] = [added_index, row.sentence, row.object_entity, row.subject_entity, "org:members", row.source]
    elif row.label == "per:parents":
        df.loc[added_index] = [added_index, row.sentence, row.object_entity, row.subject_entity, "per:children", row.source]
    elif row.label == "per:children":
        df.loc[added_index] = [added_index, row.sentence, row.object_entity, row.subject_entity, "per:parents", row.source]
    added_index += 1
    
df.drop_duplicates(subset=["sentence", "subject_entity", "object_entity", "label", "source"], inplace=True)

print("-----Finished Changing Entities-----")
print(time() - st)

print("-----AEDA-----")
st = time()

aeda = AEDA(
    morpheme_analyzer="Okt", punc_ratio=0.3, punctuations=[".", ",", "!", "?", ";", ":"]
)

added_index = df.id.to_numpy()[-1] + 1

for idx, row in df.iterrows():
    if row.label in label_order_top:
        continue
    sentence_aeda = aeda(row.sentence)
    df.loc[added_index] = [added_index, sentence_aeda, row.subject_entity, row.object_entity, row.label, row.source]
    added_index += 1
    print(idx)

df.to_csv("/opt/ml/klue-level2-nlp-17/dataset/train/train_entities_aeda_augmented.csv", index=False)

print("-----Finished AEDA-----")
print(time() - st)
