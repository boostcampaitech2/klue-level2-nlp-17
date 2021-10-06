import argparse
import json
from importlib import import_module
from pororo import Pororo
import os
from tqdm import tqdm
import torch

def round_trip_translation(args, dataset):
    language_list = args['language_list'].split()
    not_augmentate_label_list = args['not_augmentate_label_list'].split()
    mt = Pororo(task="translation", lang="multi")
    for language in language_list:
        for index, data in tqdm(dataset.iterrows(),total=dataset.shape[0]):
            if data['label'] in not_augmentate_label_list:
                continue
            sentence = mt(data['sentence'], src="ko", tgt=language)
            sentence = mt(sentence, src=language, tgt="ko")
            sentence = sentence
            new_row = {'id':dataset.iloc[-1]['id']+1, 'sentence':sentence, 'subject_entity':data['subject_entity'], 'object_entity':data['object_entity'], 'label':data['label']}
            dataset = dataset.append(new_row, ignore_index=True)

    save_file = os.path.join(args["train_dir"],"translate_"+"_".join(language_list)+"_train.csv")
    dataset.to_csv(save_file)
    return dataset


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    args = argparse.ArgumentParser(description='how to use argparser')
    args.add_argument('-c', '--config', default=None, type=str, help='config.json file path')
    args = args.parse_args()

    args = json.load(open(args.config,'rt'))

    dataset = getattr(import_module('load_data'),args['load_data'])(args['train_data'])
    print(dataset.head(5))

    augmentation_method_list = args['augmentation_method_list'].split()

    for augmentation_method in augmentation_method_list:
        dataset = getattr(import_module('data_augmentation_main'),augmentation_method)(args,dataset)

    save_file = os.path.join(args["train_dir"],"_".join(augmentation_method_list)+"_train.csv")
    dataset.to_csv(save_file)