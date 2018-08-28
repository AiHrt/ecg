import argparse
import glob
import json
import numpy as np
import os
import random
import tqdm

def genfakelabels():
    # TODO change this when we have labels 
    all_labels = ["sinus", "atrial_fibrillation",
            "ventricular_tachycardia", "supraventricular_tachycardia"]
    labels = [l for l in all_labels if random.random() > 0.5]
    if not labels:
        labels = [random.choice(all_labels)]
    return labels

def load_ecg(ecg_file):
    # *NB* we assume 16-bit integer type
    ecg = np.genfromtxt(ecg_file, delimiter=",", dtype=np.int16)
    ecg = ecg[1:, :-1]
    return ecg

def resample(ecg):
    # Assume 500Hz, resample to 250hz
    if ecg.shape[0] == 5000:
        return ecg[::2, :]
    return ecg

def load_all(data_path):
    ecgs = glob.glob(os.path.join(data_path, "*.csv"))
    dataset = []
    for ecg_file in tqdm.tqdm(ecgs):
        ecg = load_ecg(ecg_file)        
        ecg = resample(ecg)
        new_file = os.path.splitext(ecg_file)[0] + ".npy"
        np.save(new_file, ecg)
        dataset.append((new_file, genfakelabels()))
    return dataset 

def split(dataset, dev_frac, test_frac):
    num_dev = int(dev_frac * len(dataset))
    num_test = int(test_frac * len(dataset))
    random.shuffle(dataset)
    dev = dataset[:num_dev]
    test = dataset[num_dev:num_dev+num_test]
    train = dataset[num_dev+num_test:]
    return train, dev, test

def make_json(save_path, dataset):
    with open(save_path, 'w') as fid:
        for d in dataset:
            datum = {'ecg' : d[0],
                     'labels' : d[1]}
            json.dump(datum, fid)
            fid.write('\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make data JSONS.")
    parser.add_argument("data_path", help="Path to the data.")
    parser.add_argument("--seed", "-s", default=2018,
            help="Seed for splitting data.")
    parser.add_argument("--devfrac", "-d", default=0.0,
            help="Fraction used for dev set.")
    parser.add_argument("--testfrac", "-t", default=0.0,
            help="Fraction used for test set.")
    args = parser.parse_args()

    random.seed(args.seed)
    dataset = load_all(args.data_path)
    train, dev, test = split(dataset, args.devfrac, args.testfrac)

    make_json("train.json", train)
    make_json("dev.json", dev)
    make_json("test.json", dev)

