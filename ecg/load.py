from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import json
import keras
import numpy as np
import os
import random
import scipy.io as sio
import tqdm

# TODO, make this a config option
STEP = 1

def data_generator(batch_size, preproc, x, y):
    num_examples = len(x)
    examples = zip(x, y)
    examples = sorted(examples, key = lambda x: x[0].shape[0])
    end = num_examples - batch_size + 1
    batches = [examples[i:i+batch_size]
                for i in range(0, end, batch_size)]
    random.shuffle(batches)
    while True:
        for batch in batches:
            x, y = zip(*batch)
            yield preproc.process(x, y)

class Preproc:

    def __init__(self, ecg, labels):
        self.mean, self.std = compute_mean_std(ecg)
        self.classes = sorted(set(l for label in labels for l in label))
        self.int_to_class = dict( zip(range(len(self.classes)), self.classes))
        self.class_to_int = {c : i for i, c in self.int_to_class.items()}

    def process(self, x, y):
        return self.process_x(x), self.process_y(y)

    def process_x(self, x):
        x = np.stack(x)
        x = (x - self.mean) / self.std
        return x

    def process_y(self, y):
        y = [[self.class_to_int[l] for l in label] for label in y]
        y_mat = np.zeros((len(y), len(self.class_to_int)), np.int32)
        for e, label in enumerate(y):
            y_mat[e, label] = 1
        return y_mat

def pad(x, val=0, dtype=np.float32):
    max_len = max(len(i) for i in x)
    padded = np.full((len(x), max_len), val, dtype=dtype)
    for e, i in enumerate(x):
        padded[e, :len(i)] = i
    return padded

def compute_mean_std(x):
    x = np.vstack(x)
    return (np.mean(x, axis=0, keepdims=True).astype(np.float32),
           np.std(x, axis=0, keepdims=True).astype(np.float32))

def load_dataset(data_json):
    with open(data_json, 'r') as fid:
        data = [json.loads(l) for l in fid]
    labels = []; ecgs = []
    for d in tqdm.tqdm(data):
        labels.append(d['labels'])
        ecgs.append(load_ecg(d['ecg']))
    return ecgs, labels

def load_ecg(record):
    ext = os.path.splitext(record)[1]
    if ext == ".npy":
        ecg = np.load(record)
    elif ext == ".mat":
        ecg = sio.loadmat(record)['val'].squeeze()
    else: # Assumes binary 16 bit integers
        with open(record, 'r') as fid:
            ecg = np.fromfile(fid, dtype=np.int16)

    # TODO, optional truncate here (since not needed 
    # for set level label)
    trunc_samp = STEP * int(len(ecg) / STEP)
    return ecg[:trunc_samp]

if __name__ == "__main__":
    data_json = "examples/twelvelead/train.json"
    train = load_dataset(data_json)
    preproc = Preproc(*train)
    x, y = preproc.process(*train)
    print(x.shape)
    print(y.shape)
