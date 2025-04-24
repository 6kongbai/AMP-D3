import os

import numpy as np
import torch
from transformers import AutoTokenizer
import pickle


def load_properties():
    with open(os.path.join(os.path.dirname(__file__), 'properties.pkl'), 'rb') as f:
        properties = pickle.load(f)
    return properties


properties = load_properties()


def get_properties_features(sequences, max_length=30):
    features = []
    for seq in sequences:
        sequence_len = len(seq)
        feature = np.zeros((max_length, 14), dtype=np.float32)
        for i in range(min(sequence_len, max_length)):
            aa = seq[i]
            feature[i, :] = properties[aa]
        features.append(feature)

    features_array = np.array(features)

    return torch.from_numpy(features_array).float()


def load_AAI_dict():
    f = open(os.path.join(os.path.dirname(__file__), './vocab/AAindex.txt'))
    text = f.read()
    f.close()
    text = text.split('\n')
    while '' in text:
        text.remove('')
    cha = text[0].split('\t')
    while '' in cha:
        cha.remove('')
    cha = cha[1:]
    index = []
    for i in range(1, len(text)):
        temp = text[i].split('\t')
        while '' in temp:
            temp.remove('')
        temp = temp[1:]
        for j in range(len(temp)):
            temp[j] = float(temp[j])
        index.append(temp)
    index = np.array(index)
    AAI_dict = {}
    for j in range(len(cha)):
        AAI_dict[cha[j]] = index[:, j]
    AAI_dict['X'] = np.zeros(531)
    return AAI_dict


AAI_dict = load_AAI_dict()


def AAI_embedding(seq, max_len=200) -> torch.Tensor:
    all_embeddings = []
    for each_seq in seq:
        temp_embeddings = []
        for each_char in each_seq:
            temp_embeddings.append(AAI_dict[each_char])
        if max_len > len(each_seq):
            zero_padding = np.zeros((max_len - len(each_seq), 531))
            data_pad = np.vstack((temp_embeddings, zero_padding))
        elif max_len == len(each_seq):
            data_pad = temp_embeddings
        else:
            data_pad = temp_embeddings[:max_len]
        all_embeddings.append(data_pad)
    all_embeddings = np.array(all_embeddings)
    return torch.from_numpy(all_embeddings).float()


def load_PAAC_dict():
    f = open(os.path.join(os.path.dirname(__file__), './vocab/PAAC.txt'))
    text = f.read()
    f.close()
    text = text.split('\n')
    while '' in text:
        text.remove('')
    cha = text[0].split('\t')
    while '' in cha:
        cha.remove('')
    cha = cha[1:]
    index = []
    for i in range(1, len(text)):
        temp = text[i].split('\t')
        while '' in temp:
            temp.remove('')
        temp = temp[1:]
        for j in range(len(temp)):
            temp[j] = float(temp[j])
        index.append(temp)
    index = np.array(index)
    PAAC_dict = {}
    for j in range(len(cha)):
        PAAC_dict[cha[j]] = index[:, j]
    PAAC_dict['X'] = np.zeros(3)
    return PAAC_dict


PAAC_dict = load_PAAC_dict()


def PAAC_embedding(seq, max_len=200) -> torch.Tensor:
    all_embeddings = []
    for each_seq in seq:
        temp_embeddings = []
        for each_char in each_seq:
            temp_embeddings.append(PAAC_dict[each_char])
        if max_len > len(each_seq):
            zero_padding = np.zeros((max_len - len(each_seq), 3))
            data_pad = np.vstack((temp_embeddings, zero_padding))
        elif max_len == len(each_seq):
            data_pad = temp_embeddings
        else:
            data_pad = temp_embeddings[:max_len]
        all_embeddings.append(data_pad)
    all_embeddings = np.array(all_embeddings)
    return torch.from_numpy(all_embeddings).float()


def load_BLOSUM62_dict():
    f = open(os.path.join(os.path.dirname(__file__), 'vocab/blosum62.txt'))
    text = f.read()
    f.close()
    text = text.split('\n')
    while '' in text:
        text.remove('')
    cha = text[0].split(' ')
    while '' in cha:
        cha.remove('')
    index = []
    for i in range(1, len(text)):
        temp = text[i].split(' ')
        while '' in temp:
            temp.remove('')
        for j in range(len(temp)):
            temp[j] = float(temp[j])
        index.append(temp)
    index = np.array(index)
    BLOSUM62_dict = {}
    for j in range(len(cha)):
        BLOSUM62_dict[cha[j]] = index[:, j]

    return BLOSUM62_dict


BLOSUM62_dict = load_BLOSUM62_dict()


def BLOSUM62_embedding(seq, max_len=200) -> torch.Tensor:
    all_embeddings = []
    for each_seq in seq:
        temp_embeddings = []
        for each_char in each_seq:
            temp_embeddings.append(BLOSUM62_dict[each_char])
        if max_len > len(each_seq):
            zero_padding = np.zeros((max_len - len(each_seq), 23))
            data_pad = np.vstack((temp_embeddings, zero_padding))
        elif max_len == len(each_seq):
            data_pad = temp_embeddings
        else:
            data_pad = temp_embeddings[:max_len]
        all_embeddings.append(data_pad)
    all_embeddings = np.array(all_embeddings)
    return torch.from_numpy(all_embeddings).float()


def onehot_embedding(seq, max_len=200) -> torch.Tensor:
    char_list = 'ARNDCQEGHILKMFPSTWYVX'
    char_dict = {}
    for i in range(len(char_list)):
        char_dict[char_list[i]] = i
    all_embeddings = []
    for each_seq in seq:
        temp_embeddings = []
        for each_char in each_seq:
            codings = np.zeros(21)
            if each_char in char_dict.keys():
                codings[char_dict[each_char]] = 1
            else:
                codings[20] = 1
            temp_embeddings.append(codings)
        if max_len > len(each_seq):
            zero_padding = np.zeros((max_len - len(each_seq), 21))
            data_pad = np.vstack((temp_embeddings, zero_padding))
        elif max_len == len(each_seq):
            data_pad = temp_embeddings
        else:
            data_pad = temp_embeddings[:max_len]

        all_embeddings.append(data_pad)
    all_embeddings = np.array(all_embeddings)
    return torch.from_numpy(all_embeddings).float()


class MetagenesisData(torch.utils.data.Dataset):
    def __init__(self, datas):
        self.datas = datas

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        return self.datas[index]


class ProteinEncode(object):
    def __init__(self, sequences, labels=None):
        assert isinstance(sequences, (list, np.ndarray, tuple)), 'sequences must be a list or numpy array.'
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t12_35M_UR50D')

    def build_data(self, max_length):
        seq_enc_onehot = onehot_embedding(self.sequences, max_len=max_length)
        seq_enc_blosum62 = BLOSUM62_embedding(self.sequences, max_len=max_length)
        seq_enc_AAI = AAI_embedding(self.sequences, max_len=max_length)
        seq_enc_PAAC = PAAC_embedding(self.sequences, max_len=max_length)
        seq_enc_properties = get_properties_features(self.sequences, max_length=max_length)

        sample = self.tokenizer(self.sequences.tolist(), truncation=True, padding=True, max_length=max_length,
                                return_tensors="pt")
        input_ids = sample['input_ids'].numpy()
        attention_mask = sample['attention_mask'].numpy()

        samples = []
        for i in range(len(self.sequences)):
            sample = {
                'sequence': self.sequences[i],
                'input_ids': input_ids[i],
                'attention_mask': attention_mask[i],
                'onehot_feat': seq_enc_onehot[i],
                'BLOSUM62_feat': seq_enc_blosum62[i],
                'PAAC_feat': seq_enc_PAAC[i],
                'AAI_feat': seq_enc_AAI[i],
                'properties_feat': seq_enc_properties[i],
                'label': self.labels[i] if self.labels is not None else None
            }
            samples.append(sample)

        data = MetagenesisData(samples)

        return data

    def get_dataloader(self, max_length, batch_size):
        data = self.build_data(max_length)
        data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size)
        return data_loader
