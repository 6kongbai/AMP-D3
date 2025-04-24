import numpy as np
import torch.utils.data


# from transformers import T5Tokenizer,XLNetTokenizer
def AAI_embedding(seq, max_len=200):
    f = open('data/AAindex.txt')
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


def PAAC_embedding(seq, max_len=200):
    f = open('data/PAAC.txt')
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
    AAI_dict['X'] = np.zeros(3)
    all_embeddings = []
    for each_seq in seq:
        temp_embeddings = []
        for each_char in each_seq:
            temp_embeddings.append(AAI_dict[each_char])
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


def PC6_embedding(seq, max_len=200):
    f = open('data/6-pc')
    text = f.read()
    f.close()
    text = text.split('\n')
    while '' in text:
        text.remove('')
    text = text[1:]
    AAI_dict = {}
    for each_line in text:
        temp = each_line.split(' ')
        while '' in temp:
            temp.remove('')
        for i in range(1, len(temp)):
            temp[i] = float(temp[i])
        AAI_dict[temp[0]] = temp[1:]
    AAI_dict['X'] = np.zeros(6)
    all_embeddings = []
    for each_seq in seq:
        temp_embeddings = []
        for each_char in each_seq:
            temp_embeddings.append(AAI_dict[each_char])
        if max_len > len(each_seq):
            zero_padding = np.zeros((max_len - len(each_seq), 6))
            data_pad = np.vstack((temp_embeddings, zero_padding))
        elif max_len == len(each_seq):
            data_pad = temp_embeddings
        else:
            data_pad = temp_embeddings[:max_len]
        all_embeddings.append(data_pad)
    all_embeddings = np.array(all_embeddings)
    return torch.from_numpy(all_embeddings).float()


def BLOSUM62_embedding(seq, max_len=200):
    f = open('data/blosum62.txt')
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


def onehot_embedding(seq, max_len=200):
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
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class Dataset(object):
    def __init__(self,sequences):
        self.sequences = sequences

    def encode_seq_enc(self, sequences, max_length):
        pass

    def build_data(self, max_length):
        seq_enc_onehot = onehot_embedding(self.sequences, max_len=max_length)
        seq_enc_blosum62 = BLOSUM62_embedding(self.sequences, max_len=max_length)
        seq_enc_AAI = AAI_embedding(self.sequences, max_len=max_length)
        seq_enc_PAAC = PAAC_embedding(self.sequences, max_len=max_length)
        samples = []

        for i in range(len(self.sequences)):
            sample = {
                'sequence': self.sequences[i],
                'seq_enc_onehot': seq_enc_onehot[i],
                'seq_enc_BLOSUM62': seq_enc_blosum62[i],
                'seq_enc_PAAC': seq_enc_PAAC[i],
                'seq_enc_AAI': seq_enc_AAI[i]
            }
            samples.append(sample)

        data = MetagenesisData(samples)

        return data

    def get_dataloader(self, max_length=200, batch_size=128):
        data = self.build_data(max_length)
        data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size)
        return data_loader
        # return data


import torch

device = torch.device("cpu")

if __name__ == '__main__':
    dataset = Dataset(
        fasta='static/uploads/Insect.fasta',
        sep=',')
    train_loader = dataset.get_dataloader(
        batch_size=32, max_length=200)
    # y_pred=[]
    y_true = []
    all_seqs = []

