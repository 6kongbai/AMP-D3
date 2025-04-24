import random

import numpy as np
import pytorch_lightning as pl
import torch
from Bio import SeqIO
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from .dataSets import *


def set_random_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def get_fasta_list(dataType):
    assert dataType in {"AMP", "nonAMP"}, print("input type should be AMP or nonAMP")

    if dataType == "AMP":
        fasta_path = "data/source/fasta/AMP.fasta"
    else:
        fasta_path = "data/source/fasta/nonAMP.fasta"

    fasta_data = SeqIO.parse(fasta_path, "fasta")
    fasta_id_list = []
    fasta_seq_list = []

    for fasta in tqdm(fasta_data):
        fasta_id = fasta.id
        fasta_seq = str(fasta.seq)

        fasta_id_list.append(fasta_id)
        fasta_seq_list.append(fasta_seq)

    return fasta_id_list, fasta_seq_list


def custom_collate_fn(batch, max_length):
    amp = [item["AMP"] for item in batch]
    nonamp = [item["nonAMP"] for item in batch]
    amp_padded = pad_sequence(amp, batch_first=True, padding_value=0)
    nonamp_padded = pad_sequence(nonamp, batch_first=True, padding_value=0)
    amp_lengths = torch.tensor([min(len(t), max_length) for t in amp], dtype=torch.long)
    nonamp_lengths = torch.tensor([min(len(t), max_length) for t in nonamp], dtype=torch.long)
    if amp_padded.shape[1] < max_length:
        padding = torch.zeros(amp_padded.shape[0], max_length - amp_padded.shape[1], amp_padded.shape[2])
        amp_padded = torch.cat([amp_padded, padding], dim=1)
    if nonamp_padded.shape[1] < max_length:
        padding = torch.zeros(nonamp_padded.shape[0], max_length - nonamp_padded.shape[1], nonamp_padded.shape[2])
        nonamp_padded = torch.cat([nonamp_padded, padding], dim=1)

    amp_mask = torch.zeros(amp_padded.shape[0], max_length, dtype=torch.float32)
    nonamp_mask = torch.zeros(nonamp_padded.shape[0], max_length, dtype=torch.float32)

    for i, length in enumerate(amp_lengths):
        amp_mask[i, :length] = 1.0
    for i, length in enumerate(nonamp_lengths):
        nonamp_mask[i, :length] = 1.0

    return {
        "logit": amp_padded,
        "nonamp_logit": nonamp_padded,
        "amp_mask": amp_mask,
        "nonamp_mask": nonamp_mask
    }


class ProteinEncode(pl.LightningDataModule):
    def __init__(self, batch_size: int = 256, max_length=50):
        super().__init__()
        self.batch_size = batch_size

        self.AMP_id_list, self.AMP_seq_list = get_fasta_list("AMP")
        self.nonAMP_id_list, self.nonAMP_seq_list = get_fasta_list("nonAMP")
        self.dataset = ProteinDataset(self.AMP_seq_list, self.nonAMP_seq_list)
        self.max_length = max_length

    def train_dataloader(self):
        return DataLoader(self.dataset,
                          batch_size=self.batch_size,
                          collate_fn=lambda batch: custom_collate_fn(batch, max_length=self.max_length),
                          pin_memory=True,
                          shuffle=True)
