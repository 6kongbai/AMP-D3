import torch
from attr.validators import max_len
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import numpy as np
from .sequence import *


def compute_embeddings(fasta):
    embedding_1 = onehot_encoding(fasta)
    embedding_2 = get_bio_embedding_for_sequence(fasta)
    embedding = np.concatenate((embedding_1, embedding_2), axis=1)

    return embedding


def sequence_embedding(sequences=None, indexes=None, mask=None, max_length=50, device="cpu"):
    if sequences is None:
        assert indexes is not None, "Either sequences or indexes must be provided"
        batch_size_actual = indexes.shape[0]
        sequences = [index_to_fasta(indexes[i], mask[i] if mask is not None else None)
                     for i in range(batch_size_actual)]
    else:
        batch_size_actual = len(sequences)

    embeddings_list = [torch.tensor(compute_embeddings(seq), device=device, dtype=torch.float32) for seq in sequences]
    padded_embeddings = pad_sequence(embeddings_list, batch_first=True, padding_value=0)
    batch_size, seq_len, D = padded_embeddings.shape

    if seq_len < max_length:
        padding = torch.zeros(batch_size, max_length - seq_len, D, device=device)
        padded_embeddings = torch.cat([padded_embeddings, padding], dim=1)  # [batch_size, max_length, 20]

    return padded_embeddings, mask
