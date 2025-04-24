from torch.utils.data import Dataset

from src.modules.sequence.encode import index_to_onehot
from src.utils.embed.sequence import fasta_to_index


class ProteinDataset(Dataset):
    def __init__(self, amp_seq_list, nonamp_seq_list):
        self.amp_seq_list = amp_seq_list
        self.nonamp_seq_list = nonamp_seq_list

    def __getitem__(self, index):
        amp_seq = self.amp_seq_list[index]
        nonamp_seq = self.nonamp_seq_list[index]

        AMP_logit = index_to_onehot(fasta_to_index(amp_seq))
        nonAMP_logit = index_to_onehot(fasta_to_index(nonamp_seq))

        return {"AMP": AMP_logit, "nonAMP": nonAMP_logit}

    def __len__(self):
        return len(self.amp_seq_list)