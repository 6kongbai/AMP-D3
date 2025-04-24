import os
import sys
import time

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from .models.MyModel import MMCD

model = MMCD(
    n_timestep=1000,
    beta_schedule='linear',
    beta_start=1.e-7,
    beta_end=2.e-2,
    temperature=0.1,
    learning_rate_struct=1e-4,
    learning_rate_seq=1e-4,
    learning_rate_cont=1e-4,
)
# checkpoint_path = r"../checkpoints/"
model = MMCD.load_from_checkpoint(checkpoint_path).to('cuda')
model.eval()
seq_fasta, record_path, out_seq_traj = model.denoise_seq_sample(n_seq=50)

def save_seq_to_fasta(seq_fasta, output_path=""):
    record_list = []
    for i, fasta in enumerate(seq_fasta):
        record_id = "AMP_{}".format(i)
        record = SeqRecord(Seq(fasta), id=record_id, description="")
        record_list.append(record)

    output_file = "{}/output_{}.fasta".format(output_path, time.time())
    SeqIO.write(record_list, output_file, "fasta")


save_seq_to_fasta(seq_fasta, '../data/output')
