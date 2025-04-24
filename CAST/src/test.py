import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from safetensors.torch import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_curve, auc
from data_process.data_feature import ProteinEncode
from model import OYampModel


def save_seq_to_fasta(seq_fasta, output_path=""):
    record_list = []
    for i, fasta in enumerate(seq_fasta):
        record_id = "AMP_{}".format(i)
        record = SeqRecord(Seq(fasta), id=record_id, description="")
        record_list.append(record)
    SeqIO.write(record_list, output_path, "fasta")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "../weight/model.safetensors"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = OYampModel().to(device)

load_model(model, model_path)
result = {'label': [], 'pred': [], 'logits': []}
features = []
model.eval()

df = pd.read_csv('data/val_data.csv')

pe = ProteinEncode(df['sequence'].values, df['label'].values)
dataloader = pe.get_dataloader(max_length=30, batch_size=1024)
with torch.no_grad():
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        onehot_feat = batch['onehot_feat'].to(device)
        BLOSUM62_feat = batch['BLOSUM62_feat'].to(device)
        PAAC_feat = batch['PAAC_feat'].to(device)
        AAI_feat = batch['AAI_feat'].to(device)
        properties_feat = batch['properties_feat'].to(device)
        labels = batch['label'].to(device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask, AAI_feat, onehot_feat, BLOSUM62_feat, PAAC_feat,
                            properties_feat)
            predictions = outputs['logits']
            features.append(outputs['last_output'])
            predicted_labels = (predictions > 0.5).int().cpu().numpy().tolist()
            result['logits'].extend(predictions.cpu().numpy().flatten().tolist())
            result['label'].extend(labels.cpu().numpy())
            result['pred'].extend(predicted_labels)
result['pred'] = np.array(result['pred']).squeeze().tolist()
result = pd.DataFrame(result)

y_true, y_pred = result['label'], result['pred']
ACC = accuracy_score(y_true, y_pred) * 100
PRE = precision_score(y_true, y_pred) * 100
SEN = recall_score(y_true, y_pred) * 100
SPE = recall_score(y_true, y_pred, pos_label=0) * 100  # 指定负类标签为0
MCC = matthews_corrcoef(y_true, y_pred) * 100
F1 = f1_score(y_true, y_pred) * 100
fpr, tpr, thresholds = roc_curve(y_true, result['logits'])
roc_auc = auc(fpr, tpr) * 100

print(f"Accuracy: {ACC:.2f}%")
print(f"Precision: {PRE:.2f}%")
print(f"Recall (Sensitivity): {SEN:.2f}%")
print(f"Specificity: {SPE:.2f}%")
print(f"MCC: {MCC:.2f}%")
print(f"F1-Score: {F1:.2f}%")
print(f"AUC: {roc_auc:.2f}%")
