import copy

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from tqdm import tqdm

from data_feature import Dataset
from model import SequenceMultiTypeMultiCNN_1

device = torch.device("cuda:0")


def return_y(data_iter, net):
    y_pred = []

    all_seq = []
    for batch in data_iter:
        all_seq += batch['sequence']

        AAI_feat = batch['seq_enc_AAI'].to(device)
        onehot_feat = batch['seq_enc_onehot'].to(device)
        BLOSUM62_feat = batch['seq_enc_BLOSUM62'].to(device)
        PAAC_feat = batch['seq_enc_PAAC'].to(device)
        outputs = net(AAI_feat, onehot_feat, BLOSUM62_feat, PAAC_feat)
        y_pred.extend(outputs.cpu().numpy())

    return y_pred, all_seq


def testing(batch_size, testfasta, seq_len, model_file):
    model = SequenceMultiTypeMultiCNN_1(d_input=[531, 21, 23, 3], vocab_size=21, seq_len=seq_len,
                                        dropout=0.1, d_another_h=128, k_cnn=[2, 3, 4, 5, 6], d_output=1).to(device)

    dataset = Dataset(fasta=testfasta)
    test_loader = dataset.get_dataloader(batch_size=batch_size, max_length=seq_len)

    model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu'))['state_dict'])
    model.eval()

    with torch.no_grad():
        new_y_pred, all_seq = return_y(test_loader, model)

    final_y_pred = copy.deepcopy(new_y_pred)

    final_y_pred = np.array(final_y_pred).T[0].tolist()

    pred_dict = {'seq': all_seq, 'predictions': final_y_pred}
    return pd.DataFrame(pred_dict)


all_function_names = ['antibacterial', 'antigram-positive', 'antigram-negative', 'antifungal', 'antiviral', \
                      'anti_mammalian_cells', 'antihiv', 'antibiofilm', 'anticancer', 'antimrsa', 'antiparasitic', \
                      'hemolytic', 'chemotactic', 'antitb', 'anurandefense', 'cytotoxic', \
                      'endotoxin', 'insecticidal', 'antimalarial', 'anticandida', 'antiplasmodial', 'antiprotozoal']
all_function_cols = ['antibacterial', 'anti-Gram-positive', 'anti-Gram-negative', 'antifungal', 'antiviral', \
                     'anti-mammalian-cells', 'anti-HIV', 'antibiofilm', 'anticancer', 'anti-MRSA', 'antiparasitic', \
                     'hemolytic', 'chemotactic', 'anti-TB', 'anurandefense', 'cytotoxic', \
                     'endotoxin', 'insecticidal', 'antimalarial', 'anticandida', 'antiplasmodial', 'antiprotozoal']


def predict(test_file, batch_size, seq_len=200):
    fas_id = []
    fas_seq = []
    for seq_record in SeqIO.parse(test_file, "fasta"):
        fas_seq.append(str(seq_record.seq).upper())
        fas_id.append(str(seq_record.id))

    pred_prob = []
    for cv_number in tqdm(range(10)):
        df = testing(testfasta=fas_seq,
                     model_file=f'models/AMP_1st/textcnn_cdhit_40_{cv_number}.pth.tar',
                     batch_size=batch_size, seq_len=seq_len)

        temp = df.iloc[:, 1].to_numpy()
        pred_prob.append(temp)

    pred_prob = np.mean(pred_prob, axis=0)
    pred_AMP_label = np.where(pred_prob > 0.5, 'Yes', 'No')

    all_function_pred_label = []
    for function_name in all_function_names:
        function_threshold_df = pd.read_csv(f'models/AMP_2nd_threashold/{function_name}_yd_threshold.csv',
                                            index_col=0)
        function_thresholds = function_threshold_df['threshold'].values.tolist()
        each_function_data = np.zeros((10, len(fas_seq)))

        for cv_number in tqdm(range(10), desc=f"{function_name}", ncols=100, dynamic_ncols=True):
            df = testing(testfasta=fas_seq,
                         model_file=f'models/AMP_2nd/{function_name}/textcnn_cdhit_100_{cv_number}.pth.tar',
                         batch_size=batch_size, seq_len=seq_len)

            predicted_probs = df['predictions']
            each_function_data[cv_number] = predicted_probs > function_thresholds[cv_number]

        avg_pred = np.mean(each_function_data, axis=0)
        pred_each_function_label = np.where(avg_pred > 0.5, 'Yes', 'No')
        all_function_pred_label.append(pred_each_function_label)

    pred_contents_dict = {'name': fas_id, 'sequence': fas_seq, 'AMP': pred_AMP_label}
    for i in range(len(all_function_cols)):
        pred_contents_dict[all_function_cols[i]] = all_function_pred_label[i]

    pred_contents_df = pd.DataFrame(pred_contents_dict)

    return pred_contents_df


def multi_feature_predict(input_file, output_file, batch_size=32, seq_len=200):
    print('\nMulti-feature Predict Start')

    pred_df = predict(input_file, batch_size=batch_size, seq_len=seq_len)
    pred_df.to_csv(output_file)
    print('Multi-feature Predict Finished')


if __name__ == '__main__':
    output_file_name = 'result/prediction_results.csv'
    test_file = 'samples.fasta'
    pred_df = predict(test_file, batch_size=32, seq_len=200)
    pred_df.to_csv(output_file_name)
