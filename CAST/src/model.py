import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss
from transformers import AutoTokenizer, AutoModel

from middleLayer import MultiHeadCrossModalAttention


class SequenceMultiTypeMultiCNN(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, in_channels=[531, 21, 23, 3, 14], dropout=0.1, out_channels=64, k_cnn=[3, 5, 7]):
        """
        :param in_channels:
        :param max_len:
        :param dropout:
        :param out_channels: 输出通道数
        :param k_cnn:
        """
        super(SequenceMultiTypeMultiCNN, self).__init__()
        self.channels_len = len(in_channels)
        self.batchnorm = nn.ModuleList([nn.BatchNorm1d(num_features=in_channels[i]) for i in range(self.channels_len)])
        # Define convolutional layers for each input type (AAI_feat, onehot_feat, BLOSUM62_feat, PAAC_feat)
        self.convs = nn.ModuleList(
            [self._create_conv_block(in_channels[i], out_channels, k_cnn, dropout) for i in range(self.channels_len)])
        self.pool = nn.ModuleList([nn.AdaptiveMaxPool1d(out_channels) for _ in range(self.channels_len)])

    def _create_conv_block(self, in_channels, out_channels, k_cnn, dropout):
        """Helper function to create a convolutional block."""
        return nn.ModuleList([
            nn.Sequential(
                # [batch_size, in_channels, seq_len]->[batch_size, out_channels, seq_len]
                nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=h, padding='same'),
                nn.BatchNorm1d(num_features=out_channels),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(dropout),
                # [batch_size, out_channels, seq_len]->[batch_size, out_channels, seq_len]
                nn.Conv1d(out_channels, out_channels, kernel_size=1),
            ) for h in k_cnn
        ])

    def _process_feature(self, feature, idx):
        """Helper function to process each feature through its conv block."""
        feature = feature.permute(0, 2, 1)
        feature = self.batchnorm[idx](feature)
        out = [conv(feature) for conv in self.convs[idx]]
        # [batch_size, out_channels, seq_len] -> [batch_size, seq_len, len(k_cnn)*out_channels]
        out = torch.cat(out, dim=1).permute(0, 2, 1)
        # [batch_size, seq_len, len(k_cnn)*out_channels] -> [batch_size, seq_len, out_channels]
        out = self.pool[idx](out)
        return out

    def forward(self, AAI_feat, onehot_feat, BLOSUM62_feat, PAAC_feat, properties_feat):
        """
        Process each feature type using its respective convolutional blocks
        :param AAI_feat: [batch_size, seq_len, 531]
        :param onehot_feat: [batch_size, seq_len, 21]
        :param BLOSUM62_feat: [batch_size, seq_len, 23]
        :param PAAC_feat: [batch_size, seq_len, 3]
        :param properties_feat : [batch_size, seq_len, 14]
        :return: [batch_size, seq_len, out_channels * 5]
        """
        # [batch_size, seq_len, 531] -> [batch_size, seq_len，out_channels]
        out_1 = self._process_feature(AAI_feat, 0)
        # [batch_size, seq_len, 21] -> [batch_size, seq_len，out_channels]
        out_2 = self._process_feature(onehot_feat, 1)
        # [batch_size, seq_len, 23] -> [batch_size, seq_len，out_channels]
        out_3 = self._process_feature(BLOSUM62_feat, 2)
        # [batch_size, seq_len, 3] -> [batch_size, seq_len，out_channels]
        out_4 = self._process_feature(PAAC_feat, 3)
        # [batch_size, seq_len, 14] -> [batch_size, seq_len，out_channels]
        out_5 = self._process_feature(properties_feat, 4)

        outputs = torch.cat([out_1, out_2, out_3, out_4, out_5], dim=-1)
        return outputs


class SequenceEmbedding(nn.Module):
    """
    Sequence Embedding using ESM-1b model
    """

    def __init__(self):
        super(SequenceEmbedding, self).__init__()
        self.embedding_model = AutoModel.from_pretrained('facebook/esm2_t12_35M_UR50D')
        for param in self.embedding_model.parameters():
            param.requires_grad = False
        for param in self.embedding_model.pooler.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        """
        :param attention_mask:
        :param input_ids:
        :return: output: [batch_size, seq_len, 480]
        """
        output = self.embedding_model(input_ids, attention_mask).last_hidden_state
        return output


class OYampModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, dropout=0.2, encoding_hidden_size=64, embedding_hidden_size=480, hidden_size=128,
                 in_channels=[531, 21, 23, 3, 14]):
        """
        :param dropout:
        :param encoding_hidden_size: 输出通道数
        """
        super(OYampModel, self).__init__()
        self.sequence_cnn = SequenceMultiTypeMultiCNN(in_channels=in_channels, dropout=dropout,
                                                      out_channels=encoding_hidden_size)
        self.sequence_embedding = SequenceEmbedding()

        self.batchNormLayers = nn.ModuleList([
            nn.LayerNorm(encoding_hidden_size * len(in_channels)),  # 使用 LayerNorm
            nn.LayerNorm(embedding_hidden_size),
        ])

        self.embeddingProcessLayer = nn.Sequential(
            nn.Conv1d(embedding_hidden_size, hidden_size, kernel_size=1),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.encodingProcessLayer = nn.Sequential(
            nn.Conv1d(encoding_hidden_size * len(in_channels), hidden_size, kernel_size=1),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # self.fusionLayer = nn.Sequential(
        #     nn.Conv1d(hidden_size * 2, hidden_size, kernel_size=1),
        #     nn.BatchNorm1d(hidden_size),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Dropout(dropout),
        # )
        self.fusionLayer = MultiHeadCrossModalAttention(hidden_dim=hidden_size, input_dim=hidden_size)
        self.transformer_res = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=1),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=1),
        )

        self.transformer_pool = nn.AdaptiveMaxPool2d((1, None))

        self.transformer = nn.Transformer(d_model=hidden_size, batch_first=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.output = nn.Sigmoid()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        self.loss_fct = BCEWithLogitsLoss()
        self.reset_parameters()

    def reset_parameters(self):
        for layer in [self.transformer_res, self.classifier, self.embeddingProcessLayer, self.encodingProcessLayer,
                      self.fusionLayer]:
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal(layer.weight)

    def forward(self, input_ids, attention_mask, AAI_feat, onehot_feat, BLOSUM62_feat, PAAC_feat, properties_feat,
                labels=None):
        """
        Process each feature type using its respective convolutional blocks
        :param attention_mask:
        :param input_ids:
        :param AAI_feat: [batch_size, seq_len, 531]
        :param onehot_feat: [batch_size, seq_len, 21]
        :param BLOSUM62_feat: [batch_size, seq_len, 23]
        :param PAAC_feat: [batch_size, seq_len, 3]
        :param labels: [batch_size, 1]
        :return:
        """

        # seq_encoded [batch_size, seq_len, out_channels * len(in_channels)]
        seq_encoded = self.sequence_cnn(AAI_feat, onehot_feat, BLOSUM62_feat, PAAC_feat, properties_feat)
        # seq_embedded [batch_size, seq_len, 480]
        seq_embedded = self.sequence_embedding(input_ids, attention_mask)

        seq_encoded = self.batchNormLayers[0](seq_encoded)
        seq_embedded = self.batchNormLayers[1](seq_embedded)

        # [batch_size, seq_len, 480] -> [batch_size, seq_len, hidden_size]
        seq_embedded = self.embeddingProcessLayer(seq_embedded.permute(0, 2, 1)).permute(0, 2, 1)
        # [batch_size, seq_len, out_channels * len(in_channels)] -> [batch_size, seq_len, hidden_size]
        seq_encoded = self.encodingProcessLayer(seq_encoded.permute(0, 2, 1)).permute(0, 2, 1)

        # [batch_size, seq_len, hidden_size*2] -> [batch_size, seq_len, hidden_size]
        x = self.fusionLayer(seq_encoded, seq_embedded)
        # [batch_size, seq_len, hidden_size] -> [seq_len, batch_size, hidden_size]
        transformer_x = self.transformer(x, x) + x
        transformer_x = self.drop(transformer_x)
        transformer_x = self.transformer_res(transformer_x.permute(0, 2, 1)).permute(0, 2, 1)
        transformer_x = self.transformer_pool(transformer_x).squeeze(1)
        logits = self.classifier(transformer_x)
        outputs = self.output(logits)
        if labels is None:
            return {'logits': outputs, "last_output": transformer_x}
        loss = self.loss_fct(logits.view(-1), labels.view(-1).float())
        return {"loss": loss, "logits": outputs, "last_output": transformer_x}
