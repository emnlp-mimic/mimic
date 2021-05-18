import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl
from sklearn import metrics
from star import StarTransformerClassifier
from transformers import AdamW, BertModel, BertConfig


class Cnn(nn.Module):
    def __init__(self,
                 input_size: int = 7508,
                 length: int = 48,
                 depth: int = 2,
                 filter_size: int = 3,
                 n_filters: int = 64,
                 n_neurons: int = 64,
                 dropout: float = 0.2) -> None:
        super().__init__()
        self.depth = depth
        padding = int(np.floor(filter_size / 2))

        if depth == 1:
            self.conv1 = nn.Conv1d(input_size, n_filters, filter_size, padding=padding)
            self.pool1 = nn.MaxPool1d(2, 2)
            self.fc1 = nn.Linear(int(length * n_filters / 2), n_neurons)
            self.fc1_drop = nn.Dropout(dropout)

        elif depth == 2:
            self.conv1 = nn.Conv1d(input_size, n_filters, filter_size, padding=padding)
            self.pool1 = nn.MaxPool1d(2, 2)
            self.conv2 = nn.Conv1d(n_filters, n_filters, filter_size, padding=padding)
            self.pool2 = nn.MaxPool1d(2, 2)
            self.fc1 = nn.Linear(int(length * n_filters / 4), n_neurons)
            self.fc1_drop = nn.Dropout(dropout)

        elif depth == 3:
            self.conv1 = nn.Conv1d(input_size, n_filters, filter_size, padding=padding)
            self.pool1 = nn.MaxPool1d(2, 2)
            self.conv2 = nn.Conv1d(n_filters, n_filters, filter_size, padding=padding)
            self.pool2 = nn.MaxPool1d(2, 2)
            self.conv3 = nn.Conv1d(n_filters, n_filters, filter_size, padding=padding)
            self.pool3 = nn.MaxPool1d(2, 2)
            self.fc1 = nn.Linear(int(length * n_filters / 8), n_neurons)
            self.fc1_drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x.transpose(1, 2)

        x = self.pool1(F.relu(self.conv1(x)))
        if self.depth == 2 or self.depth == 3:
            x = self.pool2(F.relu(self.conv2(x)))
        if self.depth == 3:
            x = self.pool3(F.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1_drop(self.fc1(x)))
        return x


class Lstm(nn.Module):
    def __init__(self,
                 input_size: int = 7508,
                 hidden_size: int = 1024,
                 n_neurons: int = 128,
                 num_layers: int = 1,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.project = nn.Linear(hidden_size, n_neurons)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        self.lstm.flatten_parameters()
        h_all, (h_T, c_T) = self.lstm(x)
        output = h_T[-1]
        return F.relu(self.drop(self.project(output)))


class Star(nn.Module):
    def __init__(self,
                 input_size: int = 7508,
                 hidden_size: int = 1024,
                 n_neurons: int = 128,
                 num_cycles: int = 3,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.project = nn.Linear(input_size, hidden_size)
        self.star = StarTransformerClassifier(num_cycles, hidden_size, hidden_size // 64, dropout, n_neurons)

    def forward(self, x):
        x = F.relu(self.project(x))
        return self.star(x)


class Encoder(nn.Module):
    def __init__(self,
                 input_size: int = 7508,
                 hidden_size: int = 1024,
                 num_layers: int = 3) -> None:
        super(Encoder, self).__init__()
        self.project = nn.Linear(input_size, hidden_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=hidden_size // 64)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = F.relu(self.project(x))
        return self.encoder(x.transpose(0, 1))[0]


class Bert(nn.Module):
    def __init__(self,
                 pretrained_bert_dir: str = '',
                 freeze: tuple = ()):
        super().__init__()
        self.bert = BertModel(config=BertConfig())
        model_dict = self.bert.state_dict()
        pretrained_dict = torch.load(pretrained_bert_dir)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.bert.load_state_dict(model_dict)
        for name, param in self.bert.named_parameters():
            param.requires_grad = True
            for layer in freeze:
                if layer in name:
                    param.requires_grad = False
                    break

    def forward(self, x):
        return self.bert(input_ids=x[0], token_type_ids=x[1], attention_mask=x[2])[1]


class Gate(nn.Module):
    def __init__(self, inp1_size, inp2_size, inp3_size, dropout):
        super(Gate, self).__init__()

        self.fc1 = nn.Linear(inp1_size + inp2_size, 1)
        self.fc2 = nn.Linear(inp1_size + inp3_size, 1)
        self.fc3 = nn.Linear(inp2_size + inp3_size, inp1_size)
        self.beta = nn.Parameter(torch.randn((1,)))
        self.norm = nn.LayerNorm(inp1_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp1, inp2, inp3):
        w2 = torch.sigmoid(self.fc1(torch.cat([inp1, inp2], -1)))
        w3 = torch.sigmoid(self.fc2(torch.cat([inp1, inp3], -1)))
        adjust = self.fc3(torch.cat([w2 * inp2, w3 * inp3], -1))
        one = torch.tensor(1).type_as(adjust)
        alpha = torch.min(torch.norm(inp1) / torch.norm(adjust) * self.beta, one)
        output = inp1 + alpha * adjust
        output = self.dropout(self.norm(output))
        return output


class Outer(nn.Module):
    def __init__(self,
                 inp1_size: int = 128,
                 inp2_size: int = 128,
                 n_neurons: int = 128):
        super(Outer, self).__init__()
        self.inp1_size = inp1_size
        self.inp2_size = inp2_size
        self.feedforward = nn.Sequential(
            nn.Linear((inp1_size + 1) * (inp2_size + 1), n_neurons),
            nn.ReLU(),
            nn.Linear(n_neurons, n_neurons),
            nn.ReLU(),
        )

    def forward(self, inp1, inp2):
        batch_size = inp1.size(0)
        append = torch.ones((batch_size, 1)).type_as(inp1)
        inp1 = torch.cat([inp1, append], dim=-1)
        inp2 = torch.cat([inp2, append], dim=-1)
        fusion = torch.zeros((batch_size, self.inp1_size + 1, self.inp2_size + 1)).type_as(inp1)
        for i in range(batch_size):
            fusion[i] = torch.outer(inp1[i], inp2[i])
        fusion = fusion.flatten(1)
        return self.feedforward(fusion)


class Attention(nn.Module):
    def __init__(self,
                 embed_dim: int = 768,
                 dropout: float = 0.1):
        super(Attention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=embed_dim // 64, dropout=dropout)

    def forward(self, inp1, inp2):
        inp1, inp2 = inp1.unsqueeze(1), inp2.unsqueeze(1)
        return self.attention(inp1, inp2, inp2)[0].squeeze(1)


class AUROC(pl.LightningModule):
    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = F.binary_cross_entropy(preds, y.float())
        self.log('loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        return preds, y

    def validation_step_end(self, output):
        return output

    def validation_epoch_end(self, outputs):
        all_preds, all_y = zip(*outputs)

        all_preds = torch.cat(all_preds)
        all_y = torch.cat(all_y)

        all_preds = all_preds.cpu().numpy()
        all_y = all_y.cpu().numpy()

        score = metrics.roc_auc_score(all_y, all_preds)

        self.log('score', score)

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        return preds, y

    def test_step_end(self, output):
        return output

    def test_epoch_end(self, outputs):
        all_preds, all_y = zip(*outputs)

        all_preds = torch.cat(all_preds)
        all_y = torch.cat(all_y)

        all_preds = all_preds.cpu().numpy()
        all_y = all_y.cpu().numpy()
        au_roc = metrics.roc_auc_score(all_y, all_preds)
        au_pr = metrics.auc(*metrics.precision_recall_curve(all_y, all_preds)[1::-1])
        self.log('AUROC', au_roc)
        self.log('AUPR', au_pr)

    def configure_optimizers(self):
        params = self.parameters()
        optimizer = AdamW(params, lr=self.lr)
        return optimizer


class BertAUROC(AUROC):

    def lr_lambda(self, current_step):
        num_training_steps = self.num_training_steps
        num_warmup_steps = int(num_training_steps * self.warmup_proportion)
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    def configure_optimizers(self):
        optimizer = AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
        lr_scheduler = LambdaLR(optimizer, self.lr_lambda)
        return [optimizer], [lr_scheduler]


class Recall(AUROC):

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        _, top_k = torch.topk(preds, 10)
        value = torch.gather(y, dim=-1, index=top_k)
        top_k_true = torch.sum(value, dim=-1)
        total_true = torch.sum(y, dim=-1)
        return top_k_true, total_true

    def validation_epoch_end(self, outputs):
        all_top_k_true, all_true = zip(*outputs)

        all_preds = torch.cat(all_top_k_true, 0).float()
        all_y = torch.cat(all_true, 0).float()

        recall_k = torch.mean(torch.div(all_preds, all_y))

        self.log('score', recall_k)

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        res = []
        for k in [10, 20, 30]:
            _, top_k = torch.topk(preds, k)
            value = torch.gather(y, dim=-1, index=top_k)
            top_k_true = torch.sum(value, dim=-1)
            res.append(top_k_true)
        total_true = torch.sum(y, dim=-1)
        res.append(total_true)
        return res

    def test_epoch_end(self, outputs):
        top_10, top_20, top_30, y = zip(*outputs)

        top_10 = torch.cat(top_10, 0).float()
        top_20 = torch.cat(top_20, 0).float()
        top_30 = torch.cat(top_30, 0).float()
        y = torch.cat(y, 0).float()

        recall_10 = torch.mean(torch.div(top_10, y))
        recall_20 = torch.mean(torch.div(top_20, y))
        recall_30 = torch.mean(torch.div(top_30, y))

        self.log('recall@10', recall_10)
        self.log('recall@20', recall_20)
        self.log('recall@30', recall_30)


class BertRecall(Recall):

    def lr_lambda(self, current_step):
        num_training_steps = self.num_training_steps
        num_warmup_steps = int(num_training_steps * self.warmup_proportion)
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    def configure_optimizers(self):
        optimizer = AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
        lr_scheduler = LambdaLR(optimizer, self.lr_lambda)
        return [optimizer], [lr_scheduler]
