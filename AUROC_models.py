from typing import Any
from argparse import ArgumentParser

import torch
from torch import nn

import base


class Line(base.AUROC):

    def __init__(self,
                 input_size: int = 97,
                 hidden_size: int = 64,
                 output_size: int = 1,
                 lr: float = 1e-4,
                 **kwargs: Any) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.enc = nn.Linear(input_size, hidden_size)
        self.pred = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.enc(x[0][0])
        return torch.sigmoid(self.pred(x)).squeeze(1)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--input_size', type=int, default=96)
        parser.add_argument('--hidden_size', type=int, default=64)
        parser.add_argument('--lr', type=float, default=1e-4)
        return parser


class Lstm(base.AUROC):

    def __init__(self,
                 input_size: int = 4620,
                 hidden_size: int = 512,
                 n_neurons: int = 128,
                 output_size: int = 1,
                 num_layers: int = 1,
                 dropout: float = 0.0,
                 lr: float = 1e-4,
                 **kwargs: Any) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.enc = base.Lstm(input_size=input_size, hidden_size=hidden_size, n_neurons=n_neurons,
                             num_layers=num_layers, dropout=dropout)
        self.pred = nn.Linear(n_neurons, output_size)

    def forward(self, x):
        output = self.enc(x[0][1])
        return torch.sigmoid(self.pred(output)).squeeze(1)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--input_size', type=int, default=4816)
        parser.add_argument('--hidden_size', type=int, default=512)
        parser.add_argument('--n_neurons', type=int, default=128)
        parser.add_argument('--num_layers', type=int, default=1)
        parser.add_argument('--dropout', type=float, default=0.0)
        parser.add_argument('--lr', type=float, default=1e-4)
        return parser


class Star(base.AUROC):

    def __init__(self,
                 input_size: int = 4620,
                 hidden_size: int = 512,
                 n_neurons: int = 128,
                 output_size: int = 1,
                 num_cycles: int = 3,
                 dropout: float = 0.1,
                 lr: float = 1e-4,
                 **kwargs: Any) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.enc = base.Star(input_size=input_size, hidden_size=hidden_size, n_neurons=n_neurons,
                             num_cycles=num_cycles, dropout=dropout)
        self.pred = nn.Linear(n_neurons, output_size)

    def forward(self, x):
        output = self.enc(x[0][1])
        return torch.sigmoid(self.pred(output)).squeeze(1)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--input_size', type=int, default=4816)
        parser.add_argument('--hidden_size', type=int, default=512)
        parser.add_argument('--n_neurons', type=int, default=128)
        parser.add_argument('--num_cycles', type=int, default=1)
        parser.add_argument('--dropout', type=float, default=0.0)
        parser.add_argument('--lr', type=float, default=1e-4)
        return parser


class Encoder(base.AUROC):
    def __init__(self,
                 input_size: int = 4620,
                 hidden_size: int = 128,
                 output_size: int = 1,
                 num_layers: int = 3,
                 lr: float = 1e-4,
                 **kwargs: Any) -> None:
        super(Encoder, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.enc = base.Encoder(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.pred = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output = self.enc(x[0][1])
        return torch.sigmoid(self.pred(output)).squeeze(1)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--input_size', type=int, default=4816)
        parser.add_argument('--hidden_size', type=int, default=512)
        parser.add_argument('--num_layers', type=int, default=1)
        parser.add_argument('--lr', type=float, default=1e-4)
        return parser


class Cnn(base.AUROC):
    def __init__(self,
                 input_size: int = 4620,
                 length: int = 48,
                 depth: int = 2,
                 filter_size: int = 3,
                 n_filters: int = 64,
                 n_neurons: int = 64,
                 output_size: int = 1,
                 dropout: float = 0.1,
                 lr: float = 1e-4,
                 **kwargs: Any) -> None:
        super(Cnn, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.enc = base.Cnn(input_size=input_size, length=length, depth=depth, filter_size=filter_size,
                            n_filters=n_filters, n_neurons=n_neurons, dropout=dropout)
        self.pred = nn.Linear(n_neurons, output_size)

    def forward(self, x):
        output = self.enc(x[0][1])
        return torch.sigmoid(self.pred(output)).squeeze(1)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--input_size', type=int, default=4816)
        parser.add_argument('--length', type=int, default=12)
        parser.add_argument('--depth', type=int, default=2)
        parser.add_argument('--filter_size', type=int, default=3)
        parser.add_argument('--n_filters', type=int, default=64)
        parser.add_argument('--n_neurons', type=int, default=64)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--lr', type=float, default=1e-4)
        return parser


class Bert(base.BertAUROC):

    def __init__(self,
                 pretrained_bert_dir: str = '',
                 bert_size: int = 768,
                 output_size: int = 1,
                 num_training_steps: int = 1000,
                 warmup_proportion: float = 0.1,
                 lr: float = 5e-5,
                 **kwargs: Any) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.num_training_steps = num_training_steps
        self.warmup_proportion = warmup_proportion
        self.lr = lr
        self.enc = base.Bert(pretrained_bert_dir=pretrained_bert_dir)
        self.pred = nn.Linear(bert_size, output_size)

    def forward(self, x):
        output = self.enc(x[1:])
        return torch.sigmoid(self.pred(output)).squeeze(1)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--pretrained_bert_dir', type=str,
                            default='')
        parser.add_argument('--freeze', type=tuple, default=())
        parser.add_argument('--lr', type=float, default=1e-5)
        return parser


class MBertLstm(base.BertAUROC):
    def __init__(self,
                 pretrained_bert_dir: str = '',
                 ti_input_size: int = 96,
                 ti_norm_size: int = 64,
                 ts_input_size: int = 4816,
                 ts_norm_size: int = 1024,
                 n_neurons: int = 512,
                 bert_size: int = 768,
                 output_size: int = 1,
                 num_layers: int = 1,
                 dropout: int = 0.1,
                 num_training_steps: int = 1000,
                 warmup_proportion: float = 0.1,
                 lr: float = 5e-5,
                 **kwargs: Any) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.num_training_steps = num_training_steps
        self.warmup_proportion = warmup_proportion
        self.lr = lr

        self.ti_enc = nn.Linear(ti_input_size, ti_norm_size)

        self.ts_enc = base.Lstm(input_size=ts_input_size, hidden_size=ts_norm_size,
                                n_neurons=n_neurons, num_layers=num_layers)

        self.nt_enc = base.Bert(pretrained_bert_dir=pretrained_bert_dir)

        self.gate = base.Gate(bert_size, ti_norm_size, n_neurons, dropout)

        self.pred = nn.Linear(bert_size, output_size)

    def forward(self, x):
        ti = self.ti_enc(x[0][0])
        ts = self.ts_enc(x[0][1])
        nt = self.nt_enc(x[1:])
        fusion = self.gate(nt, ti, ts)
        return torch.sigmoid(self.pred(fusion)).squeeze(1)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--ti_input_size', type=int, default=97)
        parser.add_argument('--ti_norm_size', type=int, default=64)
        parser.add_argument('--ts_input_size', type=int, default=5500)
        parser.add_argument('--ts_norm_size', type=int, default=1024)
        parser.add_argument('--n_neurons', type=int, default=512)
        parser.add_argument('--num_layers', type=int, default=1)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--lr', type=float, default=1e-4)
        return parser


class MLstmBert(base.BertAUROC):
    def __init__(self,
                 pretrained_bert_dir: str = '',
                 ti_input_size: int = 96,
                 ti_norm_size: int = 64,
                 ts_input_size: int = 4816,
                 ts_norm_size: int = 1024,
                 n_neurons: int = 512,
                 bert_size: int = 768,
                 output_size: int = 1,
                 num_layers: int = 1,
                 dropout: int = 0.1,
                 num_training_steps: int = 1000,
                 warmup_proportion: float = 0.1,
                 lr: float = 5e-5,
                 **kwargs: Any) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.num_training_steps = num_training_steps
        self.warmup_proportion = warmup_proportion
        self.lr = lr

        self.ti_enc = nn.Linear(ti_input_size, ti_norm_size)

        self.ts_enc = base.Lstm(input_size=ts_input_size, hidden_size=ts_norm_size,
                                n_neurons=n_neurons, num_layers=num_layers)

        self.nt_enc = base.Bert(pretrained_bert_dir=pretrained_bert_dir)

        self.gate = base.Gate(n_neurons, ti_norm_size, bert_size, dropout)

        self.pred = nn.Linear(n_neurons, output_size)

    def forward(self, x):
        ti = self.ti_enc(x[0][0])
        ts = self.ts_enc(x[0][1])
        nt = self.nt_enc(x[1:])
        fusion = self.gate(ts, ti, nt)
        return torch.sigmoid(self.pred(fusion)).squeeze(1)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--ti_input_size', type=int, default=96)
        parser.add_argument('--ti_norm_size', type=int, default=64)
        parser.add_argument('--ts_input_size', type=int, default=4816)
        parser.add_argument('--ts_norm_size', type=int, default=512)
        parser.add_argument('--n_neurons', type=int, default=128)
        parser.add_argument('--num_layers', type=int, default=1)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--lr', type=float, default=1e-4)
        return parser


class MBertStar(base.BertAUROC):
    def __init__(self,
                 pretrained_bert_dir: str = '',
                 ti_input_size: int = 96,
                 ti_norm_size: int = 64,
                 ts_input_size: int = 4816,
                 ts_norm_size: int = 1024,
                 n_neurons: int = 512,
                 bert_size: int = 768,
                 output_size: int = 1,
                 num_cycles: int = 3,
                 dropout: int = 0.1,
                 num_training_steps: int = 1000,
                 warmup_proportion: float = 0.1,
                 lr: float = 5e-5,
                 **kwargs: Any) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.num_training_steps = num_training_steps
        self.warmup_proportion = warmup_proportion
        self.lr = lr

        self.ti_enc = nn.Linear(ti_input_size, ti_norm_size)

        self.ts_enc = base.Star(input_size=ts_input_size, hidden_size=ts_norm_size, n_neurons=n_neurons,
                                num_cycles=num_cycles, dropout=dropout)

        self.nt_enc = base.Bert(pretrained_bert_dir=pretrained_bert_dir)

        self.gate = base.Gate(bert_size, ti_norm_size, n_neurons, dropout)

        self.pred = nn.Linear(bert_size, output_size)

    def forward(self, x):
        ti = self.ti_enc(x[0][0])
        ts = self.ts_enc(x[0][1])
        nt = self.nt_enc(x[1:])
        fusion = self.gate(nt, ti, ts)
        return torch.sigmoid(self.pred(fusion)).squeeze(1)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--ti_input_size', type=int, default=97)
        parser.add_argument('--ti_norm_size', type=int, default=64)
        parser.add_argument('--ts_input_size', type=int, default=5500)
        parser.add_argument('--ts_norm_size', type=int, default=1024)
        parser.add_argument('--n_neurons', type=int, default=512)
        parser.add_argument('--num_cycles', type=int, default=6)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--lr', type=float, default=1e-4)
        return parser


class MStarBert(base.BertAUROC):
    def __init__(self,
                 pretrained_bert_dir: str = '',
                 ti_input_size: int = 96,
                 ti_norm_size: int = 64,
                 ts_input_size: int = 4816,
                 ts_norm_size: int = 1024,
                 n_neurons: int = 512,
                 bert_size: int = 768,
                 output_size: int = 1,
                 num_cycles: int = 3,
                 dropout: int = 0.1,
                 num_training_steps: int = 1000,
                 warmup_proportion: float = 0.1,
                 lr: float = 5e-5,
                 **kwargs: Any) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.num_training_steps = num_training_steps
        self.warmup_proportion = warmup_proportion
        self.lr = lr

        self.ti_enc = nn.Linear(ti_input_size, ti_norm_size)

        self.ts_enc = base.Star(input_size=ts_input_size, hidden_size=ts_norm_size, n_neurons=n_neurons,
                                num_cycles=num_cycles, dropout=dropout)

        self.nt_enc = base.Bert(pretrained_bert_dir=pretrained_bert_dir)

        self.gate = base.Gate(n_neurons, ti_norm_size, bert_size, dropout)

        self.pred = nn.Linear(n_neurons, output_size)

    def forward(self, x):
        ti = self.ti_enc(x[0][0])
        ts = self.ts_enc(x[0][1])
        nt = self.nt_enc(x[1:])
        fusion = self.gate(ts, ti, nt)
        return torch.sigmoid(self.pred(fusion)).squeeze(1)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--ti_input_size', type=int, default=97)
        parser.add_argument('--ti_norm_size', type=int, default=64)
        parser.add_argument('--ts_input_size', type=int, default=5500)
        parser.add_argument('--ts_norm_size', type=int, default=1024)
        parser.add_argument('--n_neurons', type=int, default=512)
        parser.add_argument('--num_cycles', type=int, default=6)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--lr', type=float, default=1e-4)
        return parser


class MBertEncoder(base.BertAUROC):
    def __init__(self,
                 pretrained_bert_dir: str = '',
                 ti_input_size: int = 96,
                 ti_norm_size: int = 64,
                 ts_input_size: int = 4816,
                 ts_norm_size: int = 768,
                 bert_size: int = 768,
                 output_size: int = 1,
                 num_layers: int = 3,
                 dropout: int = 0.1,
                 num_training_steps: int = 1000,
                 warmup_proportion: float = 0.1,
                 lr: float = 1e-5,
                 **kwargs: Any) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.num_training_steps = num_training_steps
        self.warmup_proportion = warmup_proportion
        self.lr = lr

        self.ti_enc = nn.Linear(ti_input_size, ti_norm_size)

        self.ts_enc = base.Encoder(input_size=ts_input_size, hidden_size=ts_norm_size, num_layers=num_layers)

        self.nt_enc = base.Bert(pretrained_bert_dir=pretrained_bert_dir)

        self.gate = base.Gate(bert_size, ti_norm_size, ts_norm_size, dropout)

        self.pred = nn.Linear(bert_size, output_size)

    def forward(self, x):
        ti = self.ti_enc(x[0][0])
        ts = self.ts_enc(x[0][1])
        nt = self.nt_enc(x[1:])
        fusion = self.gate(nt, ti, ts)
        return torch.sigmoid(self.pred(fusion)).squeeze(1)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--ti_input_size', type=int, default=97)
        parser.add_argument('--ti_norm_size', type=int, default=64)
        parser.add_argument('--ts_input_size', type=int, default=5500)
        parser.add_argument('--ts_norm_size', type=int, default=1024)
        parser.add_argument('--num_layers', type=int, default=6)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--lr', type=float, default=1e-4)
        return parser


class MEncoderBert(base.BertAUROC):
    def __init__(self,
                 pretrained_bert_dir: str = '',
                 ti_input_size: int = 96,
                 ti_norm_size: int = 64,
                 ts_input_size: int = 4816,
                 ts_norm_size: int = 1024,
                 bert_size: int = 768,
                 output_size: int = 1,
                 num_layers: int = 3,
                 dropout: int = 0.1,
                 num_training_steps: int = 1000,
                 warmup_proportion: float = 0.1,
                 lr: float = 5e-5,
                 **kwargs: Any) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.num_training_steps = num_training_steps
        self.warmup_proportion = warmup_proportion
        self.lr = lr

        self.ti_enc = nn.Linear(ti_input_size, ti_norm_size)

        self.ts_enc = base.Encoder(input_size=ts_input_size, hidden_size=ts_norm_size, num_layers=num_layers)

        self.nt_enc = base.Bert(pretrained_bert_dir=pretrained_bert_dir)

        self.gate = base.Gate(ts_norm_size, ti_norm_size, bert_size, dropout)

        self.pred = nn.Linear(ts_norm_size, output_size)

    def forward(self, x):
        ti = self.ti_enc(x[0][0])
        ts = self.ts_enc(x[0][1])
        nt = self.nt_enc(x[1:])
        fusion = self.gate(ts, ti, nt)
        return torch.sigmoid(self.pred(fusion)).squeeze(1)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--ti_input_size', type=int, default=97)
        parser.add_argument('--ti_norm_size', type=int, default=64)
        parser.add_argument('--ts_input_size', type=int, default=5500)
        parser.add_argument('--ts_norm_size', type=int, default=1024)
        parser.add_argument('--num_layers', type=int, default=6)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--lr', type=float, default=1e-4)
        return parser


class MBertCnn(base.BertAUROC):
    def __init__(self,
                 pretrained_bert_dir: str = '',
                 ti_input_size: int = 96,
                 ti_norm_size: int = 64,
                 ts_input_size: int = 4816,
                 length: int = 12,
                 ts_norm_size: int = 512,
                 filter_size: int = 3,
                 n_filters: int = 64,
                 bert_size: int = 768,
                 output_size: int = 1,
                 depth: int = 2,
                 dropout: int = 0.2,
                 num_training_steps: int = 1000,
                 warmup_proportion: float = 0.1,
                 lr: float = 5e-5,
                 **kwargs: Any) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.num_training_steps = num_training_steps
        self.warmup_proportion = warmup_proportion
        self.lr = lr

        self.ti_enc = nn.Linear(ti_input_size, ti_norm_size)

        self.ts_enc = base.Cnn(input_size=ts_input_size, length=length, depth=depth, filter_size=filter_size,
                               n_filters=n_filters, n_neurons=ts_norm_size, dropout=dropout)

        self.nt_enc = base.Bert(pretrained_bert_dir=pretrained_bert_dir)

        self.gate = base.Gate(bert_size, ti_norm_size, ts_norm_size, dropout)

        self.pred = nn.Linear(bert_size, output_size)

    def forward(self, x):
        ti = self.ti_enc(x[0][0])
        ts = self.ts_enc(x[0][1])
        nt = self.nt_enc(x[1:])
        fusion = self.gate(nt, ti, ts)
        return torch.sigmoid(self.pred(fusion)).squeeze(1)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--ti_input_size', type=int, default=97)
        parser.add_argument('--ti_norm_size', type=int, default=64)
        parser.add_argument('--ts_input_size', type=int, default=5500)
        parser.add_argument('--ts_norm_size', type=int, default=1024)
        parser.add_argument('--length', type=int, default=12)
        parser.add_argument('--depth', type=int, default=1)
        parser.add_argument('--filter_size', type=int, default=3)
        parser.add_argument('--n_filters', type=int, default=128)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--lr', type=float, default=1e-4)
        return parser


class MCnnBert(base.BertAUROC):
    def __init__(self,
                 pretrained_bert_dir: str = '',
                 ti_input_size: int = 96,
                 ti_norm_size: int = 64,
                 ts_input_size: int = 4816,
                 length: int = 12,
                 ts_norm_size: int = 512,
                 filter_size: int = 3,
                 n_filters: int = 64,
                 bert_size: int = 768,
                 output_size: int = 1,
                 depth: int = 2,
                 dropout: int = 0.2,
                 num_training_steps: int = 1000,
                 warmup_proportion: float = 0.1,
                 lr: float = 5e-5,
                 **kwargs: Any) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.num_training_steps = num_training_steps
        self.warmup_proportion = warmup_proportion
        self.lr = lr

        self.ti_enc = nn.Linear(ti_input_size, ti_norm_size)

        self.ts_enc = base.Cnn(input_size=ts_input_size, length=length, depth=depth, filter_size=filter_size,
                               n_filters=n_filters, n_neurons=ts_norm_size, dropout=dropout)

        self.nt_enc = base.Bert(pretrained_bert_dir=pretrained_bert_dir)

        self.gate = base.Gate(ts_norm_size, ti_norm_size, bert_size, dropout)

        self.pred = nn.Linear(ts_norm_size, output_size)

    def forward(self, x):
        ti = self.ti_enc(x[0][0])
        ts = self.ts_enc(x[0][1])
        nt = self.nt_enc(x[1:])
        fusion = self.gate(ts, ti, nt)
        return torch.sigmoid(self.pred(fusion)).squeeze(1)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--ti_input_size', type=int, default=97)
        parser.add_argument('--ti_norm_size', type=int, default=64)
        parser.add_argument('--ts_input_size', type=int, default=5500)
        parser.add_argument('--ts_norm_size', type=int, default=1024)
        parser.add_argument('--length', type=int, default=12)
        parser.add_argument('--depth', type=int, default=1)
        parser.add_argument('--filter_size', type=int, default=3)
        parser.add_argument('--n_filters', type=int, default=128)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--lr', type=float, default=1e-4)
        return parser


class LstmBertAttn(base.BertAUROC):
    def __init__(self,
                 pretrained_bert_dir: str = '',
                 ts_input_size: int = 4816,
                 ts_norm_size: int = 1024,
                 n_neurons: int = 512,
                 bert_size: int = 768,
                 output_size: int = 1,
                 num_layers: int = 1,
                 dropout: int = 0.1,
                 num_training_steps: int = 1000,
                 warmup_proportion: float = 0.1,
                 lr: float = 5e-5,
                 **kwargs: Any) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.num_training_steps = num_training_steps
        self.warmup_proportion = warmup_proportion
        self.lr = lr

        self.ts_enc = base.Lstm(input_size=ts_input_size, hidden_size=ts_norm_size,
                                n_neurons=n_neurons, num_layers=num_layers)

        self.nt_enc = base.Bert(pretrained_bert_dir=pretrained_bert_dir)

        self.attn = base.Attention(embed_dim=bert_size, dropout=dropout)

        self.pred = nn.Linear(n_neurons, output_size)

    def forward(self, x):
        ts = self.ts_enc(x[0])
        nt = self.nt_enc(x[1:])
        fusion = self.attn(nt, ts)
        return torch.sigmoid(self.pred(fusion)).squeeze(1)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--ts_input_size', type=int, default=4912)
        parser.add_argument('--ts_norm_size', type=int, default=512)
        parser.add_argument('--n_neurons', type=int, default=768)
        parser.add_argument('--num_layers', type=int, default=1)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--lr', type=float, default=1e-4)
        return parser


class BertLstmAttn(base.BertAUROC):
    def __init__(self,
                 pretrained_bert_dir: str = '',
                 ts_input_size: int = 4816,
                 ts_norm_size: int = 1024,
                 n_neurons: int = 512,
                 bert_size: int = 768,
                 output_size: int = 1,
                 num_layers: int = 1,
                 dropout: int = 0.1,
                 num_training_steps: int = 1000,
                 warmup_proportion: float = 0.1,
                 lr: float = 5e-5,
                 **kwargs: Any) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.num_training_steps = num_training_steps
        self.warmup_proportion = warmup_proportion
        self.lr = lr

        self.ts_enc = base.Lstm(input_size=ts_input_size, hidden_size=ts_norm_size,
                                n_neurons=n_neurons, num_layers=num_layers)

        self.nt_enc = base.Bert(pretrained_bert_dir=pretrained_bert_dir)

        self.attn = base.Attention(embed_dim=bert_size, dropout=dropout)

        self.pred = nn.Linear(n_neurons, output_size)

    def forward(self, x):
        ts = self.ts_enc(x[0])
        nt = self.nt_enc(x[1:])
        fusion = self.attn(ts, nt)
        return torch.sigmoid(self.pred(fusion)).squeeze(1)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--ts_input_size', type=int, default=4912)
        parser.add_argument('--ts_norm_size', type=int, default=512)
        parser.add_argument('--n_neurons', type=int, default=768)
        parser.add_argument('--num_layers', type=int, default=1)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--lr', type=float, default=1e-4)
        return parser


class LstmBertOuter(base.BertAUROC):
    def __init__(self,
                 pretrained_bert_dir: str = '',
                 ts_input_size: int = 4816,
                 ts_norm_size: int = 1024,
                 n_neurons: int = 512,
                 bert_size: int = 768,
                 output_size: int = 1,
                 num_layers: int = 1,
                 num_training_steps: int = 1000,
                 warmup_proportion: float = 0.1,
                 lr: float = 5e-5,
                 **kwargs: Any) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.num_training_steps = num_training_steps
        self.warmup_proportion = warmup_proportion
        self.lr = lr

        self.ts_enc = base.Lstm(input_size=ts_input_size, hidden_size=ts_norm_size,
                                n_neurons=n_neurons, num_layers=num_layers)

        self.nt_enc = base.Bert(pretrained_bert_dir=pretrained_bert_dir)

        self.outer = base.Outer(inp1_size=n_neurons, inp2_size=bert_size, n_neurons=n_neurons)

        self.pred = nn.Linear(n_neurons, output_size)

    def forward(self, x):
        ts = self.ts_enc(x[0])
        nt = self.nt_enc(x[1:])
        fusion = self.outer(ts, nt)
        return torch.sigmoid(self.pred(fusion)).squeeze(1)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--ts_input_size', type=int, default=4912)
        parser.add_argument('--ts_norm_size', type=int, default=512)
        parser.add_argument('--n_neurons', type=int, default=128)
        parser.add_argument('--num_layers', type=int, default=1)
        parser.add_argument('--dropout', type=float, default=0.1)
        parser.add_argument('--lr', type=float, default=1e-4)
        return parser
