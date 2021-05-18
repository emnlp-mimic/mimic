from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import Recall_models as Models
# import AUROC_models as Models
from data_module import MimicDataModule


models = {
    'Lstm': Models.Lstm,
    'Star': Models.Star,
    'Encoder': Models.Encoder,
    'Cnn': Models.Cnn,
    'Bert': Models.Bert,
    'MBertLstm': Models.MBertLstm,
    'MBertStar': Models.MBertStar,
    'MBertEncoder': Models.MBertEncoder,
    'MBertCnn': Models.MBertCnn,
    'MLstmBert': Models.MLstmBert,
    'MStarBert': Models.MStarBert,
    'MEncoderBert': Models.MEncoderBert,
    'MCnnBert': Models.MCnnBert,
    # 'LstmBertAttn': Models.LstmBertAttn,
    # 'BertLstmAttn': Models.BertLstmAttn,
    # 'LstmBertOuter': Models.LstmBertOuter,
    'EncoderBertAttn': Models.EncoderBertAttn,
    'BertEncoderAttn': Models.BertEncoderAttn,
    'BertEncoderOuter': Models.BertEncoderOuter,
    'Line': Models.Line
}


def parse_args(args=None):
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = MimicDataModule.add_argparse_args(parser)
    parser.add_argument('--model', type=str, default='Line')
    temp_args, _ = parser.parse_known_args()
    parser = models[temp_args.model].add_model_specific_args(parser)
    parser.add_argument('--seed', type=int, default=7)
    return parser.parse_args(args)


def main(args):
    pl.seed_everything(args.seed)
    dm = MimicDataModule.from_argparse_args(args)
    model = models[args.model](**vars(args))
    name = args.model
    project = f'{args.task}_{args.duration}_{args.model}'
    print(f'Run {project}: {name}')
    wandb_logger = WandbLogger(project=project, name=name, offline=False)
    trainer = pl.Trainer.from_argparse_args(args, logger=wandb_logger, callbacks=[
        EarlyStopping(monitor="score", mode="max"),
        ModelCheckpoint(monitor="score", mode='max',
                        dirpath=f'')
        ], gpus=-1, accelerator='dp', gradient_clip_val=1.0)
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)


if __name__ == '__main__':
    arguments = parse_args()
    main(arguments)
