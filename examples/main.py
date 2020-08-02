from __future__ import absolute_import

import sys
import os

try:
    from dotenv import find_dotenv, load_dotenv
except:
    pass

import argparse

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torchcontrib.optim import SWA
from torch.optim import Adam, SGD 
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, CyclicLR, \
                                     CosineAnnealingWarmRestarts

from consNLP.data import load_data, data_utils
from consNLP.models import transformer_models, activations, layers, losses, scorers
from consNLP.visualization import visualize
from consNLP.trainer import BasicTrainer, PLTrainer, test_pl_trainer
from consNLP.trainer_utils import set_seed, _has_apex, _torch_lightning_available, _has_wandb, _torch_gpu_available, _num_gpus, _torch_tpu_available
from consNLP.preprocessing.custom_tokenizer import BERTweetTokenizer

if _has_apex:
    #from torch.cuda import amp
    from apex import amp

if _torch_tpu_available:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp

if _has_wandb:
    import wandb
    try:
        load_dotenv(find_dotenv())
        wandb.login(key=os.environ['WANDB_API_KEY'])
    except:
        _has_wandb = False

if _torch_lightning_available:
    import pytorch_lightning as pl
    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.loggers import WandbLogger
    from pytorch_lightning.metrics.metric import NumpyMetric
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback

import tokenizers
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup, AutoConfig

def run(args):

    train_df = load_data.load_custom_text_as_pd(args.train_data,sep='\t',header=True, \
                              text_column=['Text'],target_column=['Label'])
    val_df = load_data.load_custom_text_as_pd(args.val_data,sep='\t', header=True, \
                          text_column=['Text'],target_column=['Label'])

    train_df = pd.DataFrame(train_df,copy=False)
    val_df = pd.DataFrame(val_df,copy=False)

    model_save_dir = args.model_save_path
    try:
        os.makedirs(model_save_dir)
    except OSError:
        pass

    train_df.labels, label2idx = data_utils.convert_categorical_label_to_int(train_df.labels, \
                                                             save_path=os.path.join(model_save_dir,'label2idx.pkl'))

    val_df.labels, _ = data_utils.convert_categorical_label_to_int(val_df.labels, \
                                                             save_path=os.path.join(model_save_dir,'label2idx.pkl'))

    if args.berttweettokenizer_path:
        tokenizer = BERTweetTokenizer(args.berttweettokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.transformer_model_pretrained_path)

    if not args.berttweettokenizer_path:
        bpetokenizer = tokenizers.ByteLevelBPETokenizer(args.bpe_vocab_path, \
                                                args.bpe_merges_path)
    else:
        bpetokenizer = None

    if bpetokenizer:
        train_corpus = data_utils.Corpus(train_df.copy(),tokenizer=bpetokenizer.encode)
        val_corpus = data_utils.Corpus(val_df.copy(),tokenizer=bpetokenizer.encode)
    else:
        train_corpus = data_utils.Corpus(train_df.copy(),tokenizer=tokenizer.tokenize)
        val_corpus = data_utils.Corpus(val_df.copy(),tokenizer=tokenizer.tokenize)

    train_dataset = data_utils.TransformerDataset(train_corpus.data.words, bpetokenizer=bpetokenizer, tokenizer=tokenizer, MAX_LEN=args.max_text_len, \
                  target_label=train_corpus.data.labels, sequence_target=False, target_text=None, conditional_label=None, conditional_all_labels=None)

    val_dataset = data_utils.TransformerDataset(val_corpus.data.words, bpetokenizer=bpetokenizer, tokenizer=tokenizer, MAX_LEN=args.max_text_len, \
                  target_label=val_corpus.data.labels, sequence_target=False, target_text=None, conditional_label=None, conditional_all_labels=None)

    if _torch_tpu_available and args.use_TPU:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
          train_dataset,
          num_replicas=xm.xrt_world_size(),
          rank=xm.get_ordinal(),
          shuffle=True
        )

        val_sampler = torch.utils.data.distributed.DistributedSampler(
          val_dataset,
          num_replicas=xm.xrt_world_size(),
          rank=xm.get_ordinal(),
          shuffle=False
        )

    if _torch_tpu_available and args.use_TPU:
        train_data_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.train_batch_size, sampler=train_sampler,
            drop_last=True,num_workers=2)

        val_data_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.eval_batch_size, sampler=val_sampler,
            drop_last=False,num_workers=1)
    else:
        train_data_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.train_batch_size)

        val_data_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.eval_batch_size)

    config = AutoConfig.from_pretrained(args.transformer_config_path, output_hidden_states=True, output_attentions=True)
    basemodel = AutoModel.from_pretrained(args.transformer_model_pretrained_path,config=config)
    model = transformer_models.TransformerWithCLS(basemodel)

    if args.use_torch_trainer:
        device = torch.device("cuda" if _torch_gpu_available and args.use_gpu else "cpu")
        if _torch_tpu_available and args.use_TPU:
            device=xm.xla_device()

        if args.use_TPU and _torch_tpu_available and args.num_tpus > 1:
            train_data_loader = torch_xla.distributed.parallel_loader.ParallelLoader(train_data_loader, [device])
            train_data_loader = train_data_loader.per_device_loader(device)


        trainer = BasicTrainer(model, train_data_loader, val_data_loader, device, args.transformer_model_pretrained_path, test_data_loader=val_data_loader)

        param_optimizer = list(trainer.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.001,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        num_train_steps = int(len(train_data_loader) * args.epochs)

        if _torch_tpu_available and args.use_TPU:
            optimizer = AdamW(optimizer_parameters, lr=args.lr*xm.xrt_world_size())
        else:
            optimizer = AdamW(optimizer_parameters, lr=args.lr)
        
        if args.use_apex and _has_apex:
            model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
            

        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)

        def _mp_fn(rank, flags, trainer, epochs, lr, metric, loss_function, optimizer, scheduler, model_save_path, num_gpus, num_tpus,  \
                    max_grad_norm, early_stopping_rounds, snapshot_ensemble, is_amp, seed):
            torch.set_default_tensor_type('torch.FloatTensor')
            a = trainer.train(epochs, lr, metric, loss_function, optimizer, scheduler, model_save_path, num_gpus, num_tpus,  \
                    max_grad_norm, early_stopping_rounds, snapshot_ensemble, is_amp, seed)

        FLAGS = {}
        if _torch_tpu_available and args.use_TPU:
            xmp.spawn(_mp_fn, args=(FLAGS, trainer, args.epochs, args.lr, args.metric, args.loss_function, optimizer, scheduler, args.model_save_path, args.num_gpus, args.num_tpus, \
                     1, 3, False, args.use_apex, args.seed), nprocs=8, start_method='fork')
        else:
            trainer.train(args.epochs, args.lr, args.metric, args.loss_function, optimizer, scheduler, args.model_save_path, args.num_gpus, args.num_tpus,  \
                    max_grad_norm=1, early_stopping_rounds=3, snapshot_ensemble=False, is_amp=args.use_apex, seed=args.seed)

        test_output = trainer.test_output

    elif args.use_lightning_trainer and _torch_lightning_available:
        from pytorch_lightning import Trainer, seed_everything
        seed_everything(args.seed)

        log_args = {'description': args.transformer_model_pretrained_path, 'loss': args.loss_function, 'epochs': args.epochs, 'learning_rate': args.lr}

        if _has_wandb and not _torch_tpu_available and args.wandb_logging:
            wandb.init(project="WNUT-Task-2",config=log_args)
            wandb_logger = WandbLogger()

        checkpoint_callback = ModelCheckpoint(
                    filepath=args.model_save_path,
                    save_top_k=1,
                    verbose=True,
                    monitor='val_loss',
                    mode='min'
                    )
        earlystop = EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                   verbose=False,
                   mode='min'
                   )

        if args.use_gpu and _torch_gpu_available:
            print ("using GPU")
            if args.wandb_logging:
                if _has_apex:
                    trainer = Trainer(gpus=args.num_gpus, max_epochs=args.epochs, logger=wandb_logger, precision=16, \
                                checkpoint_callback=checkpoint_callback, callbacks=[earlystop])
                else:
                    trainer = Trainer(gpus=args.num_gpus, max_epochs=args.epochs, logger=wandb_logger, \
                                checkpoint_callback=checkpoint_callback, callbacks=[earlystop])
            else:
                if _has_apex:
                    trainer = Trainer(gpus=args.num_gpus, max_epochs=args.epochs, precision=16, \
                                checkpoint_callback=checkpoint_callback, callbacks=[earlystop])
                else:
                    trainer = Trainer(gpus=args.num_gpus, max_epochs=args.epochs, \
                                checkpoint_callback=checkpoint_callback, callbacks=[earlystop])

        elif args.use_TPU and _torch_tpu_available:
            print ("using TPU")
            if _has_apex:
                trainer = Trainer(num_tpu_cores=args.num_tpus, max_epochs=args.epochs, precision=16, \
                            checkpoint_callback=checkpoint_callback, callbacks=[earlystop])
            else:
                trainer = Trainer(num_tpu_cores=args.num_tpus, max_epochs=args.epochs, \
                            checkpoint_callback=checkpoint_callback, callbacks=[earlystop])

        else:
            print ("using CPU")
            if args.wandb_logging:
                if _has_apex:
                    trainer = Trainer(max_epochs=args.epochs, logger=wandb_logger, precision=16, \
                            checkpoint_callback=checkpoint_callback, callbacks=[earlystop])
                else:
                    trainer = Trainer(max_epochs=args.epochs, logger=wandb_logger, \
                            checkpoint_callback=checkpoint_callback, callbacks=[earlystop])
            else:
                if _has_apex:
                    trainer = Trainer(max_epochs=args.epochs, precision=16, \
                            checkpoint_callback=checkpoint_callback, callbacks=[earlystop])
                else:
                    trainer = Trainer(max_epochs=args.epochs, checkpoint_callback=checkpoint_callback, callbacks=[earlystop])

        num_train_steps = int(len(train_data_loader) * args.epochs)

        pltrainer = PLTrainer(num_train_steps, model, args.metric, args.loss_function, args.lr, seed=42)

        #try:
        #    print ("Loaded model from previous checkpoint")
        #    pltrainer = PLTrainer.load_from_checkpoint(args.model_save_path)
        #except:
        #    pass

        trainer.fit(pltrainer, train_data_loader, val_data_loader) 

        test_output = (pltrainer, val_data_loader)

    idx2label = {value:key for key,value in label2idx.items()}

    test_output = [idx2label[i] for i in test_output]

    return test_output 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Torch trainer function',conflict_handler='resolve')

    parser.add_argument('--train_data', type=str, default='../data/raw/COVID19Tweet-master/train.tsv', required=False,
                        help='train data')
    parser.add_argument('--val_data', type=str, default='../data/raw/COVID19Tweet-master/valid.tsv', required=False,
                        help='validation data')
    parser.add_argument('--test_data', type=str, default=None, required=False,
                        help='test data')

    parser.add_argument('--transformer_model_pretrained_path', type=str, default='roberta-base', required=False,
                        help='transformer model pretrained path or huggingface model name')
    parser.add_argument('--transformer_config_path', type=str, default='roberta-base', required=False,
                        help='transformer config file path or huggingface model name')
    parser.add_argument('--transformer_tokenizer_path', type=str, default='roberta-base', required=False,
                        help='transformer tokenizer file path or huggingface model name')
    parser.add_argument('--bpe_vocab_path', type=str, default='../models/vocab.json', required=False,
                        help='bytepairencoding vocab file path')
    parser.add_argument('--bpe_merges_path', type=str, default='../models/merges.txt', required=False,
                        help='bytepairencoding merges file path')
    parser.add_argument('--berttweettokenizer_path', type=str, default='../models/BERTweet/', required=False,
                        help='BERTweet tokenizer path')

    parser.add_argument('--max_text_len', type=int, default=200, required=False,
                        help='maximum length of text')
    parser.add_argument('--epochs', type=int, default=5, required=False,
                        help='number of epochs')
    parser.add_argument('--lr', type=float, default=.00003, required=False,
                        help='learning rate')
    parser.add_argument('--loss_function', type=str, default='bcelogit', required=False,
                        help='loss function')
    parser.add_argument('--metric', type=str, default='f1', required=False,
                        help='scorer metric')

    parser.add_argument('--use_lightning_trainer', type=bool, default=False, required=False,
                        help='if lightning trainer needs to be used')
    parser.add_argument('--use_torch_trainer', type=bool, default=False, required=False,
                        help='if custom torch trainer needs to be used')
    parser.add_argument('--use_apex', type=bool, default=False, required=False,
                        help='if apex needs to be used')
    parser.add_argument('--use_gpu', type=bool, default=False, required=False,
                        help='GPU mode')
    parser.add_argument('--use_TPU', type=bool, default=False, required=False,
                        help='TPU mode')
    parser.add_argument('--num_gpus', type=int, default=1, required=False,
                        help='Number of GPUs')
    parser.add_argument('--num_tpus', type=int, default=1, required=False,
                        help='Number of TPUs')

    parser.add_argument('--train_batch_size', type=int, default=16, required=False,
                        help='train batch size')
    parser.add_argument('--eval_batch_size', type=int, default=16, required=False,
                        help='eval batch size')

    parser.add_argument('--model_save_path', type=str, default='../models/model2_roberta-base/', required=False,
                        help='seed')

    parser.add_argument('--wandb_logging', type=bool, default=False, required=False,
                        help='wandb logging needed')

    parser.add_argument('--seed', type=int, default=42, required=False,
                        help='seed')

    args, _ = parser.parse_known_args()

    print ("Wandb Logging: {} GPU: {} Pytorch Lighning: {} TPU: {} Apex: {}".format(\
                _has_wandb and args.wandb_logging, _torch_gpu_available, _torch_lightning_available, _torch_tpu_available, _has_apex))

    test_output = run(args)
