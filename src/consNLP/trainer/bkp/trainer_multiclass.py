#import .config as config
from __future__ import absolute_import

import sys
import os

try:
    from dotenv import find_dotenv, load_dotenv
except:
    pass

import argparse
from tqdm import tqdm
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data import load_data, data_utils
from models import transformer_models, activations, layers, losses, scorers

from .trainer_utils import set_seed, _has_apex, _torch_lightning_available, _has_wandb, _torch_gpu_available, _num_gpus, _torch_tpu_available

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torchcontrib.optim import SWA
from torch.optim import Adam, SGD 
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, CyclicLR, \
                                     CosineAnnealingWarmRestarts

from transformers import AdamW, get_linear_schedule_with_warmup

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

if _torch_lightning_available:
    class PLMetric(NumpyMetric):

        def __init__(self, metric_name):
            super(PLMetric, self).__init__(metric_name)
            self.metric = scorers.get_scorer(metric_name)

        def forward(self,x,y):
            x = np.argmax(x,axis=-1)
            #y = np.argmax(y,axis=-1)
            # for debugging
            #print (x,y)
            #print (x.shape, y.shape)
            return self.metric(x,y)

def test_fn(data_loader, model, device):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)

            outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
            fin_outputs.extend(outputs.cpu().detach().numpy().tolist())
            
    return fin_outputs

def test_pl_trainer(data_loader, pltrainer):
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            outputs = pltrainer(d)
            fin_outputs.extend(outputs.numpy().tolist())
            
    return fin_outputs

class BasicTrainer:
    
    def __init__(self, model, train_data_loader, val_data_loader, device, model_description, test_data_loader=None):
        self.model = model
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.device = device
        self.description = model_description
        self.test_data_loader = test_data_loader

        self.model = self.model.to(self.device)

        self.print_stats()

    def print_stats(self):
        print ("[LOG] Total number of parameters to learn {}".format(sum(p.numel() for p in self.model.parameters() \
                                                                 if p.requires_grad)))

    def train_fn(self, data_loader):
        self.model.train()

        total_loss = 0
        pbar = tqdm(enumerate(data_loader), total=len(data_loader))
        for bi, d in pbar:
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            targets = d["targets"]

            ids = ids.to(self.device, dtype=torch.long)
            token_type_ids = token_type_ids.to(self.device, dtype=torch.long)
            mask = mask.to(self.device, dtype=torch.long)
            targets = targets.to(self.device, dtype=torch.float)

            self.optimizer.zero_grad()

            if self.is_amp:
                #with amp.autocast():
                outputs = self.model(ids=ids, mask=mask, token_type_ids=token_type_ids)
                loss = self.loss_fn(outputs, targets)
            else:
                outputs = self.model(ids=ids, mask=mask, token_type_ids=token_type_ids)
                loss = self.loss_fn(outputs, targets)

            #nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            if self.scaler:
                #self.scaler.scale(loss.backward())
                #self.scaler.step(self.optimizer.step())
                #self.scaler.update()
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            else:
                loss.backward()
                if self.num_tpus == 0:
                    self.optimizer.step()
                else:
                    xm.optimizer_step(self.optimizer, barrier=True)
                self.scheduler.step()

            total_loss += loss

            pbar.set_description("Loss: {}".format(loss.cpu().detach().numpy()))
        return total_loss

    def eval_fn(self, data_loader):
        self.model.eval()
        fin_targets = []
        fin_outputs = []

        total_loss = 0
        with torch.no_grad():
            pbar = tqdm(enumerate(data_loader), total=len(data_loader))
            for bi, d in pbar:
                ids = d["ids"]
                token_type_ids = d["token_type_ids"]
                mask = d["mask"]
                targets = d["targets"]

                ids = ids.to(self.device, dtype=torch.long)
                token_type_ids = token_type_ids.to(self.device, dtype=torch.long)
                mask = mask.to(self.device, dtype=torch.long)
                targets = targets.to(self.device, dtype=torch.float)

                outputs = self.model(ids=ids, mask=mask, token_type_ids=token_type_ids)
                loss = self.loss_fn(outputs,targets)
                total_loss += loss

                pbar.set_description("Loss: {}".format(loss.cpu().detach().numpy()))
                fin_targets.extend(targets.cpu().detach().numpy().tolist())
                fin_outputs.extend(outputs.cpu().detach().numpy().tolist())
                
        return fin_outputs, total_loss, fin_targets

    def train(self, epochs, lr, metric, loss_fn, optimizer, scheduler, MODEL_PATH, num_gpus=0, max_grad_norm=1, early_stopping_rounds=3, \
                    snapshot_ensemble=False, is_amp=False, seed=42):

        self.metric = metric
        self.loss_fn = losses.get_loss(loss_fn)
        self.optimizer = optimizer
        self.scheduler = scheduler

        set_seed(seed)

        self.args = {'description': self.description, 'loss': loss_fn, 'epochs': epochs, 'learning_rate': lr, \
                    'optimizer': self.optimizer.__class__.__name__, 'scheduler': self.scheduler.__class__.__name__}

        if _has_wandb and not _torch_tpu_available:
            wandb.init(project="WNUT-Task-2",config=self.args)
            wandb.watch(self.model)
        else:
            with open(os.path.join(MODEL_PATH, 'parameters.json'),'w') as outfile:
                json.dump(self.args, outfile)

        scorer = scorers.get_scorer(self.metric)
        #loss_fn = losses.get_loss(self.loss_fn)

        if _num_gpus > 1 and (num_gpus != 0 and num_gpus != 1):
            self.model = nn.DataParallel(self.model)

        if is_amp and _has_apex:
            #scaler = amp.GradScaler()
            scaler = 1
        else:
            scaler = None

        if os.path.exists(os.path.join(MODEL_PATH,'model.bin')):
            try:
                try:
                    #self.model.load_state_dict(torch.load(os.path.join(MODEL_PATH,'model.bin')))
                    self.model = load_state_dict(torch.load(os.path.join(MODEL_PATH,'model.bin')))
                except:
                    self.model = torch.load(os.path.join(MODEL_PATH,'model.bin'))
                print (self.model)
                print ("Loaded model from previous checkpoint")
            except:
                pass

        self.is_amp = is_amp
        self.max_grad_norm = max_grad_norm
        self.scaler = scaler

        best_metric = 0
        bad_epochs = 0
        if snapshot_ensemble:
            test_outputs = []

        for epoch in range(epochs):
            if bad_epochs < early_stopping_rounds:
                train_loss = self.train_fn(self.train_data_loader)

                train_outputs, train_loss, train_targets = self.eval_fn(self.train_data_loader)
                val_outputs, val_loss, val_targets = self.eval_fn(self.val_data_loader)
                
                train_loss = train_loss/len(self.train_data_loader)
                val_loss = val_loss/len(self.val_data_loader)

                train_outputs = np.argmax(train_outputs,axis=-1)
                val_outputs = np.argmax(val_outputs,axis=-1) #np.array(outputs) >= 0.5

                train_metric = scorer(train_targets, train_outputs)
                val_metric = scorer(val_targets,val_outputs)

                print("Train loss = {} Train metric = {} Val loss = {} Val metric = {}".format(train_loss,train_metric,val_loss,val_metric))
                
                if val_metric > best_metric:
                    #torch.save(self.model.state_dict(), os.path.join(MODEL_PATH,'model.bin'))
                    torch.save(self.model, os.path.join(MODEL_PATH,'model.bin'))
                    best_metric = val_metric
                    bad_epochs = 0

                else:
                    bad_epochs += 1

                if snapshot_ensemble and self.test_data_loader:
                    test_out = test_fn(self.test_data_loader, self.model, self.device)

                    test_outputs += [np.expand_dims(np.array(test_out),axis=-1)]

            stats.update({"epoch_{}".format(epoch): {"train_loss": train_loss, "val_loss": val_loss, "train_metric": train_metric, "val_metric": val_metric}})

            if _has_wandb:
                wandb.log({"train_loss": train_loss, "val_loss": val_loss, "train_metric": train_metric, "val_metric": val_metric})

        if self.test_data_loader:
            if snapshot_ensemble:
                self.test_output = np.argmax(np.concatenate(test_outputs,axis=-1).mean(axis=-1),axis=-1)
            else:
                self.test_output = np.argmax(test_fn(self.test_data_loader, self.model, self.device),axis=-1)
        else:
            self.test_output = []
            
        if _has_wandb == False:
            with open(os.path.join(MODEL_PATH,'all_stats.json'), 'w') as outfile:
                json.dump(stats, outfile)

            with open(os.path.join(MODEL_PATH,'final_stats.json'), 'w') as outfile:
                d = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "train_metric": train_metric, "val_metric": val_metric}
                json.dump(d, outfile)

class PLTrainer(pl.LightningModule):

    def __init__(self, num_train_steps, model, metric, loss_fn, lr, seed=42):
        super(PLTrainer, self).__init__()

        seed_everything(seed)

        self.model = model
        self.loss_fn = losses.get_loss(loss_fn)
        self.metric_name = metric
        self.metric = PLMetric(metric)
        self.num_train_steps = num_train_steps
        self.lr = lr

        self.save_hyperparameters()

        self.print_stats()

    def print_stats(self):
        print ("[LOG] Total number of parameters to learn {}".format(sum(p.numel() for p in self.model.parameters() \
                                                                 if p.requires_grad)))

    def forward(self, x):

        return self.model(ids=x["ids"], mask=x["mask"], token_type_ids=x["token_type_ids"])

    def training_step(self, batch, batch_idx):
        di = batch
        ids = di["ids"]
        token_type_ids = di["token_type_ids"]
        mask = di["mask"]
        targets = di["targets"]

        outputs = self.model(ids=ids, mask=mask, token_type_ids=token_type_ids)
        loss = self.loss_fn(outputs, targets)
        metric_value = self.metric(outputs,targets)

        tensorboard_logs = {'train_loss': loss, "train {}".format(self.metric_name): metric_value}

        return {'loss': loss, 'train_metric': metric_value, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        di = batch
        ids = di["ids"]
        token_type_ids = di["token_type_ids"]
        mask = di["mask"]
        targets = di["targets"]

        outputs = self.model(ids=ids, mask=mask, token_type_ids=token_type_ids)
        loss = self.loss_fn(outputs, targets)
        metric_value = self.metric(outputs,targets)

        tensorboard_logs = {'val_loss': loss, "val {}".format(self.metric_name): metric_value}

        return {'val_loss': loss, 'val_metric': metric_value, 'log': tensorboard_logs}

    def configure_optimizers(self):

        param_optimizer = list(self.model.named_parameters())
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

        optimizer = AdamW(optimizer_parameters, lr=self.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=self.num_train_steps)

        return [optimizer], [{'scheduler': scheduler}]

    def training_epoch_end(self, outputs):
        train_loss_mean = torch.stack([x['loss'] for x in outputs]).mean()
        train_metric_mean = torch.stack([x['train_metric'] for x in outputs]).mean()
        print ("Train loss = {} Train metric = {}".format(train_loss_mean,train_metric_mean))

        return {'train_loss': train_loss_mean, 'train_metric': train_metric_mean}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_metric_mean = torch.stack([x['val_metric'] for x in outputs]).mean()
        print ("val loss = {} val metric = {} ".format(val_loss_mean,val_metric_mean))

        return {'val_loss': val_loss_mean, 'val_metric': val_metric_mean}

    
