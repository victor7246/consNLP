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

def test_fn(data_loader, model, device, final_activation=None):
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
            if final_activation == 'sigmoid':
                fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
            else:
                fin_outputs.extend(outputs.cpu().detach().numpy().tolist())
            
    return fin_outputs

def test_fn_multi(data_loader1, data_loader2, model, device, final_activation=None):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for bi, (d1, d2) in tqdm(enumerate(zip(data_loader1,data_loader2)), total=len(data_loader1)):
            ids1 = d1["ids"]
            token_type_ids1= d1["token_type_ids"]
            mask1 = d1["mask"]

            ids2 = d2["ids"]
            token_type_ids2= d2["token_type_ids"]
            mask2 = d2["mask"]

            ids1 = ids1.to(device, dtype=torch.long)
            token_type_ids1 = token_type_ids1.to(device, dtype=torch.long)
            mask1 = mask1.to(device, dtype=torch.long)

            ids2 = ids2.to(device, dtype=torch.long)
            token_type_ids2 = token_type_ids2.to(device, dtype=torch.long)
            mask2 = mask2.to(device, dtype=torch.long)

            outputs = model(ids1,ids2, mask1,mask2,token_type_ids1,token_type_ids2)
            if final_activation == 'sigmoid':
                fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
            else:
                fin_outputs.extend(outputs.cpu().detach().numpy().tolist())
            
    return fin_outputs

def test_fn_qa(data_loader, model, device, final_activation=None):
    model.eval()
    fin_outputs_start = []
    fin_outputs_end = []

    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)

            outputs_start, outputs_end = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
            if final_activation == 'sigmoid':
                fin_outputs_start.extend(torch.sigmoid(outputs_start).cpu().detach().numpy().tolist())
                fin_outputs_end.extend(torch.sigmoid(outputs_end).cpu().detach().numpy().tolist())
            else:
                fin_outputs_start.extend(outputs_start.cpu().detach().numpy().tolist())
                fin_outputs_end.extend(outputs_end.cpu().detach().numpy().tolist())
            
    return fin_outputs_start, fin_outputs_end

def test_pl_trainer(data_loader, pltrainer, final_activation=None):
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            outputs = pltrainer(d)
            if final_activation == 'sigmoid':
                fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
            else:
                fin_outputs.extend(outputs.cpu().detach().numpy().tolist())
            
    return fin_outputs

class BasicTrainer:
    
    def __init__(self, model, train_data_loader, val_data_loader, device, model_description, final_activation=None, test_data_loader=None):
        self.model = model
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.device = device
        self.description = model_description
        self.test_data_loader = test_data_loader
        self.final_activation = final_activation

        self.model.to(self.device)

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

            ids = ids.to(self.device)
            token_type_ids = token_type_ids.to(self.device)
            mask = mask.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            if self.is_amp:
                #with amp.autocast():
                outputs = self.model(ids=ids, mask=mask, token_type_ids=token_type_ids)
                if self.loss_fn.__class__.__name__ == 'masked_CELoss':
                    loss = self.loss_fn(outputs, targets, mask)
                else:
                    loss = self.loss_fn(outputs, targets)
            else:
                outputs = self.model(ids=ids, mask=mask, token_type_ids=token_type_ids)
                if self.loss_fn.__class__.__name__ == 'masked_CELoss':
                    loss = self.loss_fn(outputs, targets, mask)
                else:
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
            with np.printoptions(precision=3):
                v = round(loss.cpu().detach().numpy().item(), 3)
                pbar.set_description("Current training Loss {}".format(v))

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

                ids = ids.to(self.device)
                token_type_ids = token_type_ids.to(self.device)
                mask = mask.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(ids=ids, mask=mask, token_type_ids=token_type_ids)
                if self.loss_fn.__class__.__name__ == 'masked_CELoss':
                    loss = self.loss_fn(outputs, targets, mask)
                else:
                    loss = self.loss_fn(outputs, targets)

                total_loss += loss

                with np.printoptions(precision=3):
                    v = round(loss.cpu().detach().numpy().item(), 3)
                    pbar.set_description("Current eval Loss {}".format(v))

                fin_targets.extend(targets.cpu().detach().numpy().tolist())
                if self.final_activation == 'sigmoid':
                    fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
                else:
                    fin_outputs.extend(outputs.cpu().detach().numpy().tolist())
                
        return fin_outputs, total_loss, fin_targets

    def train(self, epochs, lr, scorer, loss_fn, optimizer, scheduler, MODEL_PATH, num_gpus=0, num_tpus=0, max_grad_norm=1, early_stopping_rounds=3, \
                    snapshot_ensemble=False, is_amp=False, use_wandb=False, seed=42):

        self.scorer = scorer
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_gpus = num_gpus
        self.num_tpus = num_tpus

        set_seed(seed)

        self.args = {'description': self.description, 'loss': loss_fn.__class__.__name__, 'epochs': epochs, 'learning_rate': lr, \
                    'optimizer': self.optimizer.__class__.__name__, 'scheduler': self.scheduler.__class__.__name__}

        if use_wandb and not _torch_tpu_available:
            wandb.init(project="Project",config=self.args)
            wandb.watch(self.model)
        else:
            with open(os.path.join(MODEL_PATH, 'parameters.json'),'w') as outfile:
                json.dump(self.args, outfile)

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
                    self.model.load_state_dict(torch.load(os.path.join(MODEL_PATH,'model.bin')))
                except:
                    self.model = torch.load(os.path.join(MODEL_PATH,'model.bin'))

                #print (self.model)
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
            val_outputs = []

        stats = {}
        for epoch in range(epochs):
            if bad_epochs < early_stopping_rounds:
                train_loss = self.train_fn(self.train_data_loader)

                print ("Running evaluation on whole training data")
                train_out, train_loss, train_targets = self.eval_fn(self.train_data_loader)
                print ("Running evaluation on validation data")
                val_out, val_loss, val_targets = self.eval_fn(self.val_data_loader)
                
                train_loss = train_loss/len(self.train_data_loader)
                val_loss = val_loss/len(self.val_data_loader)

                #train_out = np.round(train_out)

                if snapshot_ensemble:
                    val_outputs += [np.expand_dims(np.array(val_out),axis=-1)]
                    val_out = np.concatenate(val_outputs,axis=-1).mean(axis=-1)
                #else:
                #    val_out = np.round(val_out) #np.array(outputs) >= 0.5

                train_metric = scorer(train_targets, train_out)
                val_metric = scorer(val_targets,val_out)

                with np.printoptions(precision=3):
                    print("Train loss = {} Train metric = {} Val loss = {} Val metric = {}".format(round(train_loss.detach().cpu().numpy().item(), 3),round(train_metric,3),\
                        round(val_loss.detach().cpu().numpy().item(), 3),round(val_metric, 3)))
                
                if val_metric > best_metric:
                    #torch.save(self.model.state_dict(), os.path.join(MODEL_PATH,'model.bin'))
                    torch.save(self.model, os.path.join(MODEL_PATH,'model.bin'))
                    best_metric = val_metric
                    bad_epochs = 0

                else:
                    bad_epochs += 1

                if snapshot_ensemble and self.test_data_loader:
                    test_out = test_fn(self.test_data_loader, self.model, self.device, self.final_activation)

                    test_outputs += [np.expand_dims(np.array(test_out),axis=-1)]

            stats.update({"epoch_{}".format(epoch): {"train_loss": round(train_loss.detach().cpu().numpy().item(), 3), "train_metric": round(train_metric,3), \
                "val_loss": round(val_loss.detach().cpu().numpy().item(), 3),  "val_metric": round(val_metric,3)}})

            if use_wandb:
                wandb.log({"train_loss": train_loss, "val_loss": val_loss, "train_metric": train_metric, "val_metric": val_metric})

        if self.test_data_loader:
            if snapshot_ensemble:
                self.test_output = np.concatenate(test_outputs,axis=-1).mean(axis=-1)
            else:
                self.test_output = test_fn(self.test_data_loader, self.model, self.device, self.final_activation)
        else:
            self.test_output = []

        #with np.printoptions(precision=3):
        #    print (stats)

        if use_wandb == False:
            with open(os.path.join(MODEL_PATH,'all_stats.json'), 'w') as outfile:
                json.dump(stats, outfile)

            with open(os.path.join(MODEL_PATH,'final_stats.json'), 'w') as outfile:
                d = {"epoch": epoch, "train_loss": round(train_loss.detach().cpu().numpy().item(), 3), "train_metric": round(train_metric,3), \
                "val_loss": round(val_loss.detach().cpu().numpy().item(), 3),  "val_metric": round(val_metric,3)}
                json.dump(d, outfile)

class BasicMultiTrainer:
    
    def __init__(self, model, train_data_loader1, train_data_loader2, val_data_loader1, val_data_loader2, device, model_description, final_activation=None, \
                 test_data_loader1=None, test_data_loader2=None):
        self.model = model
        self.train_data_loader1 = train_data_loader1
        self.train_data_loader2 = train_data_loader2
        self.val_data_loader1 = val_data_loader1
        self.val_data_loader2 = val_data_loader2
        self.device = device
        self.description = model_description
        self.final_activation = final_activation
        self.test_data_loader1 = test_data_loader1
        self.test_data_loader2 = test_data_loader2

        self.model.to(self.device)

        self.print_stats()

    def print_stats(self):
        print ("[LOG] Total number of parameters to learn {}".format(sum(p.numel() for p in self.model.parameters() \
                                                                 if p.requires_grad)))

    def train_fn(self, data_loader1, data_loader2):
        self.model.train()

        total_loss = 0
        pbar = tqdm(enumerate(zip(data_loader1,data_loader2)), total=len(data_loader1))
        for bi, (d1, d2) in pbar:
            ids1 = d1["ids"]
            token_type_ids1= d1["token_type_ids"]
            mask1 = d1["mask"]

            ids2 = d2["ids"]
            token_type_ids2= d2["token_type_ids"]
            mask2 = d2["mask"]

            targets = d1["targets"]

            ids1 = ids1.to(self.device)
            token_type_ids1 = token_type_ids1.to(self.device)
            mask1 = mask1.to(self.device)

            ids2 = ids2.to(self.device)
            token_type_ids2 = token_type_ids2.to(self.device)
            mask2 = mask2.to(self.device)

            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            if self.is_amp:
                #with amp.autocast():
                outputs = self.model(ids1,ids2, mask1,mask2,token_type_ids1,token_type_ids2)
                if self.loss_fn.__class__.__name__ == 'masked_CELoss':
                    loss = self.loss_fn(outputs, targets, mask1)
                else:
                    loss = self.loss_fn(outputs, targets)

            else:
                outputs = self.model(ids1,ids2, mask1,mask2,token_type_ids1,token_type_ids2)
                if self.loss_fn.__class__.__name__ == 'masked_CELoss':
                    loss = self.loss_fn(outputs, targets, mask1)
                else:
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

            with np.printoptions(precision=3):
                v = round(loss.cpu().detach().numpy().item(), 3)
                pbar.set_description("Current training Loss {}".format(v))

        return total_loss

    def eval_fn(self, data_loader1, data_loader2):
        self.model.eval()
        fin_targets = []
        fin_outputs = []

        total_loss = 0
        with torch.no_grad():
            pbar = tqdm(enumerate(zip(data_loader1,data_loader2)), total=len(data_loader1))
            for bi, (d1, d2) in pbar:
                ids1 = d1["ids"]
                token_type_ids1= d1["token_type_ids"]
                mask1 = d1["mask"]

                ids2 = d2["ids"]
                token_type_ids2= d2["token_type_ids"]
                mask2 = d2["mask"]

                targets = d1["targets"]

                ids1 = ids1.to(self.device)
                token_type_ids1 = token_type_ids1.to(self.device)
                mask1 = mask1.to(self.device)

                ids2 = ids2.to(self.device)
                token_type_ids2 = token_type_ids2.to(self.device)
                mask2 = mask2.to(self.device)

                targets = targets.to(self.device)

                outputs = self.model(ids1,ids2, mask1,mask2,token_type_ids1,token_type_ids2)
                if self.loss_fn.__class__.__name__ == 'masked_CELoss':
                    loss = self.loss_fn(outputs, targets, mask1)
                else:
                    loss = self.loss_fn(outputs, targets)

                total_loss += loss

                with np.printoptions(precision=3):
                    v = round(loss.cpu().detach().numpy().item(), 3)
                    pbar.set_description("Current eval Loss {}".format(v))

                fin_targets.extend(targets.cpu().detach().numpy().tolist())
                if self.final_activation == 'sigmoid':
                    fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
                else:
                    fin_outputs.extend(outputs.cpu().detach().numpy().tolist())
                
        return fin_outputs, total_loss, fin_targets

    def train(self, epochs, lr, scorer, loss_fn, optimizer, scheduler, MODEL_PATH, num_gpus=0, num_tpus=0, max_grad_norm=1, early_stopping_rounds=3, \
                    snapshot_ensemble=False, is_amp=False, use_wandb=False, seed=42):

        self.scorer = scorer
        self.loss_fn = loss_fn #losses.get_loss(loss_fn)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_gpus = num_gpus
        self.num_tpus = num_tpus

        set_seed(seed)

        self.args = {'description': self.description, 'loss': loss_fn.__class__.__name__, 'epochs': epochs, 'learning_rate': lr, \
                    'optimizer': self.optimizer.__class__.__name__, 'scheduler': self.scheduler.__class__.__name__}

        if use_wandb and not _torch_tpu_available:
            wandb.init(project="Project",config=self.args)
            wandb.watch(self.model)
        else:
            with open(os.path.join(MODEL_PATH, 'parameters.json'),'w') as outfile:
                json.dump(self.args, outfile)

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
                    self.model.load_state_dict(torch.load(os.path.join(MODEL_PATH,'model.bin')))
                except:
                    self.model = torch.load(os.path.join(MODEL_PATH,'model.bin'))

                #print (self.model)
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
            val_outputs = []

        for epoch in range(epochs):
            if bad_epochs < early_stopping_rounds:
                train_loss = self.train_fn(self.train_data_loader1, self.train_data_loader2)

                print ("Running evaluation on whole training data")
                train_out, train_loss, train_targets = self.eval_fn(self.train_data_loader1, self.train_data_loader2)
                print ("Running evaluation on evaluation data")
                val_out, val_loss, val_targets = self.eval_fn(self.val_data_loader1, self.val_data_loader2)
                
                train_loss = train_loss/len(self.train_data_loader1)
                val_loss = val_loss/len(self.val_data_loader1)

                #train_out = np.round(train_out)

                if snapshot_ensemble:
                    val_outputs += [np.expand_dims(np.array(val_out),axis=-1)]
                    val_out = np.concatenate(val_outputs,axis=-1).mean(axis=-1)
                #else:
                #    val_out = np.round(val_out) #np.array(outputs) >= 0.5

                train_metric = scorer(train_targets, train_out)
                val_metric = scorer(val_targets,val_out)

                with np.printoptions(precision=3):
                    print("Train loss = {} Train metric = {} Val loss = {} Val metric = {}".format(round(train_loss.detach().cpu().numpy().item(), 3),round(train_metric,3),\
                        round(val_loss.detach().cpu().numpy().item(), 3),round(val_metric,3)))
                
                if val_metric > best_metric:
                    #torch.save(self.model.state_dict(), os.path.join(MODEL_PATH,'model.bin'))
                    torch.save(self.model, os.path.join(MODEL_PATH,'model.bin'))
                    best_metric = val_metric
                    bad_epochs = 0

                else:
                    bad_epochs += 1

                if snapshot_ensemble and self.test_data_loader1 and self.test_data_loader2:
                    test_out = test_fn_multi(self.test_data_loader1,self.test_data_loader2, self.model, self.device, self.final_activation)

                    test_outputs += [np.expand_dims(np.array(test_out),axis=-1)]

            stats.update({"epoch_{}".format(epoch): {"train_loss": round(train_loss.detach().cpu().numpy().item(), 3), "train_metric": round(train_metric,3), \
                "val_loss": round(val_loss.detach().cpu().numpy().item(), 3), "val_metric": round(val_metric,3)}})

            if use_wandb:
                wandb.log({"train_loss": train_loss, "val_loss": val_loss, "train_metric": train_metric, "val_metric": val_metric})

        if self.test_data_loader1 and self.test_data_loader2:
            if snapshot_ensemble:
                self.test_output = np.concatenate(test_outputs,axis=-1).mean(axis=-1)
            else:
                self.test_output = test_fn_multi(self.test_data_loader1, self.test_data_loader2, self.model, self.device, self.final_activation)
        else:
            self.test_output = []

        #with np.printoptions(precision=3):
        #    print (stats)

        if use_wandb == False:
            with open(os.path.join(MODEL_PATH,'all_stats.json'), 'w') as outfile:
                json.dump(stats, outfile)

            with open(os.path.join(MODEL_PATH,'final_stats.json'), 'w') as outfile:
                d = {"epoch": epoch, "train_loss": round(train_loss.detach().cpu().numpy().item(), 3), "train_metric": round(train_metric,3), \
                "val_loss": round(val_loss.detach().cpu().numpy().item(), 3), "val_metric": round(val_metric,3)}
                json.dump(d, outfile)

class QATrainer:
    
    def __init__(self, model, train_data_loader, val_data_loader, device, model_description, final_activation=None, test_data_loader=None):
        self.model = model
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.device = device
        self.description = model_description
        self.test_data_loader = test_data_loader
        self.final_activation = final_activation

        self.model.to(self.device)

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
            targets_start = d["targets_start"]
            targets_end = d["targets_end"]

            ids = ids.to(self.device)
            token_type_ids = token_type_ids.to(self.device)
            mask = mask.to(self.device)
            targets_start = targets_start.to(self.device)
            targets_end = targets_end.to(self.device)

            self.optimizer.zero_grad()

            #with amp.autocast():
            outputs_start, outputs_end = self.model(ids=ids, mask=mask, token_type_ids=token_type_ids)
            loss = self.loss_fn(outputs_start, outputs_end, targets_start, targets_end)

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
            with np.printoptions(precision=3):
                v = round(loss.cpu().detach().numpy().item(), 3)
                pbar.set_description("Current training Loss {}".format(v))

        return total_loss

    def eval_fn(self, data_loader):
        self.model.eval()
        fin_targets_start = []
        fin_targets_end = []
        fin_outputs_start = []
        fin_outputs_end = []

        total_loss = 0
        with torch.no_grad():
            pbar = tqdm(enumerate(data_loader), total=len(data_loader))
            for bi, d in pbar:
                ids = d["ids"]
                token_type_ids = d["token_type_ids"]
                mask = d["mask"]
                targets_start = d["targets_start"]
                targets_end = d["targets_end"]

                ids = ids.to(self.device)
                token_type_ids = token_type_ids.to(self.device)
                mask = mask.to(self.device)
                targets_start = targets_start.to(self.device)
                targets_end = targets_end.to(self.device)

                outputs_start, outputs_end = self.model(ids=ids, mask=mask, token_type_ids=token_type_ids)
                loss = self.loss_fn(outputs_start, outputs_end, targets_start, targets_end)

                total_loss += loss

                with np.printoptions(precision=3):
                    v = round(loss.cpu().detach().numpy().item(), 3)
                    pbar.set_description("Current eval Loss {}".format(v))

                fin_targets_start.extend(targets_start.cpu().detach().numpy().tolist())
                fin_targets_end.extend(targets_end.cpu().detach().numpy().tolist())

                if self.final_activation == 'sigmoid':
                    fin_outputs_start.extend(torch.sigmoid(outputs_start).cpu().detach().numpy().tolist())
                    fin_outputs_end.extend(torch.sigmoid(outputs_end).cpu().detach().numpy().tolist())
                else:
                    fin_outputs_start.extend(outputs_start.cpu().detach().numpy().tolist())
                    fin_outputs_end.extend(outputs_end.cpu().detach().numpy().tolist())
                
        return fin_outputs_start, fin_outputs_end, total_loss, fin_targets_start, fin_targets_end

    def train(self, epochs, lr, scorer, loss_fn, optimizer, scheduler, MODEL_PATH, num_gpus=0, num_tpus=0, max_grad_norm=1, early_stopping_rounds=3, \
                    snapshot_ensemble=False, is_amp=False, use_wandb=False, seed=42):

        self.scorer = scorer
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_gpus = num_gpus
        self.num_tpus = num_tpus

        set_seed(seed)

        self.args = {'description': self.description, 'loss': loss_fn.__class__.__name__, 'epochs': epochs, 'learning_rate': lr, \
                    'optimizer': self.optimizer.__class__.__name__, 'scheduler': self.scheduler.__class__.__name__}

        if use_wandb and not _torch_tpu_available:
            wandb.init(project="Project",config=self.args)
            wandb.watch(self.model)
        else:
            with open(os.path.join(MODEL_PATH, 'parameters.json'),'w') as outfile:
                json.dump(self.args, outfile)

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
                    self.model.load_state_dict(torch.load(os.path.join(MODEL_PATH,'model.bin')))
                except:
                    self.model = torch.load(os.path.join(MODEL_PATH,'model.bin'))

                #print (self.model)
                print ("Loaded model from previous checkpoint")
            except:
                pass

        self.is_amp = is_amp
        self.max_grad_norm = max_grad_norm
        self.scaler = scaler

        best_metric = 0
        bad_epochs = 0
        if snapshot_ensemble:
            test_outputs_start = []
            val_outputs_start = []
            test_outputs_end = []
            val_outputs_end = []

        stats = {}
        for epoch in range(epochs):
            if bad_epochs < early_stopping_rounds:
                train_loss = self.train_fn(self.train_data_loader)

                print ("Running evaluation on whole training data")
                train_out_start, train_out_end, train_loss, train_targets_start, train_targets_end = self.eval_fn(self.train_data_loader)
                print ("Running evaluation on validation data")
                val_out_start, val_out_end, val_loss, val_targets_start, val_targets_end = self.eval_fn(self.val_data_loader)
                
                train_loss = train_loss/len(self.train_data_loader)
                val_loss = val_loss/len(self.val_data_loader)

                #train_out = np.round(train_out)

                if snapshot_ensemble:
                    val_outputs_start += [np.expand_dims(np.array(val_out_start),axis=-1)]
                    val_outputs_end += [np.expand_dims(np.array(val_out_end),axis=-1)]
                    
                    val_out_start = np.concatenate(val_outputs_start,axis=-1).mean(axis=-1)
                    val_out_end = np.concatenate(val_outputs_end,axis=-1).mean(axis=-1)
                #else:
                #    val_out = np.round(val_out) #np.array(outputs) >= 0.5

                train_metric = scorer(train_targets_start, train_targets_end, train_out_start, train_out_end)
                val_metric = scorer(val_targets_start, val_targets_end, val_out_start, val_out_end)

                with np.printoptions(precision=3):
                    print("Train loss = {} Train metric = {} Val loss = {} Val metric = {}".format(round(train_loss.detach().cpu().numpy().item(), 3),round(train_metric,3),\
                        round(val_loss.detach().cpu().numpy().item(), 3),round(val_metric, 3)))
                
                if val_metric > best_metric:
                    #torch.save(self.model.state_dict(), os.path.join(MODEL_PATH,'model.bin'))
                    torch.save(self.model, os.path.join(MODEL_PATH,'model.bin'))
                    best_metric = val_metric
                    bad_epochs = 0

                else:
                    bad_epochs += 1

                if snapshot_ensemble and self.test_data_loader:
                    test_out_start, test_out_end = test_fn_qa(self.test_data_loader, self.model, self.device, self.final_activation)

                    test_outputs_start += [np.expand_dims(np.array(test_out_start),axis=-1)]
                    test_outputs_end += [np.expand_dims(np.array(test_out_end),axis=-1)]

            stats.update({"epoch_{}".format(epoch): {"train_loss": round(train_loss.detach().cpu().numpy().item(), 3), "train_metric": round(train_metric,3), \
                "val_loss": round(val_loss.detach().cpu().numpy().item(), 3),  "val_metric": round(val_metric,3)}})

            if use_wandb:
                wandb.log({"train_loss": train_loss, "val_loss": val_loss, "train_metric": train_metric, "val_metric": val_metric})

        if self.test_data_loader:
            if snapshot_ensemble:
                self.test_output_start = np.concatenate(test_outputs_start,axis=-1).mean(axis=-1)
                self.test_output_end = np.concatenate(test_outputs_end,axis=-1).mean(axis=-1)
            else:
                self.test_output_start, self.test_output_end = test_fn_qa(self.test_data_loader, self.model, self.device, self.final_activation)
        else:
            self.test_output_start = []
            self.test_output_end = []

        #with np.printoptions(precision=3):
        #    print (stats)

        if use_wandb == False:
            with open(os.path.join(MODEL_PATH,'all_stats.json'), 'w') as outfile:
                json.dump(stats, outfile)

            with open(os.path.join(MODEL_PATH,'final_stats.json'), 'w') as outfile:
                d = {"epoch": epoch, "train_loss": round(train_loss.detach().cpu().numpy().item(), 3), "train_metric": round(train_metric,3), \
                "val_loss": round(val_loss.detach().cpu().numpy().item(), 3),  "val_metric": round(val_metric,3)}
                json.dump(d, outfile)

class PLTrainer(pl.LightningModule):

    def __init__(self, num_train_steps, model, metric, loss_fn, lr, final_activation=None, seed=42):
        super(PLTrainer, self).__init__()

        seed_everything(seed)

        self.model = model
        self.loss_fn = loss_fn #losses.get_loss(loss_fn)
        self.metric_name = metric.__class__.__name__
        self.metric = metric
        self.num_train_steps = num_train_steps
        self.lr = lr
        self.final_activation = final_activation

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
        if self.loss_fn.__class__.__name__ == 'masked_CELoss':
            loss = self.loss_fn(outputs, targets, mask)
        else:
            loss = self.loss_fn(outputs, targets)

        if self.final_activation == 'sigmoid':
            metric_value = self.metric(targets, torch.sigmoid(outputs))
        else:
            metric_value = self.metric(targets, outputs)

        tensorboard_logs = {'train_loss': loss, "train {}".format(self.metric_name): metric_value}

        return {'loss': loss, 'train_metric': metric_value, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        di = batch
        ids = di["ids"]
        token_type_ids = di["token_type_ids"]
        mask = di["mask"]
        targets = di["targets"]

        outputs = self.model(ids=ids, mask=mask, token_type_ids=token_type_ids)
        if self.loss_fn.__class__.__name__ == 'masked_CELoss':
            loss = self.loss_fn(outputs, targets, mask)
        else:
            loss = self.loss_fn(outputs, targets)

        if self.final_activation == 'sigmoid':
            metric_value = self.metric(targets,torch.sigmoid(outputs))
        else:
            metric_value = self.metric(targets,outputs)

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
        print ("Train loss = {} Train metric = {}".format(round(train_loss_mean.detach().cpu().numpy().item(), 3),round(train_metric_mean.detach().cpu().numpy().item(), 3)))

        return {'train_loss': train_loss_mean, 'train_metric': train_metric_mean}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_metric_mean = torch.stack([x['val_metric'] for x in outputs]).mean()
        print ("val loss = {} val metric = {} ".format(round(val_loss_mean.detach().cpu().numpy().item(), 3),round(val_metric_mean.detach().cpu().numpy().item(), 3)))

        return {'val_loss': val_loss_mean, 'val_metric': val_metric_mean}

    
class PLTrainerQA(pl.LightningModule):

    def __init__(self, num_train_steps, model, metric, loss_fn, lr, final_activation=None, seed=42):
        super(PLTrainerQA, self).__init__()

        seed_everything(seed)

        self.model = model
        self.loss_fn = loss_fn #losses.get_loss(loss_fn)
        self.metric_name = metric.__class__.__name__
        self.metric = metric
        self.num_train_steps = num_train_steps
        self.lr = lr
        self.final_activation = final_activation

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
        targets_start = di["targets_start"]
        targets_end = di["targets_end"]

        outputs_start, outputs_end = self.model(ids=ids, mask=mask, token_type_ids=token_type_ids)
        loss = self.loss_fn(outputs_start, outputs_end, targets_start, targets_end)

        if self.final_activation == 'sigmoid':
            metric_value = self.metric(targets_start, targets_end, torch.sigmoid(outputs_start), torch.sigmoid(outputs_end))
        else:
            metric_value = self.metric(targets_start, targets_end, outputs_start, outputs_end)

        tensorboard_logs = {'train_loss': loss, "train {}".format(self.metric_name): metric_value}

        return {'loss': loss, 'train_metric': metric_value, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        di = batch
        ids = di["ids"]
        token_type_ids = di["token_type_ids"]
        mask = di["mask"]
        targets_start = di["targets_start"]
        targets_end = di["targets_end"]

        outputs_start, outputs_end = self.model(ids=ids, mask=mask, token_type_ids=token_type_ids)
        loss = self.loss_fn(outputs_start, outputs_end, targets_start, targets_end)

        if self.final_activation == 'sigmoid':
            metric_value = self.metric(targets_start, targets_end, torch.sigmoid(outputs_start), torch.sigmoid(outputs_end))
        else:
            metric_value = self.metric(targets_start, targets_end, outputs_start, outputs_end)

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
        print ("Train loss = {} Train metric = {}".format(round(train_loss_mean.detach().cpu().numpy().item(), 3),round(train_metric_mean.detach().cpu().numpy().item(), 3)))

        return {'train_loss': train_loss_mean, 'train_metric': train_metric_mean}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_metric_mean = torch.stack([x['val_metric'] for x in outputs]).mean()
        print ("val loss = {} val metric = {} ".format(round(val_loss_mean.detach().cpu().numpy().item(), 3),round(val_metric_mean.detach().cpu().numpy().item(), 3)))

        return {'val_loss': val_loss_mean, 'val_metric': val_metric_mean}