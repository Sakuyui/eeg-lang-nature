# -*- coding: utf-8 -*-

from datetime import datetime, timedelta
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class ModelTrainable():
    def __init__(self):
        pass
    
    def train(self):
        pass
    
    def evaluation(self):
        pass

    def pre_train_epoch(self):
        pass
    
    def post_train_epoch(self):
        pass
    
    def pre_test(self):
        pass
    
    def post_test(self):
        pass
    
class ModelTrainer():
    def __init__(self, args):
        self.args = args
        self.model = args['model']
        self.optimizer = args['optimizer']
        train_arg = args['train_arg']
        self.train_arg = train_arg
        self.save_model_automatically = args['save_model_automatically']

    def do_train(self, train_dataset, evaluation_dataset = None):
        total_time = 0.0
        best_metric = None
        for epoch in range(1, self.train_arg['max_epoches'] + 1):
            start = datetime.now()
            self.model.pre_train_epoch()
            self.train(train_dataset)
            print(f"Epoch {epoch} / {self.train_arg['max_epoches']}:")
            if evaluation_dataset is not None:
                evaluation_results = self.evaluate(evaluation_dataset)

            t = datetime.now() - start

            # save the model if it is the best so far
            
            if self.comparasion(evaluation_results, best_metric) > 0:
                best_metric = evaluation_results
                best_e = epoch
                if self.save_model_automatically:
                    torch.save(
                    obj=self.model.state_dict(),
                    f = self.args['save_dir'] + "/" + self.args['save_model_name'] + ".pt"
                    )
                print(f"{t}s elapsed (saved)\n")
            else:
                print(f"{t}s elapsed\n")

            total_time += t
            if self.train_arg['patience'] > 0 and epoch - best_e >= self.train_arg['patience']:
                break
    
    def train(self, train_dataset, test_dataset = None, time_sequence = False):
        INDEX_DATASET_INPUTS = 0
        INDEX_DATASET_TARGETS = 1
        shuffle = self.train_arg['shuffle_trainning_set']
        train_only = self.train_arg['train_only']
        batch_size_train = self.train_arg['batch_size_train']
        train_dataloader = DataLoader(TensorDataset(torch.FloatTensor(train_dataset[INDEX_DATASET_INPUTS]), torch.FloatTensor(train_dataset[INDEX_DATASET_TARGETS])),\
            batch_size=batch_size_train, shuffle=shuffle)
        if not train_only:
            batch_size_test = self.train_arg['batch_size_test']
            test_dataloader = DataLoader(TensorDataset(torch.FloatTensor(test_dataset[INDEX_DATASET_INPUTS]), torch.FloatTensor(test_dataset[INDEX_DATASET_TARGETS])),\
                batch_size=batch_size_test, shuffle=shuffle)
        torch.autograd.set_detect_anomaly(True)
        max_epoches = self.train_arg['max_epoches']
        for epoch in range(1, max_epoches + 1):
            self.model.train()
            if callable(getattr(self.model, 'pre_train_epoch', None)):
                self.model.pre_train_epoch()
            
            t = tqdm(train_dataloader, total=int(len(train_dataloader)),  position=0, leave=True)
            train_arg = self.train_arg
            
            # if time_sequence:
            #     for inputs, targets in t:
            #         self.optimizer.zero_grad()
            #         if self.train_arg['is_unsupervisor_learning']:
            #             loss = self.model.loss(inputs)
            #         else:
            #             loss = self.model.loss(inputs, targets)
            #         loss.backward(retain_graph = True)
            #         if train_arg['clip'] > 0:
            #             nn.utils.clip_grad_norm_(self.model.parameters(),
            #                                         train_arg['clip'])
            #         self.optimizer.step()
            #         t.set_postfix(loss=loss.item(), epoch=epoch)
            #         loss.detach()
            # else:
            for inputs, targets in t:
                    if self.train_arg['is_unsupervisor_learning']:
                        loss = self.model.loss(inputs)
                    else:
                        loss = self.model.loss(inputs, targets)
                    if train_arg['clip'] > 0:
                            nn.utils.clip_grad_norm_(self.model.parameters(),
                                                    train_arg['clip'])
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    t.set_postfix(loss=loss.item(), epoch=epoch)
                    # loss.detach()
            
            if not train_only:
                evaluation_results = self.evaluate(test_dataloader)
        return
    
    @torch.no_grad()
    def evaluate(self, evaluation_dataset):
        self.model.evaluate(evaluation_dataset[0], evaluation_dataset[1])
