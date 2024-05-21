import pickle
import copy

import torch.nn.functional as F
import numpy as np

from network.prompt_metaR import Prompt_metaR
from network.naive_metaR import Naive_metaR
import os
import sys
import torch
import logging



class Trainer:
    def __init__(self, data_loaders, dataset, parameter):
        self.parameter = parameter
        self.device = parameter['device']

        # dataset
        self.dataset = dataset

        # data loader
        self.train_data_loader = data_loaders[0]
        self.dev_data_loader = data_loaders[1]
        self.test_data_loader = data_loaders[2]
        self.fw_dev_data_loader = data_loaders[3]

        self.bfew = parameter['base_classes_few']
        self.bnq = parameter['base_classes_num_query']
        self.br = parameter['base_classes_relation']
        self.few = parameter['few']
        self.num_query = parameter['num_query']
        self.batch_size = parameter['batch_size']
        self.num_tasks = parameter['num_tasks']

        # model
        if parameter['rq']:
            self.metaR = Prompt_metaR(dataset, parameter)
        else:
            self.metaR = Naive_metaR(dataset, parameter)
  
        
        # self.naive_metaR = Lora_metaR(dataset, parameter)
        if len(list(parameter['epoch'])) <= 2:
            self.naive_metaR = Naive_metaR(dataset, parameter)
        else:
            self.naive_metaR = Naive_metaR(dataset, parameter)
            # self.naive_metaR = Prompt_metaR(dataset, parameter)

        self.metaR.to(self.device)
        self.naive_metaR.to(self.device)
        self.rq = parameter['rq']
        self.top_k = parameter['top_k']
        self.size = parameter['size']
        self.shared_prompt_pool = parameter['shared_prompt_pool']
        self.pull_constraint_coeff = parameter['pull_constraint_coeff']

        # training
        self.learning_rate = parameter['learning_rate']
        self.optimizer = torch.optim.Adam(self.metaR.parameters(), self.learning_rate)
        self.epoch = list(parameter['epoch'])
        self.print_epoch = parameter['print_epoch']
        self.early_stopping_patience = parameter['early_stopping_patience']
        self.early_NOVEL_stopping_patience = parameter['early_NOVEL_stopping_patience']
        self.eval_epoch = list(parameter['eval_epoch'])
        self.checkpoint_epoch = parameter["checkpoint_epoch"]
        self.is_prompt_tuning = parameter["is_prompt_tuning"]
        self.rel_index = parameter["rel_index"]  

        # dir
        self.save_path = parameter['save_path']
        self.state_dir = os.path.join(
            self.parameter['state_dir'], self.parameter['prefix'])
        if not os.path.isdir(self.state_dir):
            os.makedirs(self.state_dir)
        self.ckpt_dir = os.path.join(
            self.parameter['state_dir'], self.parameter['prefix'], 'checkpoint')
        if not os.path.isdir(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        self.state_dict_file = ''

        # logging
        logging_dir = os.path.join(
            self.parameter['log_dir'], self.parameter['prefix'], 'res.log')
        if not os.path.exists(logging_dir):
            os.makedirs(logging_dir)
        logging.basicConfig(
            filename=logging_dir, level=logging.INFO, format="%(asctime)s - %(message)s")
        self.csv_dir = os.path.join(
            self.parameter['log_dir'], self.parameter['prefix'])

    def train(self):
        MRR_val_mat = np.zeros((self.num_tasks, 2))
        if self.is_prompt_tuning:
            for p in self.metaR.parameters():
                    p.requires_grad = True
            if self.rq:
                print('unfreeze rq prompt')
                for n, p in self.metaR.named_parameters():
                    if n.startswith('relation_learner.prompt'):
                        print(n)
                        p.requires_grad = True
        
        for task_id in range(self.num_tasks):
            if self.is_prompt_tuning:
                if task_id == (self.rel_index - self.br) / self.batch_size:
                    print(f'load {self.save_path} stage model')
                    print(f'fastward current data idx {self.train_data_loader.curr_rel_idx}')
                    logging.info(f'fastward current data idx {self.train_data_loader.curr_rel_idx}')
                    self.transfer_base_weight(self.naive_metaR, self.metaR)
                    continue
                elif task_id < (self.rel_index - self.br) / self.batch_size:
                    continue

            print(f'task {task_id} begins')
            
            # early stop setting
            best_loss = 100
            now_waiting = 0
            best_e = 0

            # Create new optimizer for each task to clear optimizer status
            if task_id > 1 and self.is_prompt_tuning:

            for e in range(self.epoch[task_id]):
                is_last = False if e != self.epoch[task_id] - 1 else True
                is_base = True if task_id == 0 else False
                patience = 20

                # sample one batch from data_loader
                train_task, curr_rel = self.train_data_loader.next_batch(is_last, is_base)
                loss, _, _ = self.do_one_step(train_task, task_id, iseval=False, curr_rel=curr_rel)

                
                # print the loss on specific epoch
                if e % self.print_epoch == 0:
                    print("Epoch: {}\tLoss: {:.4f} {}, sofar best epo is : {}".format(e, loss_num,
                                                                                      self.train_data_loader.curr_rel_idx,
                                                                                      best_e))


                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_e = e
                    now_waiting = 0
                else:
                    now_waiting += 1

                if now_waiting > patience:
                    print(
                        f"stop at {e} for {patience} epoches, loss hasn't been better.")
                    print(f"best loss is {best_loss}, best epoch is {best_e}")
                    self.save_checkpoint(e)
                    break

                # save checkpoint on specific epoch
                if e % self.checkpoint_epoch == 0 and e != 0:
                    print('Epoch  {} has finished, saving...'.format(e))
                    self.save_checkpoint(e)

                # do evaluation on specific epoch
                if e % self.eval_epoch[task_id] == 0 and e != 0:
                    print('Epoch  {} has finished, validating few shot...'.format(e))
                    valid_data = self.fw_eval(task_id, epoch=e)  # few shot eval
                
        print('Training has finished')

    def transfer_base_weight(self, pretrain_model, empty_model):
        params1 = pretrain_model.named_parameters()
        params2 = empty_model.named_parameters()
        dict_params2 = dict(params2)


    def do_one_step(self, task, task_id, iseval=False, curr_rel=''):
        loss, p_score, n_score = 0, 0, 0
        if not iseval:
            self.metaR.train()
            self.optimizer.zero_grad()

            p_score, n_score, reduce_sim = self.metaR(task, task_id, iseval=False, curr_rel=curr_rel)
            y = torch.ones(p_score.shape[0], 1).to(self.device)

            if reduce_sim is not None:
                loss = self.metaR.loss_func(p_score, n_score, y) - self.pull_constraint_coeff * reduce_sim
            else:
                loss = self.metaR.loss_func(p_score, n_score, y)

            self.optimizer.step()
        elif curr_rel != '' and iseval:
            self.metaR.eval()
            p_score, n_score, _ = self.metaR(task, task_id, iseval=True, curr_rel=curr_rel)
            y = torch.ones(p_score.shape[0], 1).to(self.device)
            loss = self.metaR.loss_func(p_score, n_score, y)
        return loss, p_score, n_score


    def fw_eval(self, task_id, epoch=None):
        self.metaR.eval()
        # clear sharing rel_q
        self.metaR.rel_q_sharing = dict()

        data_loader.curr_tri_idx = 0

        # initial return data of validation
        data = {'MRR': 0, 'Hits@1': 0, 'Hits@5': 0, 'Hits@10': 0}
        ranks = []

        t = 0
        temp = dict()
        while True:
            # sample all the eval tasks
            # at the end of sample tasks, a symbol 'EOT' will return
            if eval_task == 'EOT':
                break
            t += 1
            self.get_epoch_score(task_id, curr_rel, data, eval_task, ranks, t, temp)

        # print overall evaluation result and return it
        for k in data.keys():
            data[k] = round(data[k] / t, 3)

        if self.parameter['step'] == 'train':
            self.logging_fw_training_data(data, epoch, task_id)
        else:
            self.logging_eval_data(data, self.state_dict_file)

        print("few shot {}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
            t, data['MRR'], data['Hits@10'], data['Hits@5'], data['Hits@1']))

        return data

    def get_epoch_score(self, task_id, curr_rel, data, eval_task, ranks, t, temp):
        self.rank_predict(data, x, ranks)

        # print current temp data dynamically
        for k in data.keys():
            temp[k] = data[k] / t
        sys.stdout.write("{}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
            t, temp['MRR'], temp['Hits@10'], temp['Hits@5'], temp['Hits@1']))
        sys.stdout.flush()

  

    def save_checkpoint(self, epoch):
        torch.save(self.metaR.state_dict(), os.path.join(
            self.ckpt_dir, 'state_dict_' + str(epoch) + '.ckpt'))

    def write_fw_validating_log(self, data, record, task, epoch):
        if epoch + self.eval_epoch[task] >= self.epoch[task]:
            record[task, 0] = data['MRR']

    def write_cl_validating_log(self, metrics, record, task):
        record[task, 1] = metrics[0]['MRR']

    def logging_fw_training_data(self, data, epoch, task):
        if epoch == self.eval_epoch[task]:
            logging.info(f"Few_Shot_Learning_task {task}")
        logging.info("Epoch: {}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
            epoch, data['MRR'], data['Hits@10'], data['Hits@5'], data['Hits@1']))

    def logging_cl_training_data(self, metrics, task):
        logging.info(f"Eval_Continual_Learning_task {task}")
        for i, data in enumerate(metrics):
            logging.info("Task: {}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
                str(i), data['MRR'], data['Hits@10'], data['Hits@5'], data['Hits@1']))

    def logging_eval_data(self, data, state_path):
        setname = 'val set'
        logging.info("Eval {} on {}".format(state_path, setname))
        logging.info("MRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
            data['MRR'], data['Hits@10'], data['Hits@5'], data['Hits@1']))

    def save_metrics(self, MRR_val_mat, early_NOVEL_stopping_patience):
        np.savetxt(os.path.join(self.csv_dir, f'{early_NOVEL_stopping_patience}_MRR.csv'),
                   MRR_val_mat, delimiter=",", fmt='%s')

