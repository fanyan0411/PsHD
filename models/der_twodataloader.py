import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler
from models.base import BaseLearner
from utils.inc_net import DERNet, IncrementalNet
from utils.toolkit import count_parameters, target2onehot, tensor2numpy
from time import time
import os, json, math
from pathlib import Path
from utils.distill_loss import matching_loss, matching_loss_topo_samplewise, matching_loss_topo_protowise
from utils.data_sampler import TwoStreamBatchSampler
#from utils.labelpropagation import LabelPropagation
EPSILON = 1e-8
from time import time
import pickle

init_epoch = 200
init_lr = 0.1  # 0.1 0.03
init_milestones = [60, 120, 170]
init_lr_decay = 0.1
init_weight_decay =  0.0005 # 0.0005   0.001 

epochs = 170
lrate = 0.1 # 0.1
milestones = [80, 120, 150]
lrate_decay = 0.1
batch_size = 128
weight_decay = 2e-4  # 0.0005  2e-4
num_workers = 2
T = 2


class DER_twodataloader(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = DERNet(args, False)
        self.threshold = 0.95
        self.uce_loss = torch.nn.CrossEntropyLoss(reduction='none') 

        self.batch_size = args["batch_size"]
        self.init_epoch = args["init_epoch"]
        self.former_epochs = args["former_epochs"]
        # Task setting

        self.labeled_batch_size =  args["labeled_batch_size"] #int(self.batch_size/2) 
        self.unlabeled_batch_size =  self.batch_size - self.labeled_batch_size 
        self.label_num = args["label_num"]
        self.dataset_name = args["dataset"]
        self.init_cls = args["init_cls"]
        self.full_supervise = args["full_supervise"]
        self.total_exp = int(args["label_size"]/self.init_cls)
        #self.repeat = args["repeat"]


        # Semi-supervised loss
        self.usp_weight = args["usp_weight"]
        self.uce_loss = torch.nn.CrossEntropyLoss(reduction='none') 
        self.insert_pse_progressive = args['insert_pse_progressive']
        self.insert_pse = args['insert_pse']
        self.pse_weight = args["pse_weight"] # weight of previous pseudo labels of weak augmentation view in uloss
        #self.lp_alpha = args["lp_alpha"]
        self.world_size = len(args["device"])
        self.train_iterations = args["train_iterations"]
        self.start_step = 0
        # Sub-graph distillation loss
        self.rw_alpha = args["rw_alpha"]
        self.match_weight = args["match_weight"]
        self.rw_T = args["rw_T"]
        self.gamma_ml = args["gamma_ml"]

        # Topo-simplex distillation loss
        self.topo_weight = args["topo_weight"]
        if args["topo_weight_class"]==None:
            self.topo_weight_class = self.topo_weight
        else:
            self.topo_weight_class = args["topo_weight_class"]
        if args["topo_weight_sample"]==None:    
            self.topo_weight_sample = self.topo_weight
        else:
            self.topo_weight_sample = args["topo_weight_sample"]
        
        self.less_train_iterations = args["less_train_iterations"]

    def after_task(self):
        self._old_network = self._network.copy().freeze()  #
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

        if not self.args["resume"]:
            if self.args["shuffle"]:
                checkpoints_folder_name = 'checkpoints'
            else:
                checkpoints_folder_name = 'checkpoints_notshuffle'
            if not os.path.exists('./{}'.format(checkpoints_folder_name)):
                os.makedirs('./{}'.format(checkpoints_folder_name))
            checkpoint_name = "./{}/uloss{}_{}_{}_{}_{}_{}_{}_{}_iter{}_inlr{}.pkl".format(checkpoints_folder_name, self.args["usp_weight"], \
                                                                                      self.args["dataset"], self.args["model_name"],\
                                                                                    self.args["init_cls"],self.args["increment"],\
                                                                                    self.init_epoch,self.args["label_num"],self._cur_task, 
                                                                                    self.train_iterations, init_lr) 
            if self._cur_task == 0 or self.args["save_all_resume"]:
                if not os.path.exists(checkpoint_name):
                    self.save_checkpoint("./{}/uloss{}_{}_{}_{}_{}_{}_{}_iter{}_inlr{}".format(checkpoints_folder_name, self.args["usp_weight"], \
                                                                                          self.args["dataset"], self.args["model_name"], \
                                                                                          self.args["init_cls"],self.args["increment"], \
                                                                                          self.init_epoch,self.args["label_num"], 
                                                                                          self.train_iterations, init_lr))
        

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        if self._cur_task > 0:
            for i in range(self._cur_task):
                for p in self._network.convnets[i].parameters():
                    p.requires_grad = False

        logging.info("All params: {}".format(count_parameters(self._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(self._network, True))
        )
        '''
        train_dataset, idxes = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
        )
        '''
        labeled_train_dataset, idxes = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
            loader_idx='labeled_train'
        )

        unlabeled_train_dataset, idxes = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
            loader_idx='unlabeled_train'
        )
        if self._cur_task > 0:
            self.train_iterations = self.less_train_iterations
        print('self.train_iterations', self.train_iterations)
        labeled_train_sampler = RandomSampler(labeled_train_dataset, replacement=True,
                                          num_samples=self.labeled_batch_size * self.world_size * self.train_iterations,  #self.train_iterations
                                          generator=None)
        unlabeled_train_sampler = RandomSampler(unlabeled_train_dataset, replacement=True,
                                          num_samples=self.unlabeled_batch_size * self.world_size * self.train_iterations,  #self.train_iterations
                                          generator=None)

        self.labeled_train_loader = DataLoader(
            labeled_train_dataset, batch_size=self.labeled_batch_size, shuffle=False, num_workers=num_workers, pin_memory=False, 
            sampler=labeled_train_sampler, drop_last=True
        ) 


        self.unlabeled_train_loader = DataLoader(
            unlabeled_train_dataset, batch_size=self.unlabeled_batch_size, shuffle=False, num_workers=num_workers, pin_memory=False,
            sampler=unlabeled_train_sampler, drop_last=True
        ) 
        
        test_dataset, _ = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        train_start_time = time()
        self._train(self.labeled_train_loader, self.unlabeled_train_loader, self.test_loader)
        train_end_time = time() - train_start_time
        logging.info("training time of task {}:{:.3f}min, {:.3f}h".format(self._cur_task, train_end_time/60, train_end_time/3600))
        self.build_rehearsal_memory(data_manager, self.samples_per_class[0], self.samples_per_class[1])
        build_memory_time = time() - train_start_time - train_end_time
        logging.info("build_memory time of task {}:{:.3f}min, {:.3f}h".format(self._cur_task,build_memory_time/60, build_memory_time/3600))
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def train(self):
        self._network.train()
        if len(self._multiple_gpus) > 1 :
            self._network_module_ptr = self._network.module
        else:
            self._network_module_ptr = self._network
        self._network_module_ptr.convnets[-1].train()
        if self._cur_task >= 1:
            for i in range(self._cur_task):
                self._network_module_ptr.convnets[i].eval()

    def _train(self, labeled_train_loader, unlabeled_train_loader, test_loader):
        if self.args["resume"] and self._cur_task == 0:
            if self.args["shuffle"]:
                checkpoints_folder_name = 'checkpoints'
            else:
                checkpoints_folder_name = 'checkpoints_notshuffle'
            checkpoint_name = "./{}/uloss{}_{}_{}_{}_{}_200_{}_iter{}_inlr{}_{}.pkl".format(checkpoints_folder_name, self.args["usp_weight"], \
                                                                                              self.args["dataset"], self.args["model_name"], \
                                                                                            self.args["init_cls"],self.args["increment"],self.args["label_num"], \
                                                                                            self.train_iterations, init_lr, self._cur_task)
            if self.args["save_resume_name"] != None:
                checkpoint_name = self.args["save_resume_name"]
                print('load checkpoint: ', checkpoint_name)
                print('load checkpoint: ', checkpoint_name)
                print('load checkpoint: ', checkpoint_name)
            if os.path.isfile(checkpoint_name):
                self._network.module.load_state_dict(torch.load(checkpoint_name)["model_state_dict"])
            else:
                print(checkpoint_name, "is none")
                self.args["resume"] = False
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)
        if self._cur_task == 0:
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                momentum=0.9,
                lr=init_lr,
                weight_decay=init_weight_decay,
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=init_milestones, gamma=init_lr_decay
            )
            '''
            # cosine lr scheduler
            lr_scheduler = optim.lr_scheduler.LambdaLR(
                optimizer,
                lambda x: math.cos(7. / 16. * math.pi * x / args.train_iterations)
            )
            '''
            if not self.args["resume"]:
                self._init_train(labeled_train_loader, unlabeled_train_loader, test_loader, optimizer, scheduler)
        else:
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=lrate,
                momentum=0.9,
                weight_decay=weight_decay,
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=milestones, gamma=lrate_decay
            )
            self._update_representation(labeled_train_loader, unlabeled_train_loader, test_loader, optimizer, scheduler)
            if len(self._multiple_gpus) > 1:
                self._network.module.weight_align(
                    self._total_classes - self._known_classes
                )
            else:
                self._network.weight_align(self._total_classes - self._known_classes)

    def _init_train(self, labeled_train_loader, unlabeled_train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.init_epoch))
        for _, epoch in enumerate(prog_bar):
            self.train()
            losses, losses_clf, ulosses = 0.0, 0.0, 0.0
            correct, total = 0, 0
            #for i, (_, inputs, inputs_s, targets, pse_targets, lab_index_task) in enumerate(train_loader):

            for global_step, (_, inputs_l, inputs_s_l, targets_l, pse_targets_l, lab_index_task_l), (_, inputs_u, inputs_s_u, targets_u, pse_targets_u, lab_index_task_u) in \
                    zip(range(self.start_step, self.train_iterations), labeled_train_loader, unlabeled_train_loader):
                
                inputs = torch.cat((inputs_l, inputs_u, inputs_s_u), dim=0)
                inputs, targets_l, targets_u = inputs.to(self._device), targets_l.to(self._device), targets_u.to(self._device)
                pse_targets_u = pse_targets_u.to(self._device)
                #inputs_s = inputs_s.to(self._device)
                #lab_index_task = lab_index_task.to(self._device)
                output = self._network(inputs)
                logits = output["logits"]
                logits_l = logits[:self.labeled_batch_size]
                logits_u, logits_s_u = logits[self.labeled_batch_size:].chunk(2, dim=0)

                targets_old = torch.cat((targets_l, targets_u), dim=0).clone()  # 
                targets = torch.cat((targets_l, pse_targets_u), dim=0)
                

                loss_clf = F.cross_entropy(logits_l, targets_l)

                if epoch>-1:
                    self.usp_weight = 1
                else:
                    self.usp_weight = 0
                with torch.no_grad():
                    predi_w = F.softmax(logits_u.clone(), 1)
                    predi_l = F.softmax(logits_l.clone(), 1)
                    wprobs, wpslab = predi_w.max(1)
                    #wprobs, wpslab = logits.clone().max(1)
                    #wpslab = (targets>-100) * targets + wpslab * (targets==-100)
                    self.label_propa = False #(self.lp_alpha>0)
                    if self.label_propa:
                        LP_model = LabelPropagation(gamma=None)
                        LP_model.fit(output["features"][:self.batch_size].detach(), targets, targets_old)
                        alpha = self.lp_alpha
                        y_pred = LP_model.predict(output["features"][:self.batch_size].detach(), torch.cat((predi_l, predi_w)).detach(), targets, alpha, targets_old)
                        wpslab = y_pred[self.labeled_batch_size:]
                        #wpslab = label_propa(logits_w, wpslab, targets)
                mask = wprobs.ge(self.threshold).float()
                #logits_s = self._network(inputs_s)["logits"]
                uloss  = torch.mean(mask * self.uce_loss(logits_s_u, wpslab)) 
                uloss *= self.usp_weight
                loss = loss_clf + uloss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                losses_clf += loss_clf.item()
                ulosses += uloss.item()

                _, preds = torch.max(logits[:self.batch_size], dim=1)
                correct += preds.eq(targets_old.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                test_acc, cnn_accy = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, uloss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.init_epoch,
                    losses / len(labeled_train_loader),
                    losses_clf / len(labeled_train_loader),
                    ulosses / len(labeled_train_loader),
                    train_acc,
                    test_acc,
                )
                logging.info(cnn_accy["grouped"])
                logging.info(info)
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.init_epoch,
                    losses / len(labeled_train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)

        logging.info(info)

    def _update_representation(self, labeled_train_loader, unlabeled_train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.former_epochs))
        for _, epoch in enumerate(prog_bar):
            self.train()
            losses = 0.0
            losses_clf = 0.0
            losses_aux = 0.0
            ulosses = 0.0
            correct, total = 0, 0
            total_supervise = 0
            losses_match_logits = 0
            aux_pseudolabel = []
            aux_pseudolabel_current = []
            aux_pseudolabel_old = []
            pseudolabel = []
            pseudolabel_current = []
            pseudolabel_old = []
            
            #for i, (_, inputs, inputs_s, targets, pse_targets, lab_index_task) in enumerate(train_loader):
            for global_step, (_, inputs_l, inputs_s_l, targets_l, pse_targets_l, lab_index_task_l), (_, inputs_u, inputs_s_u, targets_u, pse_targets_u, lab_index_task_u) in \
                    zip(range(self.start_step, self.train_iterations), labeled_train_loader, unlabeled_train_loader):


                inputs = torch.cat((inputs_l, inputs_u, inputs_s_u), dim=0)
                inputs, targets_l, targets_u = inputs.to(self._device), targets_l.to(self._device), targets_u.to(self._device)
                pse_targets_u = pse_targets_u.to(self._device)
                output = self._network(inputs)
                logits = output["logits"]
                aux_logits = output["aux_logits"]
                logits_l = logits[:self.labeled_batch_size]
                logits_u, logits_s_u = logits[self.labeled_batch_size:].chunk(2, dim=0)
                aux_logits_l = aux_logits[:self.labeled_batch_size]
                aux_logits_u, aux_logits_s_u = aux_logits[self.labeled_batch_size:].chunk(2, dim=0)

                targets_old = torch.cat((targets_l, targets_u), dim=0).clone()  # 
                targets = torch.cat((targets_l, torch.ones_like(pse_targets_u)*-100), dim=0)
                targets_withpse = torch.cat((targets_l, pse_targets_u), dim=0)

               
                loss_clf = F.cross_entropy(logits_l, targets_l)

                
                aux_targets_l = targets_l.clone()
                aux_targets_l = torch.where(
                    aux_targets_l - self._known_classes + 1 > 0,
                    aux_targets_l - self._known_classes + 1,
                    0,
                )
                
                #aux_targets = aux_targets * lab_index_task + torch.ones_like(aux_targets) * -100 * (1-lab_index_task) # 20230608
                loss_aux = F.cross_entropy(aux_logits_l, aux_targets_l)

                ## Semi-supervised Loss
                with torch.no_grad():
                    predi_w = F.softmax(logits_u.clone(), 1)
                    predi_l = F.softmax(logits_l.clone(), 1)
                    wprobs, wpslab = predi_w.max(1)
                    

                    #wpslab_target = (targets>-100) * targets + wpslab * (targets==-100)  # 标记样本是targets, 无标记样本是预测
                    #wpslab_pse = (targets_withpse>-100) * targets_withpse + wpslab * (targets_withpse==-100) # Replace wpslab of labeled data into original targets, wpslab of unlabeled data into previous pseudo labels
                    wpslab_pse = (pse_targets_u>-100) * pse_targets_u  + (pse_targets_u == -100) * wpslab
                    

                    aux_logits_w = F.softmax(aux_logits.clone(), 1)
                    aux_wprobs, aux_wpslab = aux_logits_w.max(1)
                    #aux_wpslab_target = (targets>-100) * aux_targets + aux_wpslab * (targets==-100)

                mask = ((wprobs.ge(self.threshold).float() + (pse_targets_u>-100)) > 0).float()
                

                def sigmoid_growth(x):
                    '''
                    L / (1 + np.exp(-k*(x - x0)))
                    # 
                    L = 1  # 
                    k = 1  # 
                    x0 = 5   #
                    '''
                    return 1 / (1 + np.exp(-1*(x - int(self.total_exp * 0.5))))
                
                if self.insert_pse_progressive == True:
                    if self.insert_pse == 'threshold':
                        if self._cur_task>4:
                            self.pse_weight = 0.5  # 
                    if self.insert_pse == 'logitstic':
                        self.pse_weight = sigmoid_growth(self._cur_task)
                    '''
                    if epoch<20:
                        self.usp_weight = 0
                    else:
                        self.usp_weight = 1 # sigmoid_growth(self._cur_task)
                    '''
                #self.usp_weight = 0.1
                


                uloss  = self.usp_weight * ((1-self.pse_weight) * torch.mean((mask * self.uce_loss(logits_s_u, wpslab))) + \
                                                self.pse_weight * torch.mean((mask * self.uce_loss(logits_s_u, wpslab_pse ))))
                

                #uloss_aux = self.usp_weight * torch.mean(self.uce_loss(aux_logits_s_u, aux_wpslab))
                #uloss += uloss_aux
                
                ## DSGD Loss

                logits_old = self._old_network(inputs)['logits']
                logits_lu_old = logits_old[:self.batch_size]
                logits_lu = logits[:self.batch_size]
                #logits_u, logits_s_u = logits[self.labeled_batch_size:].chunk(2, dim=0)

                if self.match_weight>0: 
                    loss_match_dsgd = self.match_weight * matching_loss(logits_lu_old, logits_lu, self._known_classes, targets_withpse, \
                                                                        targets, rw_alpha=self.rw_alpha, T=self.rw_T, gamma=self.gamma_ml)['matching_loss_seed']
                else:
                    loss_match_dsgd = torch.tensor(0)
                
                if self.topo_weight>0:    
                    embed_type=self.args["embed_type"]
                    if embed_type == 'logits':
                        if self.topo_weight_sample>-1:
                            match_topo = matching_loss_topo_samplewise(logits_lu_old, logits_lu, \
                                                                                           self._known_classes, targets_withpse, targets, relation='mse', \
                                                                                           targets_old=targets_old, rw_alpha=self.rw_alpha, T=self.rw_T, \
                                                                                           gamma=self.gamma_ml, simplex_dim=self.args["toposimpl_dim"],\
                                                                                           embed_type=self.args["embed_type"])
                            loss_match_topo = self.topo_weight_sample * match_topo['matching_loss_seed']
                            class_proto_index = match_topo['class_proto_index']
                        else:
                            loss_match_topo = torch.tensor(0)
                            
                        #print(1, self.topo_weight)
                        if self.topo_weight_class>0:
                            loss_match_topo += self.topo_weight_class * matching_loss_topo_protowise(logits_lu_old, logits_lu, \
                                                                                           self._known_classes, targets_withpse, targets, relation='mse', \
                                                                                           targets_old=targets_old, rw_alpha=self.rw_alpha, T=self.rw_T, \
                                                                                           gamma=self.gamma_ml, simplex_dim=self.args["toposimpl_dim"],\
                                                                                           embed_type=self.args["embed_type"], class_proto_index=class_proto_index)['matching_loss_seed']
                        #print(2, loss_match_topo)
                    elif embed_type == 'vector':
                        vector = self._network(inputs)['features']
                        vector_old = self._old_network(inputs)['features']
                        loss_match_topo = self.topo_weight * matching_loss_topo_samplewise(vector_old, vector, self._known_classes, targets_withpse, targets, \
                                                                                           relation='mse', targets_old=targets_old, rw_alpha=self.rw_alpha, \
                                                                                           T=self.rw_T, gamma=self.gamma_ml, simplex_dim=self.args["toposimpl_dim"], \
                                                                                            embed_type=self.args["embed_type"])['matching_loss_seed']
                    elif embed_type == 'logits_global':
                        match_topo = matching_loss_topo2(logits_lu_old, logits_lu, \
                                                                                           self._known_classes, targets_withpse, targets, relation='mse', \
                                                                                           targets_old=targets_old, rw_alpha=self.rw_alpha, T=self.rw_T, \
                                                                                           gamma=self.gamma_ml, simplex_dim=self.args["toposimpl_dim"],\
                                                                                           embed_type=self.args["embed_type"])
                        loss_match_topo = self.topo_weight * match_topo['matching_loss_seed']
                        #print(2, loss_match_topo)
                else:
                    loss_match_topo = torch.tensor(0)

                loss_match = loss_match_dsgd + loss_match_topo
                #print('loss_match_dsgd', loss_match_dsgd, 'loss_match_topo', loss_match_topo)

                loss = loss_clf + loss_aux + uloss + loss_match

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                losses_aux += loss_aux.item()
                losses_clf += loss_clf.item()
                ulosses += uloss.item()
                losses_match_logits += loss_match.item()


                _, preds = torch.max(logits[:self.batch_size], dim=1)
                correct += preds.eq(targets_old.expand_as(preds)).cpu().sum()
                total += len(targets_old)



            ##record pseudo labels
            
            info_pseudo = "aux {:.2f}, aux_cur  {:.2f}, aux_old {:.2f}, pse {:.2f}, pse_cur {:.2f}, pse_old {:.2f}".format(
                    torch.tensor(aux_pseudolabel).mean(),
                    torch.tensor(aux_pseudolabel_current).mean(),
                    torch.tensor(aux_pseudolabel_old).mean(),
                    torch.tensor(pseudolabel).mean(),
                    torch.tensor(pseudolabel_current).mean(),
                    torch.tensor(pseudolabel_old).mean(),
                    )
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if (epoch+1) % 5 == 0 or epoch==0:
                test_acc, cnn_accy = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_aux {:.3f}, Loss_u {:.3f}, Loss_match {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(labeled_train_loader),
                    losses_clf / len(labeled_train_loader),
                    losses_aux / len(labeled_train_loader),
                    ulosses / len(labeled_train_loader),
                    losses_match_logits / len(labeled_train_loader),
                    train_acc,
                    test_acc,
                )
                logging.info(cnn_accy["grouped"])
                logging.info(info)
                logging.info(info_pseudo)
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_aux {:.3f}, Loss_u {:.3f}, Loss_match {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(labeled_train_loader),
                    losses_clf / len(labeled_train_loader),
                    losses_aux / len(labeled_train_loader),
                    ulosses / len(labeled_train_loader),
                    losses_match_logits / len(labeled_train_loader),
                    train_acc,
                )
                #logging.info(info)
            prog_bar.set_description(info)
        #logging.info(info)
