import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.inc_net import CosineIncrementalNet
from utils.toolkit import target2onehot, tensor2numpy
from utils.data_sampler import TwoStreamBatchSampler
import os, json, wandb, math
from pathlib import Path
from utils.distill_loss import  matching_loss, matching_loss_topo_samplewise, matching_loss_topo_protowise
#from utils import Graph_fy
EPSILON = 1e-8
from time import time
import pickle
init_epoch =  200 #80  # 50 # 200 # 120
init_lr = 0.1
init_milestones = [60, 120, 170] #[40, 60, 70] # [30, 40] #[60, 120, 170] # [40, 80, 100]
init_lr_decay = 0.1
init_weight_decay = 0.0005


epochs = 170  #70  # 50 # 170 # 100
lrate = 0.1
milestones = [80, 120] #[50, 60] #[30, 40] #[80, 120]  # [50, 80]
lrate_decay = 0.1
batch_size = 128 # 128
weight_decay = 2e-4
num_workers = 8
T = 2


class iCaRL(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, False)
        self.batch_size = batch_size
        self.labeled_batch_size = int(self.batch_size/2) #math.floor(float(args["label_num"])/50000 * 128) #args["labeled_batch_size"]
        
        self.label_num = args["label_num"]
        self.dataset_name = args["dataset"]
        self.uce_loss = torch.nn.CrossEntropyLoss(reduction='none') #
        self.threshold = args["pse_threshold"]
        self.usp_weight = args["usp_weight"]
        
        self.init_cls = args["init_cls"]
        self.old_samp_correct = [(torch.ones(int((args["label_size"] / args["init_cls"])))*float(args["label_num"])).unsqueeze(0)]
        self.total_exp = int(args["label_size"]/self.init_cls)
        self.init_correct_epoch = torch.zeros(self.total_exp)
        #self.device = torch.device("cuda:0") args["device"]
        #self.record_wandb(args)
        #self.use_old_psed = args["use_old_psed"]
        self.pse_weight = args["pse_weight"]
        #self.knowledge_graph = Graph_fy
        self.uloss_correct = 1
        self.rw_alpha = args["rw_alpha"]
        self.match_weight = args["match_weight"]
        self.rw_T = args["rw_T"]
        self.insert_pse_progressive = args['insert_pse_progressive']
        self.insert_pse = args['insert_pse']
        self.kd_onlylabel = args['kd_onlylabel']
        self.full_supervise = args["full_supervise"]
        self.gamma_ml = args["gamma_ml"]
        self.topo_weight = args["topo_weight"]

    def record_wandb(self, args):
        run_dir = Path("../results") / args["prefix"]
        if not run_dir.exists():
            os.makedirs(str(run_dir))
        wandb.init(config=args,
                    project=args["dataset"],
                    name=args["prefix"],
                    group=args["run_name"],
                    dir=str(run_dir),
                    job_type="training",
                    reinit=True)

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))
        ## 
        if self.args["shuffle"]:
            checkpoints_folder_name = 'checkpoints'
        else:
            checkpoints_folder_name = 'checkpoints_notshuffle'
            
        if not self.args["resume"]:
            checkpoint_name = "./{}/{}_{}_{}_{}_{}_{}_{}.pkl".format(checkpoints_folder_name, self.args["dataset"], self.args["model_name"],self.args["init_cls"],self.args["increment"],init_epoch,self.args["label_num"],self._cur_task)
            if self._cur_task == 0 or self.args["save_all_resume"]:
                if not os.path.exists(checkpoint_name):
                    self.save_checkpoint("./{}/{}_{}_{}_{}_{}_{}".format(checkpoints_folder_name, self.args["dataset"], self.args["model_name"],self.args["init_cls"],self.args["increment"],init_epoch,self.args["label_num"]))
            
    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )
        if self._get_memory() is not None:
            appendent_data = self._get_memory()
            times = 1
            appendent_data_times = {}
            if times == 1:
                appendent_data_times = appendent_data
            if times >= 2:
                for j in range(len(appendent_data)):
                    appendent_data_times[j] = np.concatenate((appendent_data[j],appendent_data[j]))
            if times >= 3:
                for j in range(len(appendent_data)):
                    appendent_data_times[j] = np.concatenate((appendent_data_times[j],appendent_data[j]))
            appendent_data_times = (appendent_data_times[0], appendent_data_times[1], appendent_data_times[2], appendent_data_times[3])
            
            
        else:
            appendent_data_times = None

            

        train_dataset, idxes = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=appendent_data_times,  # self._get_memory()
        )
        
        
        if self.labeled_batch_size:
            
            fkeys_path = './data'
            dataset_name = self.dataset_name + '_labelindex'
            destination_name = 'label_map_count_' + self.label_num + '_index_0'  

            result_path = os.path.join(fkeys_path, dataset_name, destination_name)
            with open(result_path, "r") as f:
                label_index_value = json.load(f)['values']
            if 'imagenet' in self.dataset_name:
                # x_idx = list(map(lambda i: x[i].split('/')[-1].split('.')[0], range(x.shape[0])))
                # label_index_value = [i for i, num in enumerate(x_idx) if num in label_index_value]
                label_index_value = data_manager.label_index_value 
                labeled_idxs_onehot = np.zeros(130000)
            else:
                label_index_value = list(map(int, label_index_value))
                labeled_idxs_onehot = np.zeros(50000)
            labeled_idxs_onehot[label_index_value] = 1 
            labeled_idxs_onehot_batch = labeled_idxs_onehot[idxes] 
            
            labeled_idxs_onehot_batch = train_dataset.lab_index_task 

            labeled_idxs = np.where(labeled_idxs_onehot_batch>0)[0]
            unlabeled_idxs = np.where(labeled_idxs_onehot_batch==0)[0]


            batch_sampler = TwoStreamBatchSampler(
                unlabeled_idxs, labeled_idxs, self.batch_size, self.labeled_batch_size)

        if self.full_supervise == True:
            self.train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
            )
        else:
            self.train_loader = DataLoader(
                train_dataset, batch_sampler=batch_sampler, num_workers=num_workers, pin_memory=True
            )   #
       
        test_dataset, idxes  = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers  #batch_size = batch_size
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        train_start_time = time()
        self._train(self.train_loader, self.test_loader)
        train_end_time = time() - train_start_time
        logging.info("training time of task {}:{:.3f}min, {:.3f}h".format(self._cur_task, train_end_time/60, train_end_time/3600))
        if self.full_supervise == True:
            samples_per_class = list(self.samples_per_class)
            samples_per_class[1] = 0
        else:
            samples_per_class = self.samples_per_class
        self.build_rehearsal_memory(data_manager, samples_per_class[0], samples_per_class[1])
        build_memory_time = time() - train_start_time - train_end_time
        logging.info("build_memory time of task {}:{:.3f}min, {:.3f}h".format(self._cur_task,build_memory_time/60, build_memory_time/3600))
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        

    def _train(self, train_loader, test_loader):
        if self.args["resume"] and self._cur_task == 0:
            if self.args["shuffle"]:
                checkpoints_folder_name = 'checkpoints'
            else:
                checkpoints_folder_name = 'checkpoints_notshuffle'
            checkpoint_name = "./{}/{}_{}_{}_{}_200_{}_{}.pkl".format(checkpoints_folder_name, self.args["dataset"], self.args["model_name"],self.args["init_cls"],self.args["increment"],self.args["label_num"],self._cur_task)
            #checkpoint_name = "./checkpoints/multicuda_50cls_uloss_reweightsample_beta2_Lcont_logits_resnet18_ae_latent_64_initlr01_wd5e-4_SGD_200e_mile60120170.pth"
            print('checkpoint_name', checkpoint_name)
            print('checkpoint_name', checkpoint_name)
            print('checkpoint_name', checkpoint_name)
            if self.args["save_resume_name"] != None:
                checkpoint_name = self.args["save_resume_name"]
                print('load checkpoint: ', checkpoint_name)
                print('load checkpoint: ', checkpoint_name)
                print('load checkpoint: ', checkpoint_name)
            
            
            if os.path.isfile(checkpoint_name):
                #self._network.module.load_state_dict(torch.load(checkpoint_name, map_location='cuda:3')) #["model_state_dict"]
                self._network.module.load_state_dict(torch.load(checkpoint_name)["model_state_dict"])
            else:
                print(checkpoint_name, "is none")
                self.args["resume"] = False
        self._network.to(self._device)
        if self._old_network != None:
            self._old_network.to(self._device)

        if self._cur_task == 0:
            optimizer = optim.SGD(
                self._network.parameters(),
                momentum=0.9,
                lr=init_lr,
                weight_decay=init_weight_decay,
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=init_milestones, gamma=init_lr_decay
            )
            if not self.args["resume"]:
                self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            optimizer = optim.SGD(
                self._network.parameters(),
                lr=lrate,
                momentum=0.9,
                weight_decay=weight_decay,
            )  # 1e-5
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=milestones, gamma=lrate_decay
            )
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        print('init_epoch', init_epoch)
        prog_bar = tqdm(range(init_epoch))
        old_samp_correct_task = []
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses, losses_clf, ulosses = 0.0, 0.0, 0.0
            correct, total = 0, 0
            super_total = 0
            old_samp_correct_epoch = self.init_correct_epoch.clone()
            for bi, (_, inputs, inputs_s, targets, pse_targets, lab_index_task) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                inputs_s = inputs_s.to(self._device)
                lab_index_task = lab_index_task.to(self._device)
                output = self._network(inputs)
                logits = output["logits"]

                targets_old = targets.clone()
                
                targets = targets * lab_index_task + torch.ones_like(targets) * -100 * (1-lab_index_task) # 20230608
                if self.full_supervise == True:
                    targets = targets_old.clone()
                loss_clf = F.cross_entropy(logits, targets)
                ## use the outputs of weak unlabeled data as pseudo labels
                with torch.no_grad():
                    logits_w = F.softmax(logits.clone(), 1)
                    wprobs, wpslab = logits_w.max(1)

                    wpslab = (targets>-100) * targets + wpslab * (targets==-100) #
                ## cross-entropy loss for confident unlabeled data
                mask = wprobs.ge(self.threshold).float()

                mask_new_exp_conf = torch.where(mask)[0]
                if len(mask_new_exp_conf)>0:
                    old_samp_correct_epoch = old_samp_correct_epoch.to(targets.device)
                    old_samp_correct_epoch[0] += (targets_old[mask_new_exp_conf] == wpslab[mask_new_exp_conf]).sum() / (len(mask_new_exp_conf)+1e-5)

                logits_s = self._network(inputs_s)["logits"]
                if self.full_supervise == True:
                    wpslab = targets
                uloss  = torch.mean(mask * self.uce_loss(logits_s, wpslab))  # unsupervised loss
                uloss *= self.usp_weight
                loss = loss_clf + uloss
                #loop_info['uloss'].append(uloss.item())
                #wandb.log({'super_loss':loss_clf, 'unsuper_loss':uloss,'epoch':epoch})


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                losses_clf += loss_clf.item()
                ulosses += uloss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)
                super_total += len(torch.where(targets>-100)[0])


            old_samp_correct_epoch = old_samp_correct_epoch/(bi+1)
            old_samp_correct_task.append(old_samp_correct_epoch.unsqueeze(0))


            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            train_acc_supervise = np.around(tensor2numpy(correct) * 100 / super_total, decimals=2)

            if (epoch+1) % 5 == 0 or epoch==0:
                test_acc, cnn_accy = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, uloss {:.3f}, Train_accy {:.2f}, Train_accy_supervise {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    ulosses / len(train_loader),
                    train_acc,
                    train_acc_supervise,
                    test_acc,
                )
                logging.info(cnn_accy["grouped"])
                logging.info(info)
                '''
                wandb.log({'epoch':epoch, 
                           'Train_accy':train_acc, 
                           'Train_accy_supervise':train_acc_supervise, 
                           'test_acc': test_acc})
                '''
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, uloss {:.3f}, Train_accy {:.2f}, Train_accy_supervise {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    ulosses / len(train_loader),
                    train_acc,
                    train_acc_supervise,
                )

            prog_bar.set_description(info)
        
        ## record pseudo label correct ratio in the new task
        old_samp_correct_task = old_samp_correct_task[-1] # 取最后一个eoch
        self.old_samp_correct.append(old_samp_correct_task.cpu())

        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):

        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            losses_clf = 0.0
            losses_kd = 0.0
            ulosses = 0.0
            correct, total = 0, 0
            total_supervise = 0
            losses_match_logits = 0
            for bi, (_, inputs, inputs_s, targets, pse_targets, lab_index_task) in enumerate(train_loader):
                
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                inputs_s = inputs_s.to(self._device)
                logits = self._network(inputs)["logits"]
                lab_index_task = lab_index_task.to(self._device) 
                pse_targets = pse_targets.to(self._device)

                targets_old = targets.clone()
                if not self.full_supervise == True:
                    targets = targets * lab_index_task + torch.ones_like(targets) * -100 * (1-lab_index_task) # 20230608
                targets_withpse = targets * lab_index_task + pse_targets * (1-lab_index_task) # 
                loss_clf = F.cross_entropy(logits, targets)

                if self.kd_onlylabel == True:
                    kd_onlylabel = lab_index_task
                else:
                    kd_onlylabel = None
                loss_kd =  _KD_loss(
                    logits[:, : self._known_classes],
                    self._old_network(inputs)["logits"],
                    T,
                    lab_index_task=kd_onlylabel,
                    )

                with torch.no_grad():
                    #wprobs_logits, wpslab_logits = logits.clone().max(1)
                    #wprobs, wpslab = wprobs_logits, wpslab_logits
                    wprobs, wpslab = F.softmax(logits.clone(), 1).max(1)
                    wpslab_target = (targets>-100) * targets + wpslab * (targets==-100)
                    wpslab_pse    = (targets_withpse>-100) * targets_withpse + wpslab * (targets_withpse==-100)  # 20230608 targets 
                mask = ((wprobs.ge(self.threshold).float() + (targets_withpse>-100)) > 0).float()  #targets_withpse=-100
                #mask_logits = ((wprobs_logits.ge(self.threshold).float() + (targets_withpse>-100)) > 0).float()
                #assert (mask_logits>=mask).sum() == mask.shape[0]  
                logits_s = self._network(inputs_s)["logits"]

                def sigmoid_growth(x):
                    '''
                    L / (1 + np.exp(-k*(x - x0)))
                    # 
                    L = 1  # 
                    k = 1  # 
                    x0 = 5   # 
                    '''
                    return 1 / (1 + np.exp(-1*(x - int(self.total_exp * 0.5))))
                
                if self.insert_pse_progressive:
                    if self.insert_pse == 'threshold':
                        if self._cur_task>4:
                            self.pse_weight = 0.5  # 
                    if self.insert_pse == 'logitstic':
                        self.pse_weight = sigmoid_growth(self._cur_task)
                    #print('self.insert_pse_progressive', self.insert_pse_progressive, 'self.pse_weight', self.pse_weight)
                uloss = self.usp_weight * ((1-self.pse_weight) * torch.mean(mask * self.uce_loss(logits_s, wpslab_target)) + self.pse_weight * torch.mean(mask * self.uce_loss(logits_s, wpslab_pse)))
                #uloss = torch.mean(mask * self.uce_loss(logits_s, wpslab_target))
                


                if self.match_weight>0: 
                    loss_match_dsgd = self.match_weight * matching_loss(self._old_network(inputs)["logits"], logits, self._known_classes, targets_withpse, \
                                                                        targets, rw_alpha=self.rw_alpha, T=self.rw_T, gamma=self.gamma_ml)['matching_loss_seed']
                else:
                    loss_match_dsgd = torch.tensor(0)
                
                if self.topo_weight>0:    
                    embed_type=self.args["embed_type"]
                    if embed_type == 'logits':
                        '''
                        loss_match_topo = self.topo_weight * matching_loss_topo_samplewise(self._old_network(inputs)["logits"], logits, \
                                                                                           self._known_classes, targets_withpse, targets, relation='mse', \
                                                                                           targets_old=targets_old, rw_alpha=self.rw_alpha, T=self.rw_T, \
                                                                                           gamma=self.gamma_ml, simplex_dim=self.args["toposimpl_dim"],\
                                                                                           embed_type=self.args["embed_type"])['matching_loss_seed']
                        '''
                        match_topo = matching_loss_topo_samplewise(self._old_network(inputs)["logits"], logits, \
                                                                                           self._known_classes, targets_withpse, targets, relation='mse', \
                                                                                           targets_old=targets_old, rw_alpha=self.rw_alpha, T=self.rw_T, \
                                                                                           gamma=self.gamma_ml, simplex_dim=self.args["toposimpl_dim"],\
                                                                                           embed_type=self.args["embed_type"])
                        loss_match_topo = self.topo_weight * match_topo['matching_loss_seed']
                        class_proto_index = match_topo['class_proto_index']
                        #print(1, self.topo_weight)
                        '''
                        loss_match_topo += self.topo_weight * matching_loss_topo_protowise(self._old_network(inputs)["logits"], logits, \
                                                                                           self._known_classes, targets_withpse, targets, relation='mse', \
                                                                                           targets_old=targets_old, rw_alpha=self.rw_alpha, T=self.rw_T, \
                                                                                           gamma=self.gamma_ml, simplex_dim=self.args["toposimpl_dim"],\
                                                                                           embed_type=self.args["embed_type"], class_proto_index=class_proto_index)['matching_loss_seed']
                        '''
                        #print(2, loss_match_topo)
                    elif embed_type == 'vector':
                        vector = self._network(inputs)['features']
                        vector_old = self._old_network(inputs)['features']
                        loss_match_topo = self.topo_weight * matching_loss_topo_samplewise(vector_old, vector, self._known_classes, targets_withpse, targets, \
                                                                                           relation='mse', targets_old=targets_old, rw_alpha=self.rw_alpha, \
                                                                                           T=self.rw_T, gamma=self.gamma_ml, simplex_dim=self.args["toposimpl_dim"], \
                                                                                            embed_type=self.args["embed_type"])['matching_loss_seed']
                else:
                    loss_match_topo = torch.tensor(0)

                loss_match = loss_match_dsgd + loss_match_topo
                
                loss = loss_clf + loss_kd + uloss + loss_match

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                losses_clf += loss_clf.item()
                losses_kd += loss_kd.item()
                ulosses += uloss.item()
                losses_match_logits += loss_match.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if (epoch+1) % 5 == 0 or epoch==0:
                test_acc, cnn_accy = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_kd {:.3f}, Loss_u {:.3f}, Loss_match {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_kd / len(train_loader),
                    ulosses / len(train_loader),
                    losses_match_logits / len(train_loader),
                    train_acc,
                    test_acc,
                )
                logging.info(cnn_accy["grouped"])
                logging.info(info)
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Loss_clf {:.3f}, Loss_kd {:.3f}, Loss_u {:.3f}, Loss_m {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    losses_clf / len(train_loader),
                    losses_kd / len(train_loader),
                    ulosses / len(train_loader),
                    losses_match_logits / len(train_loader),
                    train_acc
                )
                logging.info(info)
            prog_bar.set_description(info)

        logging.info('training time of each epoch: {:.3f}s'.format((time() - prog_bar.start_t)/(epoch+1)))
        

def _KD_loss(pred, soft, T, lab_index_task=None):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    if lab_index_task != None:
        return -1 * torch.einsum('nm,n->nm', torch.mul(soft, pred), lab_index_task).sum()/lab_index_task.sum()
    else:
        return -1 * torch.mul(soft, pred).sum() / pred.shape[0]
