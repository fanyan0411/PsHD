import copy
import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.toolkit import tensor2numpy, accuracy
from scipy.spatial.distance import cdist

EPSILON = 1e-8
batch_size = 64
from torch.nn import functional as F
import networkx as nx
import gudhi as gd
class BaseLearner(object):
    def __init__(self, args):
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self._old_network = None
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        self._targets_memory_lab_idx = np.array([])  
        self._pse_targets_memory = np.array([])      
        self.topk = 2 #5
        

        self._memory_size_supervise = args["memory_size_supervise"]
        self._memory_size_unsupervise = args["memory_size_unsupervise"]
        self._memory_per_class = args.get("memory_per_class", None)
        self._fixed_memory = args.get("fixed_memory", False)
        self._device = args["device"][0]
        self._multiple_gpus = args["device"]
        self._incre = args["increment"]
        self._init_cls = args["init_cls"]
        self.oldpse_thre = args["oldpse_thre"]
        self.args = args


    @property
    def exemplar_size(self):
        assert len(self._data_memory) == len(
            self._targets_memory
        ), "Exemplar size error."
        return len(self._targets_memory)

    @property
    def samples_per_class(self):
        if self._fixed_memory:
            return self._memory_per_class
        else:
            assert self._total_classes != 0, "Total classes is 0"
            return self._memory_size_supervise // self._total_classes, self._memory_size_unsupervise // self._total_classes

    @property
    def feature_dim(self):
        if isinstance(self._network, nn.DataParallel):
            return self._network.module.feature_dim
        else:
            return self._network.feature_dim

    def build_rehearsal_memory(self, data_manager, per_class_super, per_class_unsuper):
        if self._fixed_memory:
            self._construct_exemplar_unified(data_manager, per_class)
        else:
            self._reduce_exemplar(data_manager, per_class_super, per_class_unsuper)
            self._construct_exemplar(data_manager, per_class_super, per_class_unsuper)
            



    def save_checkpoint(self, filename):
        self._network.cpu()
        save_dict = {
            "tasks": self._cur_task,
            "model_state_dict": self._network.state_dict(),
        }
        torch.save(save_dict, "{}_{}.pkl".format(filename, self._cur_task))

    def after_task(self):
        pass

    def _evaluate(self, y_pred, y_true, increment=10):
        ret = {}
        grouped = accuracy(y_pred.T[0], y_true, self._known_classes, increment=increment)
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        ret["top{}".format(self.topk)] = np.around(
            (y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true),
            decimals=2,
        )

        return ret

    def eval_task(self, increment=10):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true, increment=increment)

        if hasattr(self, "_class_means"):
            y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
            nme_accy = self._evaluate(y_pred, y_true, increment=increment)
        else:
            nme_accy = None

        return cnn_accy, nme_accy

    def incremental_train(self):
        pass

    def _train(self):
        pass

    def _get_memory(self):
        if len(self._data_memory) == 0:
            return None
        else:
            return (self._data_memory, self._targets_memory, self._pse_targets_memory, self._targets_memory_lab_idx)  # 20230515

    def _compute_accuracy(self, model, loader, old_class=False):
        model.eval()
        correct, total = 0, 0
        y_pred, y_true = [], []  
        for i, (_, inputs, inputs_s, targets, pse_targets, lab_index_task) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
                if old_class:
                    outputs = outputs[:, :self._known_classes]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)
            
            ## Record accuracy of each task
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        cnn_accy = self._evaluate(np.concatenate(y_pred), np.concatenate(y_true), increment=10)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2), cnn_accy

    def _compute_memory_graph(self, model, loader, old_class=False):
        model.to(self._device).eval()
        correct, total = 0, 0
        features, logits, y_pred, y_true = [], [], [], []  
        for i, (_, inputs, inputs_s, targets, pse_targets, lab_index_task) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                feature = model(inputs)["features"]
                outputs = model(inputs)["logits"]
                if old_class:
                    outputs = outputs[:, :self._known_classes]
            
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)
            
            ## Record accuracy of each task
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
            features.append(feature)
            logits.append(outputs)
        features = torch.cat(features).cpu()
        logits = torch.cat(logits).cpu()
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)

        softmax = nn.Softmax(dim=1)
        embed_type = 'logits'
        if embed_type == 'logits':
            P_memory = ((F.cosine_similarity(softmax(logits).unsqueeze(1), softmax(logits), dim=-1)))
        elif embed_type == 'vector':
            P_memory = ((F.cosine_similarity(features.unsqueeze(1), features, dim=-1)))
        
        adj_old = np.array(((P_memory > 0.8) * P_memory).cpu())  # 构建graph
        G = nx.from_numpy_array(adj_old)
        Persistency_Homology = {}
        #[g.items() for i, g in nx.all_pairs_shortest_path_length(G, cutoff=5)]
        for i in np.arange(adj_old.shape[0]) : #np.array(index_match_loss.cpu()) range(num)
            v_labels = adj_old
            '''
            tmp_subgraph_old = P_memory[np.ix_(v_labels, v_labels)]  # 邻接矩阵
            tmp_subgraph_old_dis = 1-tmp_subgraph_old
            rips_complex_old = gd.RipsComplex(distance_matrix=tmp_subgraph_old_dis,max_edge_length=1.)
            simplex_tree_old = rips_complex_old.create_simplex_tree(max_dimension=3) 
            simplex_tree_old.compute_persistence()
            p_old = simplex_tree_old.persistence_pairs()
            '''
            Persistency_Homology[i] = {'v_labels': v_labels} #, 'p_old': p_old
        return P_memory, Persistency_Homology #features, logits, y_pred, y_true
    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, inputs_s, targets, pse_targets, lab_index_task) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _eval_output(self):
        self._network.eval()
        y_pred, y_true, logits_all, features_all = [], [], [], []
        for _, (_, inputs, inputs_s, targets, pse_targets, lab_index_task) in enumerate(self.train_loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
                features = self._network(inputs)["features"]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            features_all.append(features.cpu().numpy().astype(np.float16))
            logits_all.append(outputs.cpu().numpy().astype(np.float16))
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        
        for _, (_, inputs, inputs_s, targets, pse_targets, lab_index_task) in enumerate(self.test_loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
                features = self._network(inputs)["features"]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            features_all.append(features.cpu().numpy().astype(np.float16))
            logits_all.append(outputs.cpu().numpy().astype(np.float16))
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())


        return {'y_pre':np.concatenate(y_pred), 'y_true':np.concatenate(y_true), 'logits_all':np.concatenate(logits_all), 'features_all':np.concatenate(features_all),
                'y_pre_test':np.concatenate(y_pred), 'y_true_test':np.concatenate(y_true), 'logits_all_test':np.concatenate(logits_all), 'features_all_test':np.concatenate(features_all)}  # [N, topk]

    def _eval_nme(self, loader, class_means):
        self._network.eval()
        vectors, y_true, _ = self._extract_vectors(loader)  
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T

        dists = cdist(class_means, vectors, "sqeuclidean")  # [nb_classes, N]
        scores = dists.T  # [N, nb_classes], choose the one with the smallest distance

        return np.argsort(scores, axis=1)[:, : self.topk], y_true  # [N, topk]

    def _extract_vectors(self, loader):
        self._network.eval()
        vectors, targets = [], []
        lab_index_task = []
        for _, _inputs, _inputs_s, _targets, _pse_targets, _lab_index_task in loader:  
            _targets = _targets.numpy()
            _lab_index_task = _lab_index_task.numpy()
            if isinstance(self._network, nn.DataParallel):
                _vectors = tensor2numpy(
                    self._network.module.extract_vector(_inputs.to(self._device))
                )
            else:
                _vectors = tensor2numpy(
                    self._network.extract_vector(_inputs.to(self._device))
                )

            vectors.append(_vectors)
            targets.append(_targets)
            lab_index_task.append(_lab_index_task)

        return np.concatenate(vectors), np.concatenate(targets), np.concatenate(lab_index_task)

    def _extract_vectors_and_psedolabel(self, loader):
        self._network.eval()
        vectors, targets = [], []
        lab_index_task = []
        logits, preds = [], []   
        for _, _inputs, _inputs_s, _targets, _pse_targets, _lab_index_task in loader:  
            _targets = _targets.numpy()
            _lab_index_task = _lab_index_task.numpy()
            if isinstance(self._network, nn.DataParallel):
                _vectors = self._network.module.extract_vector(_inputs.to(self._device))
                #outdim = self._network.module.aux_fc.in_features
                #_logits = self._network.module.aux_fc(_vectors[:, -outdim:])['logits']
                #_preds = _logits[:,:10].argmax(1)
                _logits = self._network.module.fc(_vectors)['logits']
                _preds = _logits[:,self._known_classes:self._total_classes].argmax(1)
            else:
                _vectors = self._network.extract_vector(_inputs.to(self._device))
                #outdim = self._network.module.aux_fc.in_features
                #_logits = self._network.module.aux_fc(_vectors)['logits']
                #_preds = _logits[:,:10].argmax(1)
                _logits = self._network.module.fc(_vectors)['logits']
                _preds = _logits[:,self._known_classes:self._total_classes].argmax(1)
                

            vectors.append(tensor2numpy(_vectors))
            targets.append(_targets)
            lab_index_task.append(_lab_index_task)
            #记录伪标签
            logits.append(tensor2numpy(_logits))
            preds.append(tensor2numpy(_preds))
        return np.concatenate(vectors), np.concatenate(targets), np.concatenate(lab_index_task), \
            np.concatenate(logits), np.concatenate(preds)


    def _reduce_exemplar(self, data_manager, m_super, m_unsuper):
        logging.info("Reducing exemplars...({} per classes)".format(m_super))
        dummy_data, dummy_targets = copy.deepcopy(self._data_memory), copy.deepcopy(self._targets_memory)
        dummy_pse_targets = copy.deepcopy(self._pse_targets_memory)     
        dummy_data_lab_idx = copy.deepcopy(self._targets_memory_lab_idx)  
        self._class_means = np.zeros((self._total_classes, self.feature_dim))
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        self._pse_targets_memory = np.array([])      
        self._targets_memory_lab_idx = np.array([])  

       


        dummy_targets_sp = dummy_targets * dummy_data_lab_idx + (1-dummy_data_lab_idx)*-100
        for class_idx in range(self._known_classes):
            mask_sp = np.where(dummy_targets_sp == class_idx)[0]    
            dd, dt = dummy_data[mask_sp][:m_super], dummy_targets_sp[mask_sp][:m_super]
            dpt = dummy_pse_targets[mask_sp][:m_super]  
            dtli = dummy_data_lab_idx[mask_sp][:m_super]



            self._data_memory = (
                np.concatenate((self._data_memory, dd))
                if len(self._data_memory) != 0
                else dd
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, dt))
                if len(self._targets_memory) != 0
                else dt
            )

            self._targets_memory_lab_idx = (
                np.concatenate((self._targets_memory_lab_idx, dtli))
                if len(self._targets_memory_lab_idx) != 0
                else dtli
            )   

            self._pse_targets_memory = (
                np.concatenate((self._pse_targets_memory, dpt))
                if len(self._pse_targets_memory) != 0
                else dpt
            )

            # Exemplar mean
            idx_dataset, _ = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(dd, dt, dtli, dpt)
            )
            #appendent=(selected_exemplars[-m_unsuper:], exemplar_targets[-m_unsuper:], pse_exemplar_targets[-m_unsuper:], selected_exemplars_lab_idx[-m_unsuper:]),
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _, _= self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

        # 无标记数据
            if class_idx == self._known_classes-1:
                m_unsuper = m_unsuper * self._known_classes
                mask_un = np.where(dummy_data_lab_idx==0)[0]
                dummy_data_un = dummy_data[mask_un]
                selected_index = torch.randperm(len(dummy_data_un))[:m_unsuper]  #随机选择一批无标记数据，不管伪标签是什么
                dummy_data_un_sele = dummy_data_un[selected_index]
                dummy_target_un_sele = dummy_targets[mask_un][selected_index]
                dummy_pse_target_un_sele = dummy_pse_targets[mask_un][selected_index]
                dummy_data_lab_idx_un = dummy_data_lab_idx[mask_un][selected_index]

                self._data_memory = (
                        np.concatenate((self._data_memory, dummy_data_un_sele))
                        if len(self._data_memory) != 0
                        else dummy_data_un_sele
                    )
                self._targets_memory = (
                    np.concatenate((self._targets_memory, dummy_target_un_sele))
                    if len(self._targets_memory) != 0
                    else dummy_target_un_sele
                )

                self._targets_memory_lab_idx = (
                    np.concatenate((self._targets_memory_lab_idx, dummy_data_lab_idx_un))
                    if len(self._targets_memory_lab_idx) != 0
                    else dummy_data_lab_idx_un
                )   #

                self._pse_targets_memory = (
                    np.concatenate((self._pse_targets_memory, dummy_pse_target_un_sele))
                    if len(self._targets_memory) != 0
                    else dummy_pse_target_un_sele
                )



    def _construct_exemplar(self, data_manager, m_super, m_unsuper):
        logging.info("Constructing supervised exemplars...({} per classes)".format(m_super))
        logging.info("Constructing unsupervised exemplars...({} per classes)".format(m_unsuper))
        logging.info("Constructing graph exemplars...({} per classes)".format(m_unsuper))

        



        for class_idx in range(self._known_classes, self._total_classes): #
            #print('class_idx', class_idx)
            

            data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )  
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )

            vectors, _, lab_index_tasl_class = self._extract_vectors(idx_loader)  #
            #vectors, targets, lab_index_tasl_class, logits, pseulabel = self._extract_vectors_and_psedolabel(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)
            ## 
            if self.full_supervise == True:
                super_idx = range(idx_dataset.lab_index_task.shape[0])
            else:
                super_idx = np.where(idx_dataset.lab_index_task==1)[0]
            

            # Select
            selected_exemplars = []
            exemplar_vectors = []  # [n, feature_dim]
            
            ## Record id of unlabeled data
            selected_exemplars_lab_idx = []
            selected_exemplars_targets = []
            selected_exemplars_pse_targets = []
        
            ##　Choose labeled samples
            if m_super:# and m_super<super_idx.shape[0]:
                vectors_super = vectors[super_idx]
                data_super = data[super_idx]
                class_mean_supervise = np.mean(vectors_super, axis=0)
                i = 0
                for k in range(1, m_super + 1):
                    if len(vectors_super)<1:
                        break
                    S = np.sum(
                        exemplar_vectors, axis=0
                    )  # [feature_dim] sum of selected exemplars vectors
                    mu_p = (vectors_super + S) / k  # [n, feature_dim] sum to all vectors
                    i = np.argmin(np.sqrt(np.sum((class_mean_supervise - mu_p) ** 2, axis=1))) #
                    
                    selected_exemplars.append(
                        np.array(data_super[i])
                    )  # New object to avoid passing by inference
                    selected_exemplars_lab_idx.append(np.array(1))
                    selected_exemplars_targets.append(np.array(class_idx))
                    selected_exemplars_pse_targets.append(np.ones_like(class_idx)*-100)
                    exemplar_vectors.append(
                        np.array(vectors_super[i])
                    )  # New object to avoid passing by inference

                    vectors_super = np.delete(
                        vectors_super, i, axis=0
                    )  # Remove it to avoid duplicative selection
                    data_super = np.delete(
                        data_super, i, axis=0
                    )  # Remove it to avoid duplicative selection
            
            


            if m_unsuper and (class_idx) == self._total_classes-1 and m_unsuper<(data.shape[0]-super_idx.shape[0]): #

                data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(self._known_classes, self._total_classes),
                source="train", 
                mode="test",
                ret_data=True,
                )
                m_unsuper = m_unsuper * (self._total_classes - self._known_classes)
                idx_loader = DataLoader(
                    idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
                )
                #vectors, _, lab_index_tasl_class = self._extract_vectors(idx_loader)  #
                vectors, _, lab_index_tasl_class, logits, pseulabel = self._extract_vectors_and_psedolabel(idx_loader) # pseulabel和
                vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
                class_mean = np.mean(vectors, axis=0)
                
                
                unsuper_idx = np.where((idx_dataset.lab_index_task==0) * (logits.max(1)>self.oldpse_thre))[0]  ## save unlabeled samples whose pseudo labels areconfident 
                if len(unsuper_idx) < m_unsuper:
                    #num_supp_idx = m_unsuper + 1 - len(unsuper_idx)
                    #unsuper_idx = np.stack(unsuper_idx,(np.random.permutation(len(unsuper_idx))[:num_supp_idx]))
                    m_unsuper = len(unsuper_idx)
                vectors_unsuper = vectors[unsuper_idx]
                data_unsuper = data[unsuper_idx]
                #targets_unsuper = targets[unsuper_idx]
                if 'icarl' in self.args["config"].split('/')[-1] or 'der' in self.args["config"].split('/')[-1]: ## 
                    pseulabel = pseulabel + self._known_classes

                targets_unsuper = targets[unsuper_idx]
                class_mean_unsupervise = np.mean(vectors_unsuper, axis=0)
                

                pse_targets_unsuper = pseulabel[unsuper_idx]  #
                


                for k in range(1, m_unsuper + 1):
                    S = np.sum(
                        exemplar_vectors, axis=0
                    )  # [feature_dim] sum of selected exemplars vectors
                    mu_p = (vectors_unsuper + S) / k  # [n, feature_dim] sum to all vectors
                    i = np.argmin(np.sqrt(np.sum((class_mean_unsupervise - mu_p) ** 2, axis=1)))
                    selected_exemplars.append(
                        np.array(data_unsuper[i])
                    )  # New object to avoid passing by inference
                    
                    selected_exemplars_lab_idx.append(np.array(0))  # 20230515
                    selected_exemplars_targets.append(np.array(targets_unsuper[i]))
                    selected_exemplars_pse_targets.append(np.array(pse_targets_unsuper[i]))
                    exemplar_vectors.append(
                        np.array(vectors_unsuper[i])
                    )  # New object to avoid passing by inference

                    vectors_unsuper = np.delete(
                        vectors_unsuper, i, axis=0
                    )  # Remove it to avoid duplicative selection
                    data_unsuper = np.delete(
                        data_unsuper, i, axis=0
                    )  # Remove it to avoid duplicative selection
                    targets_unsuper = np.delete(
                        targets_unsuper, i, axis=0
                    )

            elif m_unsuper and (class_idx) == self._total_classes-1 and m_unsuper>=(data.shape[0]-super_idx.shape[0]):
                 # 
                data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(self._known_classes, self._total_classes),
                source="train", 
                mode="test",
                ret_data=True,
                )
                m_unsuper = m_unsuper * (self._total_classes - self._known_classes)
                idx_loader = DataLoader(
                    idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
                )
                #vectors, _, lab_index_tasl_class = self._extract_vectors(idx_loader)  
                vectors, _, lab_index_tasl_class, logits, pseulabel = self._extract_vectors_and_psedolabel(idx_loader)
                vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
                class_mean = np.mean(vectors, axis=0)
                
                unsuper_idx = np.where((idx_dataset.lab_index_task==0) * (logits.max(1)>self.oldpse_thre))[0]  
                if len(unsuper_idx) < m_unsuper:
                    #num_supp_idx = m_unsuper + 1 - len(unsuper_idx)
                    #unsuper_idx = np.stack(unsuper_idx,(np.random.permutation(len(unsuper_idx))[:num_supp_idx]))
                    m_unsuper = len(unsuper_idx)
                vectors_unsuper = vectors[unsuper_idx]
                data_unsuper = data[unsuper_idx]
                #targets_unsuper = targets[unsuper_idx]
                if 'icarl' in self.args["config"].split('/')[-1]: 
                    pseulabel = pseulabel + self._known_classes

                targets_unsuper = targets[unsuper_idx]
                class_mean_unsupervise = np.mean(vectors_unsuper, axis=0)
            
                
                pse_targets_unsuper = pseulabel[unsuper_idx]  #
                
                for k in range(m_unsuper):
                    
                    selected_exemplars.append(
                        np.array(data_unsuper[k])
                    )  # New object to avoid passing by inference
                    
                    selected_exemplars_lab_idx.append(np.array(0))  # 20230515
                    selected_exemplars_targets.append(np.array(targets_unsuper[k]))
                    selected_exemplars_pse_targets.append(np.array(pse_targets_unsuper[k]))
                    exemplar_vectors.append(
                        np.array(vectors_unsuper[k])
                    )  # New object to avoid passing by inference


            # uniques = np.unique(selected_exemplars, axis=0)
            # print('Unique elements: {}'.format(len(uniques)))
            selected_exemplars = np.array(selected_exemplars)
            selected_exemplars_lab_idx = np.array(selected_exemplars_lab_idx)  #
            exemplar_targets = np.array(selected_exemplars_targets)
            pse_exemplar_targets = np.array(selected_exemplars_pse_targets)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            self._targets_memory_lab_idx = (
                np.concatenate((self._targets_memory_lab_idx, selected_exemplars_lab_idx))
                if len(self._targets_memory_lab_idx) != 0
                else selected_exemplars_lab_idx
            )

            self._pse_targets_memory = (
                np.concatenate((self._pse_targets_memory, pse_exemplar_targets))
                if len(self._pse_targets_memory) != 0
                else pse_exemplar_targets
            )
            

            # Exemplar mean  of labeled samples 
            idx_dataset, task_idxes = data_manager.get_dataset(   
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars[:m_super], exemplar_targets[:m_super], pse_exemplar_targets[:m_super], selected_exemplars_lab_idx[:m_super]),
            )  # 
            if len(idx_loader) == 0:
                print('quit')
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean




    def _construct_exemplar_unified(self, data_manager, m):
        logging.info(
            "Constructing exemplars for new classes...({} per classes)".format(m)
        )
        _class_means = np.zeros((self._total_classes, self.feature_dim))

        # Calculate the means of old classes with newly trained network
        for class_idx in range(self._known_classes):
            mask = np.where(self._targets_memory == class_idx)[0]
            class_data, class_targets = (
                self._data_memory[mask],
                self._targets_memory[mask],
            )

            class_dset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(class_data, class_targets)
            )
            class_loader = DataLoader(
                class_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean

        # Construct exemplars for new classes and calculate the means
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, class_dset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            class_loader = DataLoader(
                class_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )

            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []
            for k in range(1, m + 1):
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))

                selected_exemplars.append(
                    np.array(data[i])
                )  # New object to avoid passing by inference
                exemplar_vectors.append(
                    np.array(vectors[i])
                )  # New object to avoid passing by inference

                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                data = np.delete(
                    data, i, axis=0
                )  # Remove it to avoid duplicative selection

            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            # Exemplar mean
            exemplar_dset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            exemplar_loader = DataLoader(
                exemplar_dset, batch_size=batch_size, shuffle=False, num_workers=4
            )
            vectors, _ = self._extract_vectors(exemplar_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean

        self._class_means = _class_means
