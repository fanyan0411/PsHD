import torch
from torch import nn
from torch.nn import functional as F
from einops import repeat
import numpy as np
import gudhi as gd
from gudhi.wasserstein import wasserstein_distance
import matplotlib.pyplot as plt
import geomloss



def matching_loss(logits_old, logits, known_classes, targets_withpse, targets,  alpha=0.2, rw_alpha=0.2, T=5, gamma=1, targets_old = None, mean=False):
    index_old = torch.where((targets_withpse<known_classes) * (targets_withpse>-100))[0] ## 
    index_old_label = torch.where((targets[index_old]<known_classes) * (targets[index_old]>-100))[0] ## 
    #print(index_old)
    logits = logits[index_old]
    logits_old = logits_old[index_old]
    

    softmax = nn.Softmax(dim=0)  # dim = 0 not 1
    P = softmax(F.cosine_similarity(logits.unsqueeze(1), logits, dim=-1) * gamma)  # 
    P_old = softmax(F.cosine_similarity(logits_old.unsqueeze(1), logits_old, dim=-1) * gamma)


    
    pi, pi_old = torch.zeros_like(P), torch.zeros_like(P)
    
    for i in range(T):
        pi += matchingmatrix(P, rw_alpha, t=(i+1))
        pi_old += matchingmatrix(P_old, rw_alpha, t=(i+1))

    
    matching_matrix = pi #rw_alpha * pi  ##［N,N］
    matching_matrix_old = pi_old #rw_alpha * pi_old  ##［N,N］
    
    #matching_matrix_seed = matching_matrix[:,index_old_label]  # [N,N_old_label]
    #matching_matrix_old_seed = matching_matrix_old[:,index_old_label]  ##［N,N_old_label］

    if T == 3 and mean==True:
        matching_loss_seed = ((matching_matrix - matching_matrix_old)**2).mean(1).sum() #.mean(1).sum() .sum() 
    else:
        matching_loss_seed = ((matching_matrix - matching_matrix_old)**2).sum()
    '''
    matching_matrix_sum, matching_matrix_old_sum = 0, 0
    for i in range(matching_matrix.shape[0]):
        matching_matrix_sum += matching_matrix[:,i][targets_old[index_old]==targets_old[index_old][i]].sum()
        matching_matrix_old_sum += matching_matrix_old[:,i][targets_old[index_old]==targets_old[index_old][i]].sum()
    print('matching_matrix', matching_matrix_sum)
    print('matching_matrix_old',matching_matrix_old_sum)
    '''
    return {'matching_loss_seed': matching_loss_seed}


def matchingmatrix(P, alpha, t=2):
    matrix = ((1-alpha)**t) * torch.matrix_power(P, t)  # 
    return matrix

def matchingmatrix_inverse(P, alpha, t=2):
    I = torch.eye(P.shape[0]).to(P.device)
    matrix = ((1-alpha)**t) * torch.inverse(I-P)  # 
    return matrix

def attention_matrix(output):
    
    normal_out = F.normalize(output)
    dis_hard = get_elu_dis(normal_out) 
    #dis_hard = get_elu_dis(soft_label)
    
    gamma=1
    simi_hard = torch.exp(- dis_hard * gamma)
    simi_hard = simi_hard-torch.eye(simi_hard.shape[0]).to(simi_hard.device)
    return simi_hard

def get_elu_dis(data):
        '''
        input: data [a_1, a_2, ..., a_n].T()      dim:[n, d]
        
        dist_ij = (a_i - b_j)
        '''
        n = data.shape[0]
        A2 = repeat(torch.einsum('nd -> n', data**2), 'n -> n i', i = n)
        B2 = repeat(torch.einsum('nd -> n', data**2), 'n -> i n', i = n)
        AB = data.mm(data.t())
        dist = torch.abs(A2 + B2 -2*AB)
        return torch.sqrt(dist)


def simplicial_complex_dgm(target_matrix):
    rips_complex = gd.RipsComplex(distance_matrix=target_matrix, max_edge_length=1.)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
    diag = simplex_tree.persistence(homology_coeff_field=2, min_persistence=0) # persistence diagram of each sub-cluster, the 1st value represents dimenson, 2nd value represnts persistence time 
    res = np.zeros(shape=(np.array(diag)[:,1].shape[0],2), dtype= np.float32)
    for i in range(np.array(diag).shape[0]):
        if np.isfinite(np.array(diag)[:,1][i][1]):
            res[i, :] = np.array(diag)[:, 1][i]
    return res

def max_diameter(x, y):
    """Returns a rough estimation of the diameter of a pair of point clouds.

    This quantity is used as a maximum "starting scale" in the epsilon-scaling
    annealing heuristic.

    Args:
        x ((N, D) Tensor): First point cloud.
        y ((M, D) Tensor): Second point cloud.

    Returns:
        float: Upper bound on the largest distance between points `x[i]` and `y[j]`.
    """
    mins = torch.stack((x.min(dim=0)[0], y.min(dim=0)[0])).min(dim=0)[0]
    maxs = torch.stack((x.max(dim=0)[0], y.max(dim=0)[0])).max(dim=0)[0]
    diameter = (maxs - mins).norm().item()
    return diameter

def compute_persistence(P,dim, p1b, p1d):
    
    if dim == 0:
        #d = P[p1d[:,1],p1d[:,0]]
        b = torch.zeros_like(P[p1d[:,1],p1d[:,0]])
        #diag = torch.cat((b.unsqueeze(1),d.unsqueeze(1)),dim=1)
        d = P[p1d[:,:,None], p1d[:,None,:]].view(p1d.shape[0],-1).max(1)[0]
        diag = torch.cat((b.unsqueeze(1),d.unsqueeze(1)),dim=1)
    if dim == 1:
        b = P[p1b[:,1],p1b[:,0]]
        #d_1 = P[p1d[:,1],p1d[:,0]]
        #d_2 = P[p1d[:,1],p1d[:,2]]
        #d_3 = P[p1d[:,2],p1d[:,0]]
        #d = torch.max(d_1, torch.max(d_2, d_3))
        d = P[p1d[:,:,None], p1d[:,None,:]].view(p1d.shape[0],-1).max(1)[0]
        diag = torch.cat((b.unsqueeze(1),d.unsqueeze(1)),dim=1)
    elif dim == 2:
        '''
        b_1 = P[p1b[:,1],p1b[:,0]]
        b_2 = P[p1b[:,1],p1b[:,2]]
        b_3 = P[p1b[:,2],p1b[:,0]]
        b = torch.max(b_1, torch.max(b_2, b_3))
        d_1 = P[p1d[:,1],p1d[:,0]]
        d_2 = P[p1d[:,1],p1d[:,2]]
        d_3 = P[p1d[:,2],p1d[:,0]]
        d_4 = P[p1d[:,2],p1d[:,3]]
        d_5 = P[p1d[:,1],p1d[:,3]]
        d_6 = P[p1d[:,0],p1d[:,3]]
        d_123 = torch.max(d_1, torch.max(d_2, d_3))
        d_456 = torch.max(d_4, torch.max(d_5, d_6))
        d = torch.max(d_123, d_456)
        '''
        b = P[p1b[:,:,None], p1b[:,None,:]].view(p1b.shape[0],-1).max(1)[0]
        d = P[p1d[:,:,None], p1d[:,None,:]].view(p1d.shape[0],-1).max(1)[0]
        diag = torch.cat((b.unsqueeze(1),d.unsqueeze(1)),dim=1)
    


    return diag


def peresisdiag(simplex_tree, simplix_dim, distance_matrix):
        diagm, num_simplex = [], []
        for i in simplix_dim:
            pib = torch.tensor([j[0] for j in simplex_tree if len(j[1]) == i+2]) 
            pid = torch.tensor([j[1] for j in simplex_tree if len(j[1]) == i+2])
            if len(pib) == 0:
                diagi = torch.tensor([0.,0], requires_grad=False).unsqueeze(0).cuda(distance_matrix.device)
                num_simplex.append(0)
            else:
                diagi = compute_persistence(distance_matrix, 0, pib, pid)  #　len(i[1]) 4 -> 2  len(i[1]) 3 -> 1  len(i[1]) 2 -> 0
                num_simplex.append(diagi.shape[-1])
            diagm.append(diagi)
        return diagm, num_simplex

def peresisdiagm(simplex_tree, simplex_tree_old, simplix_dim, distance_matrix, distance_matrix_old):
    '''
    simplix_dim: dimension of simplex
    '''
    diagms_old, diagms, perstots = [], [], []
    diagms, num_simplex =  peresisdiag(simplex_tree, simplix_dim, distance_matrix)
    diagms_old, num_simplex_old = peresisdiag(simplex_tree_old, simplix_dim, distance_matrix_old)

    loss = geomloss.SamplesLoss(loss="sinkhorn", p=1)
    for i in range(len(simplix_dim)):
        diameter = max_diameter(diagms_old[i].view(-1, 2), diagms[i].view(-1, 2))#num_simplex[i]
        #diameters.append(diameter)
        if diameter==0:
            perstot = torch.tensor(0.0).cuda(diagms[i].device)
        else:
            perstot = loss(diagms[i], diagms_old[i])
        perstots.append(perstot)
    loss = torch.stack(perstots).sum() / distance_matrix.shape[0]
    return loss

def matching_loss_topo(logits_old, logits, known_classes, targets_withpse, targets, relation=None, targets_old=None, alpha=0.2, rw_alpha=0.2, T=5, gamma=1, simplex_dim=[0,1,2]):
    index_old = torch.where((targets_withpse<known_classes) * (targets_withpse>-100))[0] 
    index_old_label = torch.where((targets[index_old]<known_classes) * (targets[index_old]>-100))[0] 
    logits = logits[index_old]
    logits_old = logits_old[index_old]

    softmax = nn.Softmax(dim=0)

    P = ((F.cosine_similarity(softmax(logits).unsqueeze(1), softmax(logits), dim=-1)))
    P_old = ((F.cosine_similarity(softmax(logits_old).unsqueeze(1), softmax(logits_old), dim=-1)))
    
    dimension = 4
    rips_complex_old = gd.RipsComplex(distance_matrix=P_old,max_edge_length=1.)
    simplex_tree_old = rips_complex_old.create_simplex_tree(max_dimension=3) 
   

    simplex_tree_old.compute_persistence()
    p_old = simplex_tree_old.persistence_pairs()
    # Keep only pairs that contribute to H1, i.e. (edge, triangle), and separate birth (p1b) and death (p1d)
    #for i in range(simplex_tree_old.betti_numbers()):
    
    
    #peresisdiagms_old = peresisdiagm(p_old, [0,1], P_old)
    

    rips_complex = gd.RipsComplex(distance_matrix=P,max_edge_length=1.)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=3)
    simplex_tree.compute_persistence()
    p = simplex_tree.persistence_pairs()
    
    peresisdiagms_loss = peresisdiagm(p, p_old, np.array(simplex_dim), P, P_old)
    
    # Compute the match vector


    pi, pi_old = torch.zeros_like(P), torch.zeros_like(P)
    
    for i in range(T):
        pi += matchingmatrix(P, rw_alpha, t=(i+1))
        pi_old += matchingmatrix(P_old, rw_alpha, t=(i+1))

    matching_matrix = pi #rw_alpha * pi  ##［N,N］
    matching_matrix_old = pi_old #rw_alpha * pi_old  ##［N,N］
    
    #matching_matrix_seed = matching_matrix[:,index_old_label]  # [N,N_old_label]
    #matching_matrix_old_seed = matching_matrix_old[:,index_old_label]  ##［N,N_old_label］
    
    ## 
    #matching_loss_seed = ((matching_matrix_seed - matching_matrix_old_seed)**2).mean(1).sum()
    
    matching_loss_seed = ((matching_matrix - matching_matrix_old)**2).sum()
    return {'matching_loss_seed': peresisdiagms_loss} #perstot1  + matching_loss_seed  perstot0+perstot1+perstot2

def matching_loss_topo2(logits_old, logits, known_classes, targets_withpse, targets, relation=None, targets_old=None, alpha=0.2, rw_alpha=0.2, T=5, gamma=1, simplex_dim=[0,1,2], embed_type='vector'):

    index_old = torch.where((targets_withpse<known_classes) * (targets_withpse>-100))[0] ## 
    index_old = torch.where((targets_withpse<known_classes))[0]  ## 
    index_old_label = torch.where((targets[index_old]<known_classes) * (targets[index_old]>-100))[0] ## 
    logits = logits[index_old]
    logits_old = logits_old[index_old]
    #targets = targets[index_old]
    #index_match_loss = (targets[index_old]==-100)
    index_match_loss = (targets[index_old]>=-100)
    targets_old = targets_old[index_old]

    softmax = nn.Softmax(dim=1)

    if embed_type == 'logits' or 'logits_global':
        P = ((F.cosine_similarity(softmax(logits).unsqueeze(1), softmax(logits), dim=-1)))
        P_old = ((F.cosine_similarity(softmax(logits_old).unsqueeze(1), softmax(logits_old), dim=-1)))
    elif embed_type == 'vector':
        P = ((F.cosine_similarity(logits.unsqueeze(1), logits, dim=-1)))
        P_old = ((F.cosine_similarity(logits_old.unsqueeze(1), logits_old, dim=-1)))

    adj_old = ((P_old > 0.) * P_old)
    num = P_old.shape[0]
    path_2 = torch.einsum('nm,ml->nl', (adj_old, adj_old))
    path_3 = torch.einsum('nm,ml->nl', path_2, adj_old)
    path_total = adj_old + path_2 + path_3
    peresisdiagms_losses = []
    ##  
    j = 0
    class_proto_index = []
    for i in np.array(torch.where(index_match_loss)[0].cpu()) : #np.array(index_match_loss.cpu()) range(num)
        #v_labels = [name for name, value in nx.single_source_shortest_path_length(G, i, cutoff=3).items()]
        if i not in torch.where(index_match_loss)[0].cpu():
            continue
        v_labels = (path_total[i] > 0)
        tmp_subgraph_old = P_old[v_labels][:,v_labels]  
        tmp_subgraph = P[v_labels][:,v_labels] 
        tmp_subgraph_old_dis = 1-tmp_subgraph_old 
        tmp_subgraph_dis = 1-tmp_subgraph   
        rips_complex_old = gd.RipsComplex(distance_matrix=tmp_subgraph_old_dis,max_edge_length=1.)
        simplex_tree_old = rips_complex_old.create_simplex_tree(max_dimension=3) 
        simplex_tree_old.compute_persistence()
        p_old = simplex_tree_old.persistence_pairs()

        rips_complex = gd.RipsComplex(distance_matrix=tmp_subgraph_dis,max_edge_length=1.)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=3)
        simplex_tree.compute_persistence()
        p = simplex_tree.persistence_pairs()
        #print(np.array(simplex_dim))
        peresisdiagms_loss = peresisdiagm(p, p_old, np.array(simplex_dim), tmp_subgraph_dis, tmp_subgraph_old_dis)
        peresisdiagms_losses.append(peresisdiagms_loss)

        j = j+1
        index_match_loss[torch.where(v_labels)[0]] = 0
        if index_match_loss.sum() ==0:
            break
        
        class_proto_index.append(v_labels.unsqueeze(0))
        

    
    
    if len(index_match_loss.cpu()) == 0 or len(peresisdiagms_losses)==0:
        peresisdiagms_loss = torch.tensor(0.0).cuda(logits.device)
    else:
    # Keep only pairs that contribute to H1, i.e. (edge, triangle), and separate birth (p1b) and death (p1d)
        peresisdiagms_loss = torch.stack(peresisdiagms_losses).mean()

    return {'matching_loss_seed': peresisdiagms_loss, 'class_proto_index':class_proto_index} #

def manual_sparse(dismatrix, threshold=0):
    dismatrix = (dismatrix > threshold) * dismatrix + (dismatrix <= threshold) * 0

    return dismatrix


def matching_loss_topo_samplewise(logits_old, logits, known_classes, targets_withpse, targets, relation=None, targets_old=None, alpha=0.2, rw_alpha=0.2, T=5, gamma=1, simplex_dim=[0,1,2], embed_type='vector'):

    index_old = torch.where((targets_withpse<known_classes) * (targets_withpse>-100))[0]  
    
    index_old_label = torch.where((targets[index_old]<known_classes) * (targets[index_old]>-100))[0] 
    logits = logits[index_old]
    logits_old = logits_old[index_old]
    #targets = targets[index_old]
    index_match_loss = (targets[index_old]==-100)
    targets_old = targets_old[index_old]

    softmax = nn.Softmax(dim=1)

    if embed_type == 'logits':
        P = ((F.cosine_similarity(softmax(logits).unsqueeze(1), softmax(logits), dim=-1)))
        P_old = ((F.cosine_similarity(softmax(logits_old).unsqueeze(1), softmax(logits_old), dim=-1)))
    elif embed_type == 'vector':
        P = ((F.cosine_similarity(logits.unsqueeze(1), logits, dim=-1)))
        P_old = ((F.cosine_similarity(logits_old.unsqueeze(1), logits_old, dim=-1)))

    # torch.save([logits,targets_old], 'adj_test.pth')
    adj_old = ((P_old > 0.8) * P_old)
    num = P_old.shape[0]
    path_2 = torch.einsum('nm,ml->nl', (adj_old, adj_old))
    path_3 = torch.einsum('nm,ml->nl', path_2, adj_old)
    path_total = adj_old + path_2 + path_3
    peresisdiagms_losses = []
    
    j = 0
    class_proto_index = []
    max_dimension = np.max(np.array(simplex_dim))+2
    
    for i in np.array(torch.where(index_match_loss)[0].cpu()) : #np.array(index_match_loss.cpu()) range(num)
        #v_labels = [name for name, value in nx.single_source_shortest_path_length(G, i, cutoff=3).items()]
        if i not in torch.where(index_match_loss)[0].cpu():
            continue
        v_labels = (path_total[i] > 0)
        
        tmp_subgraph_old = P_old[v_labels][:,v_labels]  
        tmp_subgraph = P[v_labels][:,v_labels]     
        tmp_subgraph_old_dis = 1-tmp_subgraph_old 
        tmp_subgraph_dis = 1-tmp_subgraph  


        rips_complex_old = gd.RipsComplex(distance_matrix=tmp_subgraph_old_dis,max_edge_length=1.)
        simplex_tree_old = rips_complex_old.create_simplex_tree(max_dimension=max_dimension) 
        simplex_tree_old.compute_persistence()
        p_old = simplex_tree_old.persistence_pairs()

        rips_complex = gd.RipsComplex(distance_matrix=tmp_subgraph_dis,max_edge_length=1.)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dimension)
        simplex_tree.compute_persistence()
        p = simplex_tree.persistence_pairs()
        #print(np.array(simplex_dim))
        peresisdiagms_loss = peresisdiagm(p, p_old, np.array(simplex_dim), tmp_subgraph_dis, tmp_subgraph_old_dis)
        peresisdiagms_losses.append(peresisdiagms_loss)

        index_match_loss[torch.where(v_labels)[0]] = 0
        class_proto_index.append(v_labels.unsqueeze(0))
        j = j+1
        if index_match_loss.sum() ==0:
            break
        

    if len(index_match_loss.cpu()) == 0 or len(peresisdiagms_losses)==0:
        peresisdiagms_loss = torch.tensor(0.0).cuda(logits.device)
    else:
    # Keep only pairs that contribute to H1, i.e. (edge, triangle), and separate birth (p1b) and death (p1d)
        peresisdiagms_loss = torch.stack(peresisdiagms_losses).mean()

    return {'matching_loss_seed': peresisdiagms_loss, 'class_proto_index':class_proto_index} #



def matching_loss_topo_protowise(logits_old, logits, known_classes, targets_withpse, targets, relation=None, targets_old=None, alpha=0.2, rw_alpha=0.2, T=5, gamma=1, simplex_dim=[0,1,2], embed_type='vector', class_proto_index=[]):

    if len(class_proto_index) == 0:
        peresisdiagms_loss = torch.tensor(0.0).cuda(logits.device)
        return {'matching_loss_seed': peresisdiagms_loss, 'class_proto_index':class_proto_index} #
    index_old = torch.where((targets_withpse<known_classes) * (targets_withpse>-100))[0] ## 
    #index_old = torch.where((targets_withpse<known_classes))[0]
    index_old_label = torch.where((targets[index_old]<known_classes) * (targets[index_old]>-100))[0] ## 
    logits = logits[index_old]
    logits_old = logits_old[index_old]
    #targets = targets[index_old]
    index_match_loss = (targets[index_old]==-100)
    targets_old = targets_old[index_old]

    class_proto_index = torch.cat(class_proto_index, dim=0).float()
    class_logits = torch.matmul(class_proto_index, logits)/class_proto_index.sum(dim=1).unsqueeze(1)
    class_logits_old = torch.matmul(class_proto_index, logits_old)/class_proto_index.sum(dim=1).unsqueeze(1)
    softmax = nn.Softmax(dim=1)

    if embed_type == 'logits':
        P_class = ((F.cosine_similarity(softmax(class_logits).unsqueeze(1), softmax(class_logits), dim=-1)))
        P_old_class = ((F.cosine_similarity(softmax(class_logits_old).unsqueeze(1), softmax(class_logits_old), dim=-1)))
    elif embed_type == 'vector':
        P = ((F.cosine_similarity(logits.unsqueeze(1), logits, dim=-1)))
        P_old = ((F.cosine_similarity(logits_old.unsqueeze(1), logits_old, dim=-1)))

    adj_old = ((P_old_class > 0.) * P_old_class)

    path_2 = torch.einsum('nm,ml->nl', (adj_old, adj_old))
    path_3 = torch.einsum('nm,ml->nl', path_2, adj_old)
    path_total = adj_old + path_2 + path_3
    peresisdiagms_losses = []
    index_match_loss = torch.ones(P_class.shape[0])
    ## 
    j = 0
    class_proto_index = []
    max_dimension = np.max(np.array(simplex_dim))+2
    for i in np.array(torch.where(index_match_loss)[0].cpu()): #np.array(index_match_loss.cpu()) range(num)
        #v_labels = [name for name, value in nx.single_source_shortest_path_length(G, i, cutoff=3).items()]
        if i not in torch.where(index_match_loss)[0].cpu():
            continue
        v_labels = (path_total[i] > 0)
        tmp_subgraph_old = P_old_class[v_labels][:,v_labels]  # 
        tmp_subgraph = P_class[v_labels][:,v_labels] #       
        tmp_subgraph_old_dis = 1-tmp_subgraph_old 
        tmp_subgraph_dis = 1-tmp_subgraph   
        rips_complex_old = gd.RipsComplex(distance_matrix=tmp_subgraph_old_dis,max_edge_length=1.)
        simplex_tree_old = rips_complex_old.create_simplex_tree(max_dimension=max_dimension) 
        simplex_tree_old.compute_persistence()
        p_old = simplex_tree_old.persistence_pairs()

        rips_complex = gd.RipsComplex(distance_matrix=tmp_subgraph_dis,max_edge_length=1.)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dimension)
        simplex_tree.compute_persistence()
        p = simplex_tree.persistence_pairs()
        #print(np.array(simplex_dim))
        peresisdiagms_loss = peresisdiagm(p, p_old, np.array(simplex_dim), tmp_subgraph_dis, tmp_subgraph_old_dis)
        peresisdiagms_losses.append(peresisdiagms_loss)

        index_match_loss[torch.where(v_labels)[0]] = 0
        if index_match_loss.sum() ==0:
            break
        j = j+1
        
        # class level topo distillation loss


    
    if len(index_match_loss.cpu()) == 0 or len(peresisdiagms_losses)==0:
        peresisdiagms_loss = torch.tensor(0.0).cuda(logits.device)
    else:
    # Keep only pairs that contribute to H1, i.e. (edge, triangle), and separate birth (p1b) and death (p1d)
        peresisdiagms_loss = torch.stack(peresisdiagms_losses).mean()



    return {'matching_loss_seed': peresisdiagms_loss, 'class_proto_index':class_proto_index} #
