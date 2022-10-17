import torch
import statistics
from fcpr.modules.ops import apply_transform

def triplet_dist_selection(src_points, ref_points, initial_src, initial_ref, radius, bw, tri_length_threshold):
    """
    description: select two neighbor nodes to form triplet pairs
    args:
        src_points: (n_s,3)
        ref_points: (n_r,3)
        log_n_affinity: (n_s,n_r)
        initial_src: (bw,)
        initial_ref: (bw,)
    return:
        first_node_src, first_node_ref, second_node_src, second_node_ref
    """
    # pdb.set_trace()
    initial_src_vec = src_points[initial_src][:,None,:] - src_points[initial_src][None,:,:]  #(bw,bw,3)
    # mask out nearby points
    mask_src = torch.gt(torch.linalg.norm(initial_src_vec, dim = -1), radius)  #(bw,bw)
    edge_src_dist = torch.linalg.norm(initial_src_vec[:,:,None,:] - initial_src_vec[:,None,:,:], dim = -1)  #(bw,bw,bw)
    initial_ref_vec = ref_points[initial_ref][:,None,:] - ref_points[initial_ref][None,:,:]  #(bw,bw,3)
    # mask out nearby points
    mask_ref = torch.gt(torch.linalg.norm(initial_ref_vec, dim = -1), radius)  #(bw,bw)
    edge_ref_dist = torch.linalg.norm(initial_ref_vec[:,:,None,:] - initial_ref_vec[:,None,:,:], dim = -1)  #(bw,bw,bw)
    
    e_src_dist = torch.linalg.norm(initial_src_vec, dim = -1)  #(bw,bw)
    e_ref_dist = torch.linalg.norm(initial_ref_vec, dim = -1)  #(bw,bw)

    # pdb.set_trace()
    initial_ne_dist = torch.abs(e_src_dist - e_ref_dist)  #(bw,bw)
    initial_ne_dist[torch.arange(bw),torch.arange(bw)] = 1
    initial_ne_dist = torch.clamp(1 - initial_ne_dist ** 2 / tri_length_threshold ** 2, min=0.)  #(bw,bw)
    
    initial_e_dist = torch.abs(edge_src_dist - edge_ref_dist)  #(bw,bw,bw)
    initial_e_dist[:,torch.arange(bw),torch.arange(bw)] = 1
    initial_e_dist = torch.clamp(1 - initial_e_dist ** 2 / tri_length_threshold ** 2, min=0.)  #(bw,bw,bw)
    
    final_dist = initial_e_dist * initial_ne_dist.unsqueeze(-1).repeat(1,1,bw) * initial_ne_dist.unsqueeze(1).repeat(1,bw,1)  #(bw,bw,bw)
    # mask our nearby points
    final_dist = final_dist * mask_src.unsqueeze(-1).repeat(1,1,bw) * mask_ref.unsqueeze(-2).repeat(1,bw,1)
    # final_dist = tri_dist + initial_ne_dist.unsqueeze(-1).repeat(1,1,bw) + initial_ne_dist.unsqueeze(1).repeat(1,bw,1)  #(bw,bw,bw)
    indices = torch.max(final_dist.view(bw,-1),-1).indices  #(bw,)
    # indices = torch.topk(initial_ne_dist, 2, dim=-1, largest=False).indices  #(bw,2)
    # pdb.set_trace()
    # src_idx_1 = initial_src[indices[:,0]]
    # ref_idx_1 = initial_ref[indices[:,0]]

    # src_idx_2 = initial_src[indices[:,1]]
    # ref_idx_2 = initial_ref[indices[:,1]]
    
    first_node_idx = indices.div(bw, rounding_mode='floor')  #(bw,)
    second_node_idx = indices % bw  #(bw,)

    src_idx_1 = initial_src[first_node_idx]
    ref_idx_1 = initial_ref[first_node_idx]

    src_idx_2 = initial_src[second_node_idx]
    ref_idx_2 = initial_ref[second_node_idx]

    return src_idx_1, ref_idx_1, src_idx_2, ref_idx_2

def seed_sc_sm_nms_selection(src_points, ref_points, log_n_affinity, bw, fill_x, seed_length_threshold, nms_range):
    """
    description:
    args:
        s_n_features:(n_s,d)
        r_n_features:(n_r,d)
        log_n_affinity:(n_s,n_r)
    return:
    """
    ## pdb.set_trace()
    n_s,n_r = log_n_affinity.size()
    _,indices = torch.topk(log_n_affinity.view(-1), bw * fill_x, dim=-1)
    c_src_idx = (indices.div(n_r,rounding_mode='floor')).long()  #candidate src idx
    c_ref_idx = (indices % n_r).long()  #candidate ref idx
    
    node_affinity = log_n_affinity[c_src_idx,c_ref_idx]  #(bw*4,bw*4)

    c_src_dist = torch.linalg.norm(src_points[c_src_idx][:,None,:] - src_points[c_src_idx][None,:,:], dim=-1)  #(bw*4,bw*4)
    c_ref_dist = torch.linalg.norm(ref_points[c_ref_idx][:,None,:] - ref_points[c_ref_idx][None,:,:], dim=-1)  #(bw*4,bw*4)

    # c_src_dist = torch.linalg.norm(src_points[c_src_idx].unsqueeze(1).repeat(1,bw*self.fill_x,1) - src_points[c_src_idx].unsqueeze(0).repeat(bw*self.fill_x,1,1), dim=-1)  #(bw*4,bw*4)
    # c_ref_dist = torch.linalg.norm(ref_points[c_ref_idx].unsqueeze(1).repeat(1,bw*self.fill_x,1) - ref_points[c_ref_idx].unsqueeze(0).repeat(bw*self.fill_x,1,1), dim=-1)  #(bw*4,bw*4)
    beta_dist = (c_src_dist - c_ref_dist) ** 2 / seed_length_threshold ** 2
    beta_dist = torch.clamp(1 - beta_dist, min = 0.)  #(bw*4, bw*4)

    # node_affinity = node_affinity * beta_dist  #(bw*4,bw*4)
    node_affinity = beta_dist
    # set matrix diagonal to zero
    node_affinity[torch.arange(bw * fill_x),torch.arange(bw * fill_x)] = 0

    # maximum consensual pairs
    total_weight = node_affinity.sum(-1)  #(bw * fill_x)

    # _, res_idx = torch.topk(total_weight, bw * self.fill_x, dim=-1)  #(bw,)
    scores, res_idx = torch.sort(total_weight,descending=True)  #(bw * fill_x)
    cur_node_src = c_src_idx[res_idx]
    cur_node_ref = c_ref_idx[res_idx]
    
    src_dist = torch.cdist(src_points[cur_node_src],src_points[cur_node_src])  #(alpha * fill_x, alpha * fill_x)
    ref_dist = torch.cdist(ref_points[cur_node_ref],ref_points[cur_node_ref])  #(alpha * fill_x, alpha * fill_x)
    
    # pdb.set_trace()
    score_relation = scores[:,None] >= scores[None,:]  #(alpha * fill_x, alpha * fill_x)
    
    # score_relation = score_relation.bool() | ((src_dist >= self.nms_range).bool() & (ref_dist >= self.nms_range).bool())

    score_relation = score_relation.bool() | ((src_dist >= nms_range).bool() | (ref_dist >= nms_range).bool())

    is_local_max = score_relation.min(-1)[0].float()  #(fill_x * bw,)
    # print(f'current local_max number: {is_local_max.sum()}')
    
    res_idx = torch.argsort(scores * is_local_max, descending = True)[0:bw]
    
    return log_n_affinity[cur_node_src[res_idx],cur_node_ref[res_idx]], cur_node_src[res_idx], cur_node_ref[res_idx]

def evaluate_crts_transform(node_idx, src_points, ref_points, init_transform, batch_score, bw, inlier_threshold, procrustes, lgr):
    """
    description:
    args:
        node_idx: (bw, is, 2)
        src_points: (n_s,3)
        ref_points: (n_r,3)
        init_transform: (bw, 4, 4)
        batch_score: (bw,is)
    return:
        estimated_transform: (4,4)
    """
    # pdb.set_trace()
    batch_transform = init_transform  #(bw,4,4)
    src_idx = node_idx[:,:,0].contiguous().view(-1)  #(bw*is)
    ref_idx = node_idx[:,:,1].contiguous().view(-1)  #(bw*is)
    bw_score = batch_score.unsqueeze(0).repeat(bw,1,1).view(bw,-1)  #(bw,bw*is)
    bw_src_points = src_points[src_idx].unsqueeze(0).repeat(bw,1,1)  #(bw,bw*is,3)
    bw_ref_points = ref_points[ref_idx].unsqueeze(0).repeat(bw,1,1)  #(bw,bw*is.3)

    for i in range(lgr):
        #apply batch transform
        transformed_src = apply_transform(bw_src_points,batch_transform)  #(bw,bw*is,3)
        # distance between transformed src and ref points
        transformed_dist = torch.linalg.norm(transformed_src - bw_ref_points, dim=-1)  #(bw,bw*is)
        #true or false for each pair if less than radius
        transformed_idx = torch.lt(transformed_dist,inlier_threshold)  #(bw,bw*is):bool
        # transformed_idx = torch.lt(transformed_dist * transformed_dist,self.acceptance_radius)  #(bw,bw*is):bool
        # obtain new estimated transform
        batch_transform = procrustes(bw_src_points,bw_ref_points,bw_score*transformed_idx)  #(bw,4,4)

    # # Global MSE error
    transformed_src = apply_transform(bw_src_points, batch_transform)  #(bw,bw * is,3)
    transformed_dist = torch.linalg.norm(transformed_src - bw_ref_points, dim=-1)  #(bw,bw * is)
    transformed_idx = torch.lt(transformed_dist, inlier_threshold)  #(bw, bw * is):bool
    transformed_mse = (transformed_dist - inlier_threshold) * (transformed_dist - inlier_threshold) / inlier_threshold ** 2  #(bw,bw*is)
    arg_idx = torch.argmax((transformed_idx * transformed_mse).sum(-1)).squeeze()
    # arg_idx = torch.argmax(transformed_idx.sum(-1)).squeeze()

    arg_transformed_idx = transformed_idx[arg_idx]  #(bw * is)
    corr_src_points = src_points[src_idx][arg_transformed_idx]  #(<=bw * is)
    corr_ref_points = ref_points[ref_idx][arg_transformed_idx]  #(<=bw * is)

    traj_src_points = src_points[node_idx[arg_idx,:,0]]
    traj_ref_points = ref_points[node_idx[arg_idx,:,1]]

    return batch_transform, arg_idx, corr_src_points, corr_ref_points, bw_score[arg_idx][arg_transformed_idx], traj_src_points, traj_ref_points