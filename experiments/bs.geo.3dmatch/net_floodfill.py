from cProfile import label
from cgi import test
from math import radians
from mimetypes import init
from operator import gt
from optparse import Values
from os import link
from platform import node
from re import M
import timeit
from anyio import fail_after
from sklearn.metrics import SCORERS

from zmq import device
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed
import pdb
import statistics
import numpy as np
from scipy.spatial import KDTree
import open3d as o3d
from fcpr.modules.ops import apply_transform
from fcpr.modules.registration import WeightedProcrustes, test_weighted_procrustes
from fcpr.modules.registration.metrics import isotropic_transform_error

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
# from misc.fcgf import ResUNetBN2C as FCGF
# from misc.cal_fcgf import extract_features
from pytorch3d.ops import get_point_covariances
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboardX import SummaryWriter
from module import seed_sc_sm_nms_selection,triplet_dist_selection,evaluate_crts_transform
from eval import evaluate_fine, evaluate_registration

class Floodfill(nn.Module):
    def __init__(self, device=torch.device('cuda')):
        super(Floodfill, self).__init__()
        self.device = device
        # weighted SVD
        self.procrustes = WeightedProcrustes(return_transform=True)
        # number of neighbor nodes
        self.num_adj = 9
        # beam search hyper parameters
        # self.iter_search = 8
        self.bw = 512
        self.length = 0
        self.radius = 0.1
        self.nps = 1
        self.fstp = 120

        self.cos = nn.CosineSimilarity(dim=-1,eps=1e-6)
        self.acceptance_rmse = 0.2
        self.acceptance_radius = 0.1
        self.inlier_threshold = 0.1
        self.lgr = 5

        self.tri_length_threshold = 0.1
        self.seed_length_threshold = 0.05
        
        # nms selection
        self.nms_range = 0.05
        self.factor = 50
        self.fill_x = 20
        self.corr_sum = []
        self.eine_sum = []

        # instance norm, affine=false
        self.instance_norm = nn.InstanceNorm2d(1)

        #####################NSM Module ###################
        # power iteration algorithm
        self.num_iterations = 10

        self.gut_ein = 0
        self.gut_drei = 0
        self.test_step = 0
        self.gut_ein_rr2 = 0
        self.gut_drei_rr2 = 0

        # logger
        self.initial_rr = []
        self.initial_rr2 = []
        self.metrics = {'rre':[],'rte':[],'rmse':[],'rr':[],'ir':[],'rr2':[],'re':[],'te':[],'are':[],'ate':[], 'tir':[]}
        self.init_metrics = {'rre':[],'rte':[],'rmse':[],'rr':[],'ir':[],'rr2':[],'re':[],'te':[],'are':[],'ate':[], 'tir':[]}

        self.entropy_radius = 0.1

        self.ein_ir = []
        self.drei_ir = []
        self.drei_rr = []

        self.c_eine_corr = []
        self.c_zwei_corr = []

    def fast_floodfill(self, src_points_f, ref_points_f, s_n_features, r_n_features, gt_transform, src_points, ref_points):
        """
        description: geometric constraint guided walk based graph matching
        args:
            src_points_f: (n_s,3)
            ref_points_f: (n_r,3)
            src_feats_f: (n_s,D)
            ref_feats_f: (n_r,D)
            src_points: (n_s,3)
        Returns:
        """
        # pdb.set_trace()
        # # refined number of points for both reference and source
        self.gt_transform = gt_transform

        n_r = ref_points_f.shape[0]
        n_s = src_points_f.shape[0]

        key_src = get_point_covariances(src_points_f[None,:,:], torch.tensor((n_s),dtype=torch.long,device=self.device),24)[0][0]  #(n_s,3,3)
        # pdb.set_trace()
        e_key_src, _ = torch.linalg.eig(key_src)  #(n_s,)
        e_key_src = e_key_src.real.sort(-1,descending=True).values
        # pdb.set_trace()
        # e_key_src = F.normalize(e_key_src[:,0] / e_key_src[:,1],dim=0)
        # e_key_src = 1 - (e_key_src[:,1] / e_key_src[:,0])**2
        e_key_src = 1 - (e_key_src[:,1] / e_key_src[:,0])


        key_ref = get_point_covariances(ref_points_f[None,:,:], torch.tensor((n_r),dtype=torch.long,device=self.device),24)[0][0]  #(n_r,3,3)
        # pdb.set_trace()
        e_key_ref, _ = torch.linalg.eig(key_ref)  #(n_r,)
        e_key_ref = e_key_ref.real.sort(-1,descending=True).values
        # e_key_ref = 1 - (e_key_ref[:,1] / e_key_ref[:,0])**2
        e_key_ref = 1 - (e_key_ref[:,1] / e_key_ref[:,0])
        # e_key_ref = torch.clamp(e_key_ref,max=0.3)*2
        # e_key_ref = torch.clamp(e_key_ref + torch.heaviside(e_key_ref-0.3,torch.tensor([1.],device=self.device)), max = 1.)  #[0,0.3)+[1.)]
        # e_key_ref = F.normalize(e_key_ref[:,0] / e_key_ref[:,1],dim=0)

        # startTime = timeit.default_timer()
        ################################  Weight for graph matching ###########################################
        # log_n_affinity = torch.cdist(s_n_features, r_n_features)  #(n_s,n_r)
        log_n_affinity = torch.matmul(s_n_features,r_n_features.transpose(-2,-1))  #(n_s,n_r)
        # pdb.set_trace()
        # log_n_affinity = self.instance_norm(log_n_affinity.unsqueeze(1)).squeeze()  #(n_s,n_r)
        # log_n_affinity = log_n_affinity / s_n_features.shape[-1]  #(n_s,n_r)
        log_n_affinity = log_n_affinity / s_n_features.shape[-1] ** 0.5  #(n_s,n_r)
        # pdb.set_trace()
        # log_n_affinity = torch.exp(log_n_affinity)  #(exp)
        # log_n_affinity = torch.exp(-log_n_affinity)  #(exp)
        # node_affinity = F.softmax(node_affinity,dim=0) * F.softmax(node_affinity,dim=1)  #(n_s,n_r)
        # log_n_affinity = torch.exp(- log_n_affinity * log_n_affinity)  #(exp)
        log_n_affinity = F.softmax(log_n_affinity,dim=0) * F.softmax(log_n_affinity,dim=1)  #(n_s,n_r)
        # node_affinity = self.instance_norm(node_affinity.unsqueeze(1)).squeeze()  #(n_s,n_r)
        # log_n_affinity = torch.exp(log_n_affinity)  #(exp)
        # log_n_affinity = node_affinity.detach().clone()

        # points feats distance matrix
        ref_dist = torch.cdist(ref_points_f, ref_points_f) #(n_t,n_t)
        src_dist = torch.cdist(src_points_f, src_points_f) #(n_s,n_s)

        # adj matrix
        adj_ref_idx = ref_dist.topk(self.num_adj+1, dim=-1, largest=False)[1][:,1:]  #(n_r,num_adj)
        adj_src_idx = src_dist.topk(self.num_adj+1, dim=-1, largest=False)[1][:,1:]  #(n_s,num_adj)

        ####################################### initialization,larger ############################################
        self.ff_length = 3 + self.fstp * self.nps  #(total number of flood filled nodes)
        node_idx = torch.empty((self.bw,self.ff_length,2),dtype=torch.long,device=self.device)  # (bw,is,2)

        _, node_idx[:,0,0], node_idx[:,0,1] = seed_sc_sm_nms_selection(src_points_f, 
                                                                       ref_points_f,
                                                                       log_n_affinity * e_key_src[:,None].repeat(1,n_r) * e_key_ref[None,:].repeat(n_s,1),
                                                                       self.bw,
                                                                       self.fill_x,
                                                                       self.seed_length_threshold,
                                                                       self.nms_range)

        ####### select its first and second neighbor node #########
        node_idx[:,1,0], node_idx[:,1,1], node_idx[:,2,0], node_idx[:,2,1] = triplet_dist_selection(src_points_f,
                                                                                                    ref_points_f,
                                                                                                    node_idx[:,0,0],
                                                                                                    node_idx[:,0,1],
                                                                                                    self.radius,
                                                                                                    self.bw,
                                                                                                    self.tri_length_threshold)

        # # pdb.set_trace()
        # ein_corr = self.verify_correspondence(gt_transform,src_points_f[node_idx[:,0,0]],ref_points_f[node_idx[:,0,1]])
        # zwei_corr = self.verify_correspondence(gt_transform,src_points_f[node_idx[:,1,0]],ref_points_f[node_idx[:,1,1]])
        # drei_corr = self.verify_correspondence(gt_transform,src_points_f[node_idx[:,2,0]],ref_points_f[node_idx[:,2,1]])
        # self.corr_sum.append((ein_corr * zwei_corr * drei_corr).sum(-1))  #(bw,)
        # self.eine_sum.append((ein_corr.sum(-1)))  #(bw,)
        
        # initial estimated transformation
        init_est_transform = self.procrustes(src_points_f[node_idx[:,0:3,0]], ref_points_f[node_idx[:,0:3,1]], torch.ones((self.bw,3), device=self.device))  #(bw,4,4)
        est_transform = init_est_transform

        # # initial est transform compared with gt_transform
        # rr1_sum = torch.zeros((self.bw,),device=self.device)
        # rr2_sum = torch.zeros((self.bw,),device=self.device)
        # for i in range(self.bw):
        #     rre, rte, rmse, rr1_sum[i], rr2_sum[i] = self.evaluate_registration(est_transform[i], gt_transform, src_points)
        # self.initial_rr.append(rr1_sum.sum())
        # self.initial_rr2.append(rr2_sum.sum())
        # print(f'initial_rr: {statistics.fmean(self.initial_rr):.3f}, initial_rr2:{statistics.fmean(self.initial_rr2):.3f}')

        # mask for each seed, default 1, 0 if not satisfying transformation
        mask = torch.ones((self.bw, self.ff_length), device=self.device)  #(bw,is)
        mask_weight = torch.ones((self.bw, self.ff_length), device=self.device)  #(bw,is)
        
        # pdb.set_trace()
        # unvisited node mask, unvisited: 0, visited: 1
        unvisited_src = torch.zeros((self.bw, n_s), device=self.device)  #(bw,n_s)
        unvisited_ref = torch.zeros((self.bw, n_r), device=self.device)  #(bw,n_r)

        unvisited_src[torch.arange(self.bw), node_idx[:,0,0]] = 100  #(bw,N_s)
        unvisited_ref[torch.arange(self.bw), node_idx[:,0,1]] = 100  #(bw,N_r)
        unvisited_src[torch.arange(self.bw), node_idx[:,1,0]] = 100  #(bw,N_s)
        unvisited_ref[torch.arange(self.bw), node_idx[:,1,1]] = 100  #(bw,N_r)
        unvisited_src[torch.arange(self.bw), node_idx[:,2,0]] = 100  #(bw,N_s)
        unvisited_ref[torch.arange(self.bw), node_idx[:,2,1]] = 100  #(bw,N_r)

        for i in range(self.fstp):
            # pdb.set_trace()
            # obtain all neighbor nodes
            src_ne_node_idx = adj_src_idx[node_idx[:,0:i*self.nps+3,0]]  #(bw,i,na)
            ref_ne_node_idx = adj_ref_idx[node_idx[:,0:i*self.nps+3,1]]  #(bw,i,na)
            
            # src_ne_node_idx = adj_src_md[node_idx[:,0:i,0]]  #(bw,i,na)
            # ref_ne_node_idx = adj_ref_md[node_idx[:,0:i,1]]  #(bw,i,na)

            # current node neighbor nodes mask
            cur_src_ne_visited = unvisited_src.gather(1, src_ne_node_idx.view(self.bw, -1))  #(bw, i*na)
            cur_ref_ne_visited = unvisited_ref.gather(1, ref_ne_node_idx.view(self.bw, -1))  #(bw, i*na)

            src_ne_points = src_points_f[src_ne_node_idx].unsqueeze(3).repeat(1,1,1,self.num_adj,1)  #(bw,i,na,na,3)
            ref_ne_points = ref_points_f[ref_ne_node_idx].unsqueeze(2).repeat(1,1,self.num_adj,1,1)  #(bw,i,na,na,3)
            
            src_ne_points = src_ne_points.view(self.bw, -1, 3)  #(bw, i*na*na, 3)
            ref_ne_points = ref_ne_points.view(self.bw, -1, 3)  #(bw, i*na*na, 3)

            # apply transform through all estimated transform
            registered_src_points = apply_transform(src_ne_points, est_transform)  #(bw, i * na * na, 3)

            # check mean square distance error
            # pdb.set_trace()
            rmse = torch.linalg.norm(registered_src_points - ref_ne_points, dim = -1)  #(bw, i * na * na)
            
            # rmse_dist = torch.lt(rmse, self.radius).sum(-1)
            # pdb.set_trace()
            # mask visited node
            rmse = rmse + cur_src_ne_visited.unsqueeze(-1).repeat(1,1,self.num_adj).view(self.bw,-1)
            rmse = rmse + cur_ref_ne_visited.view(self.bw,-1,self.num_adj).unsqueeze(2).repeat(1,1,self.num_adj,1).view(self.bw,-1)

            # select minimum square error for each branch
            # pdb.set_trace()
            # indices = torch.min(rmse, -1).indices
            indices = torch.topk(rmse, self.nps, largest=False).indices  #(bw,nps)

            cur_node_idx = indices.div(self.num_adj**2, rounding_mode='floor')  #(bw,nps)
            cur_ne_idx = indices % (self.num_adj**2)  #(bw,nps)
            cur_ne_src = cur_ne_idx.div(self.num_adj, rounding_mode='floor')  #(bw,nps)
            cur_ne_ref = cur_ne_idx % self.num_adj  #(bw,nps)
            
            # set node_idx
            # pdb.set_trace()
            cur_pre_src = node_idx[:,:,0].gather(1,cur_node_idx)  #(bw,nps)
            cur_pre_ref = node_idx[:,:,1].gather(1,cur_node_idx)  #(bw,nps)
            node_idx[:,i*self.nps+3:(i+1)*self.nps+3,0] = adj_src_idx[cur_pre_src].gather(-1,cur_ne_src.unsqueeze(-1)).squeeze(-1)  #(bw,nps)
            node_idx[:,i*self.nps+3:(i+1)*self.nps+3,1] = adj_ref_idx[cur_pre_ref].gather(-1,cur_ne_ref.unsqueeze(-1)).squeeze(-1)  #(bw,nps)
            # node_idx[:,i,0] = adj_src_idx[node_idx[torch.arange(self.bw),cur_node_idx,0]][torch.arange(self.bw), cur_ne_src]
            # node_idx[:,i,1] = adj_ref_idx[node_idx[torch.arange(self.bw),cur_node_idx,1]][torch.arange(self.bw), cur_ne_ref]
            
            # update visited node
            # unvisited_src.gather(node_idx[:,i*self.nps+3:(i+1)*self.nps+3,0], dim=-1) = 100
            unvisited_src.scatter_(index=node_idx[:,i*self.nps+3:(i+1)*self.nps+3,0], dim=1, value=100)
            unvisited_ref.scatter_(index=node_idx[:,i*self.nps+3:(i+1)*self.nps+3,1], dim=1, value=100)
            # unvisited_src[torch.arange(self.bw), node_idx[:,i,0]] = 100  #(bw,N_s)
            # unvisited_ref[torch.arange(self.bw), node_idx[:,i,1]] = 100  #(bw,N_r)

            est_transform = self.procrustes(src_points_f[node_idx[:,0:3+(i+1)*self.nps,0]], ref_points_f[node_idx[:,0:3+(i+1)*self.nps,1]], mask_weight[:,0:3+(i+1)*self.nps])  #(bw,4,4)

            # update new mask
            registered_cur_points = apply_transform(src_points_f[node_idx[:,0:3+(i+1)*self.nps,0]], est_transform)  #(bw, i, 3)
            rmse = torch.linalg.norm(registered_cur_points - ref_points_f[node_idx[:,0:3+(i+1)*self.nps,1]], dim = -1)  #(bw,i)
            mask[:,0:3+(i+1)*self.nps] = torch.lt(rmse, self.inlier_threshold)  #(bw,i)
            mask_weight[:,0:3+(i+1)*self.nps] = torch.clamp(1 - rmse**2 / self.inlier_threshold ** 2, min = 0.)

        # crts module
        # if self.length > 3:
            # est_transform = self.procrustes(src_points_f[node_idx[:,:,0]], ref_points_f[node_idx[:,:,1]], torch.ones((self.bw,self.ff_length),device=self.device) * mask)  #(bw,4,4)
            # est_transform = self.procrustes(src_points_f[node_idx[:,:,0]], ref_points_f[node_idx[:,:,1]], torch.ones((self.bw,self.ff_length),device=self.device)*mask)  #(bw,4,4)
        est_transform, arg_idx, corr_src, corr_ref, weight, traj_src_points, traj_ref_points = evaluate_crts_transform(node_idx,
                                                                                                                      src_points_f,
                                                                                                                      ref_points_f,
                                                                                                                      est_transform,
                                                                                                                      torch.ones((self.bw,self.ff_length),device=self.device),
                                                                                                                      self.bw,
                                                                                                                      self.inlier_threshold,
                                                                                                                      self.procrustes,
                                                                                                                      self.lgr)

        # rre, rte, rmse, rr, rr2 = self.evaluate_registration(est_transform, gt_transform, src_points)
        rre, rte, rmse, rr, rr2 = evaluate_registration(est_transform[arg_idx], gt_transform, src_points, self.acceptance_radius)

        # # pose estimation stabilization effect analysis
        # # pdb.set_trace()
        # # init_est_transform, init_arg_idx, corr_src, corr_ref, weight, traj_src_points, traj_ref_points = self.evaluate_crts_transform(node_idx[:,0:3,:], src_points_f, ref_points_f, init_est_transform, torch.ones((self.bw,3),device=self.device))
        # # init_rre, init_rte, init_rmse, init_rr, init_rr2 = self.evaluate_registration(init_est_transform[init_arg_idx], gt_transform, src_points)
        # # rre, rte, rmse, rr, rr2 = self.evaluate_registration(ransac_transform, gt_transform, src_points)
        # if ein_corr.sum() > self.bw * 0.05:
        #     self.gut_ein += 1
        #     if rr2.item() > 0:
        #         self.gut_ein_rr2 += 1
        # if (ein_corr*zwei_corr*drei_corr).sum() > self.bw * 0.05:
        #     self.gut_drei += 1
        #     if rr2.item() > 0:
        #         self.gut_drei_rr2 += 1
        # self.test_step += 1
        # print(f'ein_ratio: {(self.gut_ein/self.test_step):.3f}, drei_ratio: {(self.gut_drei/self.test_step):.3f}')
        # print(f'eine_sum: {statistics.fmean(self.eine_sum):.3f}, corr_sum:{statistics.fmean(self.corr_sum):.3f}')
        # if self.test_step >= 50:
        #     print(f'gut_ein_rr: {self.gut_ein_rr2/self.gut_ein:.3f}, gut_drei_rr: {self.gut_drei_rr2/self.gut_drei:.3f}')
        # # ir = self.evaluate_fine(gt_transform, corr_src, corr_ref)
        # # tir = self.evaluate_fine(gt_transform, traj_src_points, traj_ref_points)
    
        self.metrics['rre'].append(rre.item())
        self.metrics['rte'].append(rte.item())
        self.metrics['rmse'].append(rmse.item())
        self.metrics['rr'].append(rr.item())
        self.metrics['rr2'].append(rr2.item())

        if rr2.item() > 0:
            self.metrics['re'].append(rre.item())
            self.metrics['te'].append(rte.item())
            # self.init_metrics['re'].append(init_rre.item())
            # self.init_metrics['te'].append(init_rte.item())
        
        # if self.test_step > 10:
        msg = f"rre:{statistics.fmean(self.metrics['rre']):.5f}||"\
            f"rte:{statistics.fmean(self.metrics['rte']):.5f}||"\
            f"rmse: {statistics.fmean(self.metrics['rmse']):.5f}||"\
            f"rr: {statistics.fmean(self.metrics['rr']):.5f}||"\
            f"rr2: {statistics.fmean(self.metrics['rr2']):.5f}||"\
            f"re: {statistics.fmean(self.metrics['re']):.5f}||"\
            f"te: {statistics.fmean(self.metrics['te']):.5f}||"\
            #   f"init_rr2: {statistics.fmean(self.init_metrics['rr2']):.5f}||"\
            #   f"init_re: {statistics.fmean(self.init_metrics['re']):.5f}||"\
            #   f"init_te: {statistics.fmean(self.init_metrics['te']):.5f}"
            # print(msg)

        return est_transform[arg_idx], msg

    def forward(self, output_dict):
        # self.ein_floodfill(output_dict['src_points_f'],output_dict['ref_points_f'],output_dict['src_feats_f'],output_dict['ref_feats_f'],output_dict['gt_transform'],output_dict['src_points'],output_dict['ref_points'])
        # est_transform, msg = self.fast_floodfill(output_dict['src_points_f'],output_dict['ref_points_f'],output_dict['src_feats_f'],output_dict['ref_feats_f'],output_dict['gt_transform'],output_dict['src_points'],output_dict['ref_points'],output_dict['geo_rr'],output_dict['estimated_transform'])
        est_transform, msg = self.fast_floodfill(output_dict['src_points_f'],output_dict['ref_points_f'],output_dict['src_feats_f'],output_dict['ref_feats_f'],output_dict['gt_transform'],output_dict['src_points'],output_dict['ref_points'])
        # self.ne_floodfill(output_dict['src_points_f'],output_dict['ref_points_f'],output_dict['src_feats_f'],output_dict['ref_feats_f'],output_dict['gt_transform'],output_dict['src_points'],output_dict['ref_points'])
        ## est_transform = self.beamSearch(output_dict['src_points_f'],output_dict['ref_points_f'],output_dict['src_feats_f'],output_dict['ref_feats_f'],output_dict['gt_transform'],output_dict['src_points'])
       
        output_dict['estimated_transform'] = est_transform
        
        return output_dict, msg

def create_search(config):
    model = Floodfill()
    return model
