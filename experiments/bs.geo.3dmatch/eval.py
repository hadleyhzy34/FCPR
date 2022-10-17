import torch
from fcpr.modules.ops import apply_transform
from fcpr.modules.registration.metrics import isotropic_transform_error

def verify_correspondence(gt_transform, src_points, ref_points, radius):
    """
    description
    """
    # pdb.set_trace()
    transformed_src_points = apply_transform(src_points,gt_transform)
    corr_dist = torch.linalg.norm(transformed_src_points-ref_points,dim=-1)
    gt_corr = torch.lt(corr_dist, radius)
    return gt_corr

def evaluate_registration(est_transform, gt_transform, src_points_f, acceptance_rmse):
    # pdb.set_trace()
    rre, rte = isotropic_transform_error(gt_transform, est_transform)

    realignment_transform = torch.matmul(torch.inverse(gt_transform), est_transform)
    realigned_src_points_f = apply_transform(src_points_f, realignment_transform)
    rmse = torch.linalg.norm(realigned_src_points_f - src_points_f, dim=1).mean()
    recall_1 = torch.lt(rmse, acceptance_rmse).float()

    recall_2 = torch.logical_and(torch.lt(rre, 15.0), torch.lt(rte, 0.3)).float()

    return rre, rte, rmse, recall_1, recall_2


def evaluate_fine(transform, src_corr_points, ref_corr_points, acceptance_radius):
    # pdb.set_trace()
    # temporarily modify acceptance radius for visualization
    # self.acceptance_fine = 0.05
    src_corr_points = apply_transform(src_corr_points, transform)
    corr_distances = torch.linalg.norm(ref_corr_points - src_corr_points, dim=1)
    precision = torch.lt(corr_distances, acceptance_radius).float().mean()
    # precision = torch.lt(corr_distances, self.acceptance_fine).float().mean()
    return precision