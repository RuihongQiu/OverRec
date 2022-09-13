# @Author, 13/09/2022, Ruihong Qiu
# Based on RNTK Jax implementation


import numpy as np
import torch


# pytorch
@torch.no_grad()
def RNTK_first_time_step(ver, hor, param):
    # this is for computing the first GP and RNTK for t = 1. Both for relu and erf
    sw = param["sigmaw"]
    su = param["sigmau"]
    sb = param["sigmab"]
    sh = param["sigmah"]

    # n = X.shape[0]
    # GP_new = sh ** 2 * sw ** 2 * torch.eye(n).cuda() + (su ** 2) * X + sb ** 2
    GP_new = (su ** 2) * torch.logical_and(ver == hor, ver > 0) + sb ** 2
    RNTK_new = GP_new
    return RNTK_new, GP_new


# the vt mask used here is to deal with zero padding.
# whenever there is a zero padding, both the forward and the backward vt will be zero
# vt mask used for intermediate layer is from the previous input pair of x
# vt mask used for output layer is from the current input pair of x
@torch.no_grad()
def RNTK_lin(ver, hor, RNTK_old, GP_old, param, vt_mask, output):
    sw = param["sigmaw"]
    su = param["sigmau"]
    sb = param["sigmab"]
    sv = param["sigmav"]

    if output:
        GP_new = sv ** 2 * GP_old * vt_mask
        RNTK_new = sv ** 2 * RNTK_old * vt_mask + GP_new
    else:
        GP_new = (su ** 2) * torch.logical_and(ver == hor, ver > 0) + sw ** 2 * GP_old * vt_mask + sb ** 2
        RNTK_new = sw ** 2 * RNTK_old * vt_mask + GP_new
    return RNTK_new, GP_new


@torch.no_grad()
def RNTK_relu(ver, hor, RNTK_old, GP_old, param, vt_mask, output):
    sw = param["sigmaw"]
    su = param["sigmau"]
    sb = param["sigmab"]
    sv = param["sigmav"]

    if output:
        GP_new = sv ** 2 * (1 / (2 * np.pi)) * (GP_old * (np.pi - torch.arccos(GP_old)) + torch.sqrt(1 - GP_old ** 2)) * vt_mask
        RNTK_new = sv ** 2.0 * RNTK_old * (np.pi - torch.arccos(GP_old)) / (2 * np.pi) * vt_mask + GP_new
    else:
        GP_new = su ** 2 * torch.logical_and(ver == hor, ver > 0) + sw ** 2 * (1 / (2 * np.pi)) * (GP_old * (np.pi - torch.arccos(GP_old)) + torch.sqrt(1 - GP_old ** 2)) * vt_mask + sb ** 2
        RNTK_new = sw ** 2.0 * RNTK_old * (np.pi - torch.arccos(GP_old)) / (2 * np.pi) * vt_mask + GP_new
    return RNTK_new, GP_new
