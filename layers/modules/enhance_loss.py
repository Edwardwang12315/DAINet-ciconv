from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure as ssim
from data.config import cfg

def gradient(input_tensor, direction):
    smooth_kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view((1, 1, 2, 2)).cuda()
    smooth_kernel_y = torch.transpose(smooth_kernel_x, 2, 3)

    if direction == "x":
        kernel = smooth_kernel_x
    elif direction == "y":
        kernel = smooth_kernel_y
    grad_out = torch.abs(F.conv2d(input_tensor, kernel,
                                  stride=1, padding=1))
    return grad_out

def ave_gradient(input_tensor, direction):
    return F.avg_pool2d(gradient(input_tensor, direction),
                        kernel_size=3, stride=1, padding=1)

def smooth(input_I, input_R):
    input_R = 0.299 * input_R[:, 0, :, :] + 0.587 * input_R[:, 1, :, :] + 0.114 * input_R[:, 2, :, :]
    input_R = torch.unsqueeze(input_R, dim=1)
    return torch.mean(gradient(input_I, "x") * torch.exp(-10 * ave_gradient(input_R, "x")) +
                      gradient(input_I, "y") * torch.exp(-10 * ave_gradient(input_R, "y")))

import torch
import torch.nn as nn
import torch.fft

class FourierLoss(nn.Module):
    def __init__(self, high_freq_weight=0.8, cut_ratio=0.2):
        super().__init__()
        self.high_freq_weight = high_freq_weight  # 高频分量权重
        self.cut_ratio = cut_ratio                # 低频截止比例（中心区域）

    def forward(self, pred, target):
        """ 
        pred/target形状: (B, 1, H, W) 的单通道数据
        """
        # 傅里叶变换
        pred_fft = torch.fft.fft2(pred)        # 输出复数张量 (B, 1, H, W)
        target_fft = torch.fft.fft2(target)

        # --- 幅度谱损失 ---
        pred_mag = torch.abs(pred_fft)          # 幅度谱 (B, 1, H, W)
        target_mag = torch.abs(target_fft)
        
        # --- 高频掩码生成 ---
        _, _, h, w = pred.shape
        mask = torch.ones_like(pred_mag)        # 初始全1掩码（保留所有频率）
        ch, cw = int(h * self.cut_ratio), int(w * self.cut_ratio)
        # 中心区域置零（抑制低频）
        mask[..., h//2 - ch : h//2 + ch, w//2 - cw : w//2 + cw] = 0
        
        # --- 损失计算 ---
        # 高频损失（掩码外区域）
        high_loss = F.l1_loss(pred_mag * mask, target_mag * mask)
        
        # 全频带损失（可选，根据需求调整）
        full_loss = F.l1_loss(pred_mag, target_mag)
        
        # 加权组合
        total_loss = self.high_freq_weight * high_loss + (1 - self.high_freq_weight) * full_loss
        return total_loss
    
class GramLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def gram_matrix(self, x):
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (c * h * w)
    
    def forward(self, pred, target):
        pred_gram = self.gram_matrix(pred)
        target_gram = self.gram_matrix(target)
        return torch.mean(torch.abs(pred_gram - target_gram))

class GradientLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1,1,3,3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1,1,3,3)
    
    def forward(self, pred, target):
        pred_grad_x = F.conv2d(pred, self.sobel_x.to(pred.device), padding=1)
        pred_grad_y = F.conv2d(pred, self.sobel_y.to(pred.device), padding=1)
        target_grad_x = F.conv2d(target, self.sobel_x.to(target.device), padding=1)
        target_grad_y = F.conv2d(target, self.sobel_y.to(target.device), padding=1)
        return torch.mean(torch.abs(pred_grad_x - target_grad_x) + torch.mean(torch.abs(pred_grad_y - target_grad_y)))
                          
class EnhanceLoss(nn.Module):
    def __init__(self):
        super(EnhanceLoss, self).__init__()
        self.fourier_loss = FourierLoss()
        self.gram_loss = GramLoss()
        self.grad_loss = GradientLoss()

    def forward(self, out):
        R_dark, R_light, R_dark_2, R_light_2= out

        # losses_equal_R = (F.mse_loss(R_dark, R_light.detach()) + F.mse_loss(R_dark_2, R_light_2.detach())) * cfg.WEIGHT.EQUAL_R
        # losses_decoder = F.mse_loss(R_dark * I_dark, img_dark) * 1.+ (1. - ssim(R_dark * I_dark, img_dark))
        # losses_recon_high = F.mse_loss(R_light * I_light, img) * 1.+ (1. - ssim(R_light * I_light, img))
        # losses_smooth_low = smooth(I_dark, R_dark) * cfg.WEIGHT.SMOOTH
        # losses_smooth_high = smooth(I_light, R_light) * cfg.WEIGHT.SMOOTH
        
        # Redecomposition cohering loss
        # losses_rc = (F.mse_loss(R_dark_2, R_dark.detach()) + F.mse_loss(R_light_2, R_light.detach())) * cfg.WEIGHT.RC

        # enhance_loss = losses_equal_R + losses_recon_low + losses_recon_high + losses_smooth_low \
        #                + losses_smooth_high + losses_rc

        """ 
        5.20 版本-the FIRST
        loss_decoder=0.007621681783348322,loss_ciconv=2.627662420272827,loss_dark=1.7443002462387085,loss_light=2.4949073791503906
        loss_decoder=0.004208073019981384,loss_ciconv=2.6099538803100586,loss_dark=1.66346275806427,loss_light=2.567932367324829
        loss_decoder=0.004936039447784424,loss_ciconv=2.67338490486145,loss_dark=1.770261287689209,loss_light=2.609659433364868
        loss_decoder=0.004414223600178957,loss_ciconv=2.396692991256714,loss_dark=1.6409627199172974,loss_light=2.322230100631714 
        """
        # 对同模块不同输入应该有像素级的对齐
        # loss_decoder = 1 * (self.gram_loss(R_dark, R_light.detach()) + self.grad_loss(R_dark, R_light.detach())) # decoder输出
        # loss_decoder = F.mse_loss(R_dark, R_light.detach()) * 1. + (1. - ssim(R_dark, R_light.detach())) # decoder输出
        loss_decoder = F.mse_loss(R_dark, R_light) * 1. # decoder输出

        # loss_ciconv = 1 * (self.gram_loss(R_dark_2, R_light_2.detach()) + self.grad_loss(R_dark_2, R_light_2.detach())) # ciconv输出
        # loss_ciconv = F.mse_loss(R_dark_2, R_light_2.detach()) * 1. + (1. - ssim(R_dark_2, R_light_2.detach())) # ciconv输出
        loss_ciconv = F.mse_loss(R_dark_2, R_light_2.detach()) * 1. # ciconv输出

        # 跨模块对齐应该以边缘+纹理为主，结构相似性为辅
        loss_dark = (1 * self.gram_loss(R_dark, R_dark_2.detach()) # 强调纹理
                            + self.grad_loss(R_dark, R_dark_2.detach()) # 强调轮廓
                            + (1. - ssim(R_dark, R_dark_2.detach()))  # 强调结构
                            ) # dark输出
        loss_light = (1 * self.gram_loss(R_light, R_light_2.detach()) 
                      + self.grad_loss(R_light, R_light_2.detach()) 
                      + (1. - ssim(R_light, R_light_2.detach()))
                      ) # light输出

        # if True:
        #     loss_consist = 0.3 * (1 * self.gram_loss(R_light_2, R_dark.detach()) 
        #                 + self.grad_loss(R_light_2, R_dark.detach()) 
        #                 + (1. - ssim(R_light_2, R_dark.detach()))
        #                 ) # 为检测任务服务的增强损失
            
        # print(f'loss_decoder={loss_decoder},loss_ciconv={loss_ciconv},loss_dark={loss_dark},loss_light={loss_light}')
        # if True:
        #     return loss_decoder,loss_ciconv,loss_dark,loss_light,loss_consist
        # else:
        return loss_decoder,loss_ciconv,loss_dark,loss_light
