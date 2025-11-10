# -*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable, Function

from layers import *
from data.config import cfg

import numpy as np
import matplotlib.pyplot as plt


class Interpolate(nn.Module):
	# 插值的方法对张量进行上采样或下采样
	def __init__(self, scale_factor):
		super(Interpolate, self).__init__()
		self.scale_factor = scale_factor

	def forward(self, x):
		x = nn.functional.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
		return x


class FEM(nn.Module):
	"""docstring for FEM"""

	def __init__(self, in_planes):
		super(FEM, self).__init__()
		inter_planes = in_planes // 3
		inter_planes1 = in_planes - 2 * inter_planes
		self.branch1 = nn.Conv2d(
			in_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3)

		self.branch2 = nn.Sequential(
			nn.Conv2d(in_planes, inter_planes, kernel_size=3,
					  stride=1, padding=3, dilation=3),
			nn.ReLU(inplace=True),
			nn.Conv2d(inter_planes, inter_planes, kernel_size=3,
					  stride=1, padding=3, dilation=3)
		)
		self.branch3 = nn.Sequential(
			nn.Conv2d(in_planes, inter_planes1, kernel_size=3,
					  stride=1, padding=3, dilation=3),
			nn.ReLU(inplace=True),
			nn.Conv2d(inter_planes1, inter_planes1, kernel_size=3,
					  stride=1, padding=3, dilation=3),
			nn.ReLU(inplace=True),
			nn.Conv2d(inter_planes1, inter_planes1, kernel_size=3,
					  stride=1, padding=3, dilation=3)
		)

	def forward(self, x):
		x1 = self.branch1(x)
		x2 = self.branch2(x)
		x3 = self.branch3(x)
		out = torch.cat((x1, x2, x3), dim=1)
		out = F.relu(out, inplace=True)
		return out


class DSFD(nn.Module):
	"""Single Shot Multibox Architecture
	The network is composed of a base VGG network followed by the
	added multibox conv layers.  Each multibox layer branches into
		1) conv2d for class conf scores
		2) conv2d for localization predictions
		3) associated priorbox layer to produce default bounding
		   boxes specific to the layer's feature map size.
	See: https://arxiv.org/pdf/1512.02325.pdf for more details.

	Args:
		phase: (string) Can be "test" or "train"
		size: input image size
		base: VGG16 layers for input, size of either 300 or 500
		extras: extra layers that feed to multibox loc and conf layers
		head: "multibox head" consists of loc and conf conv layers
	"""

	def __init__(self, phase, base, extras, fem, head1, head2, num_classes):
		super(DSFD, self).__init__()
		'''
		
		'''
		self.phase = phase
		self.num_classes = num_classes
		self.vgg = nn.ModuleList(base)
		if True :
			self.L2Normof1 = L2Norm(256, 10)
			self.L2Normof2 = L2Norm(512, 8)
			self.L2Normof3 = L2Norm(512, 5)
	
			self.extras = nn.ModuleList(extras)
			self.fpn_topdown = nn.ModuleList(fem[0])
			self.fpn_latlayer = nn.ModuleList(fem[1])
	
			self.fpn_fem = nn.ModuleList(fem[2])
	
			self.L2Normef1 = L2Norm(256, 10)
			self.L2Normef2 = L2Norm(512, 8)
			self.L2Normef3 = L2Norm(512, 5)
	
			self.loc_pal1 = nn.ModuleList(head1[0])#nn.ModuleList是一种存储子模块的工具
			self.conf_pal1 = nn.ModuleList(head1[1])
	
			self.loc_pal2 = nn.ModuleList(head2[0])
			self.conf_pal2 = nn.ModuleList(head2[1])

		# the reflectance decoding branch
		# 反射图解码模块
		# self.ref = nn.Sequential(
		# 	nn.Conv2d(64, 64, kernel_size=3, padding=1),
		# 	nn.ReLU(inplace=True),
		# 	Interpolate(2),#上采样
		# 	nn.Conv2d(64, 3, kernel_size=3, padding=1),
		# 	nn.ReLU(inplace=True),
		# 	nn.Conv2d(3, 1, kernel_size=1, padding=0),
		# 	# nn.Sigmoid()
		# 	# nn.BatchNorm2d(1)
		# )
		# self.ciconv2d_l = CIConv2d(invariant = 'W',k = 3,scale=0.0)
		# self.ciconv2d_d = CIConv2d(invariant = 'W',k = 3,scale=0.0)
		
		# # 计算teacher模型和学生模型的KL散度
		# self.KL = DistillKL(T=4.0)

		if self.phase == 'test':
			self.softmax = nn.Softmax(dim=-1)
			self.detect = Detect(cfg)

	def _upsample_prod(self, x, y):
		_, _, H, W = y.size()
		return F.upsample(x, size=(H, W), mode='bilinear') * y
	# 反射图解码通路
	def enh_forward(self, x):

		x = x[:1]
		for k in range(5):
			x = self.vgg[k](x)

		R = self.ref(x)

		return R

	def test_forward(self, x):
		size = x.size()[ 2 : ]
		pal1_sources = list()
		pal2_sources = list()
		loc_pal1 = list()
		conf_pal1 = list()
		loc_pal2 = list()
		conf_pal2 = list()
		
		_x = x.clone()
		
		for k in range( 16 ) :  # 检测时为 16 解码时为 5
			_x = self.vgg[ k ]( _x )
	
		if True  :
			# the following is the rest of the original detection pipeline
			of1 = _x
			s = self.L2Normof1( of1 )
			pal1_sources.append( s )
			# apply vgg up to fc7
			for k in range( 16 , 23 ) :
				_x = self.vgg[ k ]( _x )
			of2 = _x
			s = self.L2Normof2( of2 )
			pal1_sources.append( s )
			
			for k in range( 23 , 30 ) :
				_x = self.vgg[ k ]( _x )
			of3 = _x
			s = self.L2Normof3( of3 )
			pal1_sources.append( s )
			
			for k in range( 30 , len( self.vgg ) ) :
				_x = self.vgg[ k ]( _x )
			of4 = _x
			pal1_sources.append( of4 )
			# apply extra layers and cache source layer outputs
			
			for k in range( 2 ) :
				_x = F.relu( self.extras[ k ]( _x ) , inplace = True )
			of5 = _x
			pal1_sources.append( of5 )
			for k in range( 2 , 4 ) :
				_x = F.relu( self.extras[ k ]( _x ) , inplace = True )
			of6 = _x
			pal1_sources.append( of6 )
			
			conv7 = F.relu( self.fpn_topdown[ 0 ]( of6 ) , inplace = True )
			
			_x = F.relu( self.fpn_topdown[ 1 ]( conv7 ) , inplace = True )
			conv6 = F.relu( self._upsample_prod(
					_x , self.fpn_latlayer[ 0 ]( of5 ) ) , inplace = True )
			
			_x = F.relu( self.fpn_topdown[ 2 ]( conv6 ) , inplace = True )
			convfc7_2 = F.relu( self._upsample_prod(
					_x , self.fpn_latlayer[ 1 ]( of4 ) ) , inplace = True )
			
			_x = F.relu( self.fpn_topdown[ 3 ]( convfc7_2 ) , inplace = True )
			conv5 = F.relu( self._upsample_prod(
					_x , self.fpn_latlayer[ 2 ]( of3 ) ) , inplace = True )
			
			_x = F.relu( self.fpn_topdown[ 4 ]( conv5 ) , inplace = True )
			conv4 = F.relu( self._upsample_prod(
					_x , self.fpn_latlayer[ 3 ]( of2 ) ) , inplace = True )
			
			_x = F.relu( self.fpn_topdown[ 5 ]( conv4 ) , inplace = True )
			conv3 = F.relu( self._upsample_prod(
					_x , self.fpn_latlayer[ 4 ]( of1 ) ) , inplace = True )
			
			ef1 = self.fpn_fem[ 0 ]( conv3 )
			ef1 = self.L2Normef1( ef1 )
			ef2 = self.fpn_fem[ 1 ]( conv4 )
			ef2 = self.L2Normef2( ef2 )
			ef3 = self.fpn_fem[ 2 ]( conv5 )
			ef3 = self.L2Normef3( ef3 )
			ef4 = self.fpn_fem[ 3 ]( convfc7_2 )
			ef5 = self.fpn_fem[ 4 ]( conv6 )
			ef6 = self.fpn_fem[ 5 ]( conv7 )
			
			pal2_sources = (ef1 , ef2 , ef3 , ef4 , ef5 , ef6)
			for (_x , l , c) in zip( pal1_sources , self.loc_pal1 , self.conf_pal1 ) :
				loc_pal1.append( l( _x ).permute( 0 , 2 , 3 , 1 ).contiguous() )
				conf_pal1.append( c( _x ).permute( 0 , 2 , 3 , 1 ).contiguous() )
			
			for (_x , l , c) in zip( pal2_sources , self.loc_pal2 , self.conf_pal2 ) :
				loc_pal2.append( l( _x ).permute( 0 , 2 , 3 , 1 ).contiguous() )
				conf_pal2.append( c( _x ).permute( 0 , 2 , 3 , 1 ).contiguous() )
			
			features_maps = [ ]
			for i in range( len( loc_pal1 ) ) :
				feat = [ ]
				feat += [ loc_pal1[ i ].size( 1 ) , loc_pal1[ i ].size( 2 ) ]
				features_maps += [ feat ]
			
			loc_pal1 = torch.cat( [ o.view( o.size( 0 ) , -1 )
									for o in loc_pal1 ] , 1 )
			conf_pal1 = torch.cat( [ o.view( o.size( 0 ) , -1 )
									 for o in conf_pal1 ] , 1 )
			
			loc_pal2 = torch.cat( [ o.view( o.size( 0 ) , -1 )
									for o in loc_pal2 ] , 1 )
			conf_pal2 = torch.cat( [ o.view( o.size( 0 ) , -1 )
									 for o in conf_pal2 ] , 1 )
			
			priorbox = PriorBox( size , features_maps , cfg , pal = 1 )
			with torch.no_grad() :
				self.priors_pal1 = priorbox.forward()
			
			priorbox = PriorBox( size , features_maps , cfg , pal = 2 )
			with torch.no_grad() :
				self.priors_pal2 = priorbox.forward()
			
			if self.phase == 'test' :
				output = self.detect.forward(
						loc_pal2.view( loc_pal2.size( 0 ) , -1 , 4 ) ,
						self.softmax( conf_pal2.view( conf_pal2.size( 0 ) , -1 ,
													  self.num_classes ) ) ,  # conf preds
						self.priors_pal2.type( type( _x.data ) )
				)
			
			else :
				output = (
						loc_pal1.view( loc_pal1.size( 0 ) , -1 , 4 ) ,
						conf_pal1.view( conf_pal1.size( 0 ) , -1 , self.num_classes ) ,
						self.priors_pal1 ,
						loc_pal2.view( loc_pal2.size( 0 ) , -1 , 4 ) ,
						conf_pal2.view( conf_pal2.size( 0 ) , -1 , self.num_classes ) ,
						self.priors_pal2)
		
		return output
	
	# during training, the model takes the paired images, and their pseudo GT illumination maps from the Retinex Decom Net
	def forward(self, x):
		size = x.size()[2:]
		pal1_sources = list()
		pal2_sources = list()
		loc_pal1 = list()
		conf_pal1 = list()
		loc_pal2 = list()
		conf_pal2 = list()

		_x = x.clone()
		for k in range(16) if True  else range(5) : # 检测时为 16 解码时为 5
			_x = self.vgg[k](_x)
		if True :
			# the following is the rest of the original detection pipeline
			of1 = _x
			s = self.L2Normof1(of1)
			pal1_sources.append(s)
			# apply vgg up to fc7
			for k in range(16, 23):
				_x = self.vgg[k](_x)
			of2 = _x
			s = self.L2Normof2(of2)
			pal1_sources.append(s)
	
			for k in range(23, 30):
				_x = self.vgg[k](_x)
			of3 = _x
			s = self.L2Normof3(of3)
			pal1_sources.append(s)
	
			for k in range(30, len(self.vgg)):
				_x = self.vgg[k](_x)
			of4 = _x
			pal1_sources.append(of4)
			# apply extra layers and cache source layer outputs
	
			for k in range(2):
				_x = F.relu(self.extras[k](_x), inplace=True)
			of5 = _x
			pal1_sources.append(of5)
			for k in range(2, 4):
				_x = F.relu(self.extras[k](_x), inplace=True)
			of6 = _x
			pal1_sources.append(of6)
	
			conv7 = F.relu(self.fpn_topdown[0](of6), inplace=True)
	
			_x = F.relu(self.fpn_topdown[1](conv7), inplace=True)
			conv6 = F.relu(self._upsample_prod(
				_x, self.fpn_latlayer[0](of5)), inplace=True)
	
			_x = F.relu(self.fpn_topdown[2](conv6), inplace=True)
			convfc7_2 = F.relu(self._upsample_prod(
				_x, self.fpn_latlayer[1](of4)), inplace=True)
	
			_x = F.relu(self.fpn_topdown[3](convfc7_2), inplace=True)
			conv5 = F.relu(self._upsample_prod(
				_x, self.fpn_latlayer[2](of3)), inplace=True)
	
			_x = F.relu(self.fpn_topdown[4](conv5), inplace=True)
			conv4 = F.relu(self._upsample_prod(
				_x, self.fpn_latlayer[3](of2)), inplace=True)
	
			_x = F.relu(self.fpn_topdown[5](conv4), inplace=True)
			conv3 = F.relu(self._upsample_prod(
				_x, self.fpn_latlayer[4](of1)), inplace=True)
	
			ef1 = self.fpn_fem[0](conv3)
			ef1 = self.L2Normef1(ef1)
			ef2 = self.fpn_fem[1](conv4)
			ef2 = self.L2Normef2(ef2)
			ef3 = self.fpn_fem[2](conv5)
			ef3 = self.L2Normef3(ef3)
			ef4 = self.fpn_fem[3](convfc7_2)
			ef5 = self.fpn_fem[4](conv6)
			ef6 = self.fpn_fem[5](conv7)
	
			pal2_sources = (ef1, ef2, ef3, ef4, ef5, ef6)
			for (_x, l, c) in zip(pal1_sources, self.loc_pal1, self.conf_pal1):
				loc_pal1.append(l(_x).permute(0, 2, 3, 1).contiguous())
				conf_pal1.append(c(_x).permute(0, 2, 3, 1).contiguous())
	
			for (_x, l, c) in zip(pal2_sources, self.loc_pal2, self.conf_pal2):
				loc_pal2.append(l(_x).permute(0, 2, 3, 1).contiguous())
				conf_pal2.append(c(_x).permute(0, 2, 3, 1).contiguous())
	
			features_maps = []
			for i in range(len(loc_pal1)):
				feat = []
				feat += [loc_pal1[i].size(1), loc_pal1[i].size(2)]
				features_maps += [feat]
	
			loc_pal1 = torch.cat([o.view(o.size(0), -1)
								  for o in loc_pal1], 1)
			conf_pal1 = torch.cat([o.view(o.size(0), -1)
								   for o in conf_pal1], 1)
	
			loc_pal2 = torch.cat([o.view(o.size(0), -1)
								  for o in loc_pal2], 1)
			conf_pal2 = torch.cat([o.view(o.size(0), -1)
								   for o in conf_pal2], 1)
	
			priorbox = PriorBox(size, features_maps, cfg, pal=1)
			with torch.no_grad():
				self.priors_pal1 = priorbox.forward()
	
			priorbox = PriorBox(size, features_maps, cfg, pal=2)
			with torch.no_grad():
				self.priors_pal2 = priorbox.forward()
	
			if self.phase == 'test':
				output = self.detect.forward(
					loc_pal2.view(loc_pal2.size(0), -1, 4),
					self.softmax(conf_pal2.view(conf_pal2.size(0), -1,
												self.num_classes)),  # conf preds
					self.priors_pal2.type(type(_x.data))
				)
	
			else:
				output = (
					loc_pal1.view(loc_pal1.size(0), -1, 4),
					conf_pal1.view(conf_pal1.size(0), -1, self.num_classes),
					self.priors_pal1,
					loc_pal2.view(loc_pal2.size(0), -1, 4),
					conf_pal2.view(conf_pal2.size(0), -1, self.num_classes),
					self.priors_pal2)

		if True :
			return output
		else:
			return x,[ R_dark , R_light , R_dark_c , R_light_c ] , loss_mutual

	def load_weights(self, base_file):
		other, ext = os.path.splitext(base_file)
		if ext == '.pkl' or '.pth':
			print('Loading weights into state dict...')
			mdata = torch.load(base_file,
							   map_location=lambda storage, loc: storage)

			epoch = 0 # lr=1.2e-06
			self.load_state_dict(mdata)
			print('Finished!')
		else:
			print('Sorry only .pth and .pkl files supported.')
		return epoch

	def xavier(self, param):
		init.xavier_uniform_(param)

	def weights_init(self, m):
		if isinstance(m, nn.Conv2d):
			self.xavier(m.weight.data)
			m.bias.data.zero_()

		if isinstance(m, nn.ConvTranspose2d):
			self.xavier(m.weight.data)
			if 'bias' in m.state_dict().keys():
				m.bias.data.zero_()

		if isinstance(m, nn.BatchNorm2d):
			m.weight.data[...] = 1
			m.bias.data.zero_()


vgg_cfg_full = [64, 64, 'M',
				128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512, 'M'
			]
vgg_cfg = vgg_cfg_full if True  else vgg_cfg_full[:3]

extras_cfg = [256, 'S', 512, 128, 'S', 256]

fem_cfg = [256, 512, 512, 1024, 512, 256]


def fem_module(cfg):
	topdown_layers = []
	lat_layers = []
	fem_layers = []

	topdown_layers += [nn.Conv2d(cfg[-1], cfg[-1],
								 kernel_size=1, stride=1, padding=0)]
	for k, v in enumerate(cfg):
		fem_layers += [FEM(v)]
		cur_channel = cfg[len(cfg) - 1 - k]
		if len(cfg) - 1 - k > 0:
			last_channel = cfg[len(cfg) - 2 - k]
			topdown_layers += [nn.Conv2d(cur_channel, last_channel,
										 kernel_size=1, stride=1, padding=0)]
			lat_layers += [nn.Conv2d(last_channel, last_channel,
									 kernel_size=1, stride=1, padding=0)]
	return (topdown_layers, lat_layers, fem_layers)


def vgg(cfg, i, batch_norm=False):
	layers = []
	in_channels = i
	for v in cfg:
		if v == 'M':
			layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
		elif v == 'C':
			layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
		else:
			conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
			if batch_norm:
				layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
			else:
				layers += [conv2d, nn.ReLU(inplace=True)]
			in_channels = v
	if True :
		conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=3, dilation=3)
		conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
		layers += [conv6,
				   nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
	return layers


def add_extras(cfg, i, batch_norm=False):
	# Extra layers added to VGG for feature scaling
	layers = []
	in_channels = i
	flag = False
	for k, v in enumerate(cfg):
		if in_channels != 'S':
			if v == 'S':
				layers += [nn.Conv2d(in_channels, cfg[k + 1],
									 kernel_size=(1, 3)[flag], stride=2, padding=1)]
			else:
				layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
			flag = not flag
		in_channels = v
	return layers


def multibox(vgg, extra_layers, num_classes):
	loc_layers = []
	conf_layers = []
	vgg_source = [14, 21, 28, -2]

	for k, v in enumerate(vgg_source):
		loc_layers += [nn.Conv2d(vgg[v].out_channels,
								 4, kernel_size=3, padding=1)]
		conf_layers += [nn.Conv2d(vgg[v].out_channels,
								  num_classes, kernel_size=3, padding=1)]
	for k, v in enumerate(extra_layers[1::2], 2):
		loc_layers += [nn.Conv2d(v.out_channels,
								 4, kernel_size=3, padding=1)]
		conf_layers += [nn.Conv2d(v.out_channels,
								  num_classes, kernel_size=3, padding=1)]
	return (loc_layers, conf_layers)


def build_net_dark(phase, num_classes=2):
	base = vgg(vgg_cfg, 3)
	
	if True :
		extras = add_extras(extras_cfg, 1024)
		head1 = multibox(base, extras, num_classes)
		head2 = multibox(base, extras, num_classes)
		fem = fem_module(fem_cfg)
	else:
		extras = None
		head1 = None
		head2 = None
		fem = None
	
	return DSFD(phase, base, extras, fem, head1, head2, num_classes)


class DistillKL(nn.Module):
	"""KL divergence for distillation"""
	# 知识蒸馏模块，处理KL散度
	def __init__(self, T):
		super(DistillKL, self).__init__()
		self.T = T

	def forward(self, y_s, y_t):
		# y_s学生模型的输出，y_t 教师模型的输出
		p_s = F.log_softmax(y_s / self.T, dim=1)#对数概率分布
		p_t = F.softmax(y_t / self.T, dim=1)#概率分布
		# 计算KL散度
		# size_average不使用平均损失，而是返回总损失，(self.T ** 2)补偿温度缩放，/ y_s.shape[0]计算平均损失
		loss = F.kl_div(p_s, p_t, size_average=False) * (self.T ** 2) / y_s.shape[0]
		return loss

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


# ==================================
# ======== Gaussian filter =========
# ==================================

def gaussian_basis_filters( scale , use_cuda , gpu , k = 3 ) :
	std = torch.pow( 2 , scale )
	
	# Define the basis vector for the current scale
	filtersize = torch.ceil( k * std + 0.5 )
	x = torch.arange( start = -filtersize.item() , end = filtersize.item() + 1 )
	if use_cuda : x = x.cuda( gpu )
	x = torch.meshgrid( [ x , x ] )
	
	# Calculate Gaussian filter base
	# Only exponent part of Gaussian function since it is normalized anyway
	g = torch.exp( -(x[ 0 ] / std) ** 2 / 2 ) * torch.exp( -(x[ 1 ] / std) ** 2 / 2 )
	g = g / torch.sum( g )  # Normalize
	
	# Gaussian derivative dg/dx filter base
	dgdx = -x[ 0 ] / (std ** 3 * 2 * math.pi) * torch.exp( -(x[ 0 ] / std) ** 2 / 2 ) * torch.exp( -(x[ 1 ] / std) ** 2 / 2 )
	dgdx = dgdx / torch.sum( torch.abs( dgdx ) )  # Normalize
	
	# Gaussian derivative dg/dy filter base
	dgdy = -x[ 1 ] / (std ** 3 * 2 * math.pi) * torch.exp( -(x[ 1 ] / std) ** 2 / 2 ) * torch.exp( -(x[ 0 ] / std) ** 2 / 2 )
	dgdy = dgdy / torch.sum( torch.abs( dgdy ) )  # Normalize
	
	# Stack and expand dim
	basis_filter = torch.stack( [ g , dgdx , dgdy ] , dim = 0 )[ : , None , : , : ]
	
	return basis_filter


# =================================
# == Color invariant definitions ==
# =================================

eps = 1e-5


def E_inv( E , Ex , Ey , El , Elx , Ely , Ell , Ellx , Elly ) :
	E = Ex ** 2 + Ey ** 2 + Elx ** 2 + Ely ** 2 + Ellx ** 2 + Elly ** 2
	return E


def W_inv( E , Ex , Ey , El , Elx , Ely , Ell , Ellx , Elly ) :
	Wx = Ex / (E + eps)
	Wlx = Elx / (E + eps)
	Wllx = Ellx / (E + eps)
	Wy = Ey / (E + eps)
	Wly = Ely / (E + eps)
	Wlly = Elly / (E + eps)
	
	W = Wx ** 2 + Wy ** 2 + Wlx ** 2 + Wly ** 2 + Wllx ** 2 + Wlly ** 2
	return W


def C_inv( E , Ex , Ey , El , Elx , Ely , Ell , Ellx , Elly ) :
	Clx = (Elx * E - El * Ex) / (E ** 2 + 1e-5)
	Cly = (Ely * E - El * Ey) / (E ** 2 + 1e-5)
	Cllx = (Ellx * E - Ell * Ex) / (E ** 2 + 1e-5)
	Clly = (Elly * E - Ell * Ey) / (E ** 2 + 1e-5)
	
	C = Clx ** 2 + Cly ** 2 + Cllx ** 2 + Clly ** 2
	return C


def N_inv( E , Ex , Ey , El , Elx , Ely , Ell , Ellx , Elly ) :
	Nlx = (Elx * E - El * Ex) / (E ** 2 + 1e-5)
	Nly = (Ely * E - El * Ey) / (E ** 2 + 1e-5)
	Nllx = (Ellx * E ** 2 - Ell * Ex * E - 2 * Elx * El * E + 2 * El ** 2 * Ex) / (E ** 3 + 1e-5)
	Nlly = (Elly * E ** 2 - Ell * Ey * E - 2 * Ely * El * E + 2 * El ** 2 * Ey) / (E ** 3 + 1e-5)
	
	N = Nlx ** 2 + Nly ** 2 + Nllx ** 2 + Nlly ** 2
	return N


def H_inv( E , Ex , Ey , El , Elx , Ely , Ell , Ellx , Elly ) :
	Hx = (Ell * Elx - El * Ellx) / (El ** 2 + Ell ** 2 + 1e-5)
	Hy = (Ell * Ely - El * Elly) / (El ** 2 + Ell ** 2 + 1e-5)
	H = Hx ** 2 + Hy ** 2
	return H


# =================================
# == Color invariant convolution ==
# =================================

inv_switcher = {  # 也是一种函数调用
		'E' : E_inv ,
		'W' : W_inv ,
		'C' : C_inv ,
		'N' : N_inv ,
		'H' : H_inv }


class CIConv2d( nn.Module ) :
	def __init__( self , invariant , k = 3 , scale = 0.0 ) :
		super( CIConv2d , self ).__init__()
		assert invariant in [ 'E' , 'H' , 'N' , 'W' , 'C' ] , 'invalid invariant'
		self.inv_function = inv_switcher[ invariant ]
		
		self.use_cuda = torch.cuda.is_available()
		self.gpu = torch.cuda.current_device()
		
		# Constants
		self.gcm = torch.tensor( [ [ 0.06 , 0.63 , 0.27 ] , [ 0.3 , 0.04 , -0.35 ] , [ 0.34 , -0.6 , 0.17 ] ] )
		if self.use_cuda : self.gcm = self.gcm.cuda( self.gpu )
		self.k = k
		
		# Learnable parameters
		self.scale = torch.nn.Parameter( torch.tensor( [ scale ] ) , requires_grad = True )
	
	def forward( self , batch ) :
		# Make sure scale does not explode: clamp to max abs value of 2.5
		self.scale.data = torch.clamp( self.scale.data , min = -2.5 , max = 2.5 )
		
		# Measure E, El, Ell by Gaussian color model
		in_shape = batch.shape  # bchw
		batch = batch.view( (in_shape[ :2 ] + (-1 ,)) )  # [0:2]的形状不变，后面的合并为一维
		batch = torch.matmul( self.gcm , batch )  # estimate 相乘得到 E,El,Ell
		batch = batch.view( (in_shape[ 0 ] ,) + (3 ,) + in_shape[ 2 : ] )  # reshape to original image size
		
		E , El , Ell = torch.split( batch , 1 , dim = 1 )
		# Convolve with Gaussian filters
		w = gaussian_basis_filters( scale = self.scale , use_cuda = self.use_cuda , gpu = self.gpu )  # KCHW
		
		# the padding here works as "same" for odd kernel sizes
		E_out = F.conv2d( input = E , weight = w , padding = int( w.shape[ 2 ] / 2 ) )
		El_out = F.conv2d( input = El , weight = w , padding = int( w.shape[ 2 ] / 2 ) )
		Ell_out = F.conv2d( input = Ell , weight = w , padding = int( w.shape[ 2 ] / 2 ) )
		
		E , Ex , Ey = torch.split( E_out , 1 , dim = 1 )
		El , Elx , Ely = torch.split( El_out , 1 , dim = 1 )
		Ell , Ellx , Elly = torch.split( Ell_out , 1 , dim = 1 )
		
		inv_out = self.inv_function( E , Ex , Ey , El , Elx , Ely , Ell , Ellx , Elly )
		inv_out = F.instance_norm( torch.log( inv_out + eps ) )
		
		# print(f'out max={inv_out.max()},min = {inv_out.min()},mean = {inv_out.mean()}，var = {inv_out.var()}')
		# # 以下为单通道边缘图显示方法
		# image = inv_out[ 0 ].detach().cpu().numpy().squeeze()  # 维度 [H, W]
		
		# # 归一化到对称范围
		# vmax = np.max( np.abs( image ) )
		# image_normalized = image / vmax  # 范围[-1, 1]
		
		# # 使用红蓝颜色映射可视化
		# plt.imshow( image_normalized , cmap = 'RdBu' , vmin = -1 , vmax = 1 )
		# plt.axis( 'off' )
		# plt.colorbar( label = 'Edge Strength (Red: Positive, Blue: Negative)' )
		# plt.show()
		# # 保存图像到文件
		# plt.savefig( f'ciconv.png' , bbox_inches = 'tight' , pad_inches = 0 , dpi = 800 )
		# exit()
		
		return inv_out
