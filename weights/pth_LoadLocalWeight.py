# --coding:utf-8--
import torch
from models.factory import build_net

def LoadLocalW(net,path_oriMod):
	# 加载预训练权重
	ori_module=torch.load(path_oriMod)
	
	if path_oriMod.endswith('.pth'):
		print('load pth')
		ori_module_dict=ori_module # 当权重文件直接保存了模型权重时使用
	elif path_oriMod.endswith('.pt'):
		print('load pt')
		ori_module_dict=ori_module['model'] # 当权重文件用一个字典包装了模型权重时使用
	
	tar_module_dict=net.state_dict()

	# 筛选出名称和结构相同的模块权重
	matched_dict = {
			name : weight for name , weight in ori_module_dict.items()
			if name in tar_module_dict and weight.shape == tar_module_dict[ name ].shape
	}
	# torch.save(matched_dict,'matched_dict.pth')

	# 更新新模型的权重字典
	tar_module_dict.update( matched_dict )

	# 加载到新模型
	net.load_state_dict( tar_module_dict )
	
	# tar_module_dict=net.state_dict()
	# torch.save(tar_module_dict,'dsfd_wait.pth')
	return net

# net=build_net('train',1, 'dark')
# tar_module_dict = net.state_dict()
# torch.save(tar_module_dict,'test.pth')
# LoadLocalW(net)
