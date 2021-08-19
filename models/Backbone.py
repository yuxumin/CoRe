import torch.nn as nn
import torch
from .i3d import I3D
import logging

class I3D_backbone(nn.Module):
    def __init__(self, I3D_class):
        super(I3D_backbone, self).__init__()
        print('Using I3D backbone')
        self.backbone = I3D(num_classes = I3D_class, modality = 'rgb', dropout_prob = 0.5)
        
    def load_pretrain(self, I3D_ckpt_path):
        self.backbone.load_state_dict(torch.load(I3D_ckpt_path))
        print('loading ckpt done')

    def get_feature_dim(self):
        return self.backbone.get_logits_dim()

    def forward(self, target, exemplar, is_train, label, theta):
        # spatiotemporal feature
        total_video = torch.cat((target,exemplar),0)  # 2N C H W
        start_idx = [0,10,20,30,40,50,60,70,80,86]
        video_pack = torch.cat([total_video[:, :, i : i + 16] for i in start_idx])  # 10*2N, c, 16, h, w
        total_feature = self.backbone(video_pack).reshape(10,len(total_video),-1).transpose(0,1)  # 2N * 10 * 1024
        total_feature = total_feature.mean(1)

        feature_1 = total_feature[:total_feature.shape[0]//2]   
        feature_2 = total_feature[total_feature.shape[0]//2:]
        if is_train:
            combined_feature_1 = torch.cat((feature_1,feature_2,label[0] / theta),1)   # 1 is exemplar
            combined_feature_2 = torch.cat((feature_2,feature_1,label[1] / theta),1)   # 2 is exemplar
            return combined_feature_1, combined_feature_2 
        else:
            combined_feature = torch.cat((feature_2,feature_1,label[0] /theta),1) # 2 is exemplar
            return combined_feature

