import torch.nn as nn


class HeightCompression(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        # 将稀疏卷积结构转换成普通的torch.tensor
        spatial_features = encoded_spconv_tensor.dense()
        # batch,channel,depth,height,width
        N, C, D, H, W = spatial_features.shape
        # 将128维feature在depth方向上进行cat,得到[batch,cat_featrue,width,height]
        spatial_features = spatial_features.view(N, C * D, H, W)
        # => 压缩Z轴后的spconv输出层feature volume为spatial_features
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        return batch_dict
