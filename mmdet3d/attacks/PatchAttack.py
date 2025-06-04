import torch
import torch.nn as nn
import torch.optim as optim
from .registry import ATTACKS

@ATTACKS.register_module()
class PatchAttack:
    def __init__(self,
                 patch_size=32,
                 num_steps=20,
                 step_size=5,
                 epsilon=255,
                 rand_init=True):
        self.patch_size = patch_size
        self.num_steps = num_steps
        self.step_size = step_size
        self.epsilon = epsilon
        self.rand_init = rand_init

    def run(self, model, data):
        device = next(model.parameters()).device
        img = data['inputs']['img'][0].to(device).detach()  # [N, C, H, W]
        points = data['inputs']['points'][0].to(device)
        data_samples = data['data_samples']

        N, C, H, W = img.shape   # <<--- 直接取 [N, C, H, W]，不要多 unpack

        # 初始化 patch (N個相機各一個patch)
        if self.rand_init:
            patch = torch.rand(N, C, self.patch_size, self.patch_size, device=device) * self.epsilon
        else:
            patch = torch.zeros(N, C, self.patch_size, self.patch_size, device=device)
        patch.requires_grad_()

        # 貼在畫面中央
        y0 = H // 2 - self.patch_size // 2
        x0 = W // 2 - self.patch_size // 2

        optimizer = optim.Adam([patch], lr=self.step_size)

        for step in range(self.num_steps):
            optimizer.zero_grad()
            adv_img = img.clone()
            for view in range(N):
                adv_img[view, :, y0:y0+self.patch_size, x0:x0+self.patch_size] = patch[view]

            new_inputs = {'img': [adv_img], 'points': [points]}
            adv_data = dict(inputs=new_inputs, data_samples=data_samples)
            pro_data = model.data_preprocessor(adv_data, False)
            inputs = pro_data['inputs']
            data_samples = pro_data['data_samples']

            loss_dict = model.loss(batch_inputs_dict=inputs, batch_data_samples=data_samples)
            loss = sum(_loss for _loss in loss_dict.values() if isinstance(_loss, torch.Tensor))
            (-loss).backward()  # 最大化損失

            optimizer.step()
            patch.data.clamp_(0, self.epsilon)

        # 最終將 patch 貼回原圖
        for view in range(N):
            img[view, :, y0:y0+self.patch_size, x0:x0+self.patch_size] = patch[view].detach()

        data['inputs']['img'][0] = img.detach().cpu()
        return data
