import torch
import torch.nn as nn
import torch.optim as optim
from .registry import ATTACKS

@ATTACKS.register_module()
class CWAttack:
    def __init__(self,
                 max_iterations=50,
                 learning_rate=0.01,
                 initial_const=0.1,
                 epsilon=8,
                 img_norm=None,
                 rand_init=True):
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.initial_const = initial_const
        self.epsilon = epsilon
        self.rand_init = rand_init
        self.img_norm = img_norm

    def run(self, model, data):
        device = next(model.parameters()).device
        img = data['inputs']['img'][0].to(device).detach()
        points = data['inputs']['points'][0].to(device)
        data_samples = data['data_samples']

        # 初始化對抗圖像
        if self.rand_init:
            delta = torch.empty_like(img).uniform_(-self.epsilon, self.epsilon)
        else:
            delta = torch.zeros_like(img)
        delta = delta.to(device)
        adv_img = torch.clamp(img + delta, 0, 255).clone().detach().requires_grad_(True)

        optimizer = optim.Adam([adv_img], lr=self.learning_rate)

        for iteration in range(self.max_iterations):
            optimizer.zero_grad()

            # Forward pass
            current_img = torch.clamp(adv_img, 0, 255)
            new_inputs = {'img': [current_img], 'points': [points]}
            adv_data = dict(inputs=new_inputs, data_samples=data_samples)
            pro_data = model.data_preprocessor(adv_data, False)
            inputs = pro_data['inputs']
            data_samples = pro_data['data_samples']

            # 用 detection loss，和PGD相同
            loss_dict = model.loss(batch_inputs_dict=inputs, batch_data_samples=data_samples)
            loss_adv = sum(_loss for _loss in loss_dict.values() if isinstance(_loss, torch.Tensor))

            # L2 距離項，若你的資料有normalize，需加 unnormalize
            l2_loss = ((self.unnormalized(adv_img) - self.unnormalized(img)) ** 2).mean()
            loss = loss_adv + self.initial_const * l2_loss

            loss.backward()
            optimizer.step()

            # Clamp perturbation在 epsilon 範圍內
            adv_img.data = torch.max(torch.min(adv_img.data, img + self.epsilon), img - self.epsilon)
            adv_img.data = torch.clamp(adv_img.data, 0, 255)

        # 對抗圖像結果覆蓋回原資料
        data['inputs']['img'][0] = adv_img.detach().cpu()
        return data

    def unnormalized(self, img):
        """
        img: [B, N, C, H, W] or [N, C, H, W]
        """
        if self.img_norm is None:
            # 預設不做任何處理
            return img
        mean = torch.tensor(self.img_norm['mean'], device=img.device).view(1, 1, 3, 1, 1)
        std = torch.tensor(self.img_norm['std'], device=img.device).view(1, 1, 3, 1, 1)
        return img * std + mean
