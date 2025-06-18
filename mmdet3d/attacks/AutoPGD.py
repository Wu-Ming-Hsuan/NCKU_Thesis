import torch
from .registry import ATTACKS

@ATTACKS.register_module()
class AutoPGD:
    def __init__(self,
                 epsilon=4,
                 step_size=None,
                 num_steps=20,
                 alpha=0.75,
                 rand_init=True):
        self.epsilon = epsilon
        self.step_size = step_size if step_size is not None else 0.2 * epsilon
        self.num_steps = num_steps
        self.alpha = alpha
        self.rand_init = rand_init

    def run(self, model, data, mode):
        device = next(model.parameters()).device
        img = data['inputs']['img'][0].to(device).detach()
        points = data['inputs']['points'][0].to(device)
        data_samples = data['data_samples']

        # 初始化
        if self.rand_init:
            delta = torch.empty_like(img).uniform_(-self.epsilon, self.epsilon)
        else:
            delta = torch.zeros_like(img)
        delta = delta.to(device)
        adv_img = torch.clamp(img + delta, 0, 255).clone().detach().requires_grad_(True)

        step = self.step_size
        momentum = torch.zeros_like(adv_img)
        best_adv_img = adv_img.clone().detach()
        best_loss = -float('inf')

        for i in range(self.num_steps):
            # forward
            current_img = torch.clamp(adv_img, 0, 255)
            new_inputs = {'img': [current_img], 'points': [points]}
            adv_data = dict(inputs=new_inputs, data_samples=data_samples)
            pro_data = model.module.data_preprocessor(adv_data, False)
            inputs = pro_data['inputs']
            data_samples = pro_data['data_samples']

            loss_dict = model.module.loss(batch_inputs_dict=inputs, batch_data_samples=data_samples, mode=mode)
            loss = sum(_loss for _loss in loss_dict.values() if isinstance(_loss, torch.Tensor))

            # 記錄最佳
            if loss.item() > best_loss:
                best_loss = loss.item()
                best_adv_img = adv_img.clone().detach()

            # 梯度計算
            if adv_img.grad is not None:
                adv_img.grad.zero_()
            model.zero_grad()
            loss.backward()

            # 更新動量
            eta = step * adv_img.grad.sign()
            z_adv = adv_img.detach() + eta
            z_adv = torch.clamp(z_adv, img - self.epsilon, img + self.epsilon)
            z_adv = torch.clamp(z_adv, 0, 255)

            adv_img = (adv_img + self.alpha * (z_adv - adv_img) + (1 - self.alpha) * momentum).detach()
            adv_img = torch.clamp(adv_img, img - self.epsilon, img + self.epsilon)
            adv_img = torch.clamp(adv_img, 0, 255).requires_grad_(True)
            momentum = (adv_img - best_adv_img).detach()

        # 用最佳對抗圖像
        data['inputs']['img'][0] = best_adv_img.detach().cpu()
        return data
