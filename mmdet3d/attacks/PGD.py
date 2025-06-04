import torch
import copy
from .registry import ATTACKS

@ATTACKS.register_module()
class PGD:
    def __init__(self,
                 epsilon=4,
                 step_size=2,
                 num_steps=10,
                 rand_init=True):
        self.epsilon = epsilon
        self.step_size = step_size
        self.num_steps = num_steps
        self.rand_init = rand_init

    def run(self, model, data):
        device = next(model.parameters()).device
        img = data['inputs']['img'][0].to(device).detach()
        points = data['inputs']['points'][0].to(device)
        data_samples = data['data_samples']

        if self.rand_init:
            delta = torch.empty_like(img).uniform_(-self.epsilon, self.epsilon)
        else:
            delta = torch.zeros_like(img)
        delta = delta.to(device)
        delta.requires_grad = True

        for step in range(self.num_steps):
            adv_img = torch.clamp(img + delta, 0, 255)
            new_inputs = {'img': [adv_img], 'points': [points]}
            adv_data = dict(inputs=new_inputs, data_samples=data_samples)
            pro_data = model.data_preprocessor(adv_data, False)
            inputs = pro_data['inputs']
            data_samples = pro_data['data_samples']

            loss_dict = model.loss(batch_inputs_dict=inputs, batch_data_samples=data_samples)
            loss = sum(_loss for _loss in loss_dict.values() if isinstance(_loss, torch.Tensor))

            # Important: 先清梯度
            if delta.grad is not None:
                delta.grad.zero_()
            model.zero_grad()
            loss.backward()
            
            # 保險地確認梯度不為None
            if delta.grad is not None:
                delta.data.add_(self.step_size * delta.grad.sign())
                delta.data.clamp_(-self.epsilon, self.epsilon)
            else:
                raise RuntimeError('PGD: delta.grad is None!')

        # 完成後將對抗圖像覆蓋回原資料
        data['inputs']['img'][0] = torch.clamp(img + delta, 0, 255).detach().cpu()
        return data
