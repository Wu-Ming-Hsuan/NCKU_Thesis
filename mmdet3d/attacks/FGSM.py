import torch
from .registry import ATTACKS

@ATTACKS.register_module()
class FGSM:
    def __init__(self,
                 epsilon=4,
                 rand_init=False):
        self.epsilon = epsilon
        self.rand_init = rand_init

    def run(self, model, data):
        device = next(model.parameters()).device
        img = data['inputs']['img'][0].to(device).detach()
        points = data['inputs']['points'][0].to(device)
        data_samples = data['data_samples']

        img_adv = img.clone().detach()
        img_adv.requires_grad = True

        if self.rand_init:
            # 若要支援隨機起點（很少見，但某些實作有）
            img_adv = img_adv + torch.empty_like(img_adv).uniform_(-self.epsilon, self.epsilon)
            img_adv = torch.clamp(img_adv, 0, 255)
            img_adv.requires_grad = True

        new_inputs = {'img': [img_adv], 'points': [points]}
        adv_data = dict(inputs=new_inputs, data_samples=data_samples)
        pro_data = model.data_preprocessor(adv_data, False)
        inputs = pro_data['inputs']
        data_samples = pro_data['data_samples']

        loss_dict = model.loss(batch_inputs_dict=inputs, batch_data_samples=data_samples)
        loss = sum(_loss for _loss in loss_dict.values() if isinstance(_loss, torch.Tensor))

        # 計算梯度
        if img_adv.grad is not None:
            img_adv.grad.zero_()
        model.zero_grad()
        loss.backward()

        # 單步 FGSM
        grad_sign = img_adv.grad.sign()
        img_adv = img_adv + self.epsilon * grad_sign
        img_adv = torch.clamp(img_adv, 0, 255).detach().cpu()

        data['inputs']['img'][0] = img_adv
        return data
