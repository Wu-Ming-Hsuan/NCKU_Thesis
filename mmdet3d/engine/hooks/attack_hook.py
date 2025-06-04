# file: mmdet3d/hooks/adv_hook.py
import torch
from mmengine.hooks import Hook
from mmdet3d.attacks import ATTACKS
from mmengine.registry import HOOKS

@HOOKS.register_module()
class AttackHook(Hook):
    def __init__(self, attack_cfg):
        self.attack = ATTACKS.build(attack_cfg)

    def before_test_iter(self, runner, batch_idx, data_batch):
        with torch.enable_grad():
            adv_data = self.attack.run(runner.model, data_batch)
        for k in data_batch:
            data_batch[k] = adv_data[k]
