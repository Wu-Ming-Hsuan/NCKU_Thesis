# file: mmdet3d/hooks/adv_hook.py
import torch
from mmengine.hooks import Hook
from mmdet3d.attacks import ATTACKS
from mmengine.registry import HOOKS

@HOOKS.register_module()
class AttackHook(Hook):
    def __init__(self, attack_mode='none', attack_cfg=None):
        self.attack_mode = attack_mode
        self.attack = ATTACKS.build(attack_cfg)

    def before_train_iter(self, runner, batch_idx, data_batch):
        with torch.enable_grad():
            data_batch = self.attack.run(runner.model, data_batch, self.attack_mode)

    def before_test_iter(self, runner, batch_idx, data_batch):
        with torch.enable_grad():
            data_batch = self.attack.run(runner.model, data_batch, self.attack_mode)
