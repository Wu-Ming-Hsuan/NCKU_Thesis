import torch
import torch.optim as optim
import numpy as np
from .registry import ATTACKS
from mmdet3d.structures.bbox_3d import BaseInstance3DBoxes
from chamferdist import ChamferDistance

dist_func = ChamferDistance()

@ATTACKS.register_module()
class IOUSAttack:
    def __init__(self,
                 attack_name='iou_per',
                 attack_lr=0.01,
                 steps=10,
                 sub_loss='iou',
                 num_add=128,
                 num_drop=128,
                 k_drop_round=32,
                 epsilon=0.05):
        self.attack_name = attack_name
        self.attack_lr = attack_lr
        self.steps = steps
        self.sub_loss = sub_loss
        self.num_add = num_add
        self.num_drop = num_drop
        self.k_drop_round = k_drop_round
        self.epsilon = epsilon
        self.eps = 1e-8

    def enforce_fixed_dims(self, points, reference):
        points[:, -2:] = reference[:, -2:]
        return points

    def run(self, model, data):
        device = next(model.parameters()).device
        points_ori = data['inputs']['points'][0].to(device).detach()
        ori_xyz = points_ori[:, :3]
        gt_boxes_3d_ori = data['data_samples'][0].eval_ann_info['gt_bboxes_3d'].to(device)

        attack_name = self.attack_name
        steps = self.steps
        sub_loss = self.sub_loss
        attack_lr = self.attack_lr
        num_add = self.num_add
        num_drop = self.num_drop
        k_drop_round = self.k_drop_round

        best_attack_dist = float('inf')
        best_attack_score = float('inf')
        best_attack_points = points_ori.clone()

        if attack_name == 'iou_per':
            adv_point = (points_ori + torch.empty_like(points_ori).uniform_(-self.epsilon, self.epsilon)).clone().detach()
            adv_point = self.enforce_fixed_dims(adv_point, points_ori).clone().detach()
            adv_point.requires_grad_(True)
            optimizer = optim.Adam([adv_point], lr=attack_lr)

            for step in range(steps):
                cur_adv = adv_point.clone().detach().requires_grad_(True)
                new_inputs = {'img': data['inputs']['img'], 'points': [cur_adv]}
                new_data = dict(inputs=new_inputs, data_samples=data['data_samples'])
                pro_data = model.data_preprocessor(new_data, False)
                inputs = pro_data['inputs']
                data_samples = pro_data['data_samples']

                model.zero_grad()
                result = model.predict(batch_inputs_dict=inputs, batch_data_samples=data_samples)
                pre_bbox = result[0].pred_instances_3d.bboxes_3d
                pre_score = result[0].pred_instances_3d.scores_3d

                if pre_bbox.tensor.shape[0] == 0:
                    continue

                gt_bbox = gt_boxes_3d_ori
                ious_3d = BaseInstance3DBoxes.overlaps(gt_bbox, pre_bbox)
                ious_3d_sorted, idx = ious_3d.topk(k=pre_bbox.tensor.shape[0], dim=-1)
                if ious_3d_sorted.shape[1] == 0:
                    continue

                if sub_loss == 'iou':
                    loss_tensor = -torch.log(1 - ious_3d_sorted + self.eps)
                elif sub_loss == 'score':
                    loss_tensor = -torch.log(1 - pre_score[idx] + self.eps)
                elif sub_loss == 'all':
                    loss_tensor = -(torch.log(1 - pre_score[idx] + self.eps) +
                                    torch.log(1 - ious_3d_sorted + self.eps))
                else:
                    raise ValueError("Invalid sub_loss type")
                adv_loss = loss_tensor.sum(dim=1).mean()

                adv_xyz = cur_adv[:, :3]
                dist1 = dist_func(adv_xyz.unsqueeze(0), ori_xyz.unsqueeze(0))
                dist2 = dist_func(ori_xyz.unsqueeze(0), adv_xyz.unsqueeze(0))
                euclidean_loss = torch.sqrt(torch.sum((adv_xyz - ori_xyz) ** 2) + 1e-4)
                dist_loss = dist1 + dist2 + euclidean_loss

                loss_all = adv_loss + dist_loss

                loss_all.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                adv_point = cur_adv.detach()

                current_dist = torch.sqrt(torch.mean((adv_xyz - ori_xyz) ** 2)).item()
                current_loss = loss_all.item()
                if current_dist < best_attack_dist and current_loss < best_attack_score:
                    best_attack_dist = current_dist
                    best_attack_score = current_loss
                    best_attack_points = adv_point.clone().detach()

            data['inputs']['points'][0] = best_attack_points.detach().cpu()
            return data

        elif attack_name == 'iou_drop':
            num_rounds = int(np.ceil(float(num_drop) / float(k_drop_round)))
            adv_point = points_ori.clone()
            for round_idx in range(num_rounds):
                input_pc = adv_point.clone().detach().requires_grad_(True)
                new_inputs = {'img': data['inputs']['img'], 'points': [input_pc]}
                new_data = dict(inputs=new_inputs, data_samples=data['data_samples'])
                pro_data = model.data_preprocessor(new_data, False)
                inputs = pro_data['inputs']
                data_samples = pro_data['data_samples']

                model.zero_grad()
                result = model.predict(batch_inputs_dict=inputs, batch_data_samples=data_samples)
                pre_bbox = result[0].pred_instances_3d.bboxes_3d
                pre_score = result[0].pred_instances_3d.scores_3d

                gt_bbox = gt_boxes_3d_ori
                ious_3d = BaseInstance3DBoxes.overlaps(gt_bbox, pre_bbox)
                ious_3d_sorted, idx = ious_3d.topk(k=pre_bbox.tensor.shape[0], dim=-1)
                if ious_3d_sorted.shape[1] == 0:
                    continue

                if sub_loss == 'iou':
                    loss_tensor = torch.log(1 - ious_3d_sorted + self.eps)
                elif sub_loss == 'score':
                    loss_tensor = torch.log(1 - pre_score[idx] + self.eps)
                elif sub_loss == 'all':
                    loss_tensor = torch.log(1 - pre_score[idx] + self.eps) + torch.log(1 - ious_3d_sorted + self.eps)
                else:
                    raise ValueError("Invalid sub_loss type")
                loss_all = loss_tensor.sum()

                loss_all.backward()

                grad = input_pc.grad
                grad_norm = torch.sum(grad ** 2, dim=1)
                K = input_pc.shape[0]
                k_round = min(k_drop_round, num_drop - round_idx * k_drop_round)
                _, idx_keep = torch.topk(-grad_norm, k=K - k_round, dim=-1)
                adv_point = input_pc[idx_keep].detach()  # 剩下的點雲

            data['inputs']['points'][0] = adv_point.detach().cpu()
            return data

        elif attack_name == 'iou_add':
            input_pc = points_ori.clone().detach().requires_grad_(True)
            new_inputs = {'img': data['inputs']['img'], 'points': [input_pc]}
            new_data = dict(inputs=new_inputs, data_samples=data['data_samples'])
            pro_data = model.data_preprocessor(new_data, False)
            inputs = pro_data['inputs']
            data_samples = pro_data['data_samples']

            model.zero_grad()
            result = model.predict(batch_inputs_dict=inputs, batch_data_samples=data_samples)
            pre_bbox = result[0].pred_instances_3d.bboxes_3d
            pre_score = result[0].pred_instances_3d.scores_3d

            gt_bbox = gt_boxes_3d_ori
            ious_3d = BaseInstance3DBoxes.overlaps(gt_bbox, pre_bbox)
            ious_3d_sorted, idx = ious_3d.topk(k=pre_bbox.tensor.shape[0], dim=-1)
            if sub_loss == 'iou':
                loss_tensor = torch.log(1 - ious_3d_sorted + self.eps)
            elif sub_loss == 'score':
                loss_tensor = torch.log(1 - pre_score[idx] + self.eps)
            elif sub_loss == 'all':
                loss_tensor = torch.log(1 - pre_score[idx] + self.eps) + torch.log(1 - ious_3d_sorted + self.eps)
            else:
                raise ValueError("Invalid sub_loss type")
            loss_all = loss_tensor.sum()
            loss_all.backward()

            grad = input_pc.grad
            grad_norm = torch.sum(grad ** 2, dim=1)
            _, idx_crit = torch.topk(grad_norm, k=num_add, dim=-1)
            critical_points = points_ori[idx_crit]

            delta = torch.randn_like(critical_points) * 1e-7
            adv_point = (critical_points + delta).clone().detach().requires_grad_(True)
            adv_point = self.enforce_fixed_dims(adv_point, critical_points)
            optimizer = optim.Adam([adv_point], lr=attack_lr)

            for step in range(steps):
                cat_data = torch.cat([points_ori, adv_point], dim=0)
                new_inputs = {'img': data['inputs']['img'], 'points': [cat_data]}
                new_data = dict(inputs=new_inputs, data_samples=data['data_samples'])
                pro_data = model.data_preprocessor(new_data, False)
                inputs = pro_data['inputs']
                data_samples = pro_data['data_samples']

                model.zero_grad()
                result = model.predict(batch_inputs_dict=inputs, batch_data_samples=data_samples)
                pre_bbox = result[0].pred_instances_3d.bboxes_3d
                pre_score = result[0].pred_instances_3d.scores_3d

                gt_bbox = gt_boxes_3d_ori
                ious_3d = BaseInstance3DBoxes.overlaps(gt_bbox, pre_bbox)
                ious_3d_sorted, idx = ious_3d.topk(k=pre_bbox.tensor.shape[0], dim=-1)
                if ious_3d_sorted.shape[1] == 0:
                    continue

                if sub_loss == 'iou':
                    loss_tensor = -torch.log(1 - ious_3d_sorted + self.eps)
                elif sub_loss == 'score':
                    loss_tensor = -torch.log(1 - pre_score[idx] + self.eps)
                elif sub_loss == 'all':
                    loss_tensor = -(torch.log(1 - pre_score[idx] + self.eps) +
                                    torch.log(1 - ious_3d_sorted + self.eps))
                else:
                    raise ValueError("Invalid sub_loss type")
                attack_loss = loss_tensor.sum()

                dist1 = dist_func(adv_point[:, :3].unsqueeze(0), ori_xyz.unsqueeze(0))
                loss_all = attack_loss + dist1

                loss_all.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                adv_point = adv_point.detach().requires_grad_(True)  # 每步都new leaf

            cat_data = torch.cat([points_ori, adv_point], dim=0)
            data['inputs']['points'][0] = cat_data.detach().cpu()
            return data

        else:
            raise NotImplementedError(f"Unknown attack_name {attack_name}")
