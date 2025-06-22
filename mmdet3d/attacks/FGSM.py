import torch
from .registry import ATTACKS

@ATTACKS.register_module()
class FGSM:
    def __init__(self,
                 epsilon=8,
                 rand_init=False):
        """
        FGSM Attack for BEVFusion model
        
        Args:
            epsilon (int): Maximum perturbation magnitude (0-255 scale)
            rand_init (bool): Whether to use random initialization
        """
        self.epsilon = epsilon
        self.rand_init = rand_init
        
    def run(self, model, data, mode):
        device = next(model.parameters()).device
        
        # Get original image and normalize to [0, 1]
        img = data['inputs']['img'][0].to(device).float() / 255.0
        points = data['inputs']['points'][0].to(device)
        data_samples = data['data_samples']
        
        # Convert epsilon to [0, 1] scale
        eps = self.epsilon / 255.0
        
        # Store original image
        ori_img = img.clone().detach()
        
        # Initialize adversarial image
        if self.rand_init:
            # Random initialization within epsilon ball
            delta = torch.zeros_like(img).uniform_(-eps, eps)
            adv_img = torch.clamp(ori_img + delta, 0, 1)
        else:
            adv_img = ori_img.clone()
        
        adv_img.requires_grad_(True)
        
        model.eval()  # Ensure model is in eval mode for attack
        
        # Prepare input data (convert back to 0-255 scale for model)
        adv_inputs = {'img': [adv_img * 255.0], 'points': [points]}
        adv_data = dict(inputs=adv_inputs, data_samples=data_samples)
        
        # Forward pass
        pro_data = model.module.data_preprocessor(adv_data, False)
        loss_dict = model.module.loss(
            batch_inputs_dict=pro_data['inputs'], 
            batch_data_samples=pro_data['data_samples'], 
            mode=mode
        )
        
        # Calculate total loss
        loss = sum(_loss for _loss in loss_dict.values() 
                  if isinstance(_loss, torch.Tensor))
        
        # Backward pass
        model.zero_grad()
        if adv_img.grad is not None:
            adv_img.grad.zero_()
            
        loss.backward()
        
        # FGSM attack: single step with sign of gradient
        if adv_img.grad is not None:
            grad_sign = adv_img.grad.sign()
            
            if self.rand_init:
                # If random init was used, apply perturbation from original image
                perturbation = eps * grad_sign
                final_adv_img = ori_img + perturbation
            else:
                # Standard FGSM: add perturbation to current image
                final_adv_img = adv_img + eps * grad_sign
            
            # Clamp to valid range [0, 1]
            final_adv_img = torch.clamp(final_adv_img, 0, 1)
            
            # Ensure perturbation doesn't exceed epsilon
            perturbation = final_adv_img - ori_img
            perturbation = torch.clamp(perturbation, -eps, eps)
            final_adv_img = ori_img + perturbation
            final_adv_img = torch.clamp(final_adv_img, 0, 1)
            
        else:
            print("Warning: adv_img.grad is None")
            final_adv_img = adv_img.detach()
        
        # Convert back to 0-255 scale and update data
        final_adv_img_scaled = final_adv_img * 255.0
        data['inputs']['img'][0] = final_adv_img_scaled.detach().cpu()
        
        return data