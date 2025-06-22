import torch
import copy
from .registry import ATTACKS

@ATTACKS.register_module()
class PGD:
    def __init__(self,
                 epsilon=8,
                 steps=10,
                 alpha=None):
        """
        PGD Attack for BEVFusion model
        
        Args:
            epsilon (int): Maximum perturbation magnitude (0-255 scale)
            steps (int): Number of PGD steps
            alpha (float): Step size. If None, will be set to epsilon/steps
        """
        self.epsilon = epsilon
        self.steps = steps
        self.alpha = alpha if alpha is not None else epsilon / steps
        
    def run(self, model, data, mode):
        device = next(model.parameters()).device
        
        # Get original image and normalize to [0, 1]
        img = data['inputs']['img'][0].to(device).float() / 255.0
        points = data['inputs']['points'][0].to(device)
        data_samples = data['data_samples']
        
        # Convert epsilon and alpha to [0, 1] scale
        eps = self.epsilon / 255.0
        alpha = self.alpha / 255.0
        
        # Initialize random perturbation
        delta = torch.zeros_like(img).uniform_(-eps, eps)
        delta.requires_grad_(True)
        
        # Store original image
        ori_img = img.clone().detach()
        
        model.eval()  # Ensure model is in eval mode for attack
        
        for step in range(self.steps):
            # Create adversarial image
            adv_img = torch.clamp(ori_img + delta, 0, 1)
            
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
            if delta.grad is not None:
                delta.grad.zero_()
                
            loss.backward()
            
            # Update perturbation
            if delta.grad is not None:
                grad_sign = delta.grad.sign()
                delta.data = delta.data + alpha * grad_sign
                # Project perturbation to epsilon ball
                delta.data = torch.clamp(delta.data, -eps, eps)
                # Ensure adversarial image is in valid range
                delta.data = torch.clamp(ori_img + delta.data, 0, 1) - ori_img
            else:
                print(f"Warning: delta.grad is None at step {step}")
        
        # Apply final perturbation and convert back to 0-255 scale
        final_adv_img = torch.clamp(ori_img + delta, 0, 1) * 255.0
        data['inputs']['img'][0] = final_adv_img.detach().cpu()
        
        return data