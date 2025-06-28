import torch
from .registry import ATTACKS

# ------------------------------------------------------------------- helpers
def _loss_sum(loss_dict):
    return sum(v for v in loss_dict.values() if isinstance(v, torch.Tensor))

def _forward_loss(model, img, pts, data_samples, mode):
    feed = dict(inputs={"img": [img], "points": [pts]},
                data_samples=data_samples)
    pro = model.module.data_preprocessor(feed, False)
    return _loss_sum(model.module.loss(
        pro["inputs"], pro["data_samples"], mode=mode))

# ------------------------------------------------------------------- FGSM
@ATTACKS.register_module()
class FGSM:
    def __init__(self, epsilon=16, rand_init=False):
        self.eps_px = epsilon
        self.rand = rand_init

    def run(self, model, data, mode):
        dev = next(model.parameters()).device
        img = data["inputs"]["img"][0].float().to(dev) / 255.0
        pts = data["inputs"]["points"][0].to(dev)

        eps = self.eps_px / 255.0
        delta = (torch.empty_like(img).uniform_(-eps, eps)
                 if self.rand else torch.zeros_like(img)).requires_grad_()

        model.eval()
        loss = _forward_loss(model, (img + delta) * 255.0, pts,
                             data["data_samples"], mode)
        model.zero_grad()
        loss.backward()

        if delta.grad is not None and delta.grad.abs().sum() != 0:
            delta.data = (eps * delta.grad.sign()).clamp(-eps, eps)

        data["inputs"]["img"][0] = ((img + delta).clamp(0, 1) * 255.0).cpu()
        return data

# ------------------------------------------------------------------- PGD
@ATTACKS.register_module()
class PGD:
    def __init__(self, epsilon=8, steps=10, alpha=None):
        self.eps_px = epsilon
        self.T = steps
        self.step = (alpha if alpha is not None else epsilon / steps) / 255.0

    def run(self, model, data, mode):
        dev = next(model.parameters()).device
        img = data["inputs"]["img"][0].float().to(dev) / 255.0
        pts = data["inputs"]["points"][0].to(dev)
        eps = self.eps_px / 255.0
        delta = torch.empty_like(img).uniform_(-eps, eps)

        model.eval()
        for _ in range(self.T):
            delta.requires_grad_()
            loss = _forward_loss(model, (img + delta).clamp(0, 1) * 255.0,
                                 pts, data["data_samples"], mode)
            model.zero_grad()
            loss.backward()

            if delta.grad is None or delta.grad.abs().sum() == 0:
                break
            with torch.no_grad():
                delta.data += self.step * delta.grad.sign()
                delta.data.clamp_(-eps, eps)
                delta.data = (img + delta.data).clamp(0, 1) - img

        data["inputs"]["img"][0] = ((img + delta).clamp(0, 1) * 255.0).cpu()
        return data

# ------------------------------------------------------------------- AutoPGD
@ATTACKS.register_module()
class AutoPGD:
    def __init__(self, epsilon=4, num_steps=20,
                 step_size=None, alpha=0.75, rand_init=True):
        self.eps_px = epsilon
        self.T = num_steps
        self.step = (step_size if step_size is not None else 0.2 * epsilon) / 255.0
        self.alpha = alpha
        self.rand = rand_init

    def run(self, model, data, mode):
        dev = next(model.parameters()).device
        img = data["inputs"]["img"][0].float().to(dev) / 255.0
        pts = data["inputs"]["points"][0].to(dev)
        eps = self.eps_px / 255.0

        delta = (torch.empty_like(img).uniform_(-eps, eps)
                 if self.rand else torch.zeros_like(img))
        adv = (img + delta).clamp(0, 1)
        best, best_loss = adv.clone(), -1e9
        momentum = torch.zeros_like(img)

        model.eval()
        for _ in range(self.T):
            adv.requires_grad_()
            loss = _forward_loss(model, adv * 255.0, pts,
                                 data["data_samples"], mode)

            if loss.item() > best_loss:
                best_loss, best = loss.item(), adv.detach()

            model.zero_grad()
            loss.backward()
            g = adv.grad.sign()

            z = (adv + self.step * g).clamp(img - eps, img + eps).clamp(0, 1)
            adv = (adv + self.alpha * (z - adv) +
                   (1 - self.alpha) * momentum).detach()
            adv = adv.clamp(img - eps, img + eps).clamp(0, 1)
            momentum = (adv - best).detach()

        data["inputs"]["img"][0] = (best * 255.0).cpu()
        return data
