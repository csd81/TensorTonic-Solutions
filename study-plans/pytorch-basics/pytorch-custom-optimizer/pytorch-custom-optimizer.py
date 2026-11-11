import torch

class CustomSGD(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 0.01, momentum: float = 0.0):
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Returns the closure loss when provided, otherwise None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                # Initialize momentum buffer to zero on the first update
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)

                v = state["momentum_buffer"]
                
                # Update velocity in place: v_t = mu * v_{t-1} + g_t
                v.mul_(momentum).add_(g)
                
                # Update parameter in place: theta_t = theta_{t-1} - eta * v_t
                p.add_(v, alpha=-lr)

        return loss
