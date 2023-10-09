"""
    Automatic differentiation utilities
    @author: Qianyue He
    @date: 2023.10.9
    
    - [x] Gradient updates restricted to specified objects (only if the BSDF has needs_grad item in xml)
    - [ ] learning rate scheduler
    - [ ] Stopping criteria
"""

import taichi as ti
from bxdf.brdf import BRDF
from bxdf.bsdf import BSDF

@ti.kernel
def loss_backward(rdr: ti.template(), gt: ti.template(), loss: ti.template(), size_norm: float):
    for i, j in gt:
        loss += size_norm * ((rdr.pixels[i, j] - gt[i, j]) ** 2)
        
# TODO: ADAM optimizer to be added
# TODO: note that world medium is not allowed to update

@ti.data_oriented
class MomentumOptimizer:
    def __init__(self, num_objects, alpha_old = 0.8, lr = 0.001):
        self.brdf_grad = BRDF.field()
        self.bsdf_grad = BSDF.field()
        self.num_objects = num_objects
        self.alpha_old = alpha_old
        self.lr = lr
        # used for momentum computation (I decide to use dense since BSDFs are not memory consuming)
        ti.root.dense(ti.i, self.num_objects).place(self.brdf_grad, self.bsdf_grad)
        # values allowed to optimize
        self.brdf_grad_vals = {"k_d", "k_s"}
        self.bsdf_grad_vals = {"k_d", "k_s", "u_s", "u_a", "ior"}
        
    def step(self, rdr):
        """ Loss update for renderer """
        for i in range(self.num_objects):
            if ti.is_active(rdr.brdf_field, i):
                if not rdr.brdf_field[i].needs_grad(): continue
                for item in self.brdf_grad_vals:
                    grad = getattr(self.brdf_grad[i], item) * self.alpha_old + getattr(rdr.brdf_field.grad[i], item) * (1. - self.alpha_old)
                    # update variable field value
                    setattr(rdr.brdf_field[i], item, getattr(rdr.brdf_field[i], item) - self.lr * grad)
                    # update gradient momentum
                    setattr(self.brdf_grad[i], item, grad)
            else:   # for BSDF value update
                if not rdr.brdf_field[i].needs_grad(): continue
                for item in self.bsdf_grad_vals:
                    grad = getattr(self.bsdf_grad[i], item) * self.alpha_old + getattr(rdr.bsdf_field.grad[i], item) * (1. - self.alpha_old)
                    setattr(rdr.bsdf_field[i], item, getattr(rdr.bsdf_field[i], item) - self.lr * grad)
                    setattr(self.bsdf_grad[i], item, grad)
                if "u_s" in self.bsdf_grad_vals or "u_a" in self.bsdf_grad_vals:
                    rdr.bsdf_field[i].medium.u_e = rdr.bsdf_field[i].medium.u_a + rdr.bsdf_field[i].medium.u_s
                    