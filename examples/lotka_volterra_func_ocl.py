from typing import Any

import matplotlib.pyplot as plt
import torch
from torch.autograd import Function

from lv_fut_ocl import lotka_volterra as lv_constr

run_lv_fut = lv_constr().main
adj_lv = lv_constr().runge_kutta_fwd


class RunLV(Function):
    @staticmethod
    def forward(ctx: Any, *args):
        ctx.save_for_backward(*args)
        evolution = run_lv_fut(*args)
        return torch.tensor(evolution.get())

    @staticmethod
    def backward(ctx, grad_output):
        grad_step_size = grad_num_steps = grad_prey = grad_predator = None
        grad_growth_prey = grad_predation = grad_growth_predator = grad_decline_predator = None
        if ctx.needs_input_grad[2]:
            grad_prey = grad_output * adj_lv(*ctx.saved_tensors, 1, 0, 0, 0, 0, 0).get()
        if ctx.needs_input_grad[3]:
            grad_predator = grad_output * adj_lv(*ctx.saved_tensors, 0, 1, 0, 0, 0, 0).get()
        if ctx.needs_input_grad[4]:
            grad_growth_prey = grad_output * adj_lv(*ctx.saved_tensors, 0, 0, 1, 0, 0, 0).get()
        if ctx.needs_input_grad[5]:
            grad_predation = grad_output * adj_lv(*ctx.saved_tensors, 0, 0, 0, 1, 0, 0).get()
        if ctx.needs_input_grad[6]:
            grad_growth_predator = grad_output * adj_lv(*ctx.saved_tensors, 0, 0, 0, 0, 1, 0).get()
        if ctx.needs_input_grad[7]:
            grad_decline_predator = grad_output * adj_lv(*ctx.saved_tensors, 0, 0, 0, 0, 0, 1).get()

        return grad_step_size, grad_num_steps, grad_prey, grad_predator, \
               grad_growth_prey, grad_predation, grad_growth_predator, grad_decline_predator


run_lv = RunLV.apply

if __name__ == '__main__':
    evolution = run_lv(.1, 1000,
                       torch.Tensor([1.]),
                       torch.Tensor([1.]),
                       torch.Tensor([.8]),
                       torch.Tensor([.4]),
                       torch.Tensor([.2]),
                       torch.Tensor([.3]))

    plt.plot(evolution[:, 0], label='prey')
    plt.plot(evolution[:, 1], '--', label='predator')
    plt.show()
