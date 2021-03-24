from typing import Any

import matplotlib.pyplot as plt
import torch
from torch.autograd import Function

from lotka_volterra_fut import lotka_volterra as lv_constr

run_lv_fut = lv_constr().main
adj_lv = lv_constr().runge_kutta_fwd


class RunLV(Function):
    @staticmethod
    def forward(ctx: Any,
                step_size: float, num_steps: int, prey_init: float, predator_init: float, growth_prey: float,
                predation: float, growth_predator: float, decline_predator: float) -> Any:
        ctx.save_for_backward(step_size, num_steps, prey_init, predator_init, growth_prey, predation, growth_predator,
                              decline_predator)
        evolution = run_lv_fut(step_size, num_steps, prey_init, predator_init, growth_prey, predation, growth_predator,
                               decline_predator)
        return torch.tensor(evolution)

    @staticmethod
    def backward(ctx, grad_output):
        grad_step_size = grad_num_steps = grad_prey = grad_predator = None
        grad_growth_prey = grad_predation = grad_growth_predator = grad_decline_predator = None
        if ctx.needs_input_grad[2]:
            grad_prey = grad_output @ adj_lv(*ctx.saved_tensors, 1, 0, 0, 0, 0, 0)
        if ctx.needs_input_grad[3]:
            grad_predator = grad_output @ adj_lv(*ctx.saved_tensors, 0, 1, 0, 0, 0, 0)
        if ctx.needs_input_grad[4]:
            grad_growth_prey = grad_output @ adj_lv(*ctx.saved_tensors, 0, 0, 1, 0, 0, 0)
        if ctx.needs_input_grad[5]:
            grad_predation = grad_output @ adj_lv(*ctx.saved_tensors, 0, 0, 0, 1, 0, 0)
        if ctx.needs_input_grad[6]:
            grad_growth_predator = grad_output @ adj_lv(*ctx.saved_tensors, 0, 0, 0, 0, 1, 0)
        if ctx.needs_input_grad[7]:
            grad_decline_predator = grad_output @ adj_lv(*ctx.saved_tensors, 0, 0, 0, 0, 0, 1)

        return grad_step_size, grad_num_steps, grad_prey, grad_predator, \
               grad_growth_prey, grad_predation, grad_growth_predator, grad_decline_predator


run_lv = RunLV.apply


def elbo(model, guide, *args, **kwargs):
    guide_trace = trace(guide).get_trace(*args, **kwargs)
    model_trace = trace(replay(model, guide_trace)).get_trace(*args, **kwargs)
    elbo = 0.
    for site in model_trace.values():
        if site["type"] == "sample":
            elbo = elbo + site["fn"].log_prob(site["value"]).sum()
    for site in guide_trace.values():
        if site["type"] == "sample":
            elbo = elbo - site["fn"].log_prob(site["value"]).sum()
    return -elbo



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
