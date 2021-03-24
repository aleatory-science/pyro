from typing import Tuple, Any

import torch
from torch.autograd import Function, gradcheck

from lv_step import lv_step as lv_step_constr
lv_step = lv_step_constr()

class LotkaVolterraStepFunction(Function):
    @staticmethod
    def forward(ctx: Any,
                state: Tuple[float, float],  # prey, predator
                growth_prey: float,  # prey_growth_rate
                predation: float,  # predation_rate
                growth_predator: float,  # predator_growth_rate
                decline_predator: float) -> Any:
        ctx.save_for_backward(state, growth_prey, predation, growth_predator, decline_predator)
        return torch.tensor(lv_step.lv_step(growth_prey,
                                            predation,
                                            growth_predator,
                                            decline_predator,
                                            state[0],
                                            state[1]))

    @staticmethod
    def backward(ctx, grad_output):
        state, growth_prey, predation, growth_predator, decline_predator = ctx.saved_tensors

        grad_state = grad_growth_prey = grad_predation = grad_growth_predator = grad_decline_predator = None

        if ctx.needs_input_grad[0]:
            s = state
            gprey = growth_prey
            p = predation
            gpred = growth_predator
            decl = decline_predator
            grad_state = torch.tensor(((gprey - p * s[1], -p * s[0]),
                                       (gpred * s[1], gpred * s[0] - decl)),
                                      dtype=grad_output.dtype)
            grad_state = grad_output @ grad_state.reshape(2, 2)
            print(grad_state)
        if ctx.needs_input_grad[1]:
            grad_growth_prey = torch.zeros(2, dtype=grad_output.dtype)
            grad_growth_prey[0] = state[0]
            grad_growth_prey = grad_output @ grad_growth_prey
        if ctx.needs_input_grad[2]:
            grad_predation = torch.zeros(2, dtype=grad_output.dtype)
            grad_predation[0] = - state[0] * state[1]
            grad_predation = grad_output @ grad_predation
        if ctx.needs_input_grad[3]:
            grad_growth_predator = torch.zeros(2, dtype=grad_output.dtype)
            grad_growth_predator[1] = state[0] * state[1]
            grad_growth_predator = grad_output @ grad_growth_predator
        if ctx.needs_input_grad[4]:
            grad_decline_predator = torch.zeros(2, dtype=grad_output.dtype)
            grad_decline_predator[1] = -state[1]
            grad_decline_predator = grad_output @ grad_decline_predator

        return grad_state, grad_growth_prey, grad_predation, grad_growth_predator, grad_decline_predator




if __name__ == '__main__':
    print(lotka_volterra_step(
        torch.Tensor((1., 1.)),
        torch.Tensor([.8]),
        torch.Tensor([.4]),
        torch.Tensor([.2]),
        torch.Tensor([.3])))

    step = lotka_volterra_step(
        torch.Tensor((1., 1.)),
        torch.Tensor([.8]),
        torch.Tensor([.4]),
        torch.Tensor([.2]),
        torch.Tensor([.3]))

    input = (torch.tensor((1., 1.), dtype=torch.double, requires_grad=True),
             torch.tensor([.8], dtype=torch.double, requires_grad=True),
             torch.tensor([.4], dtype=torch.double, requires_grad=True),
             torch.tensor([.2], dtype=torch.double, requires_grad=True),
             torch.tensor([.3], dtype=torch.double, requires_grad=True))
    test = gradcheck(lotka_volterra_step, input, eps=1e-6, atol=1e-4)
    print(test)
