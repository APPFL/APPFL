import copy

"""Functions for server."""


def primal_recover_from_local_states(self, local_states):
    for i, states in enumerate(local_states):
        if states is not None:
            for sid, state in states.items():
                self.primal_states[sid] = copy.deepcopy(state["primal"])


def dual_recover_from_local_states(self, local_states):
    for i, states in enumerate(local_states):
        if states is not None:
            for sid, state in states.items():
                self.dual_states[sid] = copy.deepcopy(state["dual"])


"""Functions for clients."""


def optimizer_setting(self):
    momentum = 0
    if "momentum" in self.optim_args.keys():
        momentum = self.optim_args.momentum
    weight_decay = 0
    if "weight_decay" in self.optim_args.keys():
        weight_decay = self.optim_args.weight_decay
    dampening = 0
    if "dampening" in self.optim_args.keys():
        dampening = self.optim_args.dampening
    nesterov = False

    return momentum, weight_decay, dampening, nesterov


def iiadmm_step(self, coefficient, penalty, optimizer):

    momentum, weight_decay, dampening, nesterov = optimizer_setting(self)

    for name, param in self.model.named_parameters():

        grad = copy.deepcopy(param.grad * coefficient)

        if weight_decay != 0:
            grad.add_(weight_decay, self.primal_state[name])
        if momentum != 0:
            param_state = optimizer.state[param]
            if "momentum_buffer" not in param_state:
                buf = param_state["momentum_buffer"] = grad.clone()
            else:
                buf = param_state["momentum_buffer"]
                buf.mul_(momentum).add_(1 - dampening, grad)
            if nesterov:
                grad = self.grad[name].add(momentum, buf)
            else:
                grad = buf

        ## Update primal
        self.primal_state[name] = self.global_state[name] + (1.0 / penalty) * (
            self.dual_state[name] - grad
        )
