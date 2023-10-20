from typing import Optional,Tuple, Callable

import torch 
import torch.nn as nn

from moe.top2_gating import Top2Gating
from moe.expert import Experts
from moe_util import default

class HeirarchicalMoE(nn.Module):
    def __init__(self,
                 dim: int,
                 num_experts: Tuple[int, int] = (4, 4),
                 hidden_dim: Optional[int] = None,
                 activation: nn.Module = nn.ReLU,
                 second_policy_train: str = 'random',
                 second_policy_eval: str = 'random',
                 second_threshold_train: float = 0.2,
                 second_threshold_eval: float = 0.2,
                 capacity_factor_train: float = 1.25,
                 capacity_factor_eval: float = 2.,
                 loss_coef: float = 1e-2,
                 experts: Optional[Callable] = None):
        """
        Initialize the Hierarchical Mixture of Experts (MoE) model.

        Args:
            dim (int): Dimension of the input.
            num_experts (Tuple[int, int], optional): Number of experts at each level. Defaults to (4, 4).
            hidden_dim (Optional[int], optional): Hidden dimension for the experts. Defaults to None.
            activation (nn.Module, optional): Activation function to use. Defaults to nn.ReLU.
            second_policy_train (str, optional): Second expert policy during training. Defaults to 'random'.
            second_policy_eval (str, optional): Second expert policy during evaluation. Defaults to 'random'.
            second_threshold_train (float, optional): Second expert threshold during training. Defaults to 0.2.
            second_threshold_eval (float, optional): Second expert threshold during evaluation. Defaults to 0.2.
            capacity_factor_train (float, optional): Capacity factor during training. Defaults to 1.25.
            capacity_factor_eval (float, optional): Capacity factor during evaluation. Defaults to 2.0.
            loss_coef (float, optional): Loss coefficient. Defaults to 1e-2.
            experts (Optional[Callable], optional): Custom experts function. Defaults to None.
        """
        super().__init__()

        assert len(num_experts) == 2, 'only 2 levels of heirarchy for experts allowed for now'
        num_experts_outer, num_experts_inner = num_experts
        self.num_experts_outer = num_experts_outer
        self.num_experts_inner = num_experts_inner

        gating_kwargs = {'second_policy_train': second_policy_train, 'second_policy_eval': second_policy_eval, 'second_threshold_train': second_threshold_train, 'second_threshold_eval': second_threshold_eval, 'capacity_factor_train': capacity_factor_train, 'capacity_factor_eval': capacity_factor_eval}

        self.gate_outer = Top2Gating(dim, num_gates = num_experts_outer, **gating_kwargs)
        self.gate_inner = Top2Gating(dim, num_gates = num_experts_inner, outer_expert_dims = (num_experts_outer,), **gating_kwargs)

        self.experts = default(experts, lambda: Experts(dim, num_experts = num_experts, hidden_dim = hidden_dim, activation = activation))
        self.loss_coef = loss_coef

    def forward(self, inputs: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the hierarchical MoE model.

        Args:
            inputs (torch.Tensor): Input tensor with shape [batch_size, num_tokens, dim].
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - Output tensor after passing through the experts.
                - Loss term.
        """
        b, n, d, eo, ei = *inputs.shape, self.num_experts_outer, self.num_experts_inner
        dispatch_tensor_outer, combine_tensor_outer, loss_outer = self.gate_outer(inputs)
        expert_inputs_outer = torch.einsum('bnd,bnec->ebcd', inputs, dispatch_tensor_outer)

        # we construct an "importance" Tensor for the inputs to the second-level
        # gating.  The importance of an input is 1.0 if it represents the
        # first-choice expert-group and 0.5 if it represents the second-choice expert
        # group.  This is used by the second-level gating.
        importance = combine_tensor_outer.permute(2, 0, 3, 1).sum(dim=-1)
        importance = 0.5 * ((importance > 0.5).float() + (importance > 0.).float())

        dispatch_tensor_inner, combine_tensor_inner, loss_inner = self.gate_inner(expert_inputs_outer, importance = importance)
        expert_inputs = torch.einsum('ebnd,ebnfc->efbcd', expert_inputs_outer, dispatch_tensor_inner)

        # Now feed the expert inputs through the experts.
        orig_shape = expert_inputs.shape
        expert_inputs = expert_inputs.reshape(eo, ei, -1, d)
        expert_outputs = self.experts(expert_inputs)
        expert_outputs = expert_outputs.reshape(*orig_shape)

        # NOW COMBINE EXPERT OUTPUTS (reversing everything we have done)
        # expert_output has shape [y0, x1, h, d, n]

        expert_outputs_outer = torch.einsum('efbcd,ebnfc->ebnd', expert_outputs, combine_tensor_inner)
        output = torch.einsum('ebcd,bnec->bnd', expert_outputs_outer, combine_tensor_outer)
        return output, (loss_outer + loss_inner) * self.loss_coef