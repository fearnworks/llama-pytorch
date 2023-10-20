import torch
import torch.nn as nn
from typing import Optional, Union

from moe.top2_gating import Top2Gating
from moe.expert import Experts
from moe.moe_util import default


class MoE(nn.Module):
    def __init__(
        self,
        dim: int,
        num_experts: int = 16,
        hidden_dim: Optional[int] = None,
        activation: nn.Module = nn.ReLU,
        second_policy_train: str = "random",
        second_policy_eval: str = "random",
        second_threshold_train: float = 0.2,
        second_threshold_eval: float = 0.2,
        capacity_factor_train: float = 1.25,
        capacity_factor_eval: float = 2.0,
        loss_coef: float = 1e-2,
        experts: Optional[Union[nn.Module, str]] = None,
    ) -> None:
        """
        Initialize the Mixture of Experts (MoE) layer.
        
        Args:
            dim (int): Input and output dimension.
            num_experts (int, optional): Number of experts. Defaults to 16.
            hidden_dim (int, optional): Hidden layer dimension. Defaults to None.
            activation (nn.Module, optional): Activation function. Defaults to nn.ReLU.
            second_policy_train (str, optional): Policy for second expert during training. Defaults to 'random'.
            second_policy_eval (str, optional): Policy for second expert during evaluation. Defaults to 'random'.
            second_threshold_train (float, optional): Second expert threshold during training. Defaults to 0.2.
            second_threshold_eval (float, optional): Second expert threshold during evaluation. Defaults to 0.2.
            capacity_factor_train (float, optional): Capacity factor during training. Defaults to 1.25.
            capacity_factor_eval (float, optional): Capacity factor during evaluation. Defaults to 2.0.
            loss_coef (float, optional): Loss coefficient. Defaults to 1e-2.
            experts (nn.Module or str, optional): Pre-defined experts or type of experts to use. Defaults to None.
        """
        self.num_experts = num_experts

        gating_kwargs = {
            "second_policy_train": second_policy_train,
            "second_policy_eval": second_policy_eval,
            "second_threshold_train": second_threshold_train,
            "second_threshold_eval": second_threshold_eval,
            "capacity_factor_train": capacity_factor_train,
            "capacity_factor_eval": capacity_factor_eval,
        }
        # Initialize Gating and Experts
        self.gate = Top2Gating(dim, num_gates=num_experts, **gating_kwargs)
        self.experts = default(
            experts,
            lambda: Experts(
                dim,
                num_experts=num_experts,
                hidden_dim=hidden_dim,
                activation=activation,
            ),
        )
        self.loss_coef = loss_coef

    def forward(self, inputs, **kwargs):
        """
        Forward pass for the MoE layer.
        
        Args:
            inputs (torch.Tensor): Input tensor of shape [batch_size, num_elements, dim].
            **kwargs: Additional keyword arguments.
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, num_elements, dim].
        """
        b, n, d, e = *inputs.shape, self.num_experts
        
        # Perform gating to obtain dispatch and combine tensors
        dispatch_tensor, combine_tensor, loss = self.gate(inputs)
        expert_inputs = torch.einsum("bnd,bnec->ebcd", inputs, dispatch_tensor)

        # Pass expert inputs through experts
        orig_shape = expert_inputs.shape
        expert_inputs = expert_inputs.reshape(e, -1, d)
        expert_outputs = self.experts(expert_inputs)
        expert_outputs = expert_outputs.reshape(*orig_shape)
        
        # Combine expert outputs
        output = torch.einsum("ebcd,bnec->bnd", expert_outputs, combine_tensor)
        return output, loss * self.loss_coef
