from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from moe.moe_util import top1, cumsum_exclusive, safe_one_hot
from moe.expert import MIN_EXPERT_CAPACITY

class Top2Gating(nn.Module):
    """
    Implements Top-2 Gating mechanism to decide which tokens go to which experts.

    Args:
        dim (int): Dimension of the input.
        num_gates (int): Number of gates or experts.
        eps (float, optional): Small epsilon for numerical stability. Defaults to 1e-9.
        outer_expert_dims (tuple, optional): Outer dimensions for expert matrix. Defaults to tuple().
        second_policy_train (str, optional): Policy for second expert during training. Defaults to 'random'.
        second_policy_eval (str, optional): Policy for second expert during evaluation. Defaults to 'random'.
        second_threshold_train (float, optional): Threshold for second expert during training. Defaults to 0.2.
        second_threshold_eval (float, optional): Threshold for second expert during evaluation. Defaults to 0.2.
        capacity_factor_train (float, optional): Capacity factor during training. Defaults to 1.25.
        capacity_factor_eval (float, optional): Capacity factor during evaluation. Defaults to 2.
    """
    def __init__(self,
                 dim: int,
                 num_gates: int,
                 eps: float = 1e-9,
                 outer_expert_dims: Optional[Tuple[int]] = None,
                 second_policy_train: str = 'random',
                 second_policy_eval: str = 'random',
                 second_threshold_train: float = 0.2,
                 second_threshold_eval: float = 0.2,
                 capacity_factor_train: float = 1.25,
                 capacity_factor_eval: float = 2.0):
        super().__init__()
        # Initialize parameters
        self.eps = eps
        self.num_gates = num_gates
        self.w_gating = nn.Parameter(torch.randn(*outer_expert_dims, dim, num_gates))
        # Policies and thresholds
        self.second_policy_train = second_policy_train
        self.second_policy_eval = second_policy_eval
        self.second_threshold_train = second_threshold_train
        self.second_threshold_eval = second_threshold_eval
        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval

    def forward(self, x: torch.Tensor, 
                importance: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for Top2Gating.

        Args:
            x (torch.Tensor): Input tensor of shape [..., batch_size, group_size, dim].
            importance (torch.Tensor, optional): Importance weights for the tokens. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Dispatch tensor, combine tensor, and loss.
        """
        # Initial setup
        *_, b, group_size, dim = x.shape
        num_gates = self.num_gates
        policy, threshold, capacity_factor = self._get_policy_and_threshold()

        # Gating mechanism
        raw_gates = self._compute_raw_gates(x)

        # Top-2 Experts
        gate_1, index_1, mask_1, density_1_proxy = self._find_top1_expert(raw_gates, importance)
        gate_2, index_2, mask_2 = self._find_top2_expert(raw_gates, mask_1, importance)

        # Normalize and apply policy
        gate_1, gate_2, mask_2 = self._normalize_and_apply_policy(gate_1, gate_2, mask_2, policy, threshold)

        # Compute loss for balancing
        loss = self._compute_loss(density_1_proxy, mask_1)

        # Compute expert assignment
        dispatch_tensor, combine_tensor = self._compute_assignment(gate_1, index_1, gate_2, index_2, mask_1, mask_2, capacity_factor, group_size)

        return dispatch_tensor, combine_tensor, loss
    
    def _get_policy_and_threshold(self) -> Tuple[str, float, float]:
        """
        Get the current policy, threshold, and capacity factor based on the training state.

        Returns:
            Tuple[str, float, float]: 
                - The second expert policy ("random", "all", "none", "threshold").
                - The second expert threshold value.
                - The capacity factor.
        """
        if self.training:
            return self.second_policy_train, self.second_threshold_train, self.capacity_factor_train
        else:
            return self.second_policy_eval, self.second_threshold_eval, self.capacity_factor_eval

    def _compute_raw_gates(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute raw gate values using the gating mechanism.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The computed raw gate values.
        """
        return torch.einsum('...bnd,...de->...bne', x, self.w_gating).softmax(dim=-1)

    def _find_top1_expert(self, raw_gates: torch.Tensor, importance: Optional[torch.Tensor] = None
                          ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Find the top-1 expert based on the raw gate values.

        Args:
            raw_gates (torch.Tensor): The raw gate values.
            importance (torch.Tensor, optional): Importance weights for the tokens.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - The gate values for the top-1 expert.
                - The index values for the top-1 expert.
                - The mask for the top-1 expert.
                - The density proxy for the top-1 expert.
        """
        gate_1, index_1 = top1(raw_gates)
        mask_1 = F.one_hot(index_1, self.num_gates).float()
        density_1_proxy = raw_gates
        if importance is not None:
            equals_one_mask = (importance == 1.).float()
            mask_1 *= equals_one_mask[..., None]
            gate_1 *= equals_one_mask
            density_1_proxy = density_1_proxy * equals_one_mask[..., None]
        return gate_1, index_1, mask_1, density_1_proxy

    def _find_top2_expert(self, 
                          raw_gates: torch.Tensor, 
                          mask_1: torch.Tensor, 
                          importance: Optional[torch.Tensor] = None
                          ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Find the top-2 expert based on the raw gate values and the mask of the top-1 expert.

        Args:
            raw_gates (torch.Tensor): The raw gate values.
            mask_1 (torch.Tensor): The mask for the top-1 expert.
            importance (torch.Tensor, optional): Importance weights for the tokens.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - The gate values for the top-2 expert.
                - The index values for the top-2 expert.
                - The mask for the top-2 expert.
        """
        gates_without_top_1 = raw_gates * (1. - mask_1)
        gate_2, index_2 = top1(gates_without_top_1)
        mask_2 = F.one_hot(index_2, self.num_gates).float()
        if importance is not None:
            greater_zero_mask = (importance > 0.).float()
            mask_2 *= greater_zero_mask[..., None]
        return gate_2, index_2, mask_2

    def _normalize_and_apply_policy(self, gate_1: torch.Tensor,
                                    gate_2: torch.Tensor,
                                    mask_2: torch.Tensor, 
                                    policy: str, 
                                    threshold: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Normalize the gate values and apply the policy for the second gate.

        Args:
            gate_1 (torch.Tensor): The gate values for the top-1 expert.
            gate_2 (torch.Tensor): The gate values for the top-2 expert.
            mask_2 (torch.Tensor): The mask for the top-2 expert.
            policy (str): The policy for the second gate ("random", "all", "none", "threshold").
            threshold (float): The threshold for the second gate.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - The normalized gate values for the top-1 expert.
                - The normalized gate values for the top-2 expert.
                - The modified mask for the top-2 expert based on the policy.
        """
        denom = gate_1 + gate_2 + self.eps
        gate_1 /= denom
        gate_2 /= denom
        if policy == "all":
            pass
        elif policy == "none":
            mask_2 = torch.zeros_like(mask_2)
        elif policy == "threshold":
            mask_2 *= (gate_2 > threshold).float()
        elif policy == "random":
            probs = torch.zeros_like(gate_2).uniform_(0., 1.)
            mask_2 *= (probs < (gate_2 / max(threshold, self.eps))).float().unsqueeze(-1)
        else:
            raise ValueError(f"Unknown policy {policy}")
        return gate_1, gate_2, mask_2

    def _compute_loss(self, density_1_proxy: torch.Tensor, mask_1: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss for balancing the experts.

        Args:
            density_1_proxy (torch.Tensor): The density proxy for the top-1 expert.
            mask_1 (torch.Tensor): The mask for the top-1 expert.

        Returns:
            torch.Tensor: The computed loss for expert balancing.
        """
        density_1 = mask_1.mean(dim=-2)
        return (density_1_proxy * density_1).mean() * float(self.num_gates ** 2)

    def _compute_assignment(self, gate_1: torch.Tensor, 
                            index_1: torch.Tensor, 
                            gate_2: torch.Tensor, 
                            index_2: torch.Tensor, 
                            mask_1: torch.Tensor, 
                            mask_2: torch.Tensor, 
                            capacity_factor: float, 
                            group_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the assignment tensors, dispatch_tensor and combine_tensor.

        Args:
            gate_1 (torch.Tensor): Gate values for top-1 experts.
            index_1 (torch.Tensor): Index values for top-1 experts.
            gate_2 (torch.Tensor): Gate values for top-2 experts.
            index_2 (torch.Tensor): Index values for top-2 experts.
            mask_1 (torch.Tensor): Mask for top-1 experts.
            mask_2 (torch.Tensor): Mask for top-2 experts.
            capacity_factor (float): Capacity factor for gating.
            group_size (int): Size of each group in the input.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - The dispatch tensor that specifies which token goes to which expert.
                - The combine tensor used to reassemble the tokens back from the experts.
        """
        # Compute expert capacity
        expert_capacity = min(group_size, int((group_size * capacity_factor) / self.num_gates))
        expert_capacity = max(expert_capacity, MIN_EXPERT_CAPACITY)
        expert_capacity_f = float(expert_capacity)

        # Compute position in expert for top-1 experts
        position_in_expert_1 = cumsum_exclusive(mask_1, dim=-2) * mask_1
        mask_1 *= (position_in_expert_1 < expert_capacity_f).float()
        mask_1_count = mask_1.sum(dim=-2, keepdim=True)
        mask_1_flat = mask_1.sum(dim=-1)
        position_in_expert_1 = position_in_expert_1.sum(dim=-1)

        # Compute position in expert for top-2 experts
        position_in_expert_2 = cumsum_exclusive(mask_2, dim=-2) + mask_1_count
        position_in_expert_2 *= mask_2
        mask_2 *= (position_in_expert_2 < expert_capacity_f).float()
        mask_2_flat = mask_2.sum(dim=-1)
        position_in_expert_2 = position_in_expert_2.sum(dim=-1)

        # Create combine tensor
        combine_tensor = (
            gate_1[..., None, None]
            * mask_1_flat[..., None, None]
            * F.one_hot(index_1, self.num_gates)[..., None]
            * safe_one_hot(position_in_expert_1.long(), expert_capacity)[..., None, :] +
            gate_2[..., None, None]
            * mask_2_flat[..., None, None]
            * F.one_hot(index_2, self.num_gates)[..., None]
            * safe_one_hot(position_in_expert_2.long(), expert_capacity)[..., None, :]
        )

        # Create dispatch tensor
        dispatch_tensor = combine_tensor.bool().to(combine_tensor)

        return dispatch_tensor, combine_tensor