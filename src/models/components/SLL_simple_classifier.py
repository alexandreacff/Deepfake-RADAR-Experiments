from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

from .mlp import MLPBase
from .pooling import AttentiveStatisticsPooling

class SSLDeepFakeBaseModel(nn.Module, ABC):
    """
    Base Model for DeepFake detection that handles:
    - MLP initialization
    - Layer weight strategy: 'per_layer', 'weighted_sum'
    - Pooling strategy: 'mean' or 'attpool' (attpool only with per_layer)
    """

    def __init__(
        self,
        mlp_input_dim: int = 768,
        mlp_hidden_dim: int = 1024,
        mlp_num_layers: int = 2,
        mlp_output_size: int = 2,
        mlp_dropout: float = 0.1,
        mlp_activation_func: str = "relu",
        layer_weight_strategy: str = "per_layer",
        num_feature_layers: int = 25,
        specific_layer_idx: int = -1,
        pooling_strategy: str = "mean", # "mean" or "attpool"

    ):
        super().__init__()

        self.mlp = MLPBase(
            input_size=mlp_input_dim,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            output_size=mlp_output_size,
            dropout=mlp_dropout,
            activation_func=mlp_activation_func,
        )

        self.layer_weight_strategy = layer_weight_strategy
        self.specific_layer_idx = specific_layer_idx
        self.pooling_strategy = pooling_strategy
        self.num_feature_layers = num_feature_layers

        # Validate layer_weight_strategy
        if layer_weight_strategy == "weighted_sum":
            self.layer_weights = nn.ParameterList(
                [nn.Parameter(torch.zeros(1)) for _ in range(num_feature_layers)]
            )
        elif layer_weight_strategy == "per_layer":
            if specific_layer_idx < 0:
                specific_layer_idx = num_feature_layers - 1

            self.specific_layer_idx = specific_layer_idx
        else:
            raise ValueError(f"Invalid layer weight strategy: {layer_weight_strategy}, choose 'per_layer' or 'weighted_sum'.")

        if pooling_strategy not in ["mean", "attpool"]:
            raise ValueError(
                f"Invalid pooling strategy: {pooling_strategy}. Choose 'mean' or 'attpool'."
            )


        if pooling_strategy == "attpool":
            # AttentiveStatisticsPooling requires initialization once we know F
            # Which is half of the MLP input dimension because we concatenate mean and std
            # Consider that mlp_input_dim is always equal to the F dimension of the embeddings
            self.attpool = AttentiveStatisticsPooling(input_size=int(mlp_input_dim/2))

    def get_layer_weights(self):
        layer_weights = [w.detach().item() for w in self.layer_weights]
        layer_weights = F.softmax(torch.tensor(layer_weights), dim=0)
        return layer_weights

    def _weighted_sum(self, x: torch.Tensor) -> torch.Tensor:
        """
        Weighted sum over the layers dimension.
        Args:
            x: Input tensor of shape [B, NUM_LAYERS, SEQUENCE_LENGTH, FEATURE_DIM]
        Returns:
            Weighted sum tensor of shape [B, SEQUENCE_LENGTH, FEATURE_DIM]
        """

        B, NUM_LAYERS, SEQ_LEN, FEAT_DIM = x.shape

        layer_weights = torch.stack([w for w in self.layer_weights])
        layer_weights = F.softmax(layer_weights, dim=0)
        layer_weights = layer_weights.view(NUM_LAYERS, 1, 1)

        expanded_weights = layer_weights.expand(B, NUM_LAYERS, SEQ_LEN, FEAT_DIM)
        # Apply weights to the input
        weighted_layers = x * expanded_weights
        # Sum over the layers dimension
        # Shape: [B, SEQ_LEN, FEAT_DIM]
        weighted_sum = weighted_layers.sum(dim=1)

        return weighted_sum


    def _specific_layer(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        return x[:, layer_idx, :]

    @abstractmethod
    def _get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Method to get embeddings:
        For dynamic model: returns [B,num_layers+1,T,F]
        For embedding model: returns [B,num_feature_layers,F]
        """
        pass

    @abstractmethod
    def _get_embedding_dim(self) -> int:
        pass

    # @abstractmethod
    # def _get_num_feature_layers(self) -> int:
    #     pass

    def _apply_layer_weighting(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Apply the chosen layer_weight_strategy.
        After weighting:
        - per_layer: [B,F] -> reshape to [B,1,F]
        - weighted_sum: [B,F] -> reshape to [B,1,F]
        - transformer: [B,F] -> reshape to [B,1,F]
        """
        if self.layer_weight_strategy == "per_layer":
            embeddings = self._specific_layer(embeddings, self.specific_layer_idx) # [B,T,F]
        elif self.layer_weight_strategy == "weighted_sum":
            embeddings = self._weighted_sum(embeddings) # [B,T,F]
        else:
            raise ValueError(f"Invalid layer weight strategy: {self.layer_weight_strategy}")

        return embeddings

    def _apply_pooling(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Apply pooling over the time dimension.
        embeddings: [B,T,F]
        mask: [B,T] if attpool selected
        """
        if self.pooling_strategy == "mean":
            # Mean pooling over T
            return embeddings.mean(dim=1)  # [B,F]

        elif self.pooling_strategy == "attpool":
            return self.attpool(embeddings)
        else:
            raise ValueError(f"Invalid pooling strategy: {self.pooling_strategy}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get embeddings
        embeddings = self._get_embeddings(x)
        # Apply layer weighting
        embeddings = self._apply_layer_weighting(embeddings)
        # weighted_sum_2 and routing already returns [B,F], so we skip pooling

        # Apply pooling
        logits_input = self._apply_pooling(embeddings)  # [B,F] for "mean" or [B,2F] for "attpool" / [B, T, F] -> [B, F]

        # MLP classification
        logits = self.mlp(logits_input).squeeze(-1)

        return logits

class SSLDeepFakeLastLayerEmbeddingModel(nn.Module):
    """
    Model that uses the last layer of the embeddings model.

    It does not apply layer weighting strategies.
    """

    def __init__(
        self,
        mlp_input_dim: int = 768,
        mlp_hidden_dim: int = 1024,
        mlp_num_layers: int = 2,
        mlp_output_size: int = 7,
        mlp_dropout: float = 0.1,
        mlp_activation_func: str = "relu",
        pooling_strategy: str = "mean", # "mean" or "attpool"
    ):
        super().__init__()
        self.pooling_strategy = pooling_strategy
        if pooling_strategy not in ["mean", "attpool"]:
            raise ValueError(
                f"Invalid pooling strategy: {pooling_strategy}. Choose 'mean' or 'attpool'."
            )

        if pooling_strategy == "attpool":
            # AttentiveStatisticsPooling requires initialization once we know F
            # Which is half of the MLP input dimension because we concatenate mean and std
            # Consider that mlp_input_dim is always equal to the F dimension of the embeddings
            self.attpool = AttentiveStatisticsPooling(input_size=int(mlp_input_dim/2))

        self.mlp = MLPBase(
            input_size=mlp_input_dim,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            output_size=mlp_output_size,
            dropout=mlp_dropout,
            activation_func=mlp_activation_func,
        )

    def _apply_pooling(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Apply pooling over the time dimension.
        embeddings: [B,T,F]
        mask: [B,T] if attpool selected
        """
        if self.pooling_strategy == "mean":
            # Mean pooling over T
            return embeddings.mean(dim=1)  # [B,F]

        elif self.pooling_strategy == "attpool":
            return self.attpool(embeddings)
        else:
            raise ValueError(f"Invalid pooling strategy: {self.pooling_strategy}")

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        # Apply pooling
        logits_input = self._apply_pooling(embeddings) # [B, F] or [B,2*F]
        # MLP classification
        logits = self.mlp(logits_input)
        return logits

class SSLDeepFakeDynamicModel(SSLDeepFakeBaseModel):
    """
    Uses a pretrained backbone (e.g. WavLM).
    """

    def __init__(
        self,
        model_name: str = "microsoft/wavlm-large",
        freeze_backbone: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        # config = AutoConfig.from_pretrained(model_name, output_hidden_states=True , trust_remote_code=True)
        try:
            self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        except AttributeError as exc:
            is_gramt_max_length_bug = (
                model_name == "labhamlet/gramt-binaural-time"
                and "'GRAMTBinauralTimeConfig' object has no attribute 'max_length'" in str(exc)
            )
            if not is_gramt_max_length_bug or not self._patch_gramt_dynamic_config():
                raise
            self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            self._freeze_backbone()
            self.backbone.eval()

        # if self.num_feature_layers <=  self._get_num_feature_layers():
        #     self.num_feature_layers = self._get_num_feature_layers()
        #     print(f"Warning: num_feature_layers is less than the actual number of layers in the model. Setting num_feature_layers to {self.num_feature_layers}.")
        
        
    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        modality_input = x

        if self.freeze_backbone:
            with torch.no_grad():
                outputs = self.backbone(**modality_input, output_hidden_states=True)
        else:
            outputs = self.backbone(**modality_input, output_hidden_states=True)

            
        hidden_states = outputs.hidden_states  # tuple of (layer_0,...,layer_n)
        # print(f"Number of hidden states: {len(hidden_states)}")
        # [num_layers,B,T,F]
        all_layers = torch.stack(hidden_states)
        # transform to [B,num_layers,T,F]
        all_layers = all_layers.permute(1, 0, 2, 3)
        return all_layers

    def _get_embedding_dim(self) -> int:
        return self.mlp.layers[0].in_features
    
    @staticmethod
    def _patch_gramt_dynamic_config() -> bool:
        """Patch a GRAMT remote-code bug where input_length is not exposed as max_length."""
        cache_root = Path.home() / ".cache/huggingface/modules/transformers_modules/labhamlet"
        if not cache_root.exists():
            return False

        patched = False
        for config_path in cache_root.rglob("configuration_gramt_binaural_time.py"):
            text = config_path.read_text(encoding="utf-8")
            if "self.max_length = input_length" in text:
                patched = True
                continue

            old = "        self.input_length = input_length\n        self.num_mel_bins = num_mel_bins\n"
            new = (
                "        self.input_length = input_length\n"
                "        self.max_length = input_length\n"
                "        self.num_mel_bins = num_mel_bins\n"
            )
            if old in text:
                config_path.write_text(text.replace(old, new), encoding="utf-8")
                patched = True

        return patched

    # def _get_num_feature_layers(self) -> int:
    #     dummy_input = torch.zeros(1, 16000)  # [B, T]
    #     embeddings = self._get_embeddings(dummy_input)  # [B, num_layers, T, F]
    #     return embeddings.size(1)

class GramtDeepFakeDynamicModel(SSLDeepFakeBaseModel):
    """
    Uses a pretrained backbone (e.g. WavLM).
    """

    def __init__(
        self,
        model_name: str = "microsoft/wavlm-large",
        freeze_backbone: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        # config = AutoConfig.from_pretrained(model_name, output_hidden_states=True , trust_remote_code=True)
        try:
            self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        except AttributeError as exc:
            is_gramt_max_length_bug = (
                model_name == "labhamlet/gramt-binaural-time"
                and "'GRAMTBinauralTimeConfig' object has no attribute 'max_length'" in str(exc)
            )
            if not is_gramt_max_length_bug or not self._patch_gramt_dynamic_config():
                raise
            self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            self._freeze_backbone()
            self.backbone.eval()

        # if self.num_feature_layers <=  self._get_num_feature_layers():
        #     self.num_feature_layers = self._get_num_feature_layers()
        #     print(f"Warning: num_feature_layers is less than the actual number of layers in the model. Setting num_feature_layers to {self.num_feature_layers}.")
        
        
    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        modality_input = x

        if self.freeze_backbone:
            with torch.no_grad():
                outputs = self.backbone(modality_input)
        else:
            outputs = self.backbone(modality_input)

            
        hidden_states = outputs.hidden_states  # tuple of (layer_0,...,layer_n)
        # print(f"Number of hidden states: {len(hidden_states)}")
        # [num_layers,B,T,F]
        all_layers = torch.stack(hidden_states)
        # transform to [B,num_layers,T,F]
        all_layers = all_layers.permute(1, 0, 2, 3)
        return all_layers

    def _get_embedding_dim(self) -> int:
        return self.mlp.layers[0].in_features
    
    @staticmethod
    def _patch_gramt_dynamic_config() -> bool:
        """Patch a GRAMT remote-code bug where input_length is not exposed as max_length."""
        cache_root = Path.home() / ".cache/huggingface/modules/transformers_modules/labhamlet"
        if not cache_root.exists():
            return False

        patched = False
        for config_path in cache_root.rglob("configuration_gramt_binaural_time.py"):
            text = config_path.read_text(encoding="utf-8")
            if "self.max_length = input_length" in text:
                patched = True
                continue

            old = "        self.input_length = input_length\n        self.num_mel_bins = num_mel_bins\n"
            new = (
                "        self.input_length = input_length\n"
                "        self.max_length = input_length\n"
                "        self.num_mel_bins = num_mel_bins\n"
            )
            if old in text:
                config_path.write_text(text.replace(old, new), encoding="utf-8")
                patched = True

        return patched