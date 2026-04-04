"""
Machine Learning Models for Deepfake Detection
"""

import os
from typing import Any, Dict, Optional

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn as nn
import timm
from torchvision.models import (
    ResNet18_Weights,
    Swin_T_Weights,
    VGG16_Weights,
    resnet18,
    swin_t,
    vgg16,
)

from app.core.config import settings
from app.core.logging import logger


FRAME_BACKBONE_TYPES = {"vgg", "swin", "vit", "resnet"}
BUILTIN_MODEL_SOURCE = "builtin"
BUILTIN_MODEL_READINESS = "fallback_only"
BUILTIN_MODEL_SELECTION_POLICY = "fallback_only"
BUILTIN_MODEL_STATUS_PRETRAINED = "builtin_pretrained"
BUILTIN_MODEL_STATUS_RANDOM_INIT = "builtin_random_init"
BUILTIN_MODEL_STATUS_RANDOM_INIT_FALLBACK = "builtin_random_init_fallback"
BUILTIN_MODEL_STATUS_UNKNOWN = "builtin_unknown"
BUILTIN_MODEL_WEIGHT_STATE_PRETRAINED = "framework_pretrained"
BUILTIN_MODEL_WEIGHT_STATE_RANDOM_INIT = "random_init"
BUILTIN_MODEL_WEIGHT_STATE_RANDOM_INIT_FALLBACK = "random_init_fallback"
BUILTIN_MODEL_WEIGHT_STATE_UNKNOWN = "unknown"
MODEL_PROVENANCE_ATTR = "_model_provenance"


def _attach_builtin_model_provenance(
    model: nn.Module,
    *,
    status: str,
    weight_state: str,
    fallback_reason: Optional[str] = None,
) -> nn.Module:
    provenance: Dict[str, Any] = {
        "source": BUILTIN_MODEL_SOURCE,
        "status": status,
        "weight_state": weight_state,
        "readiness": BUILTIN_MODEL_READINESS,
        "selection_policy": BUILTIN_MODEL_SELECTION_POLICY,
    }
    if fallback_reason:
        provenance["fallback_reason"] = fallback_reason
    setattr(model, MODEL_PROVENANCE_ATTR, provenance)
    return model


def _inherit_builtin_model_provenance(
    model: nn.Module,
    component: nn.Module,
) -> nn.Module:
    component_provenance = getattr(component, MODEL_PROVENANCE_ATTR, None)
    if component_provenance:
        setattr(model, MODEL_PROVENANCE_ATTR, dict(component_provenance))
        return model
    return _attach_builtin_model_provenance(
        model,
        status=BUILTIN_MODEL_STATUS_UNKNOWN,
        weight_state=BUILTIN_MODEL_WEIGHT_STATE_UNKNOWN,
    )


def get_model_provenance(model: nn.Module) -> Optional[Dict[str, Any]]:
    provenance = getattr(model, MODEL_PROVENANCE_ATTR, None)
    if provenance is None:
        return None
    return dict(provenance)


def _load_torchvision_model(factory, weights, model_name: str, pretrained: bool):
    if not pretrained:
        return _attach_builtin_model_provenance(
            factory(weights=None),
            status=BUILTIN_MODEL_STATUS_RANDOM_INIT,
            weight_state=BUILTIN_MODEL_WEIGHT_STATE_RANDOM_INIT,
        )

    try:
        return _attach_builtin_model_provenance(
            factory(weights=weights),
            status=BUILTIN_MODEL_STATUS_PRETRAINED,
            weight_state=BUILTIN_MODEL_WEIGHT_STATE_PRETRAINED,
        )
    except Exception as exc:
        if not settings.MODEL_ALLOW_RANDOM_INIT_FALLBACK:
            raise
        logger.warning(
            "Failed to load pretrained torchvision weights; built-in model stays fallback-only with random initialization",
            model_name=model_name,
            error=str(exc),
        )
        return _attach_builtin_model_provenance(
            factory(weights=None),
            status=BUILTIN_MODEL_STATUS_RANDOM_INIT_FALLBACK,
            weight_state=BUILTIN_MODEL_WEIGHT_STATE_RANDOM_INIT_FALLBACK,
            fallback_reason=str(exc),
        )


def _load_timm_model(model_name: str, num_classes: int, pretrained: bool):
    if not pretrained:
        return _attach_builtin_model_provenance(
            timm.create_model(model_name, pretrained=False, num_classes=num_classes),
            status=BUILTIN_MODEL_STATUS_RANDOM_INIT,
            weight_state=BUILTIN_MODEL_WEIGHT_STATE_RANDOM_INIT,
        )

    try:
        return _attach_builtin_model_provenance(
            timm.create_model(model_name, pretrained=True, num_classes=num_classes),
            status=BUILTIN_MODEL_STATUS_PRETRAINED,
            weight_state=BUILTIN_MODEL_WEIGHT_STATE_PRETRAINED,
        )
    except Exception as exc:
        if not settings.MODEL_ALLOW_RANDOM_INIT_FALLBACK:
            raise
        logger.warning(
            "Failed to load pretrained timm weights; built-in model stays fallback-only with random initialization",
            model_name=model_name,
            error=str(exc),
        )
        return _attach_builtin_model_provenance(
            timm.create_model(model_name, pretrained=False, num_classes=num_classes),
            status=BUILTIN_MODEL_STATUS_RANDOM_INIT_FALLBACK,
            weight_state=BUILTIN_MODEL_WEIGHT_STATE_RANDOM_INIT_FALLBACK,
            fallback_reason=str(exc),
        )


class CustomVGG(nn.Module):
    """Custom VGG model for deepfake detection"""

    def __init__(self, num_classes=2, pretrained=True):
        super(CustomVGG, self).__init__()
        self.img_model = _load_torchvision_model(
            vgg16, VGG16_Weights.IMAGENET1K_V1, "vgg16", pretrained
        )

        # Freeze all parameters except the classifier
        for param in self.img_model.parameters():
            param.requires_grad = False

        num_features = self.img_model.classifier[0].in_features
        self.feature_dim = num_features
        self.img_model.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(512),
            nn.Linear(512, num_classes),
        )

        # Enable training for classifier parameters
        for param in self.img_model.classifier.parameters():
            param.requires_grad = True

        _inherit_builtin_model_provenance(self, self.img_model)

    def extract_features(self, x):
        x = self.img_model.features(x)
        x = self.img_model.avgpool(x)
        return x.view(x.size(0), -1)

    def forward(self, x):
        features = self.extract_features(x)
        return self.img_model.classifier(features)


class PretrainedVGG(nn.Module):
    """Pretrained VGG feature extractor"""

    def __init__(self, pretrained=True):
        super(PretrainedVGG, self).__init__()
        pretrained_cnn = _load_torchvision_model(
            vgg16, VGG16_Weights.IMAGENET1K_V1, "vgg16_features", pretrained
        )
        self.features = pretrained_cnn.features
        self.avgpool = pretrained_cnn.avgpool
        self.feature_dim = pretrained_cnn.classifier[0].in_features

        # Freeze feature extraction parameters
        for param in self.features.parameters():
            param.requires_grad = False

        _inherit_builtin_model_provenance(self, pretrained_cnn)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # Flatten features
        return x


class LRCN(nn.Module):
    """LRCN (Long-term Recurrent Convolutional Network) for video deepfake detection"""

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_classes=2,
        pretrained=True,
        frame_projection_size=128,
    ):
        super(LRCN, self).__init__()
        self.pretrained_cnn = PretrainedVGG(pretrained=pretrained)
        self.feature_dim = input_size
        self.frame_projection_size = max(32, min(frame_projection_size, input_size))
        self.frame_projection = nn.Sequential(
            nn.Linear(input_size, self.frame_projection_size),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.lstm = nn.LSTM(
            self.frame_projection_size, hidden_size, num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)
        frame_hidden = max(64, min(hidden_size, self.frame_projection_size * 2))
        self.frame_classifier = nn.Sequential(
            nn.Linear(self.frame_projection_size * 2, frame_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(frame_hidden, num_classes),
        )
        fusion_hidden = max(hidden_size, self.frame_projection_size)
        fusion_input_size = (
            (self.frame_projection_size * 2) + hidden_size + (num_classes * 2)
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_size, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_hidden, num_classes),
        )

        _inherit_builtin_model_provenance(self, self.pretrained_cnn)

    def forward(self, x):
        batch_size, seq_length, channels, height, width = x.size()
        x = x.view(batch_size * seq_length, channels, height, width)
        self.pretrained_cnn.eval()
        with torch.no_grad():
            frame_features = self.pretrained_cnn(x)
        frame_features = frame_features.view(batch_size, seq_length, -1)
        projected_features = self.frame_projection(
            frame_features.reshape(batch_size * seq_length, -1)
        )
        projected_features = projected_features.view(batch_size, seq_length, -1)

        center_index = seq_length // 2
        center_features = projected_features[:, center_index, :]
        pooled_features = projected_features.mean(dim=1)
        frame_summary = torch.cat([center_features, pooled_features], dim=1)
        frame_logits = self.frame_classifier(frame_summary)

        lstm_out, _ = self.lstm(projected_features)

        temporal_features = lstm_out[:, -1, :]
        temporal_logits = self.fc(temporal_features)

        fusion_input = torch.cat(
            [frame_summary, temporal_features, frame_logits, temporal_logits], dim=1
        )
        return self.fusion_head(fusion_input)


class VideoTemporalHybridModel(nn.Module):
    def __init__(
        self,
        backbone_type: str,
        temporal_hidden_size: int = 256,
        temporal_num_layers: int = 2,
        num_classes: int = 2,
        pretrained: bool = True,
        feature_projection_size: int = 256,
        temporal_bidirectional: bool = True,
        temporal_attention_pooling: bool = True,
    ):
        super(VideoTemporalHybridModel, self).__init__()
        if backbone_type not in FRAME_BACKBONE_TYPES:
            raise ValueError(
                f"Temporal hybrid model requires an image backbone, got: {backbone_type}"
            )

        self.backbone_type = backbone_type
        self.frame_encoder = ModelRegistry.get_model(
            backbone_type, num_classes=num_classes, pretrained=pretrained
        )
        for param in self.frame_encoder.parameters():
            param.requires_grad = False

        feature_dim = getattr(self.frame_encoder, "feature_dim", None)
        if not feature_dim or not hasattr(self.frame_encoder, "extract_features"):
            raise ValueError(
                f"Backbone {backbone_type} does not expose extract_features/feature_dim"
            )

        self.feature_dim = feature_dim
        self.temporal_bidirectional = temporal_bidirectional
        self.temporal_attention_pooling = temporal_attention_pooling
        self.feature_projection_size = max(
            32, min(feature_projection_size, feature_dim)
        )
        self.feature_projection = nn.Sequential(
            nn.Linear(feature_dim, self.feature_projection_size),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.lstm = nn.LSTM(
            self.feature_projection_size,
            temporal_hidden_size,
            temporal_num_layers,
            batch_first=True,
            bidirectional=temporal_bidirectional,
        )
        temporal_output_size = (
            temporal_hidden_size * 2 if temporal_bidirectional else temporal_hidden_size
        )
        frame_hidden = max(
            64, min(temporal_hidden_size, self.feature_projection_size * 2)
        )
        self.frame_classifier = nn.Sequential(
            nn.Linear(self.feature_projection_size * 2, frame_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(frame_hidden, num_classes),
        )
        if temporal_attention_pooling:
            attention_hidden = max(32, temporal_output_size // 2)
            self.temporal_attention = nn.Sequential(
                nn.Linear(temporal_output_size, attention_hidden),
                nn.Tanh(),
                nn.Linear(attention_hidden, 1),
            )
        else:
            self.temporal_attention = None
        self.temporal_classifier = nn.Linear(temporal_output_size, num_classes)
        fusion_hidden = max(temporal_output_size, self.feature_projection_size)
        fusion_input_size = (
            (self.feature_projection_size * 2)
            + temporal_output_size
            + (num_classes * 2)
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_size, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_hidden, num_classes),
        )

        _inherit_builtin_model_provenance(self, self.frame_encoder)

    def forward(self, x):
        batch_size, seq_length, channels, height, width = x.size()
        x = x.view(batch_size * seq_length, channels, height, width)
        self.frame_encoder.eval()
        with torch.no_grad():
            frame_features = self.frame_encoder.extract_features(x)
        frame_features = frame_features.view(batch_size, seq_length, -1)

        projected_features = self.feature_projection(
            frame_features.reshape(batch_size * seq_length, -1)
        )
        projected_features = projected_features.view(batch_size, seq_length, -1)

        center_index = seq_length // 2
        center_features = projected_features[:, center_index, :]
        pooled_features = projected_features.mean(dim=1)
        frame_summary = torch.cat([center_features, pooled_features], dim=1)
        frame_logits = self.frame_classifier(frame_summary)

        temporal_output, _ = self.lstm(projected_features)
        if self.temporal_attention is not None:
            attention_scores = torch.softmax(
                self.temporal_attention(temporal_output).squeeze(-1), dim=1
            )
            temporal_features = torch.sum(
                temporal_output * attention_scores.unsqueeze(-1), dim=1
            )
        else:
            temporal_features = temporal_output[:, -1, :]
        temporal_logits = self.temporal_classifier(temporal_features)

        fusion_input = torch.cat(
            [frame_summary, temporal_features, frame_logits, temporal_logits], dim=1
        )
        return self.fusion_head(fusion_input)


class SwinModel(nn.Module):
    """Swin Transformer model for deepfake detection"""

    def __init__(self, image_size=224, pretrained=True, num_classes=2):
        super(SwinModel, self).__init__()
        # Load Swin Transformer model
        self.swin = _load_torchvision_model(
            swin_t, Swin_T_Weights.DEFAULT, "swin_t", pretrained
        )

        # Freeze all parameters
        for param in self.swin.parameters():
            param.requires_grad = False

        # Replace classification head
        num_features = self.swin.head.in_features
        self.feature_dim = num_features
        self.swin.head = nn.Linear(num_features, num_classes)

        _inherit_builtin_model_provenance(self, self.swin)

    def extract_features(self, x):
        x = self.swin.features(x)
        x = self.swin.norm(x)
        x = self.swin.permute(x)
        x = self.swin.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x):
        features = self.extract_features(x)
        return self.swin.head(features)


class ViTModel(nn.Module):
    """Vision Transformer model for deepfake detection"""

    def __init__(self, image_size=224, pretrained=True, num_classes=2):
        super(ViTModel, self).__init__()
        # Load pre-trained ViT model
        self.vit = _load_timm_model("vit_base_patch16_224", num_classes, pretrained)
        self.feature_dim = self.vit.head.in_features

        # Freeze all parameters
        for param in self.vit.parameters():
            param.requires_grad = False

        # Replace classification head
        self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)

        _inherit_builtin_model_provenance(self, self.vit)

    def extract_features(self, x):
        features = self.vit.forward_features(x)
        return self.vit.forward_head(features, pre_logits=True)

    def forward(self, x):
        features = self.extract_features(x)
        return self.vit.head(features)


class ResNet(nn.Module):
    """ResNet model for deepfake detection"""

    def __init__(self, num_classes=2, pretrained=True):
        super(ResNet, self).__init__()
        self.img_model = _load_torchvision_model(
            resnet18, ResNet18_Weights.DEFAULT, "resnet18", pretrained
        )

        # Freeze ResNet parameters
        for param in self.img_model.parameters():
            param.requires_grad = False

        # Replace final layer
        num_features = self.img_model.fc.in_features
        self.feature_dim = num_features
        self.img_model.fc = nn.Sequential(  # type: ignore[assignment]
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(512),
            nn.Linear(512, num_classes),
        )

        # Enable training for new layer
        for param in self.img_model.fc.parameters():
            param.requires_grad = True

        _inherit_builtin_model_provenance(self, self.img_model)

    def extract_features(self, x):
        x = self.img_model.conv1(x)
        x = self.img_model.bn1(x)
        x = self.img_model.relu(x)
        x = self.img_model.maxpool(x)
        x = self.img_model.layer1(x)
        x = self.img_model.layer2(x)
        x = self.img_model.layer3(x)
        x = self.img_model.layer4(x)
        x = self.img_model.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x):
        features = self.extract_features(x)
        return self.img_model.fc(features)


class ModelRegistry:
    """Registry for managing different model types"""

    _models = {
        "vgg": CustomVGG,
        "lrcn": LRCN,
        "swin": SwinModel,
        "vit": ViTModel,
        "resnet": ResNet,
    }

    @classmethod
    def get_model(cls, model_type: str, **kwargs):
        """Get model instance by type"""
        if model_type not in cls._models:
            raise ValueError(
                f"Unsupported model type: {model_type}. Supported types: {list(cls._models.keys())}"
            )

        pretrained = kwargs.get("pretrained", settings.MODEL_USE_PRETRAINED_WEIGHTS)

        if kwargs.get("video_temporal_enabled") and model_type in FRAME_BACKBONE_TYPES:
            return VideoTemporalHybridModel(
                backbone_type=model_type,
                temporal_hidden_size=kwargs.get("temporal_hidden_size", 256),
                temporal_num_layers=kwargs.get("temporal_num_layers", 2),
                num_classes=kwargs.get("num_classes", 2),
                pretrained=pretrained,
                feature_projection_size=kwargs.get("feature_projection_size", 256),
                temporal_bidirectional=kwargs.get(
                    "temporal_bidirectional", settings.TEMPORAL_BIDIRECTIONAL
                ),
                temporal_attention_pooling=kwargs.get(
                    "temporal_attention_pooling",
                    settings.TEMPORAL_ATTENTION_POOLING,
                ),
            )

        model_class = cls._models[model_type]

        # Handle special cases for model initialization
        if model_type == "lrcn":
            # LRCN requires specific parameters
            input_size = kwargs.get("input_size", 25088)  # VGG feature size
            hidden_size = kwargs.get("hidden_size", 256)
            num_layers = kwargs.get("num_layers", 1)
            num_classes = kwargs.get("num_classes", 2)
            frame_projection_size = kwargs.get("frame_projection_size", 128)
            return model_class(
                input_size,
                hidden_size,
                num_layers,
                num_classes,
                pretrained=pretrained,
                frame_projection_size=frame_projection_size,
            )
        else:
            # Other models use standard initialization
            num_classes = kwargs.get("num_classes", 2)
            return model_class(num_classes=num_classes, pretrained=pretrained)

    @classmethod
    def list_models(cls):
        """List all available model types"""
        return list(cls._models.keys())

    @classmethod
    def register_model(cls, name: str, model_class):
        """Register a new model type"""
        cls._models[name] = model_class
        logger.info(
            "Model registered", model_name=name, model_class=model_class.__name__
        )


def create_model(model_type: str, **kwargs):
    """Factory function to create models"""
    return ModelRegistry.get_model(model_type, **kwargs)


def get_default_model():
    """Get the default model type"""
    return settings.DEFAULT_MODEL_TYPE
