"""
Machine Learning Models for Deepfake Detection
"""

import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

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


def _load_torchvision_model(factory, weights, model_name: str, pretrained: bool):
    """Load a torchvision model and optionally fall back to random init offline."""
    requested_weights = weights if pretrained else None
    try:
        return factory(weights=requested_weights)
    except Exception as exc:
        if not pretrained or not settings.MODEL_ALLOW_RANDOM_INIT_FALLBACK:
            raise
        logger.warning(
            "Failed to load pretrained torchvision weights; falling back to random initialization",
            model_name=model_name,
            error=str(exc),
        )
        return factory(weights=None)


def _load_timm_model(model_name: str, num_classes: int, pretrained: bool):
    """Load a timm model and optionally fall back to random init offline."""
    try:
        return timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes
        )
    except Exception as exc:
        if not pretrained or not settings.MODEL_ALLOW_RANDOM_INIT_FALLBACK:
            raise
        logger.warning(
            "Failed to load pretrained timm weights; falling back to random initialization",
            model_name=model_name,
            error=str(exc),
        )
        return timm.create_model(model_name, pretrained=False, num_classes=num_classes)


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

    def forward(self, x):
        return self.img_model(x)


class PretrainedVGG(nn.Module):
    """Pretrained VGG feature extractor"""

    def __init__(self, pretrained=True):
        super(PretrainedVGG, self).__init__()
        pretrained_cnn = _load_torchvision_model(
            vgg16, VGG16_Weights.IMAGENET1K_V1, "vgg16_features", pretrained
        )
        self.features = pretrained_cnn.features

        # Freeze feature extraction parameters
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten features
        return x


class LRCN(nn.Module):
    """LRCN (Long-term Recurrent Convolutional Network) for video deepfake detection"""

    def __init__(
        self, input_size, hidden_size, num_layers, num_classes=2, pretrained=True
    ):
        super(LRCN, self).__init__()
        self.pretrained_cnn = PretrainedVGG(pretrained=pretrained)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size, seq_length, channels, height, width = x.size()
        x = x.view(batch_size * seq_length, channels, height, width)
        frame_features = self.pretrained_cnn(x)
        frame_features = frame_features.view(batch_size, seq_length, -1)

        # Pass frame features through LSTM
        lstm_out, _ = self.lstm(frame_features)

        # Use the last time step output
        lstm_out = lstm_out[:, -1, :]

        # Classification
        output = self.fc(lstm_out)
        return output


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
        self.swin.head = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.swin(x)


class ViTModel(nn.Module):
    """Vision Transformer model for deepfake detection"""

    def __init__(self, image_size=224, pretrained=True, num_classes=2):
        super(ViTModel, self).__init__()
        # Load pre-trained ViT model
        self.vit = _load_timm_model("vit_base_patch16_224", num_classes, pretrained)

        # Freeze all parameters
        for param in self.vit.parameters():
            param.requires_grad = False

        # Replace classification head
        self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)

    def forward(self, x):
        return self.vit(x)


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

    def forward(self, x):
        return self.img_model(x)


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

        model_class = cls._models[model_type]
        pretrained = kwargs.get("pretrained", settings.MODEL_USE_PRETRAINED_WEIGHTS)

        # Handle special cases for model initialization
        if model_type == "lrcn":
            # LRCN requires specific parameters
            input_size = kwargs.get("input_size", 25088)  # VGG feature size
            hidden_size = kwargs.get("hidden_size", 512)
            num_layers = kwargs.get("num_layers", 2)
            num_classes = kwargs.get("num_classes", 2)
            return model_class(
                input_size,
                hidden_size,
                num_layers,
                num_classes,
                pretrained=pretrained,
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
