"""
Machine Learning Models for Deepfake Detection
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch.nn as nn
from torchvision.models import vgg16
from torchvision.models import resnet18, ResNet18_Weights
import timm
from torchvision.models import swin_t, swin_s, swin_b
from app.core.config import settings
from app.core.logging import logger


class CustomVGG(nn.Module):
    """Custom VGG model for deepfake detection"""
    
    def __init__(self, num_classes=2):
        super(CustomVGG, self).__init__()
        self.img_model = vgg16(weights='IMAGENET1K_V1')
        
        # Freeze all parameters except the classifier
        for param in self.img_model.parameters():
            param.requires_grad = False
        
        num_features = self.img_model.classifier[0].in_features
        self.img_model.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(512),
            nn.Linear(512, num_classes)
        )
        
        # Enable training for classifier parameters
        for param in self.img_model.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.img_model(x)


class PretrainedVGG(nn.Module):
    """Pretrained VGG feature extractor"""
    
    def __init__(self):
        super(PretrainedVGG, self).__init__()
        pretrained_cnn = vgg16(weights='IMAGENET1K_V1')
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
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes=2):
        super(LRCN, self).__init__()
        self.pretrained_cnn = PretrainedVGG()
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
        self.swin = swin_t(weights=None if pretrained else 'swinv2_tiny', progress=True)
        
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
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=num_classes)
        
        # Freeze all parameters
        for param in self.vit.parameters():
            param.requires_grad = False
            
        # Replace classification head
        self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)

    def forward(self, x):
        return self.vit(x)


class ResNet(nn.Module):
    """ResNet model for deepfake detection"""
    
    def __init__(self, num_classes=2):
        super(ResNet, self).__init__()
        self.img_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Freeze ResNet parameters
        for param in self.img_model.parameters():
            param.requires_grad = False
        
        # Replace final layer
        num_features = self.img_model.fc.in_features
        self.img_model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(512),
            nn.Linear(512, num_classes)
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
        "resnet": ResNet
    }
    
    @classmethod
    def get_model(cls, model_type: str, **kwargs):
        """Get model instance by type"""
        if model_type not in cls._models:
            raise ValueError(f"Unsupported model type: {model_type}. Supported types: {list(cls._models.keys())}")
        
        model_class = cls._models[model_type]
        
        # Handle special cases for model initialization
        if model_type == "lrcn":
            # LRCN requires specific parameters
            input_size = kwargs.get("input_size", 25088)  # VGG feature size
            hidden_size = kwargs.get("hidden_size", 512)
            num_layers = kwargs.get("num_layers", 2)
            num_classes = kwargs.get("num_classes", 2)
            return model_class(input_size, hidden_size, num_layers, num_classes)
        else:
            # Other models use standard initialization
            num_classes = kwargs.get("num_classes", 2)
            return model_class(num_classes=num_classes)
    
    @classmethod
    def list_models(cls):
        """List all available model types"""
        return list(cls._models.keys())
    
    @classmethod
    def register_model(cls, name: str, model_class):
        """Register a new model type"""
        cls._models[name] = model_class
        logger.info("Model registered", model_name=name, model_class=model_class.__name__)


def create_model(model_type: str, **kwargs):
    """Factory function to create models"""
    return ModelRegistry.get_model(model_type, **kwargs)


def get_default_model():
    """Get the default model type"""
    return settings.DEFAULT_MODEL_TYPE
