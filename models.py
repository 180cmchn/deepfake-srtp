import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch.nn as nn
from torchvision.models import vgg16
from torchvision.models import resnet18, ResNet18_Weights
import timm
from torchvision.models import swin_t, swin_s, swin_b  # 导入Swin Transformer模型

class CustomVGG(nn.Module):
    def __init__(self):
        super(CustomVGG, self).__init__()
        self.img_model = vgg16(weights='IMAGENET1K_V1')
        
        for param in self.img_model.parameters():
            param.requires_grad = False
        
        num_features = self.img_model.classifier[0].in_features
        self.img_model.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(512),
            nn.Linear(512, 2)
        )
        
        for param in self.img_model.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.img_model(x)
        return x

class PretrainedVGG(nn.Module):
    def __init__(self):
        super(PretrainedVGG, self).__init__()
        pretrained_cnn = vgg16(weights='IMAGENET1K_V1')
        self.features = pretrained_cnn.features # [:-9]
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 展开特征为一维
        return x

class LRCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LRCN, self).__init__()
        self.pretrained_cnn = PretrainedVGG()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size, seq_length, channels, height, width = x.size()
        x = x.view(batch_size * seq_length, channels, height, width)  # 重塑输入
        frame_features = self.pretrained_cnn(x)
        frame_features = frame_features.view(batch_size, seq_length, -1)  # 重塑为 (batch_size, seq_length, input_size)

        # 将帧特征通过 LSTM
        lstm_out, _ = self.lstm(frame_features)

        # 取最后一个时间步的输出
        lstm_out = lstm_out[:, -1, :]  # (batch_size, hidden_size)

        # 使用全连接层进行分类
        output = self.fc(lstm_out)

        return output
    

class SwinModel(nn.Module):
    def __init__(self, image_size=224, pretrained=True, num_classes=2):
        super(SwinModel, self).__init__()
        # 根据需要选择合适的Swin Transformer模型，这里以swin_t为例
        self.swin = swin_t(weights=None if pretrained else 'swinv2_tiny', progress=True)
        
        # 冻结预训练模型的所有参数
        for param in self.swin.parameters():
            param.requires_grad = False
        
        # 替换Swin的分类头以匹配任务类别数
        num_features = self.swin.head.in_features
        self.swin.head = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.swin(x)
        return x


class ViTModel(nn.Module):
    def __init__(self, image_size=224, pretrained=True):
        super(ViTModel, self).__init__()
        # 加载预训练的ViT模型
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=2)
        
        # 冻结预训练模型的所有参数
        for param in self.vit.parameters():
            param.requires_grad = False
            
        # 替换ViT的分类头以匹配任务类别数
        self.vit.head = nn.Linear(self.vit.head.in_features, 2)

    def forward(self, x):
        x = self.vit(x)
        return x
    
# import timm

# class ViTModel(nn.Module):
#     def __init__(self):
#         super(ViTModel, self).__init__()
#         # 加载预训练的ViT模型
#         self.vit_model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2)
        
#         # 冻结ViT的参数
#         for param in self.vit_model.parameters():
#             param.requires_grad = False
        
#         # 获取ViT最后一个全连接层的输入特征数
#         num_features = self.vit_model.head.in_features
#         # 替换ViT的全连接层
#         self.vit_model.head = nn.Sequential(
#             nn.Linear(num_features, 512),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.BatchNorm1d(512),
#             nn.Linear(512, 2)
#         )
        
#         # 解冻新全连接层的参数
#         for param in self.vit_model.head.parameters():
#             param.requires_grad = True

#     def forward(self, x):
#         x = self.vit_model(x)
#         return x

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.img_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # 冻结ResNet的参数
        for param in self.img_model.parameters():
            param.requires_grad = False
        
        # 获取ResNet最后一个全连接层的输入特征数
        num_features = self.img_model.fc.in_features
        # 替换ResNet的全连接层
        self.img_model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(512),
            nn.Linear(512, 2)
        )
        
        # 解冻新全连接层的参数
        for param in self.img_model.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.img_model(x)
        return x
