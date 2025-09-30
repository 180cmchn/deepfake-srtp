# 网络结构
import torch
import torch.nn as nn
from torchvision.models import vgg16  # alexnet

class PretrainedVGG(nn.Module):
    def __init__(self):
        super(PretrainedVGG, self).__init__()
        pretrained_cnn = vgg16(weights='IMAGENET1K_V1')
        self.features = nn.Sequential(*list(pretrained_cnn.features.children())[:-7])
        # self.features = pretrained_cnn.features
        for param in self.features.parameters():
            param.requires_grad = False
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(512 * 8 * 8, 2048), # 512 * 8 * 8 / 100352 / 8192
            nn.Dropout(0.5)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
    
# class PretrainedInception(nn.Module):
#     def __init__(self):
#         super(PretrainedInception, self).__init__()
#         pretrained_cnn = inception_v3(pretrained=True)
#         self.features = nn.Sequential(*list(pretrained_cnn.children())[:-12])

class LRCN(nn.Module):
    def __init__(self):
        super(LRCN, self).__init__()
        self.pretrained_cnn = PretrainedVGG()
        self.lstm = nn.GRU(input_size=2048, hidden_size=1024, num_layers=1, batch_first=True, dropout=0) # dropout=0
        self.fc = nn.Linear(1024, 2)

    def forward(self, x):
        batch_size = x.size(0)
        frameFeatures = torch.empty(size=(batch_size, 10, 2048), device=x.device)

        for t in range(0, 20):
            frame = x[:, t, :, :, :]
            frame_feature = self.pretrained_cnn(frame)
            frameFeatures[:, t, :] = frame_feature

        x, _ = self.lstm(frameFeatures)
        x = self.fc(x[:, -1, :])

        # input's dimension: (batch_size, seq_length, hidden_size)
        # output's dimension: (batch_size, seq_length, classNum)
        # x = self.fc(x)
        
        # get frame-wise's mean
        # output's dimension：(batch, class_Num)
        # x = torch.mean(x, dim=1)
        return x

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
