import torch
import torchvision.models as models
from torch import nn
from utils.convlstm import ConvLSTM
from utils.resnet import *

class ConvLSTM_net(nn.Module):
    def __init__(self, num_hiddens, num_classes):
        super(ConvLSTM_net, self).__init__()  # 继承父类的初始化

        self.convlstm = ConvLSTM(input_size=(7, 7), input_dim=64, hidden_dim=num_hiddens, kernel_size=(3, 3),
                                 num_layers=2, batch_first=True, \
                                 bias=True, return_all_layers=False)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # 输出shape为(seq_len*batch_size,num_hiddens)
        self.classifier_convlstm = nn.Sequential(
            nn.Linear(num_hiddens, num_classes, bias=False),  ##输出shape为(seq_len*batch_size,num_classes)
        )

    def forward(self, x):
        conv_lstm_output, _ = self.convlstm(x)  # list,[shape(1,10,64,7,7)]
        conv_lstm_output = conv_lstm_output[0][:, -1, ...].squeeze(dim=1)  # shape(1,64,7,7)
        avgpool = self.avgpool(conv_lstm_output)  # shape(1,64,1,1)
        avgpool = avgpool.view(avgpool.size(0), -1)
        output = self.classifier_convlstm(avgpool)  # shape(1,num_classes)

        return output



class Resnet_ConvLSTM(nn.Module):
    def __init__(self, num_hiddens, num_classes):
        super(Resnet_ConvLSTM, self).__init__()  # 继承父类的初始化

        model = models.resnet50(pretrained=True)
        net = nn.Sequential()
        net.add_module('conv1', model.conv1)
        net.add_module('bn1', model.bn1)
        net.add_module('relu', model.relu)
        net.add_module('maxpool', model.maxpool)
        net.add_module('layer1', model.layer1)
        net.add_module('layer2', model.layer2)
        net.add_module('layer3', model.layer3)
        net.add_module('layer4', model.layer4)
        self.cnn = net  # 输出shape为(batch_size,2048,7,7)

        self.cnn_layer = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=1),
                                       nn.ReLU(inplace=True))

        self.convlstm = ConvLSTM(input_size=(7, 7), input_dim=64, hidden_dim=num_hiddens, kernel_size=(3, 3),
                                 num_layers=2, batch_first=True, \
                                 bias=True, return_all_layers=False)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # 输出shape为(seq_len*batch_size,num_hiddens)
        self.classifier_convlstm = nn.Sequential(
            nn.Linear(num_hiddens, num_classes, bias=False),  ##输出shape为(seq_len*batch_size,num_classes)
        )

    def forward(self, x):
        batch_size, time_steps, channels, height, width = x.size()
        c_in = x.view(batch_size * time_steps, channels, height, width)  # (img_num,3,224,224)
        feature_map = self.cnn(c_in)  # (img_num,2048,7,7)
        feature_map = self.cnn_layer(feature_map)  # (img_num,64,7,7)
        conv_lstm_input = feature_map.unsqueeze(dim=0)  # (1,img_num,64,7,7)
        conv_lstm_output, _ = self.convlstm(conv_lstm_input)  # list,[shape(1,img_num,64,7,7)]
        conv_lstm_output = conv_lstm_output[0][:, -1, ...].squeeze(dim=1)  # shape(1,64,7,7)
        avgpool = self.avgpool(conv_lstm_output)  # shape(1,64,1,1)
        avgpool = avgpool.view(avgpool.size(0), -1)
        output = self.classifier_convlstm(avgpool)  # shape(1,num_classes)

        return output

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out =  self.sigmoid(out)
        return out*x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        map = torch.cat([avg_out, max_out], dim=1)
        map = self.conv1(map)
        map = self.sigmoid(map)
        x = map*x

        return x

class Crashattention(nn.Module) :
    def __init__(self, kernel_size=None,in_planes=None):
        super(Crashattention, self).__init__()
        self.ChannelAttention = ChannelAttention(in_planes=in_planes)
        self.SpatialAttention = SpatialAttention(kernel_size=kernel_size)
        self.conv = nn.Conv2d(4096, 2048,1 )
    def forward(self,x):
        x = self.ChannelAttention(x)
        x1 = self.SpatialAttention(x)
        output = torch.cat((x,x1),1)
        output = self.conv(output)
        return output


class ResNet_CrashAttention(nn.Module):
    def __init__(self, num_classes):
        super( ResNet_CrashAttention, self).__init__()
        crash_attention = Crashattention(7,2048)
        model = models.resnet50(pretrained=True)
        net = nn.Sequential()
        net.add_module('conv1', model.conv1)
        net.add_module('bn1', model.bn1)
        net.add_module('relu', model.relu)
        net.add_module('maxpool', model.maxpool)
        net.add_module('layer1', model.layer1)
        net.add_module('layer2', model.layer2)
        net.add_module('layer3', model.layer3)
        net.add_module('layer4', model.layer4)
        net.add_module('crashattention',crash_attention)
        self.cnn = net  # 输出shape为(batch_size,2048,7,7)

        self.cnn_layer = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=1),
                                       nn.ReLU(inplace=True))
        self.dense = nn.Sequential(
            nn.Linear(in_features=7 * 7 * 64, out_features=512, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)  # (img_num,2048,7,7)
        print(x.shape)
        x = self.cnn_layer(x)  # (img_num,64,7,7)
        x = self.dense(x.view(x.size(0), -1))
        return x

class Resnet_crashattention_ConvLSTM(nn.Module):
    def __init__(self, num_hiddens, num_classes):
        super(Resnet_crashattention_ConvLSTM, self).__init__()  # 继承父类的初始化
        crash_attention = Crashattention(7,2048)
        model = models.resnet50(pretrained=True)
        net = nn.Sequential()
        net.add_module('conv1', model.conv1)
        net.add_module('bn1', model.bn1)
        net.add_module('relu', model.relu)
        net.add_module('maxpool', model.maxpool)
        net.add_module('layer1', model.layer1)
        net.add_module('layer2', model.layer2)
        net.add_module('layer3', model.layer3)
        net.add_module('layer4', model.layer4)
        net.add_module('crashattention',crash_attention)
        self.cnn = net  # 输出shape为(batch_size,2048,7,7)

        self.cnn_layer = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=1),
                                       nn.ReLU(inplace=True))

        self.convlstm = ConvLSTM(input_size=(7, 7), input_dim=64, hidden_dim=num_hiddens, kernel_size=(3, 3),
                                 num_layers=2, batch_first=True, \
                                 bias=True, return_all_layers=False)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # 输出shape为(seq_len*batch_size,num_hiddens)
        self.classifier_convlstm = nn.Sequential(
            nn.Linear(num_hiddens, num_classes, bias=False),  ##输出shape为(seq_len*batch_size,num_classes)
        )

    def forward(self, x):
        batch_size, time_steps, channels, height, width = x.size()
        c_in = x.view(batch_size * time_steps, channels, height, width)  # (img_num,3,224,224)
        feature_map = self.cnn(c_in)  # (img_num,2048,7,7)
        feature_map = self.cnn_layer(feature_map)  # (img_num,64,7,7)
        conv_lstm_input = feature_map.unsqueeze(dim=0)  # (1,img_num,64,7,7)
        conv_lstm_output, _ = self.convlstm(conv_lstm_input)  # list,[shape(1,img_num,64,7,7)]
        conv_lstm_output = conv_lstm_output[0][:, -1, ...].squeeze(dim=1)  # shape(1,64,7,7)
        avgpool = self.avgpool(conv_lstm_output)  # shape(1,64,1,1)
        avgpool = avgpool.view(avgpool.size(0), -1)
        output = self.classifier_convlstm(avgpool)  # shape(1,num_classes)

        return output

if __name__ == '__main__':
    model = Resnet_crashattention_ConvLSTM(64,10)
    model = model.cuda()
    input = torch.randn((1,10, 3, 224, 224)).cuda()
    output = model(input)
    print(output.shape)