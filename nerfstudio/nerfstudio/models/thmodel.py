# code from Han Xue
import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, input_channels=3*2, n_layers=3, ndf=64):
        super(Discriminator, self).__init__()
        self.layers = nn.ModuleList()

        # First layer
        self.layers.append(self.create_layer(input_channels, ndf, stride=2))
        
        # Intermediate layers
        input_channels=ndf
        for i in range(n_layers):
            out_channels = ndf * min(2**(i+1), 2)
            stride = 1 if i == n_layers - 1 else 2
            self.layers.append(self.create_layer(input_channels, out_channels, stride=stride))
            input_channels=out_channels

        # Output layer
        self.layers.append(self.create_layer(input_channels, 1, stride=1,lrelu=False))
        # self.layers.append(nn.Sigmoid())

    def create_layer(self, in_channels, out_channels, stride,lrelu=True):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1)
        nn.init.normal_(conv2d.weight, mean=0, std=0.02)
        if lrelu:
            return nn.Sequential(conv2d,nn.LeakyReLU(0.2))
        else:
            return conv2d

    def forward(self, input):
        features = []
        for layer in self.layers:
            output = layer(input)
            # features.append(output)
            input = output
        output = torch.sigmoid(input)
        return output#, torch.tensor(features[-2] > 0.1, dtype=torch.float32)
EPS = 1e-12

class DiscriminatorLoss(nn.Module):
    def forward(self, predict_real, predict_fake):
        discrim_loss = torch.mean((-(torch.log(predict_real + EPS) + torch.log(1 - predict_fake + EPS))))
        return discrim_loss

class GeneratorLoss(nn.Module):
    def forward(self, predict_fake, data_color_tar=None, outputs=None,global_step=None):
        # l1_weight = 10.0 * (0.8 ** (global_step//960)).astype(torch.float32) # global_step是迭代次数，l1_weight随迭代逐渐减小
        gen_loss_GAN = -torch.mean(torch.log(predict_fake + EPS))
        # gen_loss_L1 = torch.mean(torch.abs(data_color_tar - outputs))
        # gen_loss = gen_loss_L1 * l1_weight + gen_loss_GAN
        return gen_loss_GAN

def crop(input,crop_size=70): # xh
    offsets = torch.randint(crop_size, (2,), dtype=torch.int32)
    sy = offsets[0]
    sx = offsets[1]
    return input[...,sy:,sx:]


import torch
import torch.nn as nn
import torch.nn.functional as F

# target：目标张量，用于指示输入对是相似对还是非相似对。
# 为0表示相似对，为1表示非相似对。

# margin：边界值（margin），它是一个超参数，用于控制非相似对的相似度得分阈值。
# 如果相似度得分小于边界值，则对比损失函数将惩罚这些非相似对。
# EPS=1e-12
class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    # def forward(self, anchor, positive, negative): #[batch_size, 64]
    #     distance_positive = torch.norm(anchor - positive, dim=1)
    #     distance_negative = torch.norm(anchor - negative, dim=1)
    #     loss = distance_positive.mean() + torch.relu(self.margin-distance_negative).mean()
    #     return loss
    def forward(self, anchor, positive, negative): #[batch_size, 64]
        # import pdb; pdb.set_trace()
        distance_positive = torch.norm(anchor - positive, dim=1)
        distance_negative = torch.norm(anchor - negative, dim=1)
        loss = distance_positive.mean() + (torch.relu(self.margin-distance_negative**(0.5))**2.0).mean()
        return loss


class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = torch.norm(anchor - positive, dim=1)
        distance_negative = torch.norm(anchor - negative, dim=1)
        loss = torch.relu(distance_positive - distance_negative + self.margin).mean()
        return loss
        
        
class NeRFartLoss(nn.Module):
    def __init__(self, margin):
        super(NeRFartLoss, self).__init__()
        self.margin = margin # NeRFart里取0.07。我们应该要改

    def forward(self, anchor, positive, negative):
        distance_positive = torch.sum(anchor*positive)
        distance_negative = torch.sum(anchor*negative)
        distance_positive = torch.exp(distance_positive/self.margin)
        distance_negative = torch.exp(distance_negative/self.margin)
        loss = torch.log((distance_negative/(distance_positive+EPS))+1)
        return loss




if __name__ == '__main__':

    generator_loss = GeneratorLoss()
    discriminator_loss = DiscriminatorLoss()
    discriminator = Discriminator()

    # img_cond: 条件图片，可以是before/after edited，建议和img_tar、img_out视角pose相同
    # img_tar:  novel view gt图片，after edited
    # img_out:  output图片，应该和img_tar视角pose相同
    # shape: [batch_size,3,H,W]
    input_real = torch.cat([img_cond, img_tar - img_cond], dim=1) 
    input_real=crop(input_real)
    predict_real = discriminator(input_real)

    input_fake = torch.cat([img_cond, img_out - img_cond], dim=1)
    input_fake=crop(input_fake)
    predict_fake = discriminator(input_fake)
    # 在训练循环中使用以下代码计算损失
    discrim_loss = discriminator_loss(predict_real, predict_fake)
    gen_loss = generator_loss(predict_fake, img_tar, img_out, global_step)
