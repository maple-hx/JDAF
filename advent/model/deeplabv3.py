# --------------------------------------------------------
# Backbone Structure
#
# Written by Haitao Huang
# --------------------------------------------------------

from turtle import forward
#from mmcv.cnn import ConvModule
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('..')
from domain_adaptation.config import cfg
from utils.func import patch_sml
from utils.func import prob_2_entropy
from model.memory import FeaturesMemoryDomain
affine_par = True

class DeepLabHead(nn.Module):
    def __init__(self, in_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHead, self).__init__()

        self.classifier = nn.Sequential(
            ASPP(2048, aspp_dilate),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        return self.classifier( feature )

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class MemoryModel(nn.Module):
    def __init__(self, num_classes=6,  in_channel=2048,aspp_dilate=[12, 24, 36], memory_source=None, memory_target=None):
        super(MemoryModel, self).__init__()
        # memory bank

        self.bottleneck = nn.Sequential(
            # ASPP(2048, aspp_dilate,out_channels = 512),
            nn.Conv2d(in_channel, 1024, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        self.memory_module = FeaturesMemoryDomain(
            num_classes=num_classes,
            feats_channels=1024,
            transform_channels=512,
            num_feats_per_cls=1,
            out_channels=2048,
            use_context_within_image=True,
            use_hard_aggregate=False,
            memory_sourcedata=memory_source,
            memory_targetdata=memory_target
        )
        self.classifier = DeepLabHead(in_channel, num_classes, aspp_dilate)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        
    def forward(self, x_high, x_main, mode, target, lr, i_iter, img_size):

        preds_source = None
        preds_target = None
        memory_source = None
        memory_target = None

        #迭代次数大于等于3000次的时候，开始更新memory，保证此时的特征是不变特征。
        if i_iter >= 0 or mode == 'TEST':
            # memory_output:torch.Size([4, 256, 32, 32])
            memory_input = self.bottleneck(x_high)
            
            memory_source, memory_target, memory_output = self.memory_module(memory_input, mode, x_main)
            if mode == 'TRAIN':
                memory_source, memory_target, memory_output_tar = self.memory_module(memory_input, 'TARGET', x_main)
                preds_target = self.classifier(memory_output_tar)
                preds_target = F.interpolate(preds_target, size=img_size, mode='bilinear', align_corners=False)
            
            # x:torch.Size([4, 6, 32, 32])
            preds_source = self.classifier(memory_output)
            preds_source = F.interpolate(preds_source, size=img_size, mode='bilinear', align_corners=False)
            #接下来是memory更新部分
            if mode == 'TRAIN':
                # updata memory
                with torch.no_grad():
                    self.memory_module.update_source(
                        features=F.interpolate(memory_input, size=img_size, mode='bilinear', align_corners=False),
                        segmentation=target,
                        learning_rate=lr,
                        strategy='cosine_similarity',
                        ignore_index=255,
                        base_momentum=0.9,
                        base_lr=0.01,
                        adjust_by_learning_rate=True,
                    )
            if mode == 'TARGET':
                    # updata memory
                    target_tar = preds_source.detach().max(dim=1)[1]
                    entropy = prob_2_entropy(F.softmax(preds_source.detach()))
                    entropy = torch.sum(entropy, axis=1)  # 2,512,512
                    # # # #
                    # # #高斯爬升曲线参数
                    # t = i_iter * 10e-5
                    # arfa = (1 - math.exp(-0.05 * t)) / (1 + math.exp(-0.05 * t))
                    arfa = 0.3   #original 0.3
                    target_tar[entropy > arfa] = 255
                    with torch.no_grad():
                        self.memory_module.update_target(
                            features=F.interpolate(memory_input, size=img_size, mode='bilinear', align_corners=False),
                            segmentation=target_tar,
                            learning_rate=lr,
                            strategy='cosine_similarity',
                            ignore_index=255,
                            base_momentum=0.9,
                            base_lr=0.01,
                            adjust_by_learning_rate=True,
                        )

        return memory_source, memory_target, preds_source, preds_target


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        # change
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        padding = dilation
        # change
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class ClassifierModule(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(ClassifierModule, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding,
                          dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        #将conv2d_list列表中的每一层的结果相加求和                       
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out

class ClassifierModule_GAP(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(ClassifierModule_GAP, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding,
                          dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, num_classes, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(num_classes),
                                             nn.ReLU())

    def forward(self, x):
        out = self.conv2d_list[0](x)
        out_gap = self.global_avg_pool(x)
        out_gap = F.upsample(out_gap, size=out.size()[2:], mode='bilinear', align_corners=True)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        out = out + out_gap
        return out


class TAL_1(nn.Module):
    def __init__(self, num_classes, ndf=128):
        super(TAL_1, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(num_classes, 2048, kernel_size=4, stride=2, padding=1),
                                   nn.BatchNorm2d(2048, affine=affine_par),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(num_classes, 2048, kernel_size=4, stride=2, padding=1),
                                   nn.BatchNorm2d(2048, affine=affine_par),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(num_classes, 2048, kernel_size=4, stride=2, padding=1),
                                   nn.BatchNorm2d(2048, affine=affine_par),
                                   nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(2048, ndf, kernel_size=4, stride=2, padding=1),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1))        
        #self.L1 = nn.Sequential(nn.Linear(2048 * 32 * 32 * 3, 1),
        #                        nn.Sigmoid())
        #self.L2 = nn.Sequential(nn.Linear(2048 * 32 * 32 * 3, 1),
        #                        nn.Sigmoid())
        #self.L3 = nn.Sequential(nn.Linear(2048 * 32 * 32 * 3, 1),
        #                        nn.Sigmoid())
        #self.L4 = nn.Sequential(nn.Linear(1024, 1),
        #                        nn.Sigmoid())                        
        #self.L5 = nn.Sequential(nn.Linear(1024, 1),
        #                        nn.Sigmoid())
        #self.L6 = nn.Sequential(nn.Linear(1024, 1),
        #                       nn.Sigmoid())

    def forward(self, x, random_numlists):
        x_s, x_m, x_l = patch_sml(x, random_numlists)
        x_s_0 = self.conv4(self.conv1(x_s[0]))
        x_s_1 = self.conv4(self.conv1(x_s[1]))
        x_s_2 = self.conv4(self.conv1(x_s[2]))
        x_m_0 = self.conv4(self.conv2(x_m[0]))
        x_m_1 = self.conv4(self.conv2(x_m[1]))
        x_m_2 = self.conv4(self.conv2(x_m[2]))
        x_l_0 = self.conv4(self.conv3(x_l[0]))
        x_l_1 = self.conv4(self.conv3(x_l[1]))
        x_l_2 = self.conv4(self.conv3(x_l[2]))
        x1 = torch.cat((x_s_0,x_s_1,x_s_2,x_m_0,x_m_1,x_m_2,x_l_0,x_l_1,x_l_2),1)
        x1 = x1.view(cfg.TRAIN.BATCH_SIZE_SOURCE, -1)

        #x1 = [x_s_0,x_s_1,x_s_2,x_m_0,x_m_1,x_m_2,x_l_0,x_l_1,x_l_2]
        #x1 = [x_s, x_m, x_l]
        return x1

class TAL_2(nn.Module):
    def __init__(self):
        super(TAL_2, self).__init__()    
    def forward(self, x1, x2):
        x = torch.cat((x1, x2), 1)
        return x

class TAL_4(nn.Module):
    def __init__(self, in_channels, num_blocks=4, ndf=128):
        super(TAL_4, self).__init__()
        self.base_channels = 512
        self.out_channels = 2048
        self.num_blocks = num_blocks
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 2 * in_channels, kernel_size=3, stride=1, padding=1, dilation=1),
        #                           nn.BatchNorm2d(2 * in_channels, affine=affine_par),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1),
        #                           nn.BatchNorm2d(self.out_channels, affine=affine_par),
                                   nn.Sigmoid())
        self.conv6 = nn.Sequential(nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1, stride=1, padding=0),
        #                           nn.BatchNorm2d(self.out_channels, affine=affine_par),
                                   nn.Sigmoid())
        
        self.conv3 = nn.Sequential(nn.Conv2d(self.out_channels * 2, ndf, kernel_size=4, stride=2, padding=1),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1))

        self.conv4 = nn.Sequential(nn.Conv2d(self.out_channels * 2, ndf, kernel_size=4, stride=2, padding=1),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   nn.Conv2d(ndf, ndf * 2, kernel_size=9, stride=2, padding=1),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   nn.Conv2d(ndf * 2, ndf * 4, kernel_size=9, stride=2, padding=1))
        
        self.conv5 = nn.Sequential(nn.Conv2d(self.out_channels, ndf, kernel_size=9, stride=4, padding=1),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   nn.Conv2d(ndf, 1, kernel_size=9, stride=4, padding=1))

        self.la_conv1 = nn.Sequential(nn.Conv2d(self.base_channels * 15, self.base_channels * 15 // 32 , 1),
                                      nn.ReLU(inplace=True))
        self.la_conv2 = nn.Sequential(nn.Conv2d(self.base_channels * 15 // 32, num_blocks, 1),
                                      nn.Sigmoid())

        self.add_conv1 = nn.Sequential(nn.Conv2d(self.base_channels * 15, self.base_channels * 4 , kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(inplace=True))
        self.add_conv2 = nn.Sequential(nn.Conv2d(self.base_channels * 4 , 1, kernel_size=3, stride=1, padding=1),
                                   nn.Sigmoid())
        self.reduction_conv = ConvModule(
            self.base_channels * 15,
            self.out_channels,
            1,
            stride=1,
            padding=0,
            conv_cfg=None,
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
            bias=None is None)


    def forward(self, x_l1, x_l2, x_l3):
        x_l4 = self.conv1(x_l3)
        x_l5 = prob_2_entropy(F.softmax(x_l3))
        x = torch.cat((x_l1, x_l2, x_l3, x_l4), 1)
        
        add_feature = self.add_conv2(self.add_conv1(x))
        b, c, h, w = x.shape
        weight = F.adaptive_avg_pool2d(x,(1,1))
        weight = self.la_conv2(self.la_conv1(weight))
        conv_weight = weight.reshape(b, 1, self.num_blocks, 1) * \
                          self.reduction_conv.conv.weight.reshape(1, self.out_channels, self.num_blocks, self.base_channels * 15 // 4)
        conv_weight = conv_weight.reshape(b, self.out_channels, self.base_channels * 15)
        x = x.reshape(b, self.base_channels * 15, h*w)
        x = torch.bmm(conv_weight, x).reshape(b, self.out_channels, h, w)
        x = self.reduction_conv.norm(x)
        x = self.reduction_conv.activate(x)
        x = self.conv2(x)
        #x = (add_feature * x).sqrt()
        x = (add_feature * x)
        x = x.conv6(add_feature * x)
        x = torch.cat((x,x_l5), 1)
        x = self.conv3(x)
        return x

class TAL(nn.Module):
    def __init__(self, in_channels, num_blocks=4, ndf=128):
        super(TAL, self).__init__()
        self.base_channels = 512
        self.out_channels = 2048
        self.num = 15
        self.num_blocks = num_blocks
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 2*in_channels, kernel_size=3, stride=1, padding=8, dilation=8),
                                   nn.BatchNorm2d(2*in_channels, affine=affine_par),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(self.out_channels, affine=affine_par),
                                   nn.Sigmoid())
        
        self.conv3 = nn.Sequential(nn.Conv2d(self.out_channels, ndf, kernel_size=4, stride=2, padding=1),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1))

        self.conv4 = nn.Sequential(nn.Conv2d(self.out_channels, ndf, kernel_size=9, stride=2, padding=1),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   nn.Conv2d(ndf, ndf * 2, kernel_size=9, stride=2, padding=1),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   nn.Conv2d(ndf * 4, 1, kernel_size=4, stride=2, padding=1))
        
        self.conv5 = nn.Sequential(nn.Conv2d(self.out_channels, ndf, kernel_size=9, stride=4, padding=1),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   nn.Conv2d(ndf, 1, kernel_size=9, stride=4, padding=1))

        self.conv6 = nn.Sequential(nn.Conv2d(self.base_channels, self.base_channels, kernel_size=1, stride=1, padding=0),
                                   nn.BatchNorm2d(self.base_channels, affine=affine_par),
                                   nn.ReLU(inplace=True))

        self.conv7 = nn.Sequential(nn.Conv2d(self.base_channels * 2, self.base_channels, kernel_size=1, stride=1, padding=0),
                                   nn.BatchNorm2d(self.base_channels, affine=affine_par),
                                   nn.ReLU(inplace=True))

        self.conv8 = nn.Sequential(nn.Conv2d(self.base_channels * 4, self.base_channels, kernel_size=1, stride=1, padding=0),
                                   nn.BatchNorm2d(self.base_channels, affine=affine_par),
                                   nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(nn.Conv2d(self.base_channels * 8, self.base_channels, kernel_size=1, stride=1, padding=0),
                                   nn.BatchNorm2d(self.base_channels, affine=affine_par),
                                   nn.ReLU(inplace=True))

        self.la_conv1 = nn.Sequential(nn.Conv2d(self.base_channels * self.num, self.base_channels * self.num // 32 , 1),
                                      nn.ReLU(inplace=True))
        self.la_conv2 = nn.Sequential(nn.Conv2d(self.base_channels * self.num // 32, num_blocks, 1),
                                      nn.Sigmoid())

        self.add_conv1 = nn.Sequential(nn.Conv2d(self.base_channels * self.num, self.base_channels * self.num , kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(inplace=True))
        self.add_conv2 = nn.Sequential(nn.Conv2d(self.base_channels * self.num , 1, kernel_size=3, stride=1, padding=1),
                                   nn.Sigmoid())
        self.reduction_conv = ConvModule(
            self.base_channels * self.num,
            self.out_channels,
            1,
            stride=1,
            padding=0,
            conv_cfg=None,
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
            bias=None is None)
        self.layer6 = ClassifierModule_GAP(2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes=19)

    def forward(self, x_l1, x_l2, x_l3):
        x_l4 = self.conv1(x_l3)
        #x = torch.cat((self.conv6(x_l1), self.conv7(x_l2), self.conv8(x_l3), self.conv9(x_l4)), 1)
        x = torch.cat((x_l1, x_l2, x_l3, x_l4), 1)
        add_feature = self.add_conv2(self.add_conv1(x))
        b, c, h, w = x.shape
        weight = F.adaptive_avg_pool2d(x,(1,1))
        weight = self.la_conv2(self.la_conv1(weight))
        conv_weight = weight.reshape(b, 1, self.num_blocks, 1) * \
                          self.reduction_conv.conv.weight.reshape(1, self.out_channels, self.num_blocks, self.base_channels * self.num // 4)
        conv_weight = conv_weight.reshape(b, self.out_channels, self.base_channels * self.num)
        x = x.reshape(b, self.base_channels * self.num, h*w)
        x = torch.bmm(conv_weight, x).reshape(b, self.out_channels, h, w)
        x = self.reduction_conv.norm(x)
        x = self.reduction_conv.activate(x)
        x = self.conv2(x)
        x = (add_feature * x).sqrt()
        x = self.layer6(x)
        #x = self.conv5(x)
        return x

class TAL_3(nn.Module):
    def __init__(self, num_classes, num_blocks=4, out_channel=2048, ndf=128) -> None:
        super(TAL_3, self).__init__()
        self.in_channel = out_channel * num_blocks
        #self.in_channel  = 7680
        self.out_channel = out_channel
        self.num_blocks = num_blocks
        self.conv1 = nn.Sequential(nn.Conv2d(num_classes, out_channel // 2, kernel_size=3, stride=1, padding=1, dilation=1),
                                   nn.BatchNorm2d(out_channel // 2, affine=affine_par),
                                   nn.ReLU(inplace=True))   
        self.conv2 = nn.Sequential(nn.Conv2d(out_channel // 2, out_channel // 2, kernel_size=3, stride=1, padding=2, dilation=2),
                                   nn.BatchNorm2d(out_channel // 2, affine=affine_par),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(out_channel // 2, out_channel, kernel_size=3, stride=1, padding=4, dilation=4),
                                   nn.BatchNorm2d(out_channel, affine=affine_par),
                                   nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(out_channel, out_channel * 2, kernel_size=3, stride=1, padding=8, dilation=8),
                                   nn.BatchNorm2d(out_channel * 2, affine=affine_par),
                                   nn.ReLU(inplace=True))
        self.la_conv1 = nn.Sequential(nn.Conv2d(self.in_channel, self.in_channel // 32 , 1),
                                      nn.ReLU(inplace=True))
        self.la_conv2 = nn.Sequential(nn.Conv2d(self.in_channel // 32, num_blocks, 1),
                                      nn.Sigmoid())
        self.add_conv1 = nn.Sequential(nn.Conv2d(self.in_channel, self.out_channel // 4 , kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(inplace=True))
        self.add_conv2 = nn.Sequential(nn.Conv2d(self.out_channel // 4 , 1, kernel_size=3, stride=1, padding=1),
                                   nn.Sigmoid())

        self.conv5 = nn.Sequential(nn.Conv2d(self.out_channel, out_channel // 4, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(out_channel // 4, affine=affine_par),
                                   nn.Sigmoid())

        self.conv6 = nn.Sequential(nn.Conv2d(out_channel // 4, ndf, kernel_size=4, stride=2, padding=1),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
                                   nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                   nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1))
    
        self.reduction_conv = ConvModule(
            self.in_channel,
            out_channel,
            1,
            stride=1,
            padding=0,
            conv_cfg=None,
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
            bias=None is None)
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        x5 = torch.cat((x1, x2, x3, x4), 1)
        add_feature = self.add_conv2(self.add_conv1(x5))
        b, c, h, w = x5.shape
        weight = F.adaptive_avg_pool2d(x5,(1,1))
        weight = self.la_conv2(self.la_conv1(weight))
        #weight = weight.reshape(b, 1, self.num_blocks, 1)
        conv_weight = weight.reshape(b, 1, self.num_blocks, 1) * \
                          self.reduction_conv.conv.weight.reshape(1, self.out_channel, self.num_blocks, self.out_channel)
        conv_weight = conv_weight.reshape(b, self.out_channel, self.in_channel)

        x5 = x5.reshape(b, self.in_channel, h*w)
        x5 = torch.bmm(conv_weight, x5).reshape(b, self.out_channel, h, w)
        x5 = self.reduction_conv.norm(x5)
        x5 = self.reduction_conv.activate(x5)
        #x10 = torch.cat((x1, x2, x3, x4), 1)
        x10 = self.conv5(x5)
        x10 = (add_feature * x10).sqrt()
        #x8 = x5 * x8
        x10 = self.conv6(x10)
        return x10


class ResNetMulti(nn.Module):
    def __init__(self, block, layers, num_classes, multi_level):
        self.multi_level = multi_level
        self.inplanes = 64
        super(ResNetMulti, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        if self.multi_level:
            self.low_layers = ClassifierModule(512, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        if self.multi_level:
            self.layer5 = ClassifierModule(1024, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        #self.layer6 = ClassifierModule(2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        self.layer6 = ClassifierModule_GAP(2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if (stride != 1
                or self.inplanes != planes * block.expansion
                or dilation == 2
                or dilation == 4):
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        if self.multi_level:
            x_low_level = self.low_layers(x) # produce segmap 1
        else:
            x_low_level = None
        x = self.layer3(x)
        if self.multi_level:
            x1 = self.layer5(x)  # produce segmap 2
        else:
            x1 = None
        x2 = self.layer4(x)
        x2 = self.layer6(x2)  # produce segmap 3
        return x_low_level, x1, x2

    def get_1x_lr_params_no_scale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []

        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        if self.multi_level:
            b.append(self.layer5.parameters())
        b.append(self.layer6.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, lr):
        return [{'params': self.get_1x_lr_params_no_scale(), 'lr': lr},
                {'params': self.get_10x_lr_params(), 'lr': 10 * lr}]

class ResNetSingleEL(nn.Module):
    def __init__(self, block, layers, num_classes, multi_level):
        self.multi_level = multi_level
        self.inplanes = 64
        super(ResNetSingleEL, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        if self.multi_level:
            self.layer5 = ClassifierModule(1024, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        #self.layer6 = ClassifierModule(2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        self.layer6 = ClassifierModule_GAP(2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if (stride != 1
                or self.inplanes != planes * block.expansion
                or dilation == 2
                or dilation == 4):
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.multi_level:
            x1 = self.layer5(x)  # produce segmap 1
        else:
            x1 = None
        x2 = self.layer4(x)
        x3 = self.layer6(x2)  # produce segmap 2
        return x2, x3

    def get_1x_lr_params_no_scale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []

        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        if self.multi_level:
            b.append(self.layer5.parameters())
        b.append(self.layer6.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, lr):
        return [{'params': self.get_1x_lr_params_no_scale(), 'lr': lr},
                {'params': self.get_10x_lr_params(), 'lr': 10 * lr}]

class ResNetSingleFL(nn.Module):
    def __init__(self, block, layers, num_classes, multi_level, aspp_dilate=[12, 24, 36]):
        self.multi_level = multi_level
        self.inplanes = 64
        super(ResNetSingleFL, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        if self.multi_level:
            self.layer5 = ClassifierModule(1024, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        #self.layer6 = ClassifierModule(2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        self.layer6 = ClassifierModule_GAP(2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if (stride != 1
                or self.inplanes != planes * block.expansion
                or dilation == 2
                or dilation == 4):
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)

        x_l1 = self.layer2(x)

        if self.multi_level:
            x1 = x_l1  # produce segmap 1  x1之前的是分割骨干网络backbone，x1之后的是像素级分类头部E
        else:
            x1 = None
        x_l2 = self.layer3(x_l1)

        x_l3 = self.layer4(x_l2)

        x2 = self.layer6(x_l3)  # produce segmap 2
        #x2 = self.decoder_stage1(x_l3)
        return x_l1, x_l2, x_l3, x2

    def get_1x_lr_params_no_scale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []

        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        if self.multi_level:
            b.append(self.layer5.parameters())
        b.append(self.layer6.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, lr):
        return [{'params': self.get_1x_lr_params_no_scale(), 'lr': lr},
                {'params': self.get_10x_lr_params(), 'lr': 10 * lr}]

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

def get_deeplab_v3(num_classes=19, multi_level=True):
    model = ResNetMulti(Bottleneck, [3, 4, 23, 3], num_classes, multi_level)
    return model

def get_deeplab_v3_EL_Adapt(num_classes=19, multi_level=True):
    model = ResNetSingleEL(Bottleneck, [3, 4, 23, 3], num_classes, multi_level)
    return model

def get_deeplab_v3_FL_Adapt(num_classes=19, multi_level=True):
    model = ResNetSingleFL(Bottleneck, [3, 4, 23, 3], num_classes, multi_level)
    return model
