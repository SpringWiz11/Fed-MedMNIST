import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride = 1 ):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size = 3, stride = stride, padding = 1, bias = False
        )

        self.bn1 = nn.BatchNorm2d(planes)

        self.Conv2 = nn.Conv2d(
            planes, planes, kernel_size = 3, stride = 1, padding = 1, bias = False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size = 1,
                    stride = stride,
                    bias = False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.Conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, block, num_blocks, in_channels = 3, num_classes = 9):
        super(ResNet18, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride = 1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride = 2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride = 2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride = 2)
        self.linear = nn.Linear(512 * BasicBlock.expansion, num_classes)
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = list()

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out
    

def resnet18():
    return ResNet18(BasicBlock, [2, 2, 2, 2])


class CNNTarget(nn.Module):
    def __init__(self):
        super(CNNTarget, self).__init__()

        self.in_planes = 64


        ################################################################
        ################ Initial Layer #################################
        ################################################################
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(64)


        ################################################################
        ################ Block 1 #######################################
        ################################################################
        # Layer_1 = stride = 1
        self.conv_1_1_1 = nn.Conv2d(
                                64, 
                                64, 
                                kernel_size=3, 
                                stride=1, 
                                padding=1,
                                bias=False
                            )
        self.bn1_1_1 = nn.BatchNorm2d(64)

        self.conv_1_1_2 = nn.Conv2d(
                                64,
                                64,
                                kernel_size=3,
                                stride = 1,
                                padding = 1,
                                bias = False
                            )
        self.bn1_1_2 = nn.BatchNorm2d(64)

        #stride = 1
        self.conv_1_2_1 = nn.Conv2d(
                                64, 
                                64, 
                                kernel_size=3, 
                                stride=1, 
                                padding=1,
                                bias=False
                            )
        self.bn1_2_1 = nn.BatchNorm2d(64)

        self.conv_1_2_2 = nn.Conv2d(
                                64,
                                64,
                                kernel_size = 3,
                                stride = 1,
                                padding = 1,
                                bias = False,
                            )
        self.bn1_2_2 = nn.BatchNorm2d(64)

        ################################################################
        ################ Block 2 #######################################
        ################################################################

        # since stride = 1 and inplanes = self.expansion * planes there is no skip connection
        # Layer 2, stride = 2
        self.conv_2_1_1 = nn.Conv2d(
                                64,
                                128,
                                kernel_size = 3,
                                stride = 2,
                                padding = 1,
                                bias = False,
                            )
        
        self.bn2_1_1 = nn.BatchNorm2d(128)

        self.conv_2_1_2 = nn.Conv2d(
            128,
            128,
            kernel_size=3,
            stride = 1,
            padding = 1,
            bias = False,
        )

        self.bn2_1_2 = nn.BatchNorm2d(128)

        # Since stride = 2 and in_planes != self.expansion * planes there will be a shortcut 

        self.conv_2_1_3 = nn.Conv2d(
            64,
            1 * 128,
            kernel_size= 1,
            stride = 2,
            bias = False
        )

        self.bn2_1_3 = nn.BatchNorm2d(128)


        self.conv_2_2_1  = nn.Conv2d(
            128,
            128,
            kernel_size= 3,
            stride = 1,
            padding = 1,
            bias = False,
        )

        self.bn2_2_1 = nn.BatchNorm2d(128)

        self.conv_2_2_2 = nn.Conv2d(
            128,
            128,
            kernel_size=3,
            stride = 1,
            padding = 1,
            bias = False   
        )

        self.bn2_2_2 = nn.BatchNorm2d(128)

        ################################################################
        ################ Block 3 #######################################
        ################################################################

        # Layer 3, stride = 2
        self.conv_3_1_1 = nn.Conv2d(
                                128,
                                256,
                                kernel_size = 3,
                                stride = 2,
                                padding = 1,
                                bias = False,
                            )
        
        self.bn3_1_1 = nn.BatchNorm2d(256)

        self.conv_3_1_2 = nn.Conv2d(
            256,
            256,
            kernel_size=3,
            stride = 1,
            padding = 1,
            bias = False,
        )

        self.bn3_1_2 = nn.BatchNorm2d(256)

        # Since stride = 2 and in_planes != self.expansion * planes there will be a shortcut 

        self.conv_3_1_3 = nn.Conv2d(
            128,
            256,
            kernel_size= 1,
            stride = 2,
            bias = False
        )

        self.bn3_1_3 = nn.BatchNorm2d(256)


        self.conv_3_2_1  = nn.Conv2d(
            256,
            256,
            kernel_size= 3,
            stride = 1,
            padding = 1,
            bias = False,
        )

        self.bn3_2_1 = nn.BatchNorm2d(256)

        self.conv_3_2_2 = nn.Conv2d(
            256,
            256,
            kernel_size=3,
            stride = 1,
            padding = 1,
            bias = False   
        )

        self.bn3_2_2 = nn.BatchNorm2d(256)
    

        ################################################################
        ################ Block 4 #######################################
        ################################################################

        # Layer 4, stride = 2
        self.conv_4_1_1 = nn.Conv2d(
                                256,
                                512,
                                kernel_size = 3,
                                stride = 2,
                                padding = 1,
                                bias = False,
                            )
        
        self.bn4_1_1 = nn.BatchNorm2d(512)

        self.conv_4_1_2 = nn.Conv2d(
            512,
            512,
            kernel_size=3,
            stride = 1,
            padding = 1,
            bias = False,
        )

        self.bn4_1_2 = nn.BatchNorm2d(512)

        # Since stride = 2 and in_planes != self.expansion * planes there will be a shortcut 

        self.conv_4_1_3 = nn.Conv2d(
            256,
            512,
            kernel_size= 1,
            stride = 2,
            bias = False
        )

        self.bn4_1_3 = nn.BatchNorm2d(512)


        self.conv_4_2_1  = nn.Conv2d(
            512,
            512,
            kernel_size= 3,
            stride = 1,
            padding = 1,
            bias = False,
        )

        self.bn4_2_1 = nn.BatchNorm2d(512)

        self.conv_4_2_2 = nn.Conv2d(
            512,
            512,
            kernel_size=3,
            stride = 1,
            padding = 1,
            bias = False   
        )

        self.bn4_2_2 = nn.BatchNorm2d(512)


        self.linear = nn.Linear(512 * 1, 9)
        

    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))

        ################################################################
        ################ Block 1 #######################################
        ################################################################


        out_1 = out
        out = F.relu(self.bn1_1_1(self.conv_1_1_1(out)))
        out = self.bn1_1_2(self.conv_1_1_2(out))
        out = out_1 + out
        out = F.relu(out)

        out_2 = out
        out = F.relu(self.bn1_2_1(self.conv_1_2_1(out)))
        out = self.bn1_2_2(self.conv_1_2_2(out))
        out = out_2 + out
        out = F.relu(out)

        ################################################################
        ################ Block 2 #######################################
        ################################################################

        out_3 = out
        out = F.relu(self.bn2_1_1(self.conv_2_1_1(out)))
        out = self.bn2_1_2(self.conv_2_1_2(out))
        out = out + self.bn2_1_3(self.conv_2_1_3(out_3))
        out = F.relu(out)

        out_4 = out
        out = F.relu(self.bn2_2_1(self.conv_2_2_1(out)))
        out = self.bn2_2_2(self.conv_2_2_2(out))
        out = out + out_4
        out = F.relu(out)
        

        ################################################################
        ################ Block 3 #######################################
        ################################################################

        out_5 = out
        out = F.relu(self.bn3_1_1(self.conv_3_1_1(out)))
        out = self.bn3_1_2(self.conv_3_1_2(out))
        out = out + self.bn3_1_3(self.conv_3_1_3(out_5))
        out = F.relu(out)

        out_6 = out
        out = F.relu(self.bn3_2_1(self.conv_3_2_1(out)))
        out = self.bn3_2_2(self.conv_3_2_2(out))
        out = out + out_6
        out = F.relu(out)

        ################################################################
        ################ Block 3 #######################################
        ################################################################

        out_7 = out
        out = F.relu(self.bn4_1_1(self.conv_4_1_1(out)))
        out = self.bn4_1_2(self.conv_4_1_2(out))
        out = out + self.bn4_1_3(self.conv_4_1_3(out_7))
        out = F.relu(out)

        out_8 = out
        out = F.relu(self.bn4_2_1(self.conv_4_2_1(out)))
        out = self.bn4_2_2(self.conv_4_2_2(out))
        out = out + out_8
        out = F.relu(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return out