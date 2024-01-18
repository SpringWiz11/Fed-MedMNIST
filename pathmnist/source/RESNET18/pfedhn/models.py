from collections import OrderedDict

import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm


class CNNHyper(nn.Module):
    def __init__(
            self, n_nodes, embedding_dim, in_channels=3, out_dim=9, n_kernels=64, hidden_dim=100, spec_norm=False, n_hidden=3):
        super().__init__()

        self.in_channels = in_channels
        self.out_dim = out_dim
        self.n_kernels = n_kernels
        self.embeddings = nn.Embedding(num_embeddings=n_nodes, embedding_dim=embedding_dim)

        layers = [
            spectral_norm(nn.Linear(embedding_dim, hidden_dim)) if spec_norm else nn.Linear(embedding_dim, hidden_dim),
        ]
        for _ in range(n_hidden):
            layers.append(nn.ReLU(inplace=True))
            layers.append(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim),
            )

        self.mlp = nn.Sequential(*layers)

        ################################################################
        ################ Initial Layer #################################
        ################################################################

        self.conv_initial_weights = nn.Linear(hidden_dim, self.n_kernels * self.in_channels * 3 * 3)
        self.conv_initial_bias = nn.Linear(hidden_dim, self.n_kernels)

        ################################################################
        ################ Block 1 #######################################
        ################################################################

        self.conv1_1_1_weights = nn.Linear(hidden_dim, self.n_kernels * self.n_kernels * 3 * 3)
        self.conv1_1_1_bias = nn.Linear(hidden_dim, self.n_kernels)
        self.conv1_1_2_weights = nn.Linear(hidden_dim, self.n_kernels * self.n_kernels * 3 * 3)
        self.conv1_1_2_bias = nn.Linear(hidden_dim, self.n_kernels)

        self.conv1_2_1_weights = nn.Linear(hidden_dim, self.n_kernels * self.n_kernels * 3 * 3)
        self.conv1_2_1_bias = nn.Linear(hidden_dim, self.n_kernels)
        self.conv1_2_2_weights = nn.Linear(hidden_dim, self.n_kernels * self.n_kernels * 3 * 3)
        self.conv1_2_2_bias = nn.Linear(hidden_dim, self.n_kernels)

        ################################################################
        ################ Block 2 #######################################
        ################################################################

        self.conv2_1_1_weights = nn.Linear(hidden_dim, 2 * self.n_kernels * self.n_kernels * 3 * 3)
        self.conv2_1_1_bias = nn.Linear(hidden_dim, 2 * self.n_kernels)
        self.conv2_1_2_weights = nn.Linear(hidden_dim, 2 * self.n_kernels * 2 * self.n_kernels * 3 * 3)
        self.conv2_1_2_bias = nn.Linear(hidden_dim, 2 * self.n_kernels)
        self.conv2_1_3_weights = nn.Linear(hidden_dim, 2 * self.n_kernels * self.n_kernels * 1 * 1)
        self.conv2_1_3_bias = nn.Linear(hidden_dim, 2 * self.n_kernels)

        self.conv2_2_1_weights = nn.Linear(hidden_dim, 2 * self.n_kernels * 2 * self.n_kernels * 3 * 3)
        self.conv2_2_1_bias = nn.Linear(hidden_dim, 2 * self.n_kernels)
        self.conv2_2_2_weights = nn.Linear(hidden_dim, 2 * self.n_kernels * 2 * self.n_kernels * 3 * 3)
        self.conv2_2_2_bias = nn.Linear(hidden_dim, 2 * self.n_kernels)


        ################################################################
        ################ Block 3 #######################################
        ################################################################

        self.conv3_1_1_weights = nn.Linear(hidden_dim,2 * 2 * self.n_kernels * 2 * self.n_kernels * 3 * 3)
        self.conv3_1_1_bias = nn.Linear(hidden_dim, 2 * 2 * self.n_kernels)
        self.conv3_1_2_weights = nn.Linear(hidden_dim, 2 * 2 * self.n_kernels * 2 * 2 * self.n_kernels * 3 * 3)
        self.conv3_1_2_bias = nn.Linear(hidden_dim, 2 * 2 * self.n_kernels)
        self.conv3_1_3_weights = nn.Linear(hidden_dim, 2 * 2 * self.n_kernels * 2 * self.n_kernels * 1 * 1)
        self.conv3_1_3_bias = nn.Linear(hidden_dim, 2 * 2 * self.n_kernels)

        self.conv3_2_1_weights = nn.Linear(hidden_dim, 2 * 2 * self.n_kernels * 2 * 2 * self.n_kernels * 3 * 3)
        self.conv3_2_1_bias = nn.Linear(hidden_dim, 2 * 2 * self.n_kernels)
        self.conv3_2_2_weights = nn.Linear(hidden_dim, 2 * 2 * self.n_kernels * 2 * 2 * self.n_kernels * 3 * 3)
        self.conv3_2_2_bias = nn.Linear(hidden_dim, 2 * 2 * self.n_kernels)

        ################################################################
        ################ Block 4 #######################################
        ################################################################

        self.conv4_1_1_weights = nn.Linear(hidden_dim,2 * 2 * 2 * self.n_kernels * 2 * 2 * self.n_kernels * 3 * 3)
        self.conv4_1_1_bias = nn.Linear(hidden_dim,2 * 2 * 2 * self.n_kernels)
        self.conv4_1_2_weights = nn.Linear(hidden_dim,2 * 2 * 2 * self.n_kernels * 2 * 2 * 2 * self.n_kernels * 3 * 3)
        self.conv4_1_2_bias = nn.Linear(hidden_dim,2 * 2 * 2 * self.n_kernels)
        self.conv4_1_3_weights = nn.Linear(hidden_dim,2 * 2 * 2 * self.n_kernels * 2 * 2 * self.n_kernels * 1 * 1)
        self.conv4_1_3_bias = nn.Linear(hidden_dim, 2 * 2 * 2 * self.n_kernels)

        self.conv4_2_1_weights = nn.Linear(hidden_dim,2 * 2 * 2 * self.n_kernels * 2 * 2 * 2 * self.n_kernels * 3 * 3)
        self.conv4_2_1_bias = nn.Linear(hidden_dim, 2 * 2 * 2 * self.n_kernels)
        self.conv4_2_2_weights = nn.Linear(hidden_dim,2 * 2 * 2 * self.n_kernels * 2 * 2 * 2 * self.n_kernels * 3 * 3)
        self.conv4_2_2_bias = nn.Linear(hidden_dim, 2 * 2 * 2 * self.n_kernels)

        self.linear_weight = nn.Linear(hidden_dim, 9 * 2 * 2 * 2 * self.n_kernels)
        self.linear_bias = nn.Linear(hidden_dim, 9)


    def forward(self, idx):
        emd = self.embeddings(idx)
        features = self.mlp(emd)

        weights = OrderedDict({
            ################################################################
            ################ Initial Layer #################################
            ################################################################

            "conv1.weight": self.conv_initial_weights(features).view(self.n_kernels, self.in_channels, 3, 3),
            "conv1.bias": self.conv_initial_bias(features).view(-1),

            ################################################################
            ################ Block_1 #######################################
            ################################################################

            "conv_1_1_1.weight": self.conv1_1_1_weights(features).view(self.n_kernels, self.n_kernels, 3, 3),
            "conv_1_1_1.bias": self.conv1_1_1_bias(features).view(-1),
            "conv_1_1_2.weight": self.conv1_1_2_weights(features).view(self.n_kernels, self.n_kernels, 3, 3),
            "conv_1_1_2.bias": self.conv1_1_2_bias(features).view(-1),

            "conv_1_2_1.weight": self.conv1_2_1_weights(features).view(self.n_kernels, self.n_kernels, 3, 3),
            "conv_1_2_1.bias": self.conv1_2_1_bias(features).view(-1),
            "conv_1_2_2.weight": self.conv1_2_2_weights(features).view(self.n_kernels, self.n_kernels, 3, 3),
            "conv_1_2_2.bias": self.conv1_2_2_bias(features).view(-1),

            ################################################################
            ################ Block_2 #######################################
            ################################################################

            "conv_2_1_1.weight": self.conv2_1_1_weights(features).view(2 * self.n_kernels, self.n_kernels, 3, 3),
            "conv_2_1_1.bias": self.conv2_1_1_bias(features).view(-1),
            "conv_2_1_2.weight": self.conv2_1_2_weights(features).view(2 * self.n_kernels, 2 * self.n_kernels, 3, 3),
            "conv_2_1_2.bias": self.conv2_1_2_bias(features).view(-1),
            "conv_2_1_3.weight": self.conv2_1_3_weights(features).view(2 * self.n_kernels, self.n_kernels, 1, 1),
            "conv_2_1_3.bias": self.conv2_1_3_bias(features).view(-1),

            "conv_2_2_1.weight": self.conv2_2_1_weights(features).view(2 * self.n_kernels,2 * self.n_kernels, 3, 3),
            "conv_2_2_1.bias": self.conv2_2_1_bias(features).view(-1),
            "conv_2_2_2.weight": self.conv2_2_2_weights(features).view(2 * self.n_kernels,2 * self.n_kernels, 3, 3),
            "conv_2_2_2.bias": self.conv2_2_2_bias(features).view(-1),

            ################################################################
            ################ Block_3 #######################################
            ################################################################

            "conv_3_1_1.weight": self.conv3_1_1_weights(features).view(2 * 2 * self.n_kernels, 2 * self.n_kernels, 3, 3),
            "conv_3_1_1.bias": self.conv3_1_1_bias(features).view(-1),
            "conv_3_1_2.weight": self.conv3_1_2_weights(features).view(2 * 2 * self.n_kernels, 2 * 2 * self.n_kernels, 3, 3),
            "conv_3_1_2.bias": self.conv3_1_2_bias(features).view(-1),
            "conv_3_1_3.weight": self.conv3_1_3_weights(features).view(2 * 2 * self.n_kernels, 2 * self.n_kernels, 1, 1),
            "conv_3_1_3.bias": self.conv3_1_3_bias(features).view(-1),

            "conv_3_2_1.weight": self.conv3_2_1_weights(features).view(2 * 2 * self.n_kernels,2 * 2 * self.n_kernels, 3, 3),
            "conv_3_2_1.bias": self.conv3_2_1_bias(features).view(-1),
            "conv_3_2_2.weight": self.conv3_2_2_weights(features).view(2 * 2 * self.n_kernels,2 * 2 * self.n_kernels, 3, 3),
            "conv_3_2_2.bias": self.conv3_2_2_bias(features).view(-1),

            ################################################################
            ################ Block_4 #######################################
            ################################################################

            "conv_4_1_1.weight": self.conv4_1_1_weights(features).view(2 * 2 * 2 * self.n_kernels, 2 * 2 * self.n_kernels, 3, 3),
            "conv_4_1_1.bias": self.conv4_1_1_bias(features).view(-1),
            "conv_4_1_2.weight": self.conv4_1_2_weights(features).view(2 * 2 * 2 * self.n_kernels, 2 * 2 * 2 * self.n_kernels, 3, 3),
            "conv_4_1_2.bias": self.conv4_1_2_bias(features).view(-1),
            "conv_4_1_3.weight": self.conv4_1_3_weights(features).view(2 * 2 * 2 * self.n_kernels, 2 * 2 * self.n_kernels, 1, 1),
            "conv_4_1_3.bias": self.conv4_1_3_bias(features).view(-1),

            "conv_4_2_1.weight": self.conv4_2_1_weights(features).view(2 * 2 * 2 * self.n_kernels,2 * 2 * 2 * self.n_kernels, 3, 3),
            "conv_4_2_1.bias": self.conv4_2_1_bias(features).view(-1),
            "conv_4_2_2.weight": self.conv4_2_2_weights(features).view(2 * 2 * 2 * self.n_kernels,2 * 2 * 2 * self.n_kernels, 3, 3),
            "conv_4_2_2.bias": self.conv4_2_2_bias(features).view(-1),


            "linear.weight": self.linear_weight(features).view(9, 2 * 2 * 2 * self.n_kernels),
            "linear.bias": self.linear_bias(features).view(-1),


        })
        return weights


class CNNTarget(nn.Module):
    def __init__(self, in_channels=3, n_kernels=64, out_dim=9):
        super(CNNTarget, self).__init__()

        self.in_planes = 64


        ################################################################
        ################ Initial Layer #################################
        ################################################################
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1, bias = True)
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
                                bias = True
                            )
        self.bn1_1_1 = nn.BatchNorm2d(64)

        self.conv_1_1_2 = nn.Conv2d(
                                64,
                                64,
                                kernel_size=3,
                                stride = 1,
                                padding = 1,
                                bias = True
                            )
        self.bn1_1_2 = nn.BatchNorm2d(64)

        #stride = 1
        self.conv_1_2_1 = nn.Conv2d(
                                64, 
                                64, 
                                kernel_size=3, 
                                stride=1, 
                                padding=1,
                                bias = True
                            )
        self.bn1_2_1 = nn.BatchNorm2d(64)

        self.conv_1_2_2 = nn.Conv2d(
                                64,
                                64,
                                kernel_size = 3,
                                stride = 1,
                                padding = 1,
                                bias = True,
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
                                bias = True,
                            )
        
        self.bn2_1_1 = nn.BatchNorm2d(128)

        self.conv_2_1_2 = nn.Conv2d(
                                128,
                                128,
                                kernel_size=3,
                                stride = 1,
                                padding = 1,
                                bias = True,
                            )

        self.bn2_1_2 = nn.BatchNorm2d(128)

        # Since stride = 2 and in_planes != self.expansion * planes there will be a shortcut 

        self.conv_2_1_3 = nn.Conv2d(
                                64,
                                1 * 128,
                                kernel_size= 1,
                                stride = 2,
                                bias = True
                            )

        self.bn2_1_3 = nn.BatchNorm2d(128)


        self.conv_2_2_1  = nn.Conv2d(
                                128,
                                128,
                                kernel_size= 3,
                                stride = 1,
                                padding = 1,
                                bias = True,
                            )

        self.bn2_2_1 = nn.BatchNorm2d(128)

        self.conv_2_2_2 = nn.Conv2d(
                                128,
                                128,
                                kernel_size=3,
                                stride = 1,
                                padding = 1,
                                bias = True   
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
                                bias = True,
                            )
        
        self.bn3_1_1 = nn.BatchNorm2d(256)

        self.conv_3_1_2 = nn.Conv2d(
                                256,
                                256,
                                kernel_size=3,
                                stride = 1,
                                padding = 1,
                                bias = True,
                            )

        self.bn3_1_2 = nn.BatchNorm2d(256)

        # Since stride = 2 and in_planes != self.expansion * planes there will be a shortcut 

        self.conv_3_1_3 = nn.Conv2d(
                                128,
                                256,
                                kernel_size= 1,
                                stride = 2,
                                bias = True
                            )

        self.bn3_1_3 = nn.BatchNorm2d(256)


        self.conv_3_2_1  = nn.Conv2d(
                                256,
                                256,
                                kernel_size= 3,
                                stride = 1,
                                padding = 1,
                                bias = True,
                            )

        self.bn3_2_1 = nn.BatchNorm2d(256)

        self.conv_3_2_2 = nn.Conv2d(
                                256,
                                256,
                                kernel_size=3,
                                stride = 1,
                                padding = 1,
                                bias = True   
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
                                bias = True,
                            )
        
        self.bn4_1_1 = nn.BatchNorm2d(512)

        self.conv_4_1_2 = nn.Conv2d(
                                512,
                                512,
                                kernel_size=3,
                                stride = 1,
                                padding = 1,
                                bias = True,
                            )

        self.bn4_1_2 = nn.BatchNorm2d(512)

        # Since stride = 2 and in_planes != self.expansion * planes there will be a shortcut 

        self.conv_4_1_3 = nn.Conv2d(
                                256,
                                512,
                                kernel_size= 1,
                                stride = 2,
                                bias = True
                            )

        self.bn4_1_3 = nn.BatchNorm2d(512)


        self.conv_4_2_1  = nn.Conv2d(
                                512,
                                512,
                                kernel_size= 3,
                                stride = 1,
                                padding = 1,
                                bias = True,
                            )

        self.bn4_2_1 = nn.BatchNorm2d(512)

        self.conv_4_2_2 = nn.Conv2d(
                                512,
                                512,
                                kernel_size=3,
                                stride = 1,
                                padding = 1,
                                bias = True   
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
        ################ Block 4 #######################################
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


# class CNNHyper(nn.Module):
#     def __init__(
#             self, n_nodes, embedding_dim, in_channels=3, out_dim=9, n_kernels=7, hidden_dim=100, spec_norm=False, n_hidden=3):
#         super().__init__()

#         self.in_channels = in_channels
#         self.out_dim = out_dim
#         self.n_kernels = n_kernels
#         self.embeddings = nn.Embedding(num_embeddings=n_nodes, embedding_dim=embedding_dim)

#         layers = [
#             spectral_norm(nn.Linear(embedding_dim, hidden_dim)) if spec_norm else nn.Linear(embedding_dim, hidden_dim),
#         ]
#         for _ in range(n_hidden):
#             layers.append(nn.ReLU(inplace=True))
#             layers.append(
#                 spectral_norm(nn.Linear(hidden_dim, hidden_dim)) if spec_norm else nn.Linear(hidden_dim, hidden_dim),
#             )

#         self.mlp = nn.Sequential(*layers)

#         self.c1_weights = nn.Linear(hidden_dim, self.n_kernels * self.in_channels * 3 * 3)
#         self.c1_bias = nn.Linear(hidden_dim, self.n_kernels)
#         self.c2_weights = nn.Linear(hidden_dim, 2 * self.n_kernels * self.n_kernels * 3 * 3)
#         self.c2_bias = nn.Linear(hidden_dim, 2 * self.n_kernels)
#         self.c3_weights = nn.Linear(hidden_dim, 2 * 2 * self.n_kernels * 2 * self.n_kernels * 3 * 3)
#         self.c3_bias = nn.Linear(hidden_dim, 2 * 2 * self.n_kernels)
#         self.l1_weights = nn.Linear(hidden_dim, 120 * 2 * 2 * self.n_kernels * 3 * 3)
#         self.l1_bias = nn.Linear(hidden_dim, 120)
#         self.l2_weights = nn.Linear(hidden_dim, 84 * 120)
#         self.l2_bias = nn.Linear(hidden_dim, 84)
#         self.l3_weights = nn.Linear(hidden_dim, self.out_dim * 84)
#         self.l3_bias = nn.Linear(hidden_dim, self.out_dim)


#     def forward(self, idx):
#         emd = self.embeddings(idx)
#         features = self.mlp(emd)

#         weights = OrderedDict({
#             "conv1.weight": self.c1_weights(features).view(self.n_kernels, self.in_channels, 3, 3),
#             "conv1.bias": self.c1_bias(features).view(-1),
#             "conv2.weight": self.c2_weights(features).view(2 * self.n_kernels, self.n_kernels, 3, 3),
#             "conv2.bias": self.c2_bias(features).view(-1),
#             "conv3.weight": self.c3_weights(features).view(2 * 2 * self.n_kernels, 2 * self.n_kernels, 3, 3),
#             "conv3.bias": self.c3_bias(features).view(-1),
#             "fc1.weight": self.l1_weights(features).view(120, 2 * 2 * self.n_kernels * 3 * 3),
#             "fc1.bias": self.l1_bias(features).view(-1),
#             "fc2.weight": self.l2_weights(features).view(84, 120),
#             "fc2.bias": self.l2_bias(features).view(-1),
#             "fc3.weight": self.l3_weights(features).view(self.out_dim, 84),
#             "fc3.bias": self.l3_bias(features).view(-1),
#         })
#         return weights


# class CNNTarget(nn.Module):
#     def __init__(self, in_channels=3, n_kernels=7, out_dim=9):
#         super(CNNTarget, self).__init__()

#         self.conv1 = nn.Conv2d(in_channels, n_kernels, 3)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(n_kernels, 2 * n_kernels, 3)
#         self.conv3 = nn.Conv2d(2 * n_kernels, 2 * 2 * n_kernels, 3)
#         self.fc1 = nn.Linear(2 * 2* n_kernels * 3 * 3, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, out_dim)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = F.relu(self.conv3(x))
#         x = x.view(x.shape[0], -1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
