import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class DattNet(nn.Module):
    def __init__(self, pic_width, m_patterns, input_shape):
        super(DattNet, self).__init__()
        self.m_patterns = m_patterns
        self.ims = pic_width  # must be even number
        self.input_shape = input_shape
        self.batch_size = input_shape[2]
        self.filter_depth = filter_depth
        self.kernel_size = kernel_size
        self.classes = classes
        self.c_i7 = 768

        # Datt Net
        self.fc_layer_1 = nn.Linear(self.m_patterns, 4096)
        self.fc_layer_2 = nn.Linear(4096, 16384)
        self.diconv_layer_3 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding='same', dilation=2),
            nn.MaxPool2d((2, 2)))
        self.blue_block_4 = BlueBlock(16, 26)
        self.blue_block_5 = BlueBlock(26, 31)
        self.blue_block_6 = BlueBlock(31, 33)
        self.blue_block_7 = BlueBlock(33, 34)
        self.blue_block_8 = BlueBlock(34, 35)
        self.blue_block_9 = BlueBlock(35, 36)
        self.dense_block_10 = DenseBlock(36, 36)
        self.semi_red_block_11 = SemiRedAttBlock(36, 36, 36, 36, 36)
        self.semi_red_block_12 = SemiRedAttBlock(36, 35, 36, 35, 36)
        self.red_block_13 = RedAttBlock(36, 34, 36, 34, 36)
        self.red_block_14 = RedAttBlock(36, 33, 36, 33, 36)
        self.red_block_15 = RedAttBlock(36, 31, 36, 31, 36)
        self.red_block_16 = RedAttBlock(36, 26, 36, 26, 36)
        self.layer_17 = nn.Sequential(nn.Conv2d(in_channels=36, out_channels=36, kernel_size=3, padding='same'),
                                      nn.ReLU(inplace=True),
                                      nn.Upsample(scale_factor=2),
                                      nn.Conv2d(in_channels=36, out_channels=36, kernel_size=3, padding='same'),
                                      nn.ReLU(inplace=True))
        self.layer_18 = nn.Sequential(nn.Conv2d(in_channels=36, out_channels=1, kernel_size=1, padding='same'),
                                      nn.ReLU(inplace=True))
        self.nlt_19 = nn.Sequential(nn.BatchNorm2d(1),
                                    nn.Linear(128, 128))

    def forward(self, x):
        x = self.fc_layer_1(x)
        x = self.fc_layer_2(x)
        x = x.reshape(-1, 1, 128, 128)
        x = self.diconv_layer_3(x)
        x4 = self.blue_block_4(x)
        x5 = self.blue_block_5(x4)
        x6 = self.blue_block_6(x5)
        x7 = self.blue_block_7(x6)
        x8 = self.blue_block_8(x7)
        x9 = self.blue_block_9(x8)
        x = self.dense_block_10(x9)
        x = self.semi_red_block_11(x, x9)
        x = self.semi_red_block_12(x, x8)
        x = self.red_block_13(x, x7)
        x = self.red_block_14(x, x6)
        x = self.red_block_15(x, x5)
        x = self.red_block_16(x, x4)
        x = self.layer_17(x)
        x = self.layer_18(x)
        x = self.nlt_19(x)
        return x

    def first_block(self, x):
        batch_size, num_feats = x.shape
        self.int_fc1 = nn.Linear(num_feats, 32 * 32)
        x = F.relu(self.int_fc1(x))
        x = self.dropout(x)

        x = F.relu(self.int_fc2(x))
        x = self.dropout(x)

        x = F.relu(self.conv_block1(x.view(-1, 1, 2 * self.ims, 2 * self.ims)))
        x = self.dropout(x)

        x = F.relu(self.conv_block2(x))
        x = self.dropout(x)

        return x

    def fork_block(self, x):
        # path 1
        x1 = self.res_block(x)

        # path 2
        x2 = self.maxpool2(x)
        x2 = self.res_block(x2)
        x2 = self.upsample(x2)

        # path 3
        x3 = self.maxpool4(x)
        x3 = self.res_block(x3)
        x3 = self.upsample(x3)
        x3 = self.upsample(x3)

        # path 4
        x4 = self.maxpool8(x)
        x4 = self.res_block(x4)
        x4 = self.upsample(x4)
        x4 = self.upsample(x4)
        x4 = self.upsample(x4)

        concat_x = torch.cat((x1, x2, x3, x4), 1)
        return concat_x

    def res_block(self, x):
        """ 4 blue res block, fit to all paths"""
        for _ in range(4):
            y = F.relu(self.conv_res(x))
            f_x = F.relu(self.conv_res(y))
            x = F.relu(x + f_x)
        return x

    def final_block(self, x):
        x = self.maxpool2(x)

        x = F.relu(self.conv_block3(x))
        x = self.dropout(x)

        x = F.relu(self.conv_block4(x))
        x = self.dropout(x)

        x = F.relu(self.conv_block5(x))
        x = self.dropout(x)

        x = F.relu(self.last_layer(x))
        return x


class DenseBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super(DenseBlock, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.layer = nn.Sequential(nn.BatchNorm2d(c_in),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(in_channels=c_in, out_channels=c_in, kernel_size=5, padding='same', dilation=2))
        self.transition_layers = nn.Sequential(nn.BatchNorm2d(c_in),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(in_channels=c_in, out_channels=c_in, kernel_size=1, padding='same', dilation=2),
                                    nn.Dropout(0.05),
                                    nn.AvgPool2d(2))

    def forward(self, x1):
        x2 = self.layer(x1)

        x3 = self.layer(x1 + x2)

        x4 = self.layer(x1 + x2 + x3)

        x = x1 + x2 + x3 + x4
        return x


class BlueBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super(BlueBlock, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.dense_block = DenseBlock(c_in, c_in)
        self.transition_layers = nn.Sequential(nn.BatchNorm2d(c_in),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=1, dilation=2),
                                    nn.Dropout(0.05),
                                    nn.AvgPool2d(2))

    def forward(self, x):
        x = self.dense_block(x)
        x = self.transition_layers(x)
        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.params = {'F_g': F_g, 'F_l': F_l, 'F_int': F_int}
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class RedAttBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int, c_in, c_out):
        super(RedAttBlock, self).__init__()
        self.params = {'F_g': F_g, 'F_l': F_l, 'F_int': F_int, 'c_in': c_in, 'c_out': c_out}
        self.att_gate = Attention_block(F_g, F_l, F_int)
        self.layers = nn.Sequential(
            nn.Conv2d(c_in, c_in, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(c_in, c_out, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True))
        self.dence_block = DenseBlock(c_out, c_out)

    def forward(self, g, x):
        x = self.att_gate(g, x)
        x = self.layers(x)
        x = self.dence_block(x)
        return x


class SemiRedAttBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int, c_in, c_out):
        super(SemiRedAttBlock, self).__init__()
        self.params = {'F_g': F_g, 'F_l': F_l, 'F_int': F_int, 'c_in': c_in, 'c_out': c_out}
        self.att_gate = Attention_block(F_g, F_l, F_int)
        self.layers = nn.Sequential(
            nn.Conv2d(c_in, c_in, kernel_size=1, padding='same'),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(c_in, c_out, kernel_size=1, padding='same'),
            nn.ReLU(inplace=True))
        self.dence_block = DenseBlock(c_out, c_out)

    def forward(self, g, x):
        x = self.att_gate(g, x)
        x = self.layers(x)
        x = self.dence_block(x)
        return x
