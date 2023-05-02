import math

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

class ImageGenerator(nn.Module): # ImageConvolutionalGenerator
    def __init__(self, # class Self():
                       #    self.level_uppsampling =0
                    # self = Self()
                 scat_dim, # 入力ベクトルの次元 scat_dim =
                 image_size, #  image_size = 64
                 num_final_channels, #  num_final_channels = 32 # num_final_channels
                 input_size=4):# size_input size_image image_size
        # 512 --linear--> size_first_layer * size_first_layer * n_channels_input
        super(ImageGenerator, self).__init__()

        filter_size = 5
        padding = (filter_size - 1) // 2
        # self.depth_net = math.log2(image_size/input_size)
        # self.input_size = input_size
        self.level_uppsampling = int(math.log2(image_size/input_size))
        seq_channnels = [num_final_channels * 2 ** p for p in range(self.level_uppsampling, -1, -1)]

        self.input_tensor_size = [seq_channnels[0],input_size]

        self.generator_input = nn.Linear(scat_dim, seq_channnels[0]*input_size*input_size,bias=False)
        self.input_operators = nn.Sequential(
            nn.BatchNorm2d(seq_channnels[0]), # eps=0.001, momentum=0.9),
            nn.ReLU()#inplace=True) # ^^^
        )

        modules_upsampling = [
            nn.Sequential( # <- これいらんのじゃない？
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), #   align_corners=False警告がうざいからつけた
                # nn.ReflectionPad2d(padding),
                nn.Conv2d(seq_channnels[i], seq_channnels[i + 1],
                          filter_size, bias=False,
                          padding=padding, padding_mode='reflect'),
                nn.BatchNorm2d(seq_channnels[i + 1]),  # , eps=0.001, momentum=0.9),
                nn.ReLU()#inplace=True) #^^^
            )
            for i in range(self.level_uppsampling)]

        self.upsampling_layers = nn.Sequential(*modules_upsampling) # inversion_net #  # uppsampling_layers

        # final
        self.final_layer = nn.Sequential(
            nn.Conv2d(seq_channnels[self.level_uppsampling], 3, # カラー画像限定
                      filter_size, bias=False,
                      padding=padding, padding_mode='reflect'),
            nn.BatchNorm2d(3),
            nn.Tanh()
        )

    def forward(self, input):
        g_input = self.generator_input(input)
        g_input = g_input.view(-1, self.input_tensor_size[0],
                               self.input_tensor_size[1], self.input_tensor_size[1])
        output = self.input_operators(g_input)
        output = self.upsampling_layers(output)
        return self.final_layer(output)


#######
#
# def weights_init(layer):
#     if isinstance(layer, nn.Linear):
#         layer.weight.data.normal_(0.0, 0.02)
#     elif isinstance(layer, nn.Conv2d):
#         layer.weight.data.normal_(0.0, 0.02)
#     elif isinstance(layer, nn.BatchNorm2d):
#         layer.weight.data.normal_(1.0, 0.02)
#         layer.bias.data.fill_(0)

#
# if __name__ == '__main__':
