import math

import torch.nn as nn

class ImageGenerator(nn.Module): # ImageConvolutionalGenerator
    def __init__(self,
                 scat_dim, # dimension of input vector
                 image_size, #  output image size
                 num_image_channels=3, # the number of output image channels
                 num_final_channels=32, # the number of final layer channels
                 input_size=4): # input tensor size

        super().__init__()

        filter_size = 5
        padding = (filter_size - 1) // 2
        self.level_uppsampling = int(math.log2(image_size/input_size))
        seq_channnels = [num_final_channels * 2 ** p for p in range(self.level_uppsampling, -1, -1)]

        self.input_tensor_size = [seq_channnels[0],input_size]

        self.generator_input = nn.Linear(scat_dim, seq_channnels[0]*input_size*input_size,bias=True)
        self.input_operators = nn.Sequential(
            nn.BatchNorm2d(seq_channnels[0]),
            nn.ReLU()
        )

        modules_upsampling = [
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(seq_channnels[i], seq_channnels[i + 1],
                          filter_size, bias=False,
                          padding=padding, padding_mode='reflect'),
                nn.BatchNorm2d(seq_channnels[i + 1]),
                nn.ReLU()
            )
            for i in range(self.level_uppsampling)]

        self.upsampling_layers = nn.Sequential(*modules_upsampling)

        # final
        self.final_layer = nn.Sequential(
            nn.Conv2d(seq_channnels[self.level_uppsampling], num_image_channels,
                      filter_size, bias=False,
                      padding=padding, padding_mode='reflect'),
            nn.BatchNorm2d(num_image_channels),
            nn.Tanh()
        )

    def forward(self, input):
        g_input = self.generator_input(input)
        g_input = g_input.view(-1, self.input_tensor_size[0],
                               self.input_tensor_size[1], self.input_tensor_size[1])
        output = self.input_operators(g_input)
        output = self.upsampling_layers(output)
        return self.final_layer(output)
