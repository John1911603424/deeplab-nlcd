# The MIT License (MIT)
# =====================
#
# Copyright © 2019 Azavea
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the “Software”), to deal in the Software without
# restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
#
# The code in this file is under the MIT license except where
# indicted.


class LearnedIndices(torch.nn.Module):

    output_channels = 32-13

    def __init__(self, band_count):
        super(LearnedIndices, self).__init__()
        intermediate_channels1 = 64
        kernel_size = 1
        padding_size = (kernel_size - 1) // 2

        self.conv1 = torch.nn.Conv2d(
            band_count, intermediate_channels1, kernel_size=kernel_size, padding=padding_size, bias=False)
        self.conv_numerator = torch.nn.Conv2d(
            intermediate_channels1, self.output_channels, kernel_size=1, padding=0, bias=False)
        self.conv_denominator = torch.nn.Conv2d(
            intermediate_channels1, self.output_channels, kernel_size=1, padding=0, bias=True)
        self.batch_norm_quotient = torch.nn.BatchNorm2d(
            self.output_channels)

    def forward(self, x):
        x = self.conv1(x)
        numerator = self.conv_numerator(x)
        denomenator = self.conv_denominator(x)
        x = numerator / (denomenator + 1e-7)
        x = self.batch_norm_quotient(x)
        return x


class Resnet18RegressionOnly(torch.nn.Module):

    def __init__(self, band_count, input_stride, pretrained):
        super(Resnet18RegressionOnly, self).__init__()

        self.indices = LearnedIndices(band_count)
        self.backbone = torchvision.models.resnet.resnet18(
            pretrained=pretrained)
        self.backbone.conv1 = torch.nn.Conv2d(
            band_count + self.indices.output_channels, 64, kernel_size=7, stride=input_stride, padding=3, bias=False)
        inplanes = 512
        self.backbone.fc = torch.nn.Linear(
            in_features=512, out_features=1, bias=True)

        self.input_layers = [self.backbone.conv1, self.indices]
        self.output_layers = [self.backbone.fc]

    def forward(self, x):
        x = torch.cat([self.indices(x), x], axis=1)
        regression = self.backbone(x)
        regression = regression.reshape(-1, 1)

        return {'reg': regression}


def make_model(band_count, input_stride=1, class_count=1, divisor=1, pretrained=False):
    deeplab = Resnet18RegressionOnly(
        band_count, input_stride, pretrained)
    return deeplab
