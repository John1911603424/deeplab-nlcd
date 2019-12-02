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

if False:
    import torch
    import torchvision


class DeepLabResnet18Binary(torch.nn.Module):
    def __init__(self, band_count, input_stride, divisor, pretrained):
        super(DeepLabResnet18Binary, self).__init__()
        resnet18 = torchvision.models.resnet.resnet18(
            pretrained=pretrained)
        self.backbone = torchvision.models._utils.IntermediateLayerGetter(
            resnet18, return_layers={'layer4': 'out'})
        inplanes = 512
        self.classifier = torchvision.models.segmentation.deeplabv3.DeepLabHead(
            inplanes, 1)
        self.backbone.conv1 = torch.nn.Conv2d(
            band_count, 64, kernel_size=7, stride=input_stride, padding=3, bias=False)

        if input_stride == 1:
            self.factor = 16 // divisor
        else:
            self.factor = 32 // divisor

        self.input_layers = [self.backbone.conv1]
        self.output_layers = [self.classifier[4]]

    def forward(self, x):
        [w, h] = x.shape[-2:]

        features = self.backbone(torch.nn.functional.interpolate(
            x, size=[w*self.factor, h*self.factor], mode='bilinear', align_corners=False))

        result = {}

        x = features['out']
        x = self.classifier(x)
        x = torch.nn.functional.interpolate(
            x, size=[w, h], mode='bilinear', align_corners=False)

        return x


def make_model(band_count, input_stride=1, class_count=1, divisor=1, pretrained=False):
    deeplab = DeepLabResnet18Binary(
        band_count, input_stride, divisor, pretrained)
    return deeplab
