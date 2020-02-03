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


class DeepLabResnet34(torch.nn.Module):
    def __init__(self, band_count, input_stride, class_count, divisor, pretrained):
        super(DeepLabResnet34, self).__init__()
        resnet34 = torchvision.models.resnet.resnet34(
            pretrained=pretrained)
        self.backbone = torchvision.models._utils.IntermediateLayerGetter(
            resnet34, return_layers={'layer4': 'out', 'layer3': 'aux'})
        inplanes = 512
        self.classifier = torchvision.models.segmentation.deeplabv3.DeepLabHead(
            inplanes, class_count)
        inplanes = 256
        self.aux_classifier = torchvision.models.segmentation.fcn.FCNHead(
            inplanes, class_count)
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
        result['seg'] = x

        y = features['aux']
        y = self.aux_classifier(y)
        y = torch.nn.functional.interpolate(
            y, size=[w, h], mode='bilinear', align_corners=False)
        result['aux'] = y

        return {'seg': x, 'aux': y}


def make_model(band_count, input_stride=1, class_count=2, divisor=1, pretrained=False):
    deeplab = DeepLabResnet34(
        band_count, input_stride, class_count, divisor, pretrained)
    return deeplab
