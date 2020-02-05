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


class Resnet18RegressionOnly(torch.nn.Module):

    def __init__(self, band_count, input_stride, pretrained):
        super(Resnet18RegressionOnly, self).__init__()
        self.backbone = torchvision.models.resnet.resnet18(
            pretrained=pretrained)
        self.backbone.conv1 = torch.nn.Conv2d(
            band_count, 64, kernel_size=7, stride=input_stride, padding=3, bias=False)
        inplanes = 512
        self.backbone.fc = torch.nn.Linear(
            in_features=512, out_features=1, bias=True)

        self.input_layers = [self.backbone.conv1]
        self.output_layers = [self.backbone.fc]

    def forward(self, x):
        [w, h] = x.shape[-2:]

        regression = self.backbone(x)
        regression = regression.reshape(-1, 1)

        return {'reg': regression}


def make_model(band_count, input_stride=1, class_count=1, divisor=1, pretrained=False):
    deeplab = Resnet18RegressionOnly(
        band_count, input_stride, pretrained)
    return deeplab
