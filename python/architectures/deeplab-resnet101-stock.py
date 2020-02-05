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


def make_model(band_count, input_stride=1, class_count=2, divisor=1, pretrained=False):
    deeplab = torchvision.models.segmentation.deeplabv3_resnet101(
        pretrained=pretrained)
    last_class = deeplab.classifier[4] = torch.nn.Conv2d(
        256, class_count, kernel_size=7, stride=1, dilation=1)
    last_class_aux = deeplab.aux_classifier[4] = torch.nn.Conv2d(
        256, class_count, kernel_size=7, stride=1, dilation=1)
    input_filters = deeplab.backbone.conv1 = torch.nn.Conv2d(
        band_count, 64, kernel_size=7, stride=input_stride, dilation=1, padding=(3, 3), bias=False)
    deeplab.input_layers = [deeplab.backbone.conv1]
    deeplab.output_layers = [deeplab.classifier[4]]
    return deeplab
