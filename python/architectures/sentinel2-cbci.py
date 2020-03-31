# The MIT License (MIT)
# =====================
#
# Copyright © 2020 Azavea
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


class SentinelMBSI(torch.nn.Module):

    def __init__(self, band_count):
        super(SentinelMBSI, self).__init__()
        self.no_weights = True

    def forward(self, x):
        self.red = x[:, 3:4, :, :]
        self.green = x[:, 2:3, :, :]
        return 2*(self.red - self.green)/(self.red + self.green - 2*(1 << 16))


class SentinelOSAVI(torch.nn.Module):

    def __init__(self, band_count):
        super(SentinelOSAVI, self).__init__()
        self.no_weights = True

    def forward(self, x):
        self.nir = x[:, 7:8, :, :]
        self.red = x[:, 3:4, :, :]
        return (self.nir - self.red)/(self.nir + self.red + 0.16*(1 << 16))


class SentinelCBCI(torch.nn.Module):
    def __init__(self, band_count):
        super(SentinelCBCI, self).__init__()
        self.mbsi = SentinelMBSI(band_count)
        self.osavi = SentinelOSAVI(band_count)
        self.no_weights = True

    def forward(self, x):
        a = 0.51
        return (a + 1.0) * self.mbsi(x) - self.osavi(x) + a


def make_model(band_count, input_stride=1, class_count=1, divisor=1, pretrained=False):
    cbci = SentinelCBCI(band_count)
    return cbci
