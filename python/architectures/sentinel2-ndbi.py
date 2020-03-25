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


class SentinelNDBI(torch.nn.Module):

    def __init__(self, band_count):
        super(SentinelNDBI, self).__init__()
        self.no_weights = True

    def forward(self, x):
        self.nir = x[:, 7:8, :, :]
        self.swir = x[:, 11:12, :, :]
        return (self.swir - self.nir)/(self.swir + self.nir + 1e-6)


def make_model(band_count, input_stride=1, class_count=1, divisor=1, pretrained=False):
    ndbi = SentinelNDBI(band_count)
    return ndbi
