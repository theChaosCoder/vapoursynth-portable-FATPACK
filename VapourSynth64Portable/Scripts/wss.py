# MIT License
#
#Copyright (c) 2018 Fredrik Mellbin
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import vapoursynth as vs

def Xsharpen(clip, strength = 128, threshold = 8):
    return vs.core.std.Expr([clip, clip.std.Maximum(planes=0), clip.std.Minimum(planes=0)], ["y x - x z - min {} < x z - y x - < z y ? {} * x {} * + x ?".format(threshold, strength / 256, (256 - strength) / 256), ""])
    
def UnsharpMask(clip, strength = 64, radius = 3, threshold = 8):
    blurclip = clip.std.BoxBlur(vradius=radius, hradius=radius, planes=0)
    return vs.core.std.Expr([clip, blurclip], ["x y - abs {} > x y - {} * x + x ?".format(threshold, strength/128), ""])
