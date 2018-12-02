# Shitty old function, please do not use

from vapoursynth import core, YUV, GRAY  # You need VapourSynth R37 or newer
import adjust  # https://github.com/dubhater/vapoursynth-adjust
import math


def MfTurd(src, twidth=None, theight=None, ssw=4, ssh=4, xstren=255, xthresh=255, sharpen=True,
           strength=255, wdepth=16, wblur=1, wthresh=0.5, drange=64, dboost=1.0, dlimit=30,
           doutput=None, show=False, scolor=[255,0,255], legacy=False):
    """Original avs fucntion by mf"""

    def scale8(x, newmax):
        return x * newmax // 0xFF

    def Xsharpen(clip, strength=128, threshold=8):
        """Ported by Myrsloik"""
        threshold = scale8(threshold, maxvalue)
        return core.std.Expr([clip, clip.std.Maximum(planes=0), clip.std.Minimum(planes=0)],
                             ['y x - x z - min {} < x z - y x - < z y ? {} * x {} * + x ?'
                              .format(threshold, strength / 256, (256 - strength) / 256), ''])

    # old - R37
	#def UnsharpMask(clip, strength=64, radius=3, threshold=8):
    #    """Ported by Myrsloik"""
    #    threshold = scale8(threshold, maxvalue)
    #    blurclip = clip.std.Convolution([1] * (radius * 2 + 1), planes=0, mode='v')
    #    blurclip = blurclip.std.Convolution([1] * (radius * 2 + 1), planes=0, mode='h')
    #    return core.std.Expr([clip, blurclip], ['x y - abs {} > x y - {} * x + x ?'
    #                                            .format(threshold, strength / 128), ''])
 
	
	# Need R38
    def UnsharpMask(clip, strength = 64, radius = 3, threshold = 8):
        blurclip = clip.std.BoxBlur(vradius=radius, hradius=radius, planes=0)
        return core.std.Expr([clip, blurclip], ["x y - abs {} > x y - {} * x + x ?".format(threshold, strength/128), ""])

    def CartoonEdges(clip, low=0, high=255):
        """Should behave like mt_edge(mode="cartoon")"""
        low = scale8(low, maxvalue)
        high = scale8(high, maxvalue)
        edges = core.std.Convolution(clip, matrix=[0,-2,1,0,1,0,0,0,0], saturate=True)
        return core.std.Expr(edges, ['x {high} >= {maxvalue} x {low} <= 0 x ? ?'
                                     .format(low=low, high=high, maxvalue=maxvalue), ''])

    def RobertsEdges(clip, low=0, high=255):
        """Should behave like mt_edge(mode="roberts")"""
        low = scale8(low, maxvalue)
        high = scale8(high, maxvalue)
        edges = core.std.Convolution(clip, matrix=[0,0,0,0,2,-1,0,-1,0], divisor=2, saturate=False)
        return core.std.Expr(edges, ['x {high} >= {maxvalue} x {low} <= 0 x ? ?'
                                     .format(low=low, high=high, maxvalue=maxvalue), ''])

    def Levels(clip, input_low, gamma, input_high, output_low, output_high, coring=True):
        """Equivalent to AviSynth's Levels
        Stolen from havsfunc"""
        gamma = 1 / gamma
        divisor = input_high - input_low + (input_high == input_low)

        tvLow = scale8(16, maxvalue)
        tvHigh = [scale8(235, maxvalue), scale8(240, maxvalue)]
        scaleUp = maxvalue / scale8(219, maxvalue)
        scaleDown = scale8(219, maxvalue) / maxvalue
        
        def get_lut1(x):
            p = ((x - tvLow) * scaleUp - input_low) / divisor if coring else (x - input_low) / divisor
            p = min(max(p, 0), 1) ** gamma * (output_high - output_low) + output_low
            return min(max(math.floor(p * scaleDown + tvLow + 0.5), tvLow), tvHigh[0]) if coring else min(max(math.floor(p + 0.5), 0), maxvalue)
        def get_lut2(x):
            q = math.floor((x - neutralvalue) * (output_high - output_low) / divisor + neutralvalue + 0.5)
            return min(max(q, tvLow), tvHigh[1]) if coring else min(max(q, 0), maxvalue)
        
        last = core.std.Lut(clip, planes=[0], function=get_lut1)
        if clip.format.color_family == GRAY:
            return last
        else:
            return core.std.Lut(last, planes=[1, 2], function=get_lut2)


    if twidth is None:
        twidth = src.width
    if theight is None:
        theight = src.height
    if show:
        sharpen = False
    ssw = twidth * ssw
    ssh = theight * ssh
    valuerange = (1 << src.format.bits_per_sample)
    maxvalue = valuerange - 1
    neutralvalue = round(maxvalue / 2)
    strength = scale8(strength, maxvalue)
    drange = scale8(drange, maxvalue)
    dlimit = scale8(dlimit, maxvalue)
    scolor = [scale8(x, maxvalue) for x in scolor]

    sharp = UnsharpMask(src, 300, 4, 0)
    sharp = core.std.MaskedMerge(sharp, src, src)
    sharp = core.std.MaskedMerge(sharp, src, src)

    cartoonedges = CartoonEdges(src, 3, 255)
    cartoonedges = Levels(Levels(adjust.Tweak(cartoonedges,bright=drange),
                   scale8(60, maxvalue), dboost, maxvalue, 0, maxvalue),
                   0, dboost, dlimit, maxvalue, 0) \
                   .std.Inflate().std.Deflate().std.Deflate().std.Deflate()

    robertsedges = RobertsEdges(src, 3, 255)
    robertsedges = Levels(Levels(adjust.Tweak(robertsedges,bright=drange),
                   scale8(60, maxvalue), dboost, maxvalue, 0, maxvalue),
                   0, dboost, dlimit, maxvalue, 0).std.Inflate()

    detailmaskpre = Levels(core.std.Expr([cartoonedges,robertsedges], ['x y * x y * * {} /'.format(valuerange ** 3),''])
                    .std.Convolution(matrix=[1,2,1,2,4,2,1,2,1]), 0, 1.0, scale8(190, maxvalue), 0, maxvalue)

    if legacy:
        # LOL
        detailmask = sharp.std.Deflate()
    else:
        detailmask = detailmaskpre.std.Invert().std.Inflate().std.Invert()
    white = src.std.Expr(['{}'.format(maxvalue),'{}'.format(neutralvalue)])
    linemask = core.std.MaskedMerge(white, detailmask, src.std.Invert(), planes=0).std.Invert()
    if strength != maxvalue:
        linemask = Levels(linemask, 0, 1.0, maxvalue, 0, strength)
    color = src.std.BlankClip(color=scolor)
    if show:
        sharp = color
    dark = core.std.MaskedMerge(src, sharp, linemask, planes=0)
    darkmerged = core.std.ShufflePlanes([dark,src], [0,1,2], YUV)
    finaldark = dark if show else darkmerged
    semifinal = src.resize.Bicubic(twidth, theight, filter_param_a=0, filter_param_b=0.75)
    final = dark.resize.Bicubic(ssw, ssh, filter_param_a=0, filter_param_b=0.75)
    final = Xsharpen(final, xstren, xthresh).resize.Bicubic(twidth, theight, filter_param_a=0, filter_param_b=0.75)
    final = core.std.ShufflePlanes([final,semifinal], [0,1,2], YUV) \
            .warp.AWarpSharp2(depth=wdepth // 2, blur=wblur, thresh=round(wthresh * 255))
    return final if sharpen else finaldark
