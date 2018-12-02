'''
Functions:
    LDMerge
    Compare (2)
    ExInpand
    InDeflate
    MultiRemoveGrain
    GradFun3
    AnimeMask (2)
    PolygonExInpand
    Luma
    ediaa
    nnedi3aa
    maa
    SharpAAMcmod
    TEdge
    Sort
    Soothe_mod
    TemporalSoften
    FixTelecinedFades
    TCannyHelper
    MergeChroma
    firniture
    BoxFilter
    SmoothGrad
    DeFilter
    scale
    ColorBarsHD
    SeeSaw
    abcxyz
    Sharpen
    Blur
    BlindDeHalo3
    dfttestMC
    TurnLeft
    TurnRight
    BalanceBorders
    DisplayHistogram
    GuidedFilter (Color)
    GMSD
    SSIM
    SSIM_downsample
    LocalStatisticsMatching
    LocalStatistics
    TextSub16
    TMinBlur
    mdering
    BMAFilter
    LLSURE
    YAHRmod
    RandomInterleave
    super_resolution
'''

import functools
import math
import vapoursynth as vs
from vapoursynth import core
import havsfunc as haf
import mvsfunc as mvf

def LDMerge(flt_h, flt_v, src, mrad=0, show=0, planes=None, convknl=1, conv_div=None, calc_mode=0, power=1.0):
    """Merges two filtered clips based on the gradient direction map from a source clip.

    Args:
        flt_h, flt_v: Two filtered clips.

        src: Source clip. Must be the same format as the filtered clips.

        mrad: (int) Expanding of gradient direction map. Default is 0.

        show: (bint) Whether to output gradient direction map. Default is False.

        planes: (int []) Whether to process the corresponding plane. By default, every plane will be processed.
            The unprocessed planes will be copied from the first clip, "flt_h".

        convknl: (0 or 1) Convolution kernel used to generate gradient direction map. 
            0: Seconde order center difference in one direction and average in perpendicular direction
            1: First order center difference in one direction and weighted average in perpendicular direction.
            Default is 1.

        conv_div: (int) Divisor in convolution filter. Default is the max value in convolution kernel.

        calc_mode: (0 or 1) Method used to calculate the gradient direction map. Default is 0.

        power: (float) Power coefficient in "calc_mode=0".

    Example:
        # Fast anti-aliasing
        horizontal = core.std.Convolution(clip, matrix=[1, 4, 0, 4, 1], planes=[0], mode='h')
        vertical = core.std.Convolution(clip, matrix=[1, 4, 0, 4, 1], planes=[0], mode='v')
        blur_src = core.tcanny.TCanny(clip, mode=-1, planes=[0]) # Eliminate noise
        antialiasing = muf.LDMerge(horizontal, vertical, blur_src, mrad=1, planes=[0])

    """

    funcName = 'LDMerge'

    if not isinstance(src, vs.VideoNode):
        raise TypeError(funcName + ': \"src\" must be a clip!')

    if not isinstance(flt_h, vs.VideoNode):
        raise TypeError(funcName + ': \"flt_h\" must be a clip!')
    if src.format.id != flt_h.format.id:
        raise TypeError(funcName + ': \"flt_h\" must be of the same format as \"src\"!')
    if src.width != flt_h.width or src.height != flt_h.height:
        raise TypeError(funcName + ': \"flt_h\" must be of the same size as \"src\"!')

    if not isinstance(flt_v, vs.VideoNode):
        raise TypeError(funcName + ': \"flt_v\" must be a clip!')
    if src.format.id != flt_v.format.id:
        raise TypeError(funcName + ': \"flt_v\" must be of the same format as \"src\"!')
    if src.width != flt_v.width or src.height != flt_v.height:
        raise TypeError(funcName + ': \"flt_v\" must be of the same size as \"src\"!')

    if not isinstance(mrad, int):
        raise TypeError(funcName + '\"mrad\" must be an int!')

    if not isinstance(show, int):
        raise TypeError(funcName + '\"show\" must be an int!')
    if show not in list(range(0, 4)):
        raise ValueError(funcName + '\"show\" must be in [0, 1, 2, 3]!')

    if planes is None:
        planes = list(range(flt_h.format.num_planes))
    elif isinstance(planes, int):
        planes = [planes]

    bits = flt_h.format.bits_per_sample

    if convknl == 0:
        convknl_h = [-1, -1, -1, 2, 2, 2, -1, -1, -1]
        convknl_v = [-1, 2, -1, -1, 2, -1, -1, 2, -1]
    else: # convknl == 1
        convknl_h = [-17, -61, -17, 0, 0, 0, 17, 61, 17]
        convknl_v = [-17, 0, 17, -61, 0, 61, -17, 0, 17]

    if conv_div is None:
        conv_div = max(convknl_h)

    hmap = core.std.Convolution(src, matrix=convknl_h, saturate=False, planes=planes, divisor=conv_div)
    vmap = core.std.Convolution(src, matrix=convknl_v, saturate=False, planes=planes, divisor=conv_div)

    if mrad > 0:
        hmap = haf.mt_expand_multi(hmap, sw=0, sh=mrad, planes=planes)
        vmap = haf.mt_expand_multi(vmap, sw=mrad, sh=0, planes=planes)
    elif mrad < 0:
        hmap = haf.mt_inpand_multi(hmap, sw=0, sh=-mrad, planes=planes)
        vmap = haf.mt_inpand_multi(vmap, sw=-mrad, sh=0, planes=planes)

    if calc_mode == 0:
        ldexpr = '{peak} 1 x 0.0001 + y 0.0001 + / {power} pow + /'.format(peak=(1 << bits) - 1, power=power)
    else:
        ldexpr = 'y 0.0001 + x 0.0001 + dup * y 0.0001 + dup * + sqrt / {peak} *'.format(peak=(1 << bits) - 1)
    ldmap = core.std.Expr([hmap, vmap], [(ldexpr if i in planes else '') for i in range(src.format.num_planes)])

    if show == 0:
        return core.std.MaskedMerge(flt_h, flt_v, ldmap, planes=planes)
    elif show == 1:
        return ldmap
    elif show == 2:
        return hmap
    elif show == 3:
        return vmap


def Compare(src, flt, power=1.5, chroma=False, mode=2):
    """Visualizes the difference between the source clip and filtered clip.

    Args:
        src: Source clip.

        flt: Filtered clip.

        power: (float) The variable in the processing function which controls the "strength" to increase difference. Default is 1.5.

        chroma: (bint) Whether to process chroma. Default is False.

        mode: (1 or 2) Different processing function. 1: non-linear; 2: linear.

    """

    funcName = 'Compare'

    if not isinstance(src, vs.VideoNode):
        raise TypeError(funcName + ': \"src\" must be a clip!')
    if src.format.color_family not in [vs.GRAY, vs.YUV, vs.YCOCG]:
        raise TypeError(funcName + ': \"src\" must be a YUV clip!')
    if not isinstance(flt, vs.VideoNode):
        raise TypeError(funcName + ': \"flt\" must be a clip!')
    if mode not in [1, 2]:
        raise TypeError(funcName + ': \"mode\" must be in [1, 2]!')

    Compare2(src, flt, props_list=['width', 'height', 'format.name'])

    isGray = src.format.color_family == vs.GRAY
    bits = src.format.bits_per_sample
    sample = src.format.sample_type

    expr = {}
    expr[1] = 'y x - abs 1 + {power} pow 1 -'.format(power=power)
    expr[2] = 'y x - {scale} * {neutral} +'.format(scale=32768 / (65536 ** (1 / power) - 1), neutral=32768)

    chroma = chroma or isGray

    if bits != 16:
        src = mvf.Depth(src, 16, sample=vs.INTEGER)
        flt = mvf.Depth(flt, 16, sample=vs.INTEGER)
        diff = core.std.Expr([src, flt], [expr[mode]] if chroma else [expr[mode], '{neutral}'.format(neutral=32768)])
        diff = mvf.Depth(diff, depth=bits, sample=sample, fulls=True, fulld=True, dither="none", ampo=0, ampn=0)
    else:
        diff = core.std.Expr([src, flt], [expr[mode]] if chroma else [expr[mode], '{neutral}'.format(neutral=32768)])

    return diff


def Compare2(clip1, clip2, props_list=None):
    """Compares the formats of two clips.

    TypeError will be raised when one of the format of two clips are not identical.
    Otherwise, None is returned.

    Args:
        clip1, clip2: Input.

        props_list: (list) A list containing the format to be compared. If it is none, all the formats will be compared.
            Default is None.

    """

    funcName = 'Compare2'

    if not isinstance(clip1, vs.VideoNode):
        raise TypeError(funcName + ': \"clip1\" must be a clip!')

    if not isinstance(clip2, vs.VideoNode):
        raise TypeError(funcName + ': \"clip2\" must be a clip!')

    if props_list is None:
        props_list = ['width', 'height', 'num_frames', 'fps', 'format.name']

    info = ''

    for prop in props_list:
        clip1_prop = eval('clip1.{prop}'.format(prop=prop))
        clip2_prop = eval('clip2.{prop}'.format(prop=prop))

        if clip1_prop != clip2_prop:
            info += '{prop}: {clip1_prop} != {clip2_prop}\n'.format(prop=prop, clip1_prop=clip1_prop, clip2_prop=clip2_prop)

    if info != '':
        info = '\n\n{}'.format(info)

        raise TypeError(info)

    return


def ExInpand(input, mrad=0, mode='rectangle', planes=None):
    """A filter to simplify the calls of std.Maximum()/std.Minimum() and their concatenation.

    Args:
        input: Source clip.

        mrad: (int []) How many times to use std.Maximum()/std.Minimum(). Default is 0.
            Positive value indicates to use std.Maximum().
            Negative value indicates to use std.Minimum().
            Values can be put into a list, or a list of lists.

            Example:
                mrad=[2, -1] is equvalant to clip.std.Maximum().std.Maximum().std.Minimum()
                mrad=[[2, 1], [2, -1]] is equivalant to
                    haf.mt_expand_multi(clip, sw=2, sh=1).std.Maximum().std.Maximum().std.Minimum()

        mode: (0:"rectangle", 1:"losange" or 2:"ellipse", int or string). Default is "rectangle"
            The shape of the kernel.

        planes: (int []) Whether to process the corresponding plane. By default, every plane will be processed.
            The unprocessed planes will be copied from "input".

    """

    funcName = 'ExInpand'

    if not isinstance(input, vs.VideoNode):
        raise TypeError(funcName + ': \"input\" must be a clip!')

    if planes is None:
        planes = list(range(input.format.num_planes))
    elif isinstance(planes, int):
        planes = [planes]

    if isinstance(mrad, int):
        mrad = [mrad]
    if isinstance(mode, (str, int)):
        mode = [mode]

    if not isinstance(mode, list):
        raise TypeError(funcName + ': \"mode\" must be an int, a string, a list of ints, a list of strings or a list of mixing ints and strings!')

    # internel function
    def ExInpand_process(input, mode=None, planes=None, mrad=None):
        if isinstance(mode, int):
            mode = ['rectangle', 'losange', 'ellipse'][mode]
        if isinstance(mode, str):
            mode = mode.lower()
            if mode not in ['rectangle', 'losange', 'ellipse']:
                raise ValueError(funcName + ': \"mode\" must be an int in [0, 2] or a specific string in [\"rectangle\", \"losange\", \"ellipse\"]!')
        else:
            raise TypeError(funcName + ': \"mode\" must be an int in [0, 2] or a specific string in [\"rectangle\", \"losange\", \"ellipse\"]!')

        if isinstance(mrad, int):
            sw = sh = mrad
        else:
            sw, sh = mrad

        if sw * sh < 0:
            raise TypeError(funcName + ': \"mrad\" at a time must be both positive or negative!')

        if sw > 0 or sh > 0:
            return haf.mt_expand_multi(input, mode=mode, planes=planes, sw=sw, sh=sh)
        else:
            return haf.mt_inpand_multi(input, mode=mode, planes=planes, sw=-sw, sh=-sh)

    # process
    if isinstance(mrad, list):
        if len(mode) < len(mrad):
            mode_length = len(mode)
            for i in range(mode_length, len(mrad)):
                mode.append(mode[mode_length - 1])

        for i in range(len(mrad)):
            if isinstance(mrad[i], list):
                if len(mrad[i]) != 1 and len(mrad[i]) != 2:
                    raise TypeError(funcName + ': \"mrad\" must be an int, a list of ints or a list of a list of two ints!')
                for n in mrad[i]:
                    if not isinstance(n, int):
                        raise TypeError(funcName + ': \"mrad\" must be an int, a list of ints or a list of a list of two ints!')
                if len(mrad[i]) == 1:
                    mrad[i].append(mrad[i][0])
            elif not isinstance(mrad[i], int):
                raise TypeError(funcName + ': \"mrad\" must be an int, a list of ints or a list of a list of two ints!')
            clip = ExInpand_process(input, mode=mode[i], planes=planes, mrad=mrad[i])
    else:
        raise TypeError(funcName + ': \"mrad\" must be an int, a list of ints or a list of a list of two ints!')

    return clip


def InDeflate(input, msmooth=0, planes=None):
    """A filter to simplify the calls of std.Inflate()/std.Deflate() and their concatenation.

    Args:
        input: Source clip.

        msmooth: (int []) How many times to use std.Inflate()/std.Deflate(). Default is 0.
            The behaviour is the same as "mode" in ExInpand().

        planes: (int []) Whether to process the corresponding plane. By default, every plane will be processed.
            The unprocessed planes will be copied from "input".

    """

    funcName = 'InDeFlate'

    if not isinstance(input, vs.VideoNode):
        raise TypeError(funcName + ': \"input\" must be a clip!')

    if planes is None:
        planes = list(range(input.format.num_planes))
    elif isinstance(planes, int):
        planes = [planes]

    if isinstance(msmooth, int):
        msmoooth = [msmooth]

    # internel function
    def InDeflate_process(input, planes=None, radius=None):
        if radius > 0:
            return haf.mt_inflate_multi(input, planes=planes, radius=radius)
        else:
            return haf.mt_deflate_multi(input, planes=planes, radius=-radius)

    # process
    if isinstance(msmooth, list):
        for m in msmooth:
            if not isinstance(m, int):
                raise TypeError(funcName + ': \"msmooth\" must be an int or a list of ints!')
            else:
                clip = InDeflate_process(input, planes=planes, radius=m)
    else:
        raise TypeError(funcName + ': \"msmooth\" must be an int or a list of ints!')

    return clip


def MultiRemoveGrain(input, mode=0, loop=1):
    """A filter to simplify the calls of rgvs.RemoveGrain().

    Args:
        input: Source clip.

        mode: (int []) "mode" in rgvs.RemoveGrain().
            Can be a list, the logic is similar to "mode" in ExInpand().

            Example: mode=[4, 11, 11] is equivalant to clip.rgvs.RemoveGrain(4).rgvs.RemoveGrain(11).rgvs.RemoveGrain(11)
            Default is 0.

        loop: (int) How many times the "mode" loops.

    """

    funcName = 'MultiRemoveGrain'

    if not isinstance(input, vs.VideoNode):
        raise TypeError(funcName + ': \"input\" must be a clip!')

    if isinstance(mode, int):
        mode = [mode]

    if not isinstance(loop, int):
        raise TypeError(funcName + ': \"loop\" must be an int!')
    if loop < 0:
        raise ValueError(funcName + ': \"loop\" must be positive value!')

    if isinstance(mode, list):
        for i in range(loop):
            for m in mode:
                clip = core.rgvs.RemoveGrain(input, mode=m)
    else:
        raise TypeError(funcName + ': \"mode\" must be an int, a list of ints or a list of a list of ints!')

    return clip


def GradFun3(src, thr=None, radius=None, elast=None, mask=None, mode=None, ampo=None, ampn=None,
             pat=None, dyn=None, lsb=None, staticnoise=None, smode=None, thr_det=None,
             debug=None, thrc=None, radiusc=None, elastc=None, planes=None, ref=None):
    """GradFun3 by Firesledge v0.1.1

    Port by Muonium  2016/6/18
    Port from Dither_tools v1.27.2 (http://avisynth.nl/index.php/Dither_tools)
    Internal precision is always 16 bits.

    Read the document of Avisynth version for more details.

    Notes:
        1. In this function I try to keep the original look of GradFun3 in Avisynth.
            It should be better to use Frechdachs's GradFun3 in his fvsfunc.py 
            (https://github.com/Irrational-Encoding-Wizardry/fvsfunc) which is more novel and powerful.

    Removed parameters list:
        "dthr", "wmin", "thr_edg", "subspl", "lsb_in"

    Parameters "y", "u", "v" are changed into "planes"

    """

    funcName = 'GradFun3'

    if not isinstance(src, vs.VideoNode):
        raise TypeError(funcName + ': \"src\" must be a clip!')
    if src.format.color_family not in [vs.YUV, vs.GRAY, vs.YCOCG]:
        raise TypeError(funcName + ': \"src\" must be YUV, GRAY or YCOCG color family!')

    if thr is None:
        thr = 0.35
    else:
        raise TypeError(funcName + ': \"thr\" must be an int or a float!')

    if smode is None:
        smode = 1
    elif smode not in [0, 1, 2, 3]:
        raise ValueError(funcName + ': \"smode\" must be in [0, 1, 2, 3]!')

    if radius is None:
        radius = (16 if src.width > 1024 or src.height > 576 else 12) if (smode == 1 or smode == 2) else 9
    elif isinstance(radius, int):
        if radius <= 0:
            raise ValueError(funcName + ': \"radius\" must be strictly positive.')
    else:
        raise TypeError(funcName + ': \"radius\" must be an int!')

    if elast is None:
        elast = 3.0
    elif isinstance(elast, (int, float)):
        if elast < 1:
            raise ValueError(funcName + ': Valid range of \"elast\" is [1, +inf)!')
    else:
        raise TypeError(funcName + ': \"elast\" must be an int or a float!')

    if mask is None:
        mask = 2
    elif not isinstance(mask, int):
        raise TypeError(funcName + ': \"mask\" must be an int!')

    if lsb is None:
        lsb = False

    if thr_det is None:
        thr_det = 2 + round(max(thr - 0.35, 0) / 0.3)
    elif isinstance(thr_det, (int, float)):
        if thr_det <= 0.0:
            raise ValueError(funcName + '" \"thr_det\" must be strictly positive!')
    else:
        raise TypeError(funcName + ': \"mask\" must be an int or a float!')

    if debug is None:
        debug = False
    elif not isinstance(debug, bool) and debug not in [0, 1]:
        raise TypeError(funcName + ': \"debug\" must be a bool!')

    if thrc is None:
        thrc = thr

    if radiusc is None:
        radiusc = radius
    elif isinstance(radiusc, int):
        if radiusc <= 0:
            raise ValueError(funcName + '\"radiusc\" must be strictly positive.')
    else:
        raise TypeError(funcName + '\"radiusc\" must be an int!')

    if elastc is None:
        elastc = elast
    elif isinstance(elastc, (int, float)):
        if elastc < 1:
            raise ValueError(funcName + ':valid range of \"elastc\" is [1, +inf)!')
    else:
        raise TypeError(funcName + ': \"elastc\" must be an int or a float!')

    if planes is None:
        planes = list(range(src.format.num_planes))
    elif isinstance(planes, int):
        planes = [planes]

    if ref is None:
        ref = src
    elif not isinstance(ref, vs.VideoNode):
        raise TypeError(funcName + ': \"ref\" must be a clip!')
    if ref.format.color_family not in [vs.YUV, vs.GRAY, vs.YCOCG]:
        raise TypeError(funcName + ': \"ref\" must be YUV, GRAY or YCOCG color family!')
    if src.width != ref.width or src.height != ref.height:
        raise TypeError(funcName + ': \"ref\" must be of the same size as \"src\"!')

    bits = src.format.bits_per_sample
    src_16 = core.fmtc.bitdepth(src, bits=16, planes=planes) if bits < 16 else src
    src_8 = core.fmtc.bitdepth(src, bits=8, dmode=1, planes=[0]) if bits != 8 else src
    if src == ref:
        ref_16 = src_16
    else:
        ref_16 = core.fmtc.bitdepth(ref, bits=16, planes=planes) if ref.format.bits_per_sample < 16 else ref

    # Main debanding
    chroma_flag = (thrc != thr or radiusc != radius or
                   elastc != elast) and 0 in planes and (1 in planes or 2 in planes)

    if chroma_flag:
        planes2 = [0] if 0 in planes else []
    else:
        planes2 = planes

    if not planes2:
        raise ValueError(funcName + ': no plane is processed!')

    flt_y = _GF3_smooth(src_16, ref_16, smode, radius, thr, elast, planes2)
    if chroma_flag:
        if 0 in planes2:
            planes2.remove(0)
        flt_c = _GF3_smooth(src_16, ref_16, smode, radiusc, thrc, elastc, planes2)
        flt = core.std.ShufflePlanes([flt_y, flt_c], list(range(src.format.num_planes)), src.format.color_family)
    else:
        flt = flt_y

    # Edge/detail mask
    td_lo = max(thr_det * 0.75, 1.0)
    td_hi = max(thr_det, 1.0)
    mexpr = 'x {tl} - {th} {tl} - / 255 *'.format(tl=td_lo - 0.0001, th=td_hi + 0.0001)

    if mask > 0:
        dmask = mvf.GetPlane(src_8, 0)
        dmask = _Build_gf3_range_mask(dmask)
        dmask = core.std.Expr([dmask], [mexpr])
        dmask = core.rgvs.RemoveGrain([dmask], [22])
        if mask > 1:
            dmask = core.std.Convolution(dmask, matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1])
            if mask > 2:
                dmask = core.std.Convolution(dmask, matrix=[1]*9)
        dmask = core.fmtc.bitdepth(dmask, bits=16, fulls=True, fulld=True)
        res_16 = core.std.MaskedMerge(flt, src_16, dmask, planes=planes, first_plane=True)
    else:
        res_16 = flt

    # Dithering
    result = res_16 if lsb else core.fmtc.bitdepth(res_16, bits=bits, planes=planes, dmode=mode, ampo=ampo,
                                                   ampn=ampn, dyn=dyn, staticnoise=staticnoise, patsize=pat)

    if debug:
        last = dmask
        if not lsb:
            last = core.fmtc.bitdepth(last, bits=8, fulls=True, fulld=True)
    else:
        last = result

    return last


def _GF3_smooth(src_16, ref_16, smode, radius, thr, elast, planes):
    if smode == 0:
        return _GF3_smoothgrad_multistage(src_16, ref_16, radius, thr, elast, planes)
    elif smode == 1:
        return _GF3_dfttest(src_16, ref_16, radius, thr, elast, planes)
    elif smode == 2:
        return _GF3_bilateral_multistage(src_16, ref_16, radius, thr, elast, planes)
    elif smode == 3:
        return _GF3_smoothgrad_multistage_3(src_16, radius, thr, elast, planes)
    else:
        raise ValueError(funcName + ': wrong smode value!')


def _GF3_smoothgrad_multistage(src, ref, radius, thr, elast, planes):
    ela_2 = max(elast * 0.83, 1.0)
    ela_3 = max(elast * 0.67, 1.0)
    r2 = radius * 2 // 3
    r3 = radius * 3 // 3
    r4 = radius * 4 // 4
    last = src
    last = SmoothGrad(last, radius=r2, thr=thr, elast=elast, ref=ref, planes=planes) if r2 >= 1 else last
    last = SmoothGrad(last, radius=r3, thr=thr * 0.7, elast=ela_2, ref=ref, planes=planes) if r3 >= 1 else last
    last = SmoothGrad(last, radius=r4, thr=thr * 0.46, elast=ela_3, ref=ref, planes=planes) if r4 >= 1 else last
    return last


def _GF3_smoothgrad_multistage_3(src, radius, thr, elast, planes):

    ref = SmoothGrad(src, radius=radius // 3, thr=thr * 0.8, elast=elast)
    last = BoxFilter(src, radius=radius, planes=planes)
    last = BoxFilter(last, radius=radius, planes=planes)
    last = mvf.LimitFilter(last, src, thr=thr * 0.6, elast=elast, ref=ref, planes=planes)
    return last


def _GF3_dfttest(src, ref, radius, thr, elast, planes):

    hrad = max(radius * 3 // 4, 1)
    last = core.dfttest.DFTTest(src, sigma=hrad * thr * thr * 32, sbsize=hrad * 4,
                                sosize=hrad * 3, tbsize=1, planes=planes)
    last = mvf.LimitFilter(last, ref, thr=thr, elast=elast, planes=planes)

    return last


def _GF3_bilateral_multistage(src, ref, radius, thr, elast, planes):

    last = core.bilateral.Bilateral(src, ref=ref, sigmaS=radius / 2, sigmaR=thr / 255, planes=planes, algorithm=0)

    last = mvf.LimitFilter(last, src, thr=thr, elast=elast, planes=planes)

    return last


def _Build_gf3_range_mask(src, radius=1):

    last = src

    if radius > 1:
        ma = haf.mt_expand_multi(last, mode='ellipse', planes=[0], sw=radius, sh=radius)
        mi = haf.mt_inpand_multi(last, mode='ellipse', planes=[0], sw=radius, sh=radius)
        last = core.std.Expr([ma, mi], ['x y -'])
    else:
        bits = src.format.bits_per_sample
        black = 0
        white = (1 << bits) - 1
        maxi = core.std.Maximum(last, [0])
        mini = core.std.Minimum(last, [0])
        exp = "x y -"
        exp2 = "x {thY1} < {black} x ? {thY2} > {white} x ?".format(thY1=0, thY2=255, black=black, white=white)
        last = core.std.Expr([maxi,mini],[exp])
        last = core.std.Expr([last], [exp2])

    return last


def AnimeMask(input, shift=0, expr=None, mode=1, **resample_args):
    """Generates edge/ringing mask for anime based on gradient operator.

    For Anime's ringing mask, it's recommended to set "shift" between 0.5 and 1.0.

    Args:
        input: Source clip. Only the First plane will be processed.

        shift: (float, -1.5 ~ 1.5) The distance of translation. Default is 0.

        expr: (string) Subsequent processing in std.Expr(). Default is "".

        mode: (-1 or 1) Type of the kernel, which simply inverts the pixel values and "shift". 
            Typically, -1 is for edge, 1 is for ringing. Default is 1.

        resample_args: (dict) Additional parameters passed to core.resize in the form of dict.

    """

    funcName = 'AnimeMask'

    if not isinstance(input, vs.VideoNode):
        raise TypeError(funcName + ': \"input\" must be a clip!')

    if input.format.color_family != vs.GRAY:
        input = mvf.GetPlane(input, 0)

    if mode not in [-1, 1]:
        raise ValueError(funcName + ': \'mode\' have not a correct value! [-1 or 1]')

    if mode == -1:
        input = core.std.Invert(input)
        shift = -shift

    full_args = dict(range_s="full", range_in_s="full")
    mask1 = core.std.Convolution(input, [0, 0, 0, 0, 2, -1, 0, -1, 0], saturate=True).resize.Bicubic(src_left=shift, 
        src_top=shift, **full_args, **resample_args)
    mask2 = core.std.Convolution(input, [0, -1, 0, -1, 2, 0, 0, 0, 0], saturate=True).resize.Bicubic(src_left=-shift, 
        src_top=-shift, **full_args, **resample_args)
    mask3 = core.std.Convolution(input, [0, -1, 0, 0, 2, -1, 0, 0, 0], saturate=True).resize.Bicubic(src_left=shift, 
        src_top=-shift, **full_args, **resample_args)
    mask4 = core.std.Convolution(input, [0, 0, 0, -1, 2, 0, 0, -1, 0], saturate=True).resize.Bicubic(src_left=-shift, 
        src_top=shift, **full_args, **resample_args)

    calc_expr = 'x x * y y * + z z * + a a * + sqrt '

    if isinstance(expr, str):
        calc_expr += expr

    mask = core.std.Expr([mask1, mask2, mask3, mask4], [calc_expr])

    return mask


def AnimeMask2(input, r=1.2, expr=None, mode=1):
    """Yet another filter to generate edge/ringing mask for anime.

    More specifically, it's an approximatation of the difference of gaussians filter based on resampling.

    Args:
        input: Source clip. Only the First plane will be processed.

        r: (float, positive) Radius of resampling coefficient. Default is 1.2.

        expr: (string) Subsequent processing in std.Expr(). Default is "".

        mode: (-1 or 1) Type of the kernel. Typically, -1 is for edge, 1 is for ringing. Default is 1.

    """

    funcName = 'AnimeMask2'

    if not isinstance(input, vs.VideoNode):
        raise TypeError(funcName + ': \"input\" must be a clip!')

    if input.format.color_family != vs.GRAY:
        input = mvf.GetPlane(input, 0)

    w = input.width
    h = input.height

    if mode not in [-1, 1]:
        raise ValueError(funcName + ': \'mode\' have not a correct value! [-1 or 1]')

    smooth = core.resize.Bicubic(input, haf.m4(w / r), haf.m4(h / r)).resize.Bicubic(w, h, filter_param_a=1, filter_param_b=0)
    smoother = core.resize.Bicubic(input, haf.m4(w / r), haf.m4(h / r)).resize.Bicubic(w, h, filter_param_a=1.5, filter_param_b=-0.25)

    calc_expr = 'x y - ' if mode == 1 else 'y x - '

    if isinstance(expr, str):
        calc_expr += expr

    mask = core.std.Expr([smooth, smoother], [calc_expr])

    return mask


def PolygonExInpand(input, shift=0, shape=0, mixmode=0, noncentral=False, step=1, amp=1, **resample_args):
    """Processes mask based on resampling.

    Args:
        input: Source clip. Only the First plane will be processed.

        shift: (float) Distance of expanding/inpanding. Default is 0.

        shape: (int, 0:losange, 1:square, 2:octagon) The shape of expand/inpand kernel. Default is 0.

        mixmode: (int, 0:max, 1:arithmetic mean, 2:quadratic mean)
            Method used to calculate the mix of different mask. Default is 0.

        noncentral: (bint) Whether to calculate the center pixel in mix process.

        step: (float) Step of expanding/inpanding. Default is 1.

        amp: (float) Linear multiple to strengthen the final mask. Default is 1.

        resample_args: (dict) Additional parameters passed to core.resize in the form of dict.

    """

    funcName = 'PolygonExInpand'

    if not isinstance(input, vs.VideoNode):
        raise TypeError(funcName + ': \"input\" must be a clip!')

    if shape not in list(range(3)):
        raise ValueError(funcName + ': \'shape\' have not a correct value! [0, 1 or 2]')

    if mixmode not in list(range(3)):
        raise ValueError(funcName + ': \'mixmode\' have not a correct value! [0, 1 or 2]')

    if step <= 0:
        raise ValueError(funcName + ': \'step\' must be positive!')

    invert = False
    if shift < 0:
        invert = True
        input = core.std.Invert(input)
        shift = -shift
    elif shift == 0:
        return input

    mask5 = input

    while shift > 0:
        step = min(step, shift)
        shift = shift - step

        ortho = step
        inv_ortho = -step
        dia = math.sqrt(step / 2)
        inv_dia = -math.sqrt(step / 2)

        # shift
        if shape == 0 or shape == 2:
            mask2 = core.resize.Bilinear(mask5, src_left=0, src_top=ortho, **resample_args)
            mask4 = core.resize.Bilinear(mask5, src_left=ortho, src_top=0, **resample_args)
            mask6 = core.resize.Bilinear(mask5, src_left=inv_ortho, src_top=0, **resample_args)
            mask8 = core.resize.Bilinear(mask5, src_left=0, src_top=inv_ortho, **resample_args)

        if shape == 1 or shape == 2:
            mask1 = core.resize.Bilinear(mask5, src_left=dia, src_top=dia, **resample_args)
            mask3 = core.resize.Bilinear(mask5, src_left=inv_dia, src_top=dia, **resample_args)
            mask7 = core.resize.Bilinear(mask5, src_left=dia, src_top=inv_dia, **resample_args)
            mask9 = core.resize.Bilinear(mask5, src_left=inv_dia, src_top=inv_dia, **resample_args)

        # mix
        if noncentral:
            expr_list = [
                'x y max z max a max',
                'x y + z + a + 4 /',
                'x x * y y * + z z * + a a * + 4 / sqrt',
                'x y max z max a max b max c max d max e max',
                'x y + z + a + b + c + d + e + 8 /',
                'x x * y y * + z z * + a a * + b b * + c c * + d d * + e e * + 8 / sqrt',
                ]

            if shape == 0 or shape == 1:
                expr = expr_list[mixmode] + ' {amp} *'.format(amp=amp)
                mask5 = core.std.Expr([mask2, mask4, mask6, mask8] if shape == 0 else [mask1, mask3, mask7, mask9], [expr])
            else: # shape == 2
                expr = expr_list[mixmode + 3] + ' {amp} *'.format(amp=amp)
                mask5 = core.std.Expr([mask1, mask2, mask3, mask4, mask6, mask7, mask8, mask9], [expr])
        else: # noncentral == False
            expr_list = [
                'x y max z max a max b max',
                'x y + z + a + b + 5 /',
                'x x * y y * + z z * + a a * + b b * + 5 / sqrt',
                'x y max z max a max b max c max d max e max f max',
                'x y + z + a + b + c + d + e + f + 9 /',
                'x x * y y * + z z * + a a * + b b * + c c * + d d * + e e * + f f * + 9 / sqrt',
                ]

            if (shape == 0) or (shape == 1):
                expr = expr_list[mixmode] + ' {amp} *'.format(amp=amp)
                mask5 = core.std.Expr([mask2, mask4, mask5, mask6, mask8] if shape == 0 else 
                    [mask1, mask3, mask5, mask7, mask9], [expr])
            else: # shape == 2
                expr = expr_list[mixmode + 3] + ' {amp} *'.format(amp=amp)
                mask5 = core.std.Expr([mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8, mask9], [expr])

    return core.std.Invert(mask5) if invert else mask5


def Luma(input, plane=0, power=4):
    """std.Lut() implementation of Luma() in Histogram() filter.

    Args:
        input: Source clip. Only one plane will be processed.

        plane: (int) Which plane to be processed. Default is 0.

        power: (int) Coefficient in processing. Default is 4.

    """

    funcName = 'Luma'

    if not isinstance(input, vs.VideoNode):
        raise TypeError(funcName + ': \"input\" must be a clip!')

    if (input.format.sample_type != vs.INTEGER):
        raise TypeError(funcName + ': \"input\" must be of integer format!')

    bits = input.format.bits_per_sample
    peak = (1 << bits) - 1

    clip = mvf.GetPlane(input, plane)


    def calc_luma(x):
        p = x << power
        return (peak - (p & peak)) if (p & (peak + 1)) else (p & peak)

    return core.std.Lut(clip, function=calc_luma)


def ediaa(a):
    """Suggested by Mystery Keeper in "Denoise of tv-anime" thread

    Read the document of Avisynth version for more details.

    """

    funcName = 'ediaa'

    if not isinstance(a, vs.VideoNode):
        raise TypeError(funcName + ': \"a\" must be a clip!')

    last = core.eedi2.EEDI2(a, field=1).std.Transpose()
    last = core.eedi2.EEDI2(last, field=1).std.Transpose()
    last = core.resize.Spline36(last, a.width, a.height, src_left=-0.5, src_top=-0.5)

    return last


def nnedi3aa(a):
    """Using nnedi3 (Emulgator):

    Read the document of Avisynth version for more details.

    """

    funcName = 'nnedi3aa'

    if not isinstance(a, vs.VideoNode):
        raise TypeError(funcName + ': \"a\" must be a clip!')

    last = core.nnedi3.nnedi3(a, field=1, dh=True).std.Transpose()
    last = core.nnedi3.nnedi3(last, field=1, dh=True).std.Transpose()
    last = core.resize.Spline36(last, a.width, a.height, src_left=-0.5, src_top=-0.5)

    return last


def maa(input):
    """Anti-aliasing with edge masking by martino, 
    mask using "sobel" taken from Kintaro's useless filterscripts and modded by thetoof for spline36

    Read the document of Avisynth version for more details.

    """

    funcName = 'maa'

    if not isinstance(input, vs.VideoNode):
        raise TypeError(funcName + ': \"input\" must be a clip!')

    w = input.width
    h = input.height
    bits = input.format.bits_per_sample

    if input.format.color_family != vs.GRAY:
        input_src = input
        input = mvf.GetPlane(input, 0)
    else:
        input_src = None

    mask = core.std.Convolution(input, [0, -1, 0, -1, 0, 1, 0, 1, 0], divisor=2, saturate=False).std.Binarize(scale(7, bits) + 1)
    aa_clip = core.resize.Spline36(input, w * 2, h * 2)
    aa_clip = core.sangnom.SangNom(aa_clip).std.Transpose()
    aa_clip = core.sangnom.SangNom(aa_clip).std.Transpose()
    aa_clip = core.resize.Spline36(aa_clip, w, h)
    last = core.std.MaskedMerge(input, aa_clip, mask)

    if input_src is None:
        return last
    else:
        return core.std.ShufflePlanes([last, input_src], planes=list(range(input_src.format.num_planes)), 
            colorfamily=input_src.format.color_family)


def SharpAAMcmod(orig, dark=0.2, thin=10, sharp=150, smooth=-1, stabilize=False, tradius=2, aapel=1, aaov=None, 
    aablk=None, aatype='nnedi3'):
    """High quality MoComped AntiAliasing script.

    Also a line darkener since it uses edge masking to apply tweakable warp-sharpening,
    "normal" sharpening and line darkening with optional temporal stabilization of these edges.
    Part of AnimeIVTC.

    Author: thetoof. Developed in the "fine anime antialiasing thread".

    Only the first plane (luma) will be processed.

    Args:
        orig: Source clip. Only the first plane will be processed.

        dark: (float) Strokes darkening strength. Default is 0.2.

        thin: (int) Presharpening. Default is 10.

        sharp: (int) Postsharpening. Default is 150.

        smooth: (int) Postsmoothing. Default is -1.

        stabilize: (bint) Use post stabilization with Motion Compensation. Default is False.

        tradius: (1~3) 1 = Degrain1 / 2 = Degrain2 / 3 = Degrain3. Default is 2.

        aapel: (int) Accuracy of the motion estimation. Default is 1
            (Value can only be 1, 2 or 4.
            1 means a precision to the pixel.
            2 means a precision to half a pixel,
            4 means a precision to quarter a pixel,
            produced by spatial interpolation (better but slower).)

        aaov: (int) Block overlap value (horizontal). Default is None.
            Must be even and less than block size.(Higher = more precise & slower)

        aablk: (4, 8, 16, 32, 64, 128) Size of a block (horizontal). Default is 8.
            Larger blocks are less sensitive to noise, are faster, but also less accurate.

        aatype: ("sangnom", "eedi2" or "nnedi3"). Default is "nnedi3".
            Use Sangnom() or EEDI2() or NNEDI3() for anti-aliasing.

    """

    funcName = 'SharpAAMcmod'

    if not isinstance(orig, vs.VideoNode):
        raise TypeError(funcName + ': \"orig\" must be a clip!')

    w = orig.width
    h = orig.height
    bits = orig.format.bits_per_sample

    if orig.format.color_family != vs.GRAY:
        orig_src = orig
        orig = mvf.GetPlane(orig, 0)
    else:
        orig_src = None

    if aaov is None:
        aaov = 8 if w > 1100 else 4

    if aablk is None:
        aablk = 16 if w > 1100 else 8

    m = core.std.Expr([core.std.Convolution(orig, [5, 10, 5, 0, 0, 0, -5, -10, -5], divisor=4, saturate=False), 
        core.std.Convolution(orig, [5, 0, -5, 10, 0, -10, 5, 0, -5], divisor=4, saturate=False)], 
        ['x y max {neutral} / 0.86 pow {peak} *'.format(neutral=1 << (bits-1), peak=(1 << bits)-1)])

    if thin == 0 and dark == 0:
        preaa = orig
    elif thin == 0:
        preaa = haf.Toon(orig, str=dark)
    elif dark == 0:
        preaa = core.warp.AWarpSharp2(orig, depth=thin)
    else:
        preaa = haf.Toon(orig, str=dark).warp.AWarpSharp2(depth=thin)

    aatype = aatype.lower()
    if aatype == 'sangnom':
        aa = core.resize.Spline36(preaa, w * 2, h * 2)
        aa = core.std.Transpose(aa).sangnom.SangNom()
        aa = core.std.Transpose(aa).sangnom.SangNom()
        aa = core.resize.Spline36(aa, w, h)
    elif aatype == 'eedi2':
        aa = ediaa(preaa)
    elif aatype == 'nnedi3':
        aa = nnedi3aa(preaa)
    else:
        raise ValueError(funcName + ': valid values of \"aatype\" are \"sangnom\", \"eedi2\" and \"nnedi3\"!')

    if sharp == 0 and smooth == 0:
        postsh = aa
    else:
        postsh = haf.LSFmod(aa, strength=sharp, overshoot=1, soft=smooth, edgemode=1)

    merged = core.std.MaskedMerge(orig, postsh, m)

    if stabilize:
        sD = core.std.MakeDiff(orig, merged)

        origsuper = haf.DitherLumaRebuild(orig, s0=1).mv.Super(pel=aapel)
        sDsuper = core.mv.Super(sD, pel=aapel)

        fv3 = core.mv.Analyse(origsuper, isb=False, delta=3, overlap=aaov, blksize=aablk) if tradius == 3 else None
        fv2 = core.mv.Analyse(origsuper, isb=False, delta=2, overlap=aaov, blksize=aablk) if tradius >= 2 else None
        fv1 = core.mv.Analyse(origsuper, isb=False, delta=1, overlap=aaov, blksize=aablk) if tradius >= 1 else None
        bv1 = core.mv.Analyse(origsuper, isb=True, delta=1, overlap=aaov, blksize=aablk) if tradius >= 1 else None
        bv2 = core.mv.Analyse(origsuper, isb=True, delta=2, overlap=aaov, blksize=aablk) if tradius >= 2 else None
        bv3 = core.mv.Analyse(origsuper, isb=True, delta=3, overlap=aaov, blksize=aablk) if tradius == 3 else None

        if tradius == 1:
            sDD = core.mv.Degrain1(sD, sDsuper, bv1, fv1)
        elif tradius == 2:
            sDD = core.mv.Degrain2(sD, sDsuper, bv1, fv1, bv2, fv2)
        elif tradius == 3:
            sDD = core.mv.Degrain3(sD, sDsuper, bv1, fv1, bv2, fv2, bv3, fv3)
        else:
            raise ValueError(funcName + ': valid values of \"tradius\" are 1, 2 and 3!')

        sDD = core.std.Expr([sD, sDD], ['x {neutral} - abs y {neutral} - abs < x y ?'.format(neutral=1 << (bits-1))]).std.Merge(sDD, 0.6)

        last = core.std.MakeDiff(orig, sDD)
    else:
        last = merged

    if orig_src is None:
        return last
    else:
        return core.std.ShufflePlanes([last, orig_src], planes=list(range(orig_src.format.num_planes)), 
            colorfamily=orig_src.format.color_family)


def TEdge(input, min=0, max=65535, planes=None, rshift=0):
    """Detects edge using TEdgeMask(type=2).

    Port from https://github.com/chikuzen/GenericFilters/blob/2044dc6c25a1b402aae443754d7a46217a2fddbf/src/convolution/tedge.c

    Args:
        input: Source clip.

        min: (int) If output pixel value is lower than this, it will be zero. Default is 0.

        max: (int) If output pixel value is same or higher than this, it will be maximum value of the format. Default is 65535.

        planes: (int []) Whether to process the corresponding plane. By default, every plane will be processed.
            The unprocessed planes will be copied from "input".

        rshift: (int) Shift the output values to right by this count before clamp. Default is 0.

    """

    funcName = 'TEdge'

    if not isinstance(input, vs.VideoNode):
        raise TypeError(funcName + ': \"input\" must be a clip!')

    if planes is None:
        planes = list(range(input.format.num_planes))
    elif isinstance(planes, int):
        planes = [planes]

    rshift = 1 << rshift

    bits = input.format.bits_per_sample
    floor = 0
    peak = (1 << bits) - 1

    gx = core.std.Convolution(input, [4, -25, 0, 25, -4], planes=planes, saturate=False, mode='h')
    gy = core.std.Convolution(input, [-4, 25, 0, -25, 4], planes=planes, saturate=False, mode='v')

    calcexpr = 'x x * y y * + {rshift} / sqrt'.format(rshift=rshift)
    expr = '{calc} {max} > {peak} {calc} {min} < {floor} {calc} ? ?'.format(calc=calcexpr, max=max, peak=peak, min=min, floor=floor)
    return core.std.Expr([gx, gy], [(expr if i in planes else '') for i in range(input.format.num_planes)])


def Sort(input, order=1, planes=None, mode='max'):
    """Simple filter to get nth largeest value in 3x3 neighbourhood.

    Args:
        input: Source clip.

        order: (int) The order of value to get in 3x3 neighbourhood. Default is 1.

        planes: (int []) Whether to process the corresponding plane. By default, every plane will be processed.
            The unprocessed planes will be copied from "input".

        mode: ("max" or "min") How to measure order. Default is "max".

    """

    funcName = 'Sort'

    if not isinstance(input, vs.VideoNode):
        raise TypeError(funcName + ': \"input\" must be a clip!')

    if order not in range(1, 10):
        raise ValueError(funcName + ': valid values of \"order\" are 1~9!')

    mode = mode.lower()
    if mode not in ['max', 'min']:
        raise ValueError(funcName + ': valid values of \"mode\" are \"max\" and \"min\"!')

    if planes is None:
        planes = list(range(input.format.num_planes))
    elif isinstance(planes, int):
        planes = [planes]

    if mode == 'min':
        order = 10 - order # the nth smallest value in 3x3 neighbourhood is the same as the (10-n)th largest value

    if order == 1:
        sort = core.std.Maximum(input, planes=planes)
    elif order in range(2, 5):
        sort = core.rgvs.Repair(core.std.Maximum(input, planes=planes), input,
                                [(order if i in planes else 0) for i in range(input.format.num_planes)])
    elif order == 5:
        sort = core.std.Median(input, planes=planes)
    elif order in range(6, 9):
        sort = core.rgvs.Repair(core.std.Minimum(input, planes=planes), input,
                                [((10 - order) if i in planes else 0) for i in range(input.format.num_planes)])
    else: # order == 9
        sort = core.std.Minimum(input, planes=planes)

    return sort


def Soothe_mod(input, source, keep=24, radius=1, scenechange=32, use_misc=True):
    """Modified Soothe().

    Basd on Didée, 6th September 2005, http://forum.doom9.org/showthread.php?p=708217#post708217
    Modified by TheRyuu, 14th July 2007, http://forum.doom9.org/showthread.php?p=1024318#post1024318
    Modified by Muonium, 12th, December 2016, add args "radius", "scenechange" and "use_misc"

    Requires Filters
    misc (optional)

    Args:
        input: Filtered clip.

        source: Source clip. Must match "input" clip.

        keep: (0~100). Minimum percent of the original sharpening to keep. Default is 24.

        radius: (1~7 (use_misc=True) or 1~12 (use_misc=False)) Temporal radius of AverageFrames. Default is 1.

        scenechange: (int) Argument in scenechange detection. Default is 32.

        use_misc: (bint) Whether to use miscellaneous filters. Default is True.

    Examples: (in Avisynth)
        We use LimitedSharpen() as sharpener, and we'll keep at least 20% of its result:
        dull = last
        sharpener = dull.LimitedSharpen( ss_x=1.25, ss_y=1.25, strength=150, overshoot=1 )

        Soothe( sharp, dull, 20 )

    """

    funcName = 'Soothe_mod'

    if not isinstance(input, vs.VideoNode):
        raise TypeError(funcName + ': \"input\" must be a clip!')
    if not isinstance(source, vs.VideoNode):
        raise TypeError(funcName + ': \"source\" must be a clip!')

    if input == source:
        return input

    if input.format.id != source.format.id:
        raise TypeError(funcName + ': \"source\" must be of the same format as \"input\"!')
    if input.width != source.width or input.height != source.height:
        raise TypeError(funcName + ': \"source\" must be of the same size as \"input\"!')

    if input.format.color_family != vs.GRAY:
        input = mvf.GetPlane(input, 0)

    if source.format.color_family != vs.GRAY:
        source_src = source
        source = mvf.GetPlane(source, 0)
    else:
        source_src = None

    keep = max(min(keep, 100), 0)

    if use_misc:
        if not isinstance(radius, int) or (not(1 <= radius <= 12)):
            raise ValueError(funcName + ': \'radius\' have not a correct value! [1 ~ 12]')
    else:
        if not isinstance(radius, int) or (not(1 <= radius <= 7)):
            raise ValueError(funcName + ': \'radius\' have not a correct value! [1 ~ 7]')

    bits = source.format.bits_per_sample

    diff = core.std.MakeDiff(source, input)
    if use_misc:
        diff2 = TemporalSoften(diff, radius, scenechange)
    else:
        if 'TemporalSoften' in dir(haf):
            diff2 = haf.TemporalSoften(diff, radius, (1 << bits)-1, 0, scenechange)
        else:
            raise NameError(funcName + (': \"TemporalSoften\" has been deprecated from the latest havsfunc.' 
                'If you would like to use it, copy the old function in ' 
                'https://github.com/HomeOfVapourSynthEvolution/havsfunc/blob/0f5e6c5c2f1e825caf17f6b7de6edd4a0e13d27d/havsfunc.py#L4300-L4308'
                'and function set_scenechange() at line 4320-4344 to havsfunc in your disk.'))

    expr = 'x {neutral} - y {neutral} - * 0 < x {neutral} - {KP} * {neutral} + x {neutral} - abs y {neutral} - abs > x {KP} * y {iKP} * + x ? ?'.format(
        neutral=1 << (bits-1), KP=keep/100, iKP=1-keep/100)
    diff3 = core.std.Expr([diff, diff2], [expr])

    last = core.std.MakeDiff(source, diff3)

    if source_src is None:
        return last
    else:
        return core.std.ShufflePlanes([last, source_src], planes=list(range(source_src.format.num_planes)), 
            colorfamily=source_src.format.color_family)


def TemporalSoften(input, radius=4, scenechange=15):
    """TemporalSoften filter without thresholding using Miscellaneous filters.

    There will be slight difference in result compare to havsfunc.TemporalSoften().
    It seems that this Misc-filter-based TemporalSoften is slower than the one in havsfunc.

    Read the document of Avisynth version for more details.

    """

    funcName = 'TemporalSoften'

    if not isinstance(input, vs.VideoNode):
        raise TypeError(funcName + ': \"input\" must be a clip!')

    if scenechange:
        if 'SCDetect' in dir(haf):
            input = haf.SCDetect(input, scenechange / 255)
        elif 'set_scenechange' in dir(haf):
            input = haf.set_scenechange(input, scenechange)
        else:
            raise AttributeError('module \"havsfunc\" has no attribute \"SCDetect\"!')

    return core.misc.AverageFrames(input, [1] * (2 * radius + 1), scenechange=scenechange)


def FixTelecinedFades(input, mode=0, threshold=[0.0], color=[0.0], full=None, planes=None):
    """Fix Telecined Fades filter

    The main algorithm was proposed by feisty2 (http://forum.doom9.org/showthread.php?t=174151).
    The idea of thresholding was proposed by Myrsloik (http://forum.doom9.org/showthread.php?p=1791412#post1791412).
    Corresponding C++ code written by feisty2: https://github.com/IFeelBloated/Fix-Telecined-Fades/blob/7922a339629ed8ce93b540f3bdafb99fe97096b6/Source.cpp.

    the filter gives a mathematically perfect solution to such
    (fades were done AFTER telecine which made a picture perfect IVTC pretty much impossible) problem,
    and it's now time to kiss "vinverse" goodbye cuz "vinverse" is old and low quality.
    unlike vinverse which works as a dumb blurring + contra-sharpening combo and very harmful to artifacts-free frames,
    this filter works by matching the brightness of top and bottom fields with statistical methods, and also harmless to healthy frames.

    Args:
        input: Source clip. Can be 8-16 bits integer or 32 bits floating point based. Recommend to use 32 bits float format.

        mode: (0~2 []) Default is 0.
            0: adjust the brightness of both fields to match the average brightness of 2 fields.
            1: darken the brighter field to match the brightness of the darker field
            2: brighten the darker field to match the brightness of the brighter field

        threshold: (float [], positive) Default is 0.
            If the absolute difference between filtered pixel and input pixel is less than "threshold", then just copy the input pixel.
            The value is always scaled by 8 bits integer.
            The last value in the list will be used for the remaining plane.

        color: (float [], positive) Default is 0.
            (It is difficult for me to describe the effect of this parameter.)
            The value is always scaled by 8 bits integer.
            The last value in the list will be used for the remaining plane.

        full: (bint) If not set, assume False(limited range) for Gray and YUV input, assume True(full range) for other input.

        planes: (int []) Whether to process the corresponding plane. By default, every plane will be processed.
            The unprocessed planes will be copied from "input".

    """

    funcName = 'FixTelecinedFades'

    # set parameters
    if not isinstance(input, vs.VideoNode):
        raise TypeError(funcName + ': \"input\" must be a clip!')

    bits = input.format.bits_per_sample
    isFloat = input.format.sample_type == vs.FLOAT

    if isinstance(mode, int):
        mode = [mode]
    elif not isinstance(mode, list):
        raise TypeError(funcName + ': \"mode\" must be an int!')
    if len(mode) < input.format.num_planes:
        if len(mode) == 0:
            mode = [0]
        modeLength = len(mode)
        for i in range(input.format.num_planes - modeLength):
            mode.append(mode[modeLength - 1])
    for i in mode:
        if i not in [0, 1, 2]:
            raise ValueError(funcName + ': valid values of \"mode\" are 0, 1 or 2!')

    if isinstance(threshold, (int, float)):
        threshold = [threshold]
    if not isinstance(threshold, list):
        raise TypeError(funcName + ': \"threshold\" must be a list!')
    if len(threshold) < input.format.num_planes:
        if len(threshold) == 0:
            threshold = [0.0]
        thresholdLength = len(threshold)
        for i in range(input.format.num_planes - thresholdLength):
            threshold.append(threshold[thresholdLength - 1])
    if isFloat:
        for i in range(len(threshold)):
            threshold[i] = abs(threshold[i]) / 255
    else:
        for i in range(len(threshold)):
            threshold[i] = abs(threshold[i]) * ((1 << bits) - 1) / 255

    if isinstance(color, (int, float)):
        color = [color]
    if not isinstance(color, list):
        raise TypeError(funcName + ': \"color\" must be a list!')
    if len(color) < input.format.num_planes:
        if len(color) == 0:
            color = [0.0]
        colorLength = len(color)
        for i in range(input.format.num_planes - colorLength):
            color.append(color[colorLength - 1])
    if isFloat:
        for i in range(len(color)):
            color[i] = color[i] / 255
    else:
        for i in range(len(color)):
            color[i] = abs(color[i]) * ((1 << bits) - 1) / 255

    if full is None:
        if input.format.color_family in [vs.GRAY, vs.YUV]:
            full = False
        else:
            full = True

    if planes is None:
        planes = list(range(input.format.num_planes))
    elif isinstance(planes, int):
        planes = [planes]

    # internal function
    def GetExpr(scale, color, threshold):
        if color != 0:
            flt = 'x {color} - {scale} * {color} +'.format(scale=scale, color=color)
        else:
            flt = 'x {scale} *'.format(scale=scale)
        return flt if threshold == 0 else '{flt} x - abs {threshold} > {flt} x ?'.format(flt=flt, threshold=threshold)


    def Adjust(n, f, clip, core, mode, threshold, color):
        separated = core.std.SeparateFields(clip, tff=True)
        topField = core.std.SelectEvery(separated, 2, [0])
        bottomField = core.std.SelectEvery(separated, 2, [1])

        topAvg = f[0].props['PlaneStatsAverage']
        bottomAvg = f[1].props['PlaneStatsAverage']

        if color != 0:
            if isFloat:
                topAvg -= color
                bottomAvg -= color
            else:
                topAvg -= color / ((1 << bits) - 1)
                bottomAvg -= color / ((1 << bits) - 1)

        if topAvg != bottomAvg:
            if mode == 0:
                meanAvg = (topAvg + bottomAvg) / 2
                topField = core.std.Expr([topField], [GetExpr(scale=meanAvg / topAvg, threshold=threshold, color=color)])
                bottomField = core.std.Expr([bottomField], [GetExpr(scale=meanAvg / bottomAvg, threshold=threshold, color=color)])
            elif mode == 1:
                minAvg = min(topAvg, bottomAvg)
                if minAvg == topAvg:
                    bottomField = core.std.Expr([bottomField], [GetExpr(scale=minAvg / bottomAvg, threshold=threshold, color=color)])
                else:
                    topField = core.std.Expr([topField], [GetExpr(scale=minAvg / topAvg, threshold=threshold, color=color)])
            elif mode == 2:
                maxAvg = max(topAvg, bottomAvg)
                if maxAvg == topAvg:
                    bottomField = core.std.Expr([bottomField], [GetExpr(scale=maxAvg / bottomAvg, threshold=threshold, color=color)])
                else:
                    topField = core.std.Expr([topField], [GetExpr(scale=maxAvg / topAvg, threshold=threshold, color=color)])

        woven = core.std.Interleave([topField, bottomField])
        woven = core.std.DoubleWeave(woven, tff=True).std.SelectEvery(2, [0])
        return woven

    # process
    input_src = input
    if not full and not isFloat:
        input = core.fmtc.bitdepth(input, fulls=False, fulld=True, planes=planes)

    separated = core.std.SeparateFields(input, tff=True)
    topField = core.std.SelectEvery(separated, 2, [0])
    bottomField = core.std.SelectEvery(separated, 2, [1])

    topFieldPlanes = {}
    bottomFieldPlanes = {}
    adjustedPlanes = {}
    for i in range(input.format.num_planes):
        if i in planes:
            inputPlane = mvf.GetPlane(input, i)
            topFieldPlanes[i] = mvf.GetPlane(topField, i).std.PlaneStats()
            bottomFieldPlanes[i] = mvf.GetPlane(bottomField, i).std.PlaneStats()
            adjustedPlanes[i] = core.std.FrameEval(inputPlane, functools.partial(Adjust, clip=inputPlane, core=core, mode=mode[i], 
                threshold=threshold[i], color=color[i]), prop_src=[topFieldPlanes[i], bottomFieldPlanes[i]])
        else:
            adjustedPlanes[i] = None

    adjusted = core.std.ShufflePlanes([(adjustedPlanes[i] if i in planes else input_src) for i in range(input.format.num_planes)], 
        [(0 if i in planes else i) for i in range(input.format.num_planes)], input.format.color_family)
    if not full and not isFloat:
        adjusted = core.fmtc.bitdepth(adjusted, fulls=True, fulld=False, planes=planes)
        adjusted = core.std.ShufflePlanes([(adjusted if i in planes else input_src) for i in range(input.format.num_planes)], 
            list(range(input.format.num_planes)), input.format.color_family)
    return adjusted


def TCannyHelper(input, t_h=8.0, t_l=1.0, plane=0, returnAll=False, **canny_args):
    """A helper function for tcanny.TCanny(mode=0)

    Strong edge detected by "t_h" will be highlighted in white, and weak edge detected by "t_l" will be highlighted in gray.

    Args:
        input: Source clip. Can be 8-16 bits integer or 32 bits floating point based.

        t_h: (float) TCanny's high gradient magnitude threshold for hysteresis. Default is 8.0.

        t_l: (float) TCanny's low gradient magnitude threshold for hysteresis. Default is 1.0.

        plane: (int) Which plane to be processed. Default is 0.

        returnAll: (bint) Whether to return a tuple containing every 4 temporary clips
            (strongEdge, weakEdge, view, tcannyOutput) or just "view" clip.
            Default is False.

        canny_args: (dict) Additional parameters passed to core.tcanny.TCanny (except "mode" and "planes") in the form of keyword arguments.

    """

    funcName = 'TCannyHelper'

    if not isinstance(input, vs.VideoNode):
        raise TypeError(funcName + ': \"input\" must be a clip!')

    if input.format.color_family != vs.GRAY:
        input = mvf.GetPlane(input, plane)

    if "mode" in canny_args:
        del canny_args["mode"]

    bits = input.format.bits_per_sample
    isFloat = input.format.sample_type == vs.FLOAT

    strongEdge = core.tcanny.TCanny(input, t_h=t_h+1e-4, t_l=t_h, mode=0, **canny_args)
    weakEdge = core.tcanny.TCanny(input, t_h=t_l+1e-4, t_l=t_l, mode=0, **canny_args)

    expr = "x y and {peak} y {neutral} 0 ? ?".format(peak=1.0 if isFloat else (1 << bits) - 1, neutral=0.5 if isFloat else 1 << (bits - 1))
    view = core.std.Expr([strongEdge, weakEdge], [expr])

    if returnAll:
        tcannyOutput = core.tcanny.TCanny(input, t_h=t_h, t_l=t_l, mode=0, **canny_args)
        return (strongEdge, weakEdge, view, tcannyOutput)
    else:
        return view


def MergeChroma(clip1, clip2, weight=1.0):
    """Merges the chroma from one videoclip into another. Port from Avisynth's equivalent.

    There is an optional weighting, so a percentage between the two clips can be specified.

    Args:
        clip1: The clip that has the chroma pixels merged into (the base clip).

        clip2: The clip from which the chroma pixel data is taken (the overlay clip).

        weight: (float) Defines how much influence the new clip should have. Range is 0.0–1.0.

    """

    funcName = 'MergeChroma'

    if not isinstance(clip1, vs.VideoNode):
        raise TypeError(funcName + ': \"clip1\" must be a clip!')

    if not isinstance(clip2, vs.VideoNode):
        raise TypeError(funcName + ': \"clip2\" must be a clip!')

    if weight >= 1.0:
        return core.std.ShufflePlanes([clip1, clip2], [0, 1, 2], vs.YUV)
    elif weight <= 0.0:
        return clip1
    else:
        if clip1.format.num_planes != 3:
            raise TypeError(funcName + ': \"clip1\" must have 3 planes!')
        if clip2.format.num_planes != 3:
            raise TypeError(funcName + ': \"clip2\" must have 3 planes!')

        clip1_u = mvf.GetPlane(clip1, 1)
        clip2_u = mvf.GetPlane(clip2, 1)
        output_u = core.std.Merge(clip1_u, clip2_u, weight)

        clip1_v = mvf.GetPlane(clip1, 2)
        clip2_v = mvf.GetPlane(clip2, 2)
        output_v = core.std.Merge(clip1_v, clip2_v, weight)

        output = core.std.ShufflePlanes([clip1, output_u, output_v], [0, 0, 0], vs.YUV)

        return output


def firniture(clip, width, height, kernel='binomial7', taps=None, gamma=False, fulls=False, fulld=False, curve='709', sigmoid=False, **resample_args):
    '''5 new interpolation kernels (via fmtconv)

    Proposed by *.mp4 guy (https://forum.doom9.org/showthread.php?t=166080)

    Args:
        clip: Source clip.

        width, height: (int) New picture width and height in pixels.

        kernel: (string) Default is "binomial7".
            "binomial5", "binomial7": A binomial windowed sinc filter with 5 or 7 taps. 
                Should have the least ringing of any available interpolator, except perhaps "noaliasnoring4".
            "maxflat5", "maxflat8": 5 or 8 tap interpolation that is maximally flat in the passband. 
                In English, these filters have a sharp and relatively neutral look, but can have ringing and aliasing problems.
            "noalias4": A 4 tap filter hand designed to be free of aliasing while having acceptable ringing and blurring characteristics. 
                Not always a good choice, but sometimes very useful.
            "noaliasnoring4": Derived from the "noalias4" kernel, but modified to have reduced ringing. Other attributes are slightly worse.

        taps: (int) Default is the last num in "kernel".
            "taps" in fmtc.resample. This parameter is now mostly superfluous. 
            It has been retained so that you can truncate the kernels to shorter taps then they would normally use.

        gamma: (bool) Default is False.
            Set to true to turn on gamma correction for the y channel.

        fulls: (bool) Default is False.
            Specifies if the luma is limited range (False) or full range (True) 

        fulld: (bool) Default is False.
            Same as fulls, but for output.
        
        curve: (string) Default is '709'.
            Type of gamma mapping.

        sigmoid: (bool) Default is False.
            When True, applies a sigmoidal curve after the power-like curve (or before when converting from linear to gamma-corrected). 
            This helps reducing the dark halo artefacts around sharp edges caused by resizing in linear luminance.
        
        resample_args: (dict) Additional parameters passed to core.fmtc.resample in the form of keyword arguments.

    Examples:
        clip = muvsfunc.firniture(clip, 720, 400, kernel="noalias4", gamma=False)

    '''

    funcName = 'firniture'

    if not isinstance(clip, vs.VideoNode):
        raise TypeError(funcName + ': \"clip\" must be a clip!')

    import nnedi3_resample as nnrs

    impulseCoefficents = dict(
        binomial5=[8, 0, -589, 0, 11203, 0, -93355, 0, 606836, 1048576, 606836, 0, -93355, 0, 11203, 0, -589, 0, 8],
        binomial7=[146, 0, -20294, 0, 744006, 0, -11528384, 0, 94148472, 0, -487836876, 0, 2551884458, 4294967296, 2551884458, 
            0, -487836876, 0, 94148472, 0, -11528384, 0, 744006, 0, -20294, 0, 146],
        maxflat5=[-259, 1524, -487, -12192, 17356, 42672, -105427, -85344, 559764, 1048576, 559764, -85344, -105427, 42672, 
            17356, -12192, -487, 1524, -259],
        maxflat8=[2, -26, 166, -573, 912, 412, 1524, -589, -12192, 17356, 42672, -105427, -85344, 606836, 1048576, 606836, -85344, 
            -105427, 42672, 17356, -12192, -589, 1524, 412, 912, -573, 166, -26, 2],
        noalias4=[-1, 2, 4, -6, -17, 7, 59, 96, 59, 7, -17, -6, 4, 2, -1],
        noaliasnoring4=[-1, 8, 40, -114, -512, 360, 3245, 5664, 3245, 360, -512, -114, 40, 8, -1]
        )

    if taps is None:
        taps = int(kernel[-1])

    if clip.format.bits_per_sample != 16:
        clip = mvf.Depth(clip, 16)

    if gamma:
        clip = nnrs.GammaToLinear(clip, fulls=fulls, fulld=fulld, curve=curve, sigmoid=sigmoid, planes=[0])

    clip = core.fmtc.resample(clip, width, height, kernel='impulse', impulse=impulseCoefficents[kernel], kovrspl=2, 
        taps=taps, **resample_args)

    if gamma:
        clip = nnrs.LinearToGamma(clip, fulls=fulls, fulld=fulld, curve=curve, sigmoid=sigmoid, planes=[0])

    return clip


def BoxFilter(input, radius=16, radius_v=None, planes=None, fmtc_conv=0, radius_thr=None, resample_args=None, keep_bits=True, 
    depth_args=None):
    '''Box filter

    Performs a box filtering on the input clip.
    Box filtering consists in averaging all the pixels in a square area whose center is the output pixel.
    You can approximate a large gaussian filtering by cascading a few box filters.

    Args:
        input: Input clip to be filtered.

        radius, radius_v: (int) Size of the averaged square. The size is (radius*2-1) * (radius*2-1). 
            If "radius_v" is None, it will be set to "radius".
            Default is 16.

        planes: (int []) Whether to process the corresponding plane. By default, every plane will be processed.
            The unprocessed planes will be copied from the source clip, "input".

        fmtc_conv: (0~2) Whether to use fmtc.resample for convolution.
            It's recommended to input clip without chroma subsampling when using fmtc.resample, otherwise the output may be incorrect.
            0: False. 1: True (except both "radius" and "radius_v" is strictly smaller than 4). 
                2: Auto, determined by radius_thr (exclusive).
            Default is 0.

        radius_thr: (int) Threshold of wheter to use fmtc.resample when "fmtc_conv" is 2.
            Default is 11 for integer input and 21 for float input.
            Only works when "fmtc_conv" is enabled.

        resample_args: (dict) Additional parameters passed to core.fmtc.resample in the form of dict.
            It's recommended to set "flt" to True for higher precision, like:
                flt = muf.BoxFilter(src, resample_args=dict(flt=True))
            Only works when "fmtc_conv" is enabled.
            Default is {}.

        keep_bits: (bool) Whether to keep the bitdepth of the output the same as input.
            Only works when "fmtc_conv" is enabled and input is integer.

        depth_args: (dict) Additional parameters passed to mvf.Depth in the form of dict.
            Only works when "fmtc_conv" is enabled, input is integer and "keep_bits" is True.
            Default is {}.

    '''

    funcName = 'BoxFilter'

    if not isinstance(input, vs.VideoNode):
        raise TypeError(funcName + ': \"input\" must be a clip!')

    if planes is None:
        planes = list(range(input.format.num_planes))
    elif isinstance(planes, int):
        planes = [planes]

    if radius_v is None:
        radius_v = radius

    if radius == radius_v == 1:
        return input

    if radius_thr is None:
        radius_thr = 21 if input.format.sample_type == vs.FLOAT else 11 # Values are measured from my experiment

    if resample_args is None:
        resample_args = {}

    if depth_args is None:
        depth_args = {}

    planes2 = [(3 if i in planes else 2) for i in range(input.format.num_planes)]
    width = radius * 2 - 1
    width_v = radius_v * 2 - 1
    kernel = [1 / width] * width
    kernel_v = [1 / width_v] * width_v

    # process
    if input.format.sample_type == vs.FLOAT:
        if core.version_number() < 33:
            raise NotImplementedError(funcName + (': Please update your VapourSynth.'
                'BoxBlur on float sample has not yet been implemented on current version.'))
        elif radius == radius_v == 2 or radius == radius_v == 3:
            return core.std.Convolution(input, [1] * ((radius * 2 - 1) * (radius * 2 - 1)), planes=planes, mode='s')

        else:
            if fmtc_conv == 1 or (fmtc_conv != 0 and radius > radius_thr): # Use fmtc.resample for convolution
                flt = core.fmtc.resample(input, kernel='impulse', impulseh=kernel, impulsev=kernel_v, planes=planes2, 
                    cnorm=False, fh=-1, fv=-1, center=False, **resample_args)
                return flt # No bitdepth conversion is required since fmtc.resample outputs the same bitdepth as input

            elif core.version_number() >= 39:
                return core.std.BoxBlur(input, hradius=radius-1, vradius=radius_v-1, planes=planes)

            else: # BoxBlur on float sample has not been implemented
                if radius > 1:
                    input = core.std.Convolution(input, [1] * (radius * 2 - 1), planes=planes, mode='h')
                if radius_v > 1:
                    input = core.std.Convolution(input, [1] * (radius_v * 2 - 1), planes=planes, mode='v')
                return input

    else: # input.format.sample_type == vs.INTEGER
        if radius == radius_v == 2 or radius == radius_v == 3:
            return core.std.Convolution(input, [1] * ((radius * 2 - 1) * (radius * 2 - 1)), planes=planes, mode='s')

        else:
            if fmtc_conv == 1 or (fmtc_conv != 0 and radius > radius_thr): # Use fmtc.resample for convolution
                flt = core.fmtc.resample(input, kernel='impulse', impulseh=kernel, impulsev=kernel_v, planes=planes2, 
                    cnorm=False, fh=-1, fv=-1, center=False, **resample_args)
                if keep_bits and input.format.bits_per_sample != flt.format.bits_per_sample:
                    flt = mvf.Depth(flt, depth=input.format.bits_per_sample, **depth_args)
                return flt

            elif core.std.get_functions().__contains__('BoxBlur'):
                return core.std.BoxBlur(input, hradius=radius-1, vradius=radius_v-1, planes=planes)

            else: # BoxBlur was not found
                if radius > 1:
                    input = core.std.Convolution(input, [1] * (radius * 2 - 1), planes=planes, mode='h')
                if radius_v > 1:
                    input = core.std.Convolution(input, [1] * (radius_v * 2 - 1), planes=planes, mode='v')
                return input


def SmoothGrad(input, radius=9, thr=0.25, ref=None, elast=3.0, planes=None, **limit_filter_args):
    '''Avisynth's SmoothGrad

    SmoothGrad smooths the low gradients or flat areas of a 16-bit clip. 
    It proceeds by applying a huge blur filter and comparing the result with the input data for each pixel.
    If the difference is below the specified threshold, the filtered version is taken into account, 
        otherwise the input pixel remains unchanged.

    Args:
        input: Input clip to be filtered.

        radius: (int) Size of the averaged square. Its width is radius*2-1. Range is 2–9.

        thr: (float) Threshold between reference data and filtered data, on an 8-bit scale.

        ref: Reference clip for the filter output comparison. Specify here the input clip when you cascade several SmoothGrad calls.
            When undefined, the input clip is taken as reference.

        elast: (float) To avoid artifacts, the threshold has some kind of elasticity.
            Value differences falling over this threshold are gradually attenuated, up to thr * elast > 1.

        planes: (int []) Whether to process the corresponding plane. By default, every plane will be processed.
            The unprocessed planes will be copied from the source clip, "input".

        limit_filter_args: (dict) Additional arguments passed to mvf.LimitFilter in the form of keyword arguments.

    '''

    funcName = 'SmoothGrad'

    if not isinstance(input, vs.VideoNode):
        raise TypeError(funcName + ': \"input\" must be a clip!')

    if planes is None:
        planes = list(range(input.format.num_planes))
    elif isinstance(planes, int):
        planes = [planes]

    # process
    smooth = BoxFilter(input, radius, planes=planes)

    return mvf.LimitFilter(smooth, input, ref, thr, elast, planes=planes, **limit_filter_args)


def DeFilter(input, fun, iter=10, planes=None, **fun_args):
    '''Zero-order reverse filter

    Args:
        input: Input clip to be reversed.

        fun: The function of how the input clip is filtered.

        iter: (int) Number of iterations. Default is 10.

        planes: (int []) Whether to process the corresponding plane. By default, every plane will be processed.
            The unprocessed planes will be copied from the source clip, "input".

        fun_args: (dict) Additional arguments passed to "fun" in the form of keyword arguments. Alternative to functools.partial.

    Ref:
        [1] Tao, X., Zhou, C., Shen, X., Wang, J., & Jia, J. (2017, October). Zero-Order Reverse Filtering. 
            In Computer Vision (ICCV), 2017 IEEE International Conference on (pp. 222-230). IEEE.
        [2] https://github.com/jiangsutx/DeFilter
        [3] Milanfar, P. (2018). Rendition: Reclaiming what a black box takes away. arXiv preprint arXiv:1804.08651.

    '''

    funcName = 'DeFilter'

    if not isinstance(input, vs.VideoNode):
        raise TypeError(funcName + ': \"input\" must be a clip!')

    if planes is None:
        planes = list(range(input.format.num_planes))
    elif isinstance(planes, int):
        planes = [planes]

    # initialization
    flt = input
    calc_expr = 'x y + z -'

    # iteration
    for i in range(iter):
        flt = core.std.Expr([flt, input, fun(flt, **fun_args)], [(calc_expr if i in planes else '') for i in range(input.format.num_planes)])

    return flt


def scale(val, bits):
    '''The old scale function in havsfunc.

    '''

    return val * ((1 << bits) - 1) // 255


def ColorBarsHD(clip=None, width=1288, height=720):
    '''Avisynth's ColorBarsHD()

    It produces a video clip containing SMPTE color bars (Rec. ITU-R BT.709 / arib std b28 v1.0) scaled to any image size.
    By default, a 1288×720, YV24, TV range, 29.97 fps, 1 frame clip is produced.

    Requirment:
        mt_lutspa by tp7 (https://gist.githubusercontent.com/tp7/1e39044e1b660ef0a02c)

    Args:
        clip: 'clip' in std.Blankclip(). The output clip will copy its property.
        width, height: (int) The size of the returned clip.
            Nearest 16:9 pixel exact sizes
            56*X x 12*Y
             728 x  480  ntsc anamorphic
             728 x  576  pal anamorphic
             840 x  480
            1008 x  576
            1288 x  720 <- default
            1456 x 1080  hd anamorphic
            1904 x 1080

    '''

    funcName = 'ColorBarsHD'
    from mt_lutspa import lutspa

    if clip is not None and not isinstance(clip, vs.VideoNode):
        raise TypeError(funcName + ': \"clip\" must be a clip!')

    c = round(width * 3 / 28)
    d = round((width - c * 7) / 2)

    p4 = round(height / 4)
    p23 = round(height / 12)
    p1 = height - p23 * 2 - p4

    blkclip_args = dict(format=vs.YUV444P8, length=1, fpsnum=30000, fpsden=1001)

    pattern1_colors = dict(Gray40=[104, 128, 128], White75=[180, 128, 128], Yellow=[168, 44, 136], Cyan=[145, 147, 44], Green=[134, 63, 52], 
        Magenta=[63, 193, 204], Red=[51, 109, 212], Blue=[28, 212, 120])
    Gray40 = core.std.BlankClip(clip, d, p1, color=pattern1_colors['Gray40'], **blkclip_args)
    White75 = core.std.BlankClip(clip, c, p1, color=pattern1_colors['White75'], **blkclip_args)
    Yellow = core.std.BlankClip(clip, c, p1, color=pattern1_colors['Yellow'], **blkclip_args)
    Cyan = core.std.BlankClip(clip, c, p1, color=pattern1_colors['Cyan'], **blkclip_args)
    Green = core.std.BlankClip(clip, c, p1, color=pattern1_colors['Green'], **blkclip_args)
    Magenta = core.std.BlankClip(clip, c, p1, color=pattern1_colors['Magenta'], **blkclip_args)
    Red = core.std.BlankClip(clip, c, p1, color=pattern1_colors['Red'], **blkclip_args)
    Blue = core.std.BlankClip(clip, c, p1, color=pattern1_colors['Blue'], **blkclip_args)
    pattern1 = core.std.StackHorizontal([Gray40, White75, Yellow, Cyan, Green, Magenta, Red, Blue, Gray40])

    pattern2_colors = dict(Cyan100=[188, 154, 16], plusI=[16, 98, 161], White75=[180, 128, 128], Blue100=[32, 240, 118])
    Cyan100 = core.std.BlankClip(clip, d, p23, color=pattern2_colors['Cyan100'], **blkclip_args)
    plusI = core.std.BlankClip(clip, c, p23, color=pattern2_colors['plusI'], **blkclip_args)
    White75 = core.std.BlankClip(clip, c*6, p23, color=pattern2_colors['White75'], **blkclip_args)
    Blue100 = core.std.BlankClip(clip, d, p23, color=pattern2_colors['Blue100'], **blkclip_args)
    pattern2 = core.std.StackHorizontal([Cyan100, plusI, White75, Blue100])

    pattern3_colors = dict(Yellow100=[219, 16, 138], Red100=[63, 102, 240])
    Yellow100 = core.std.BlankClip(clip, d, p23, color=pattern3_colors['Yellow100'], **blkclip_args)
    Y_Ramp_tmp = core.std.BlankClip(clip, c*7, 1, color=[0, 128, 128], **blkclip_args)
    Y_Ramp = lutspa(Y_Ramp_tmp, mode='absolute', y_expr='220 x * {c} 7 * / 16 +'.format(c=c), chroma='copy')
    Y_Ramp = core.resize.Point(Y_Ramp, c*7, p23)
    Red100 = core.std.BlankClip(clip, d, p23, color=pattern3_colors['Red100'], **blkclip_args)
    pattern3 = core.std.StackHorizontal([Yellow100, Y_Ramp, Red100])

    pattern4_colors = dict(Gray15=[49, 128, 128], Black0=[16, 128, 128], White100=[235, 128, 128], Black_neg2=[12, 128, 128], 
        Black_pos2=[20, 128, 128], Black_pos4=[25, 128, 128])
    Gray15 = core.std.BlankClip(clip, d, p4, color=pattern4_colors['Gray15'], **blkclip_args)
    Black0_1 = core.std.BlankClip(clip, round(c*3/2), p4, color=pattern4_colors['Black0'], **blkclip_args)
    White100 = core.std.BlankClip(clip, c*2, p4, color=pattern4_colors['White100'], **blkclip_args)
    Black0_2 = core.std.BlankClip(clip, round(c*5/6), p4, color=pattern4_colors['Black0'], **blkclip_args)
    Black_neg2 = core.std.BlankClip(clip, round(c/3), p4, color=pattern4_colors['Black_neg2'], **blkclip_args)
    Black0_3 = core.std.BlankClip(clip, round(c/3), p4, color=pattern4_colors['Black0'], **blkclip_args)
    Black_pos2 = core.std.BlankClip(clip, round(c/3), p4, color=pattern4_colors['Black_pos2'], **blkclip_args)
    Black0_4 = Black0_3
    Black_pos4 = core.std.BlankClip(clip, round(c/3), p4, color=pattern4_colors['Black_pos4'], **blkclip_args)
    Black0_5 = core.std.BlankClip(clip, c, p4, color=pattern4_colors['Black0'], **blkclip_args)
    pattern4 = core.std.StackHorizontal([Gray15, Black0_1, White100, Black0_2, Black_neg2, Black0_3, Black_pos2, Black0_4, 
        Black_pos4, Black0_5, Gray15])

    #pattern = core.std.StackVertical([pattern1, pattern2, pattern3, pattern4])
    #return pattern1, pattern2, pattern3, pattern4
    pattern = core.std.StackVertical([pattern1, pattern2, pattern3, pattern4])
    return pattern


def SeeSaw(clp, denoised=None, NRlimit=2, NRlimit2=None, Sstr=1.5, Slimit=None, Spower=4, SdampLo=None, SdampHi=24, Szp=18, bias=49, 
    Smode=None, sootheT=49, sootheS=0, ssx=1.0, ssy=None, diff=False):
    """Avisynth's SeeSaw v0.3e

    Author: Didée (http://avisynth.nl/images/SeeSaw.avs)

    (Full Name: "Denoiser-and-Sharpener-are-riding-the-SeeSaw" )

    This function provides a (simple) implementation of the "crystality sharpen" principle.
    In conjunction with a user-specified denoised clip, the aim is to enhance
    weak detail, hopefully without oversharpening or creating jaggies on strong
    detail, and produce a result that is temporally stable without detail shimmering,
    while keeping everything within reasonable bitrate requirements.
    This is done by intermixing source, denoised source and a modified sharpening process,
    in a seesaw-like manner.

    This version is considered alpha.

    Only the first plane (luma) will be processed.

    Args:
        clp: Input clip; the noisy source.

        deonised: Input clip; denoised clip.
            You're very much encouraged to feed your own custom denoised clip into SeeSaw.
            If the "denoised" clip parameter is omitted, a simple "spatial pressdown" filter is used.

        NRlimit: (int) Absolute limit for pixel change by denoising. Default is 2.

        NRlimit2: (int) Limit for intermediate denoising. Default is NRlimit+1.

        Sstr: (float) Sharpening strength (don't touch this too much). Default is 1.5.

        Slimit: (int) Positive: absolute limit for pixel change by sharpening.
            Negative: pixel's sharpening difference is reduced to diff = pow(diff,1/abs(limit)).
            Default is NRlimit+2.

        Spower: (float) Exponent for modified sharpener. Default is 4.

        Szp: (float) Zero point - below: overdrive sharpening - above: reduced sharpening. Default is 16+2.

        SdampLo: (float) Reduces overdrive sharpening for very small changes. Default is Spower+1.

        SdampHi: (float) Further reduces sharpening for big sharpening changes. Try 15~30. "0" disables. Default is 24.

        bias: (float) Bias towards detail ( >= 50 ), or towards calm result ( < 50 ). Default is 49.

        Smode: (int) RemoveGrain mode used in the modified sharpening function (sharpen2).
            Default: ssx<1.35 ? 11 : ssx<1.51 ? 20 : 19

        sootheT: (int) 0=minimum, 100=maximum soothing of sharpener's temporal instability.
            (-100 .. -1 : will chain 2 instances of temporal soothing.)
            Default is 49.

        sootheS: (int) 0=minimum, 100=maximum smoothing of sharpener's spatial effect. Default is 0.

        ssx, ssy: (int) SeeSaw doesn't require supersampling urgently, if at all, small values ~1.25 seem to be enough. Default is 1.0.

        diff: (bool) When True, limit the sharp-difference instead of the sharpened clip.
                     Relative limiting is more safe, less aliasing, but also less sharpening.
                     
    Usage: (in Avisynth)
        a = TheNoisySource
        b = a.YourPreferredDenoising()
        SeeSaw( a, b, [parameters] )

    """

    funcName = 'SeeSaw'

    if not isinstance(clp, vs.VideoNode) or clp.format.color_family not in [vs.GRAY, vs.YUV, vs.YCOCG]:
        raise TypeError(funcName + ': \"clp\" must be a Gray or YUV clip!')

    isGray = clp.format.color_family == vs.GRAY
    bits = clp.format.bits_per_sample

    if NRlimit2 is None:
        NRlimit2 = NRlimit + 1

    if Slimit is None:
        Slimit = NRlimit + 2

    if SdampLo is None:
        SdampLo = Spower + 1

    if Smode is None:
        if ssx < 1.35:
            Smode = 11
        elif ssx < 1.51:
            Smode = 20
        else:
            Smode = 19

    if ssy is None:
        ssy = ssx

    Szp = Szp / pow(Sstr, 0.25) / pow((ssx + ssy) / 2, 0.5)
    SdampLo = SdampLo / pow(Sstr, 0.25) / pow((ssx + ssy) / 2, 0.5)

    ox = clp.width
    oy = clp.height
    xss = haf.m4(ox * ssx)
    yss = haf.m4(oy * ssy)
    NRL = scale(NRlimit, bits)
    NRL2 = scale(NRlimit2, bits)
    NRLL = scale(round(NRlimit2 * 100 / bias - 1), bits)
    SLIM = scale(Slimit, bits) if Slimit >= 0 else abs(Slimit)
    multiple = 1 << (bits - 8)
    neutral = 1 << (bits - 1)

    if denoised is None:
        dnexpr = 'x {NRL} + y < x {NRL} + x {NRL} - y > x {NRL} - y ? ?'.format(NRL=NRL)
        denoised = core.std.Expr([clp, core.std.Median(clp, [0])], [dnexpr] if isGray else [dnexpr, ''])
    else:
        if not isinstance(denoised, vs.VideoNode):
            raise TypeError(funcName + ': \"denoised\" must be a clip!')
        if denoised.format.id != clp.format.id:
            raise TypeError(funcName + ': \"denoised\" the same format as \"clp\"!')
        if denoised.width != clp.width or denoised.height != clp.height:
            raise TypeError(funcName + ': \"denoised\" must be of the same size as \"clp\"!')

    if not isGray:
        clp_src = clp
        clp = mvf.GetPlane(clp)
        denoised_src = denoised
        denoised = mvf.GetPlane(denoised) if clp_src != denoised_src else clp

    NRdiff = core.std.MakeDiff(clp, denoised)

    tameexpr = 'x {NRLL} + y < x {NRL2} + x {NRLL} - y > x {NRL2} - x {BIAS1} * y {BIAS2} * + 100 / ? ?'.format(NRLL=NRLL, 
        NRL2=NRL2, BIAS1=bias, BIAS2=100-bias)
    tame = core.std.Expr([clp, denoised], [tameexpr])

    head = _SeeSaw_sharpen2(tame, Sstr, Spower, Szp, SdampLo, SdampHi, 4, diff)

    if ssx == 1 and ssy == 1:
        last = core.rgvs.Repair(_SeeSaw_sharpen2(tame, Sstr, Spower, Szp, SdampLo, SdampHi, Smode, diff), head, [1])
    else:
        last = core.rgvs.Repair(_SeeSaw_sharpen2(tame.resize.Lanczos(xss, yss), Sstr, Spower, Szp, SdampLo, SdampHi, Smode, diff), 
            head.resize.Bicubic(xss, yss, filter_param_a=-0.2, filter_param_b=0.6), [1]).resize.Lanczos(ox, oy)
        
    if diff:
        last = core.std.MergeDiff(tame, last)
        
    last = _SeeSaw_SootheSS(last, tame, sootheT, sootheS)
    sharpdiff = core.std.MakeDiff(tame, last)

    if NRlimit == 0 or clp == denoised:
        last = clp
    else:
        NRdiff = core.std.MakeDiff(clp, denoised)
        last = core.std.Expr([clp, NRdiff], ['y {neutral} {NRL} + > x {NRL} - y {neutral} {NRL} - < x {NRL} + x y {neutral} - - ? ?'.format(
            neutral=neutral, NRL=NRL)])

    if Slimit >= 0:
        limitexpr = 'y {neutral} {SLIM} + > x {SLIM} - y {neutral} {SLIM} - < x {SLIM} + x y {neutral} - - ? ?'.format(
            neutral=neutral, SLIM=SLIM)
        last = core.std.Expr([last, sharpdiff], [limitexpr])
    else:
        limitexpr = 'y {neutral} = x x y {neutral} - abs {multiple} / 1 {SLIM} / pow {multiple} * y {neutral} - y {neutral} - abs / * - ?'.format(
            neutral=neutral, SLIM=SLIM, multiple=multiple)
        last = core.std.Expr([last, sharpdiff], [limitexpr])

    return last if isGray else core.std.ShufflePlanes([last, clp_src], list(range(clp_src.format.num_planes)), clp_src.format.color_family)


def _SeeSaw_sharpen2(clp, strength, power, zp, lodmp, hidmp, rg, diff):
    """Modified sharpening function from SeeSaw()

    Only the first plane (luma) will be processed.

    """

    funcName = '_SeeSaw_sharpen2'

    if not isinstance(clp, vs.VideoNode) or clp.format.color_family not in [vs.GRAY, vs.YUV, vs.YCOCG]:
        raise TypeError(funcName + ': \"clp\" must be a Gray or YUV clip!')

    isGray = clp.format.color_family == vs.GRAY
    bits = clp.format.bits_per_sample
    multiple = 1 << (bits - 8)
    neutral = 1 << (bits - 1)
    peak = (1 << bits) - 1

    power = max(power, 1)
    power = 1 / power

    # copied from havsfunc
    def get_lut1(x):
        if x == neutral:
            return x
        else:
            tmp1 = abs(x - neutral) / multiple
            tmp2 = tmp1 ** 2
            tmp3 = zp ** 2
            return min(max(math.floor(neutral + (tmp1 / zp) ** power * zp * (strength * multiple) * (1 if x > neutral else -1) * 
                (tmp2 * (tmp3 + lodmp) / ((tmp2 + lodmp) * tmp3)) * ((1 + (0 if hidmp == 0 else (zp / hidmp) ** 4)) / 
                    (1 + (0 if hidmp == 0 else (tmp1 / hidmp) ** 4))) + 0.5), 0), peak)

    if rg == 4:
        method = clp.std.Median(planes=[0])
    elif rg in [11, 12]:
        method = clp.std.Convolution(matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1], planes=[0])
    elif rg == 19:
        method = clp.std.Convolution(matrix=[1, 1, 1, 1, 0, 1, 1, 1, 1], planes=[0])
    elif rg == 20:
        method = clp.std.Convolution(matrix=[1, 1, 1, 1, 1, 1, 1, 1, 1], planes=[0])
    else:
        method = clp.rgvs.RemoveGrain([rg] if isGray else [rg, 0])

    sharpdiff = core.std.MakeDiff(clp, method, [0]).std.Lut(function=get_lut1, planes=[0])
    
    return sharpdiff if diff else core.std.MergeDiff(clp, sharpdiff, [0])


def _SeeSaw_SootheSS(sharp, orig, sootheT=25, sootheS=0):
    """Soothe() function to stabilze sharpening from SeeSaw()

    Only the first plane (luma) will be processed.

    """

    funcName = '_SeeSaw_SootheSS'

    if not isinstance(sharp, vs.VideoNode) or sharp.format.color_family not in [vs.GRAY, vs.YUV, vs.YCOCG]:
        raise TypeError(funcName + ': \"sharp\" must be a Gray or YUV clip!')

    if not isinstance(orig, vs.VideoNode):
        raise TypeError(funcName + ': \"orig\" must be a clip!')
    if orig.format.id != sharp.format.id:
        raise TypeError(funcName + ': \"orig\" the same format as \"sharp\"!')
    if orig.width != sharp.width or orig.height != sharp.height:
        raise TypeError(funcName + ': \"orig\" must be of the same size as \"sharp\"!')

    sootheT = max(min(sootheT, 100), -100)
    sootheS = max(min(sootheS, 100), 0)
    ST = 100 - abs(sootheT)
    SSPT = 100 - abs(sootheS)
    last = core.std.MakeDiff(orig, sharp, [0])

    neutral = 1 << (sharp.format.bits_per_sample - 1)
    isGray = sharp.format.color_family == vs.GRAY

    if not isGray:
        sharp_src = sharp
        sharp = mvf.GetPlane(sharp)
        orig_src = orig
        orig = mvf.GetPlane(orig) if sharp_src != orig_src else sharp

    expr1 = ('x {neutral} < y {neutral} < xor x {neutral} - 100 / {SSPT} * {neutral} + x {neutral} - ' 
        'abs y {neutral} - abs > x {SSPT} * y {i} * + 100 / x ? ?'.format(neutral=neutral, SSPT=SSPT, i=100-SSPT))
    expr2 = ('x {neutral} < y {neutral} < xor x {neutral} - 100 / {ST} * {neutral} + x {neutral} - ' 
        'abs y {neutral} - abs > x {ST} * y {i} * + 100 / x ? ?'.format(neutral=neutral, ST=ST, i=100-ST))

    if sootheS != 0:
        last = core.std.Expr([last, core.std.Convolution(last, [1]*9)], [expr1])
    if sootheT != 0:
        last = core.std.Expr([last, TemporalSoften(last, 1, 0)], [expr2])
    if sootheT <= -1:
        last = core.std.Expr([last, TemporalSoften(last, 1, 0)], [expr2])

    last = core.std.MakeDiff(orig, last, [0])
    return last if isGray else core.std.ShufflePlanes([last, orig_src], list(range(orig_src.format.num_planes)), orig_src.format.color_family)


def abcxyz(clp, rad=3.0, ss=1.5):
    """Avisynth's abcxyz()

    Reduces halo artifacts that can occur when sharpening.

    Author: Didée (http://avisynth.nl/images/Abcxyz_MT2.avsi)

    Only the first plane (luma) will be processed.

    Args:
        clp: Input clip.

        rad: (float) Radius for halo removal. Default is 3.0.

        ss: (float) Radius for supersampling / ss=1.0 -> no supersampling. Range: 1.0 - ???. Default is 1.5

    """

    funcName = 'abcxyz'

    if not isinstance(clp, vs.VideoNode) or clp.format.color_family not in [vs.GRAY, vs.YUV, vs.YCOCG]:
        raise TypeError(funcName + ': \"clp\" must be a Gray or YUV clip!')

    ox = clp.width
    oy = clp.height

    isGray = clp.format.color_family == vs.GRAY
    bits = clp.format.bits_per_sample

    if not isGray:
        clp_src = clp
        clp = mvf.GetPlane(clp)

    x = core.resize.Bicubic(clp, haf.m4(ox/rad), haf.m4(oy/rad)).resize.Bicubic(ox, oy, filter_param_a=1, filter_param_b=0)
    y = core.std.Expr([clp, x], ['x {a} + y < x {a} + x {b} - y > x {b} - y ? ? x y - abs * x {c} x y - abs - * + {c} /'.format(
        a=scale(8, bits), b=scale(24, bits), c=scale(32, bits))])

    z1 = core.rgvs.Repair(clp, y, [1])

    if ss != 1:
        maxbig = core.std.Maximum(y).resize.Bicubic(haf.m4(ox*ss), haf.m4(oy*ss))
        minbig = core.std.Minimum(y).resize.Bicubic(haf.m4(ox*ss), haf.m4(oy*ss))
        z2 = core.resize.Lanczos(clp, haf.m4(ox*ss), haf.m4(oy*ss))
        z2 = core.std.Expr([z2, maxbig, minbig], ['x y min z max']).resize.Lanczos(ox, oy)
        z1 = z2  # for simplicity

    if not isGray:
        z1 = core.std.ShufflePlanes([z1, clp_src], list(range(clp_src.format.num_planes)), clp_src.format.color_family)

    return z1


def Sharpen(clip, amountH=1.0, amountV=None, planes=None):
    """Avisynth's internel filter Sharpen()

    Simple 3x3-kernel sharpening filter.

    Args:
        clip: Input clip.

        amountH, amountV: (float) Sharpen uses the kernel is [(1-2^amount)/2, 2^amount, (1-2^amount)/2].
            A value of 1.0 gets you a (-1/2, 2, -1/2) for example.
            Negative Sharpen actually blurs the image.
            The allowable range for Sharpen is from -1.58 to +1.0.
            If \"amountV\" is not set manually, it will be set to \"amountH\".
            Default is 1.0.

        planes: (int []) Whether to process the corresponding plane. By default, every plane will be processed.
            The unprocessed planes will be copied from the source clip, "clip".

    """

    funcName = 'Sharpen'

    if not isinstance(clip, vs.VideoNode):
        raise TypeError(funcName + ': \"clip\" is not a clip!')

    if amountH < -1.5849625 or amountH > 1:
        raise ValueError(funcName + ': \'amountH\' have not a correct value! [-1.58 ~ 1]')

    if amountV is None:
        amountV = amountH
    else:
        if amountV < -1.5849625 or amountV > 1:
            raise ValueError(funcName + ': \'amountV\' have not a correct value! [-1.58 ~ 1]')

    if planes is None:
        planes = list(range(clip.format.num_planes))

    center_weight_v = math.floor(2 ** (amountV - 1) * 1023 + 0.5)
    outer_weight_v = math.floor((0.25 - 2 ** (amountV - 2)) * 1023 + 0.5)
    center_weight_h = math.floor(2 ** (amountH - 1) * 1023 + 0.5)
    outer_weight_h = math.floor((0.25 - 2 ** (amountH - 2)) * 1023 + 0.5)

    conv_mat_v = [outer_weight_v, center_weight_v, outer_weight_v]
    conv_mat_h = [outer_weight_h, center_weight_h, outer_weight_h]

    if math.fabs(amountH) >= 0.00002201361136: # log2(1+1/65536)
        clip = core.std.Convolution(clip, conv_mat_v, planes=planes, mode='v')

    if math.fabs(amountV) >= 0.00002201361136:
        clip = core.std.Convolution(clip, conv_mat_h, planes=planes, mode='h')

    return clip


def Blur(clip, amountH=1.0, amountV=None, planes=None):
    """Avisynth's internel filter Blur()

    Simple 3x3-kernel blurring filter.

    In fact Blur(n) is just an alias for Sharpen(-n).

    Args:
        clip: Input clip.

        amountH, amountV: (float) Blur uses the kernel is [(1-1/2^amount)/2, 1/2^amount, (1-1/2^amount)/2].
            A value of 1.0 gets you a (1/4, 1/2, 1/4) for example.
            Negative Blur actually sharpens the image.
            The allowable range for Blur is from -1.0 to +1.58.
            If \"amountV\" is not set manually, it will be set to \"amountH\".
            Default is 1.0.

        planes: (int []) Whether to process the corresponding plane. By default, every plane will be processed.
            The unprocessed planes will be copied from the source clip, "clip".

    """

    funcName = 'Blur'

    if not isinstance(clip, vs.VideoNode):
        raise TypeError(funcName + ': \"clip\" is not a clip!')

    if amountH < -1 or amountH > 1.5849625:
        raise ValueError(funcName + ': \'amountH\' have not a correct value! [-1 ~ 1.58]')

    if amountV is None:
        amountV = amountH
    else:
        if amountV < -1 or amountV > 1.5849625:
            raise ValueError(funcName + ': \'amountV\' have not a correct value! [-1 ~ 1.58]')

    return Sharpen(clip, -amountH, -amountV, planes)


def BlindDeHalo3(clp, rx=3.0, ry=3.0, strength=125, lodamp=0, hidamp=0, sharpness=0, tweaker=0, PPmode=0, PPlimit=None, interlaced=False):
    """Avisynth's BlindDeHalo3() version: 3_MT2

    This script removes the light & dark halos from too strong "Edge Enhancement".

    Author: Didée (https://forum.doom9.org/attachment.php?attachmentid=5599&d=1143030001)

    Only the first plane (luma) will be processed.

    Args:
        clp: Input clip.

        rx, ry: (float) The radii to use for the [quasi-] Gaussian blur, on which the halo removal is based. Default is 3.0.

        strength: (float) The overall strength of the halo removal effect. Default is 125.

        lodamp, hidamp: (float) With these two values, one can reduce the basic effect on areas that would change only little anyway (lodamp),
            and/or on areas that would change very much (hidamp).
            lodamp does a reasonable job in keeping more detail in affected areas.
            hidamp is intended to keep rather small areas that are very bright or very dark from getting processed too strong.
            Works OK on sources that contain only weak haloing - for sources with strong over sharpening,
                it should not be used, mostly. (Usage has zero impact on speed.)
            Range: 0.0 to ??? (try 4.0 as a start)
            Default is 0.0.

        sharpness: (float) By setting this bigger than 0.0, the affected areas will come out with better sharpness.
            However, strength must be chosen somewhat bigger as well, then, to get the same effect than without.
            (This is the same as initial version's "maskblur" option.)
            Range: 0.0 to 1.58.
            Default is 0.

        tweaker: (float) May be used to get a stronger effect, separately from altering "strength".
            (Also in accordance to initial version's working methodology. I had no better idea for naming this parameter.)
            Range: 0.0 - 1.00.
            Default is 0.

        PPmode: (int) When set to "1" or "2", a second cleaning operation after the basic halo removal is done.
            This deals with:
                a) Removing/reducing those corona lines that sometimes are left over by BlindDeHalo
                b) Improving on mosquito noise, if some is present.
            PPmode=1 uses a simple Gaussian blur for post-cleaning. PPmode=2 uses a 3*3 average, with zero weighting of the center pixel.
            Also, PPmode can be "-1" or "-2". In this case, the main dehaloing step is completely discarded, and *only* the PP cleaning is done.
            This has less effect on halos, but can deal for sources containing more mosquito noise than halos.
            Default is 0.

        PPlimit: (int) Can be used to make the PP routine change no pixel by more than [PPlimit].
            I'm not sure if this makes much sense in this context. However the option is there - you never know what it might be good for.
            Default is 0.

        interlaced: (bool) As formerly, this is intended for sources that were originally interlaced, but then made progressive by deinterlacing.
            It aims in particular at clips that made their way through Restore24.
            Default is False.

    """

    funcName = 'BlindDeHalo3'

    if not isinstance(clp, vs.VideoNode):
        raise TypeError(funcName + ': \"clp\" is not a clip!')

    if clp.format.sample_type != vs.INTEGER:
        raise TypeError(funcName + ': Only integer clip is supported!')

    if PPlimit is None:
        PPlimit = 4 if abs(PPmode) == 3 else 0

    bits = clp.format.bits_per_sample
    isGray = clp.format.color_family == vs.GRAY
    neutral = 1 << (bits - 1)

    if not isGray:
        clp_src = clp
        clp = mvf.GetPlane(clp)

    sharpness = min(sharpness, 1.58)
    tweaker = min(tweaker, 1.0)
    strength *= 1 + sharpness * 0.25
    RR = (rx + ry) / 2
    ST = strength / 100
    LD = scale(lodamp, bits)
    HD = hidamp ** 2
    TWK0 = 'x y - {i} /'.format(i=12 / ST / RR)
    TWK = 'x y - {i} / abs'.format(i=12 / ST / RR)
    TWK_HLIGHT = ('x y - abs {i} < {neutral} {TWK} {neutral} {TWK} - {TWK} {neutral} / * + {TWK0} {TWK} {LD} + / * '
        '{neutral} {TWK} - {j} / dup * {neutral} {TWK} - {j} / dup * {HD} + / * {neutral} + ?'.format(
            i=1 << (bits-8), neutral=neutral, TWK=TWK, TWK0=TWK0, LD=LD, j=scale(20, bits), HD=HD))

    i = clp if not interlaced else core.std.SeparateFields(clp, tff=True)
    oxi = i.width
    oyi = i.height
    sm = core.resize.Bicubic(i, haf.m4(oxi/rx), haf.m4(oyi/ry))
    mm = core.std.Expr([sm.std.Maximum(), sm.std.Minimum()], ['x y - 4 *']).std.Maximum().std.Deflate().std.Convolution([1]*9)
    mm = mm.std.Inflate().resize.Bicubic(oxi, oyi, filter_param_a=1, filter_param_b=0).std.Inflate()
    sm = core.resize.Bicubic(sm, oxi, oyi, filter_param_a=1, filter_param_b=0)
    smd = core.std.Expr([Sharpen(i, tweaker), sm], [TWK_HLIGHT])
    if sharpness != 0:
        smd = Blur(smd, sharpness)
    clean = core.std.Expr([i, smd], ['x y {neutral} - -'.format(neutral=neutral)])
    clean = core.std.MaskedMerge(i, clean, mm)

    if PPmode != 0:
        LL = scale(PPlimit, bits)
        LIM = 'x {LL} + y < x {LL} + x {LL} - y > x {LL} - y ? ?'.format(LL=LL)

        base = i if PPmode < 0 else clean
        small = core.resize.Bicubic(base, haf.m4(oxi / math.sqrt(rx * 1.5)), haf.m4(oyi / math.sqrt(ry * 1.5)))
        ex1 = Blur(small.std.Maximum(), 0.5)
        in1 = Blur(small.std.Minimum(), 0.5)
        hull = core.std.Expr([ex1.std.Maximum().std.Convolution(matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1]), ex1, in1, 
            in1.std.Minimum().std.Convolution(matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1])], 
            ['x y - {i} - 5 * z a - {i} - 5 * max'.format(i=1 << (bits-8))]).resize.Bicubic(oxi, oyi, filter_param_a=1, filter_param_b=0)

        if abs(PPmode) == 1:
            postclean = core.std.MaskedMerge(base, small.resize.Bicubic(oxi, oyi, filter_param_a=1, filter_param_b=0), hull)
        elif abs(PPmode) == 2:
            postclean = core.std.MaskedMerge(base, base.std.Convolution(matrix=[1, 1, 1, 1, 0, 1, 1, 1, 1]), hull)
        elif abs(PPmode) == 3:
            postclean = core.std.MaskedMerge(base, base.std.Median(), hull)
        else:
            raise ValueError(funcName + ': \"PPmode\" must be in [-3 ... 3]!')
    else:
        postclean = clean

    if PPlimit != 0:
        postclean = core.std.Expr([base, postclean], [LIM])

    last = haf.Weave(postclean, tff=True) if interlaced else postclean

    if not isGray:
        last = core.std.ShufflePlanes([last, clp_src], list(range(clp_src.format.num_planes)), clp_src.format.color_family)

    return last


def dfttestMC(input, pp=None, mc=2, mdg=False, planes=None, sigma=None, sbsize=None, sosize=None, tbsize=None, mdgSAD=None, 
    thSAD=None, thSCD1=None, thSCD2=None, pel=None, blksize=None, search=None, searchparam=None, overlap=2, dct=None, **dfttest_params):
    """Avisynth's dfttestMC

    Motion-compensated dfttest
    Aka: Really Really Really Slow

    Author: thewebchat (https://forum.doom9.org/showthread.php?p=1295788#post1295788)

    Notes:
        \"lsb\" and \"dither\" are removed. The output always has the same bitdepth as input.
        "Y", "U" and "V" are replaced by "planes".
        "dfttest_params" is removed. Additional arguments will be passed to DFTTest by keyword arguments.
        mc can be 0, and the function will simply be a pure dfttest().

    Args:

        input: Input clip.

        pp: (clip) Clip to calculate vectors from. Default is \"input\".

        mc: (int) Number of frames in each direction to compensate. Range: 0 ~ 5. Default is 2.

        mdg: (bool) Run MDeGrain before dfttest. Default is False.

        mdgSAD: (int) thSAD for MDeGrain. Default is undefined.

        dfttest's sigma, sbsize, sosize and tbsize are supported. Extra dfttest parameters may be passed via "dfttest_params".

        pel, thSCD, thSAD, blksize, overlap, dct, search, and searchparam are also supported.

        sigma is the main control of dfttest strength.
        tbsize should not be set higher than mc * 2 + 1.

    """

    funcName = 'dfttestMC'

    if not isinstance(input, vs.VideoNode) or input.format.color_family not in [vs.GRAY, vs.YUV, vs.YCOCG]:
        raise TypeError(funcName + ': \"input\" must be a Gray or YUV clip!')

    if pp is not None:
        if not isinstance(pp, vs.VideoNode):
            raise TypeError(funcName + ': \"pp\" must be a clip!')
        if input.format.id != pp.format.id:
            raise TypeError(funcName + ': \"pp\" must be of the same format as \"input\"!')
        if input.width != pp.width or input.height != pp.height:
            raise TypeError(funcName + ': \"pp\" must be of the same size as \"input\"!')

    # Set default options. Most external parameters are passed valueless.
    if dfttest_params is None:
        dfttest_params = {}

    mc = min(mc, 5)

    # Set chroma parameters.
    if planes is None:
        planes = list(range(input.format.num_planes))
    elif not isinstance(planes, dict):
        planes = [planes]

    Y = 0 in planes
    U = 1 in planes
    V = 2 in planes

    chroma = U or V

    if not Y and U and not V:
        plane = 1
    elif not Y and not U and V:
        plane = 2
    elif not Y and chroma:
        plane = 3
    elif Y and chroma:
        plane = 4
    else:
        plane = 0

    # Prepare supersampled clips.
    pp_enabled = pp is not None
    pp_super = haf.DitherLumaRebuild(pp if pp_enabled else input, s0=1, chroma=chroma).mv.Super(pel=pel, chroma=chroma)
    super = haf.DitherLumaRebuild(input, s0=1, chroma=chroma).mv.Super(pel=pel, chroma=chroma) if pp_enabled and input != pp else pp_super

    # Motion vector search.
    analysis_args = dict(chroma=chroma, search=search, searchparam=searchparam, overlap=overlap, blksize=blksize, dct=dct)
    bvec = []
    fvec = []

    for i in range(1, mc+1):
        bvec.append(core.mv.Analyse(pp_super, delta=i, isb=True, **analysis_args))
        fvec.append(core.mv.Analyse(pp_super, delta=i, isb=False, **analysis_args))

    # Optional MDegrain.
    if mdg:
        degrain_args = dict(thsad=mdgSAD, plane=plane, thscd1=thSCD1, thscd2=thSCD2)
        if mc >= 3:
            degrained = core.mv.Degrain3(input, super, bvec[0], fvec[0], bvec[1], fvec[1], bvec[2], fvec[2], **degrain_args)
        elif mc == 2:
            degrained = core.mv.Degrain2(input, super, bvec[0], fvec[0], bvec[1], fvec[1], **degrain_args)
        elif mc == 1:
            degrained = core.mv.Degrain1(input, super, bvec[0], fvec[0], **degrain_args)
        else:
            degrained = input
    else:
        degrained = input

    # Motion Compensation.
    degrained_super = haf.DitherLumaRebuild(degrained, s0=1, chroma=chroma).mv.Super(pel=pel, levels=1, chroma=chroma) if mdg else super
    compensate_args = dict(thsad=thSAD, thscd1=thSCD1, thscd2=thSCD2)
    bclip = []
    fclip = []
    for i in range(1, mc+1):
        bclip.append(core.mv.Compensate(degrained, degrained_super, bvec[i-1], **compensate_args))
        fclip.append(core.mv.Compensate(degrained, degrained_super, fvec[i-1], **compensate_args))

    # Create compensated clip.
    fclip.reverse()
    interleaved = core.std.Interleave(fclip + [degrained] + bclip) if mc >= 1 else degrained

    # Perform dfttest.
    filtered = core.dfttest.DFTTest(interleaved, sigma=sigma, sbsize=sbsize, sosize=sosize, tbsize=tbsize, **dfttest_params)

    return core.std.SelectEvery(filtered, mc * 2 + 1, mc) if mc > 1 else filtered


def TurnLeft(clip):
    """Avisynth's internel function TurnLeft()"""

    return core.std.Transpose(clip).std.FlipVertical()


def TurnRight(clip):
    """Avisynth's internel function TurnRight()"""

    return core.std.FlipVertical(clip).std.Transpose()


def BalanceBorders(c, cTop=0, cBottom=0, cLeft=0, cRight=0, thresh=128, blur=999):
    """Avisynth's BalanceBorders() Version: v0.2

    Author: PL (https://www.dropbox.com/s/v8fm6om7hm1dz0b/BalanceBorders.avs)

    The following documentaion is mostly translated by Google Translate from Russian.

    The function changes the values of the extreme pixels of the clip,
    so that they are "more similar" to the neighboring ones,
    which, perhaps, will prevent the "strong" use of Crop () to remove the "unpleasant edges"
    that are not very different from the "main" image.

    Args:
        c: Input clip. The image area "in the middle" does not change during processing.
            The clip can be any format, which differs from Avisynth's equivalent.

        cTop, cBottom, cLeft, cRight: (int) The number of variable pixels on each side.
            There will not be anything very terrible if you specify values that are greater than the minimum required in your case,
            but to achieve a good result, "it is better not to" ...
            Range: 0 will skip the processing. For RGB input, the range is 2~inf.
                For YUV or YCbCr input, the minimum accepted value depends on chroma subsampling.
                Specifically, for YV24, the range is also 2~inf. For YV12, the range is 4~inf.
            Default is 0.

        thresh: (int) Threshold of acceptable changes for local color matching in 8 bit scale.
            Range: 0~128. Recommend: [0~16 or 128].
            Default is 128.

        blur: (int) Degree of blur for local color matching.
            Smaller values give a more accurate color match,
            larger values give a more accurate picture transfer.
            Range: 1~inf. Recommend: [1~20 or 999].
            Default is 999.

    Notes:
        1) At default values ​​of thresh = 128 blur = 999,
            you will get a series of pixels that have been changed only by selecting the color for each row in its entirety, without local selection;
            The colors of neighboring pixels may be very different in some places, but there will be no change in the nature of the picture.

            And with thresh = 128 and blur = 1 you get almost the same rows of pixels,
            i.e. The colors between them will coincide completely, but the original pattern will be lost.

        2) Beware of using a large number of pixels to change in combination with a high level of "thresh",
            and a small "blur" that can lead to unwanted artifacts "in a clean place".
            For each function call, try to set as few pixels as possible to change and as low a threshold as possible "thresh" (when using blur 0..16).

    Examples:
        The variant of several calls of the order:
        last = muf.BalanceBorders(last, 7, 6, 4, 4)                    # "General" color matching
        last = muf.BalanceBorders(last, 5, 5, 4, 4, thresh=2, blur=10) # Very slightly changes a large area (with a "margin")
        last = muf.BalanceBorders(last, 3, 3, 2, 2, thresh=8, blur=4)  # Slightly changes the "main problem area"

    """

    funcName = 'BalanceBorders'

    if not isinstance(c, vs.VideoNode):
        raise TypeError(funcName + ': \"c\" must be a clip!')

    if c.format.sample_type != vs.INTEGER:
        raise TypeError(funcName+': \"c\" must be integer format!')

    if blur <= 0:
        raise ValueError(funcName + ': \'blur\' have not a correct value! (0 ~ inf]')

    if thresh <= 0:
        raise ValueError(funcName + ': \'thresh\' have not a correct value! (0 ~ inf]')

    last = c

    if cTop > 0:
        last = _BalanceTopBorder(last, cTop, thresh, blur)

    last = TurnRight(last)

    if cLeft > 0:
        last = _BalanceTopBorder(last, cLeft, thresh, blur)

    last = TurnRight(last)

    if cBottom > 0:
        last = _BalanceTopBorder(last, cBottom, thresh, blur)

    last = TurnRight(last)

    if cRight > 0:
        last = _BalanceTopBorder(last, cRight, thresh, blur)

    last = TurnRight(last)

    return last


def _BalanceTopBorder(c, cTop, thresh, blur):
    """BalanceBorders()'s helper function"""

    cWidth = c.width
    cHeight = c.height
    cTop = min(cTop, cHeight - 1)
    blurWidth = max(4, math.floor(cWidth / blur))

    c2 = mvf.PointPower(c, 1, 1)

    last = core.std.Crop(c2, 0, 0, cTop*2, (cHeight - cTop - 1) * 2)
    last = core.resize.Point(last, cWidth * 2, cTop * 2)
    last = core.resize.Bilinear(last, blurWidth * 2, cTop * 2)
    last = core.std.Convolution(last, [1, 1, 1], mode='h')
    last = core.resize.Bilinear(last, cWidth * 2, cTop * 2)
    referenceBlur = last

    original = core.std.Crop(c2, 0, 0, 0, (cHeight - cTop) * 2)

    last = original
    last = core.resize.Bilinear(last, blurWidth * 2, cTop * 2)
    last = core.std.Convolution(last, [1, 1, 1], mode='h')
    last = core.resize.Bilinear(last, cWidth * 2, cTop * 2)
    originalBlur = last

    """
    balanced = core.std.Expr([original, originalBlur, referenceBlur], ['z y - x +'])
    difference = core.std.MakeDiff(balanced, original)

    tp = scale(128 + thresh, c.format.bits_per_sample)
    tm = scale(128 - thresh, c.format.bits_per_sample)
    difference = core.std.Expr([difference], ['x {tp} min {tm} max'.format(tp=tp, tm=tm)])

    last = core.std.MergeDiff(original, difference)
    """
    tp = scale(thresh, c.format.bits_per_sample)
    tm = -tp
    last = core.std.Expr([original, originalBlur, referenceBlur], ['z y - {tp} min {tm} max x +'.format(tp=tp, tm=tm)])

    return core.std.StackVertical([last, core.std.Crop(c2, 0, 0, cTop * 2, 0)]).resize.Point(cWidth, cHeight)


def DisplayHistogram(clip, factor=None):
    """A simple function to display the histogram of an image.

    The right and bottom of the output is the histogram along the horizontal/vertical axis,
    with the left(bottom) side of the graph represents luma=0 and the right(above) side represents luma=255.
    The bottom right is hist.Levels().

    More details of the graphs can be found at http://avisynth.nl/index.php/Histogram.

    Args:
        clip: Input clip. Must be constant format 8..16 bit integer YUV input.
            If the input's bitdepth is not 8, input will be converted to 8 bit before passing to hist.Levels().

        factor: (float) hist.Levels()'s argument.
            It specifies how the histograms are displayed, exaggerating the vertical scale.
            It is specified as percentage of the total population (that is number of luma or chroma pixels in a frame).
            Range: 0~100. Default is 100.

    """

    funcName = 'DisplayHistogram'

    if not isinstance(clip, vs.VideoNode):
        raise TypeError(funcName + ': \"clip\" must be a clip!')

    if clip.format.sample_type != vs.INTEGER or clip.format.bits_per_sample > 16 or clip.format.color_family != vs.YUV:
        raise TypeError(funcName+': \"clip\" must be 8..16 integer YUV format!')

    histogram_v = core.hist.Classic(clip)

    clip_8 = mvf.Depth(clip, 8)
    levels = core.hist.Levels(clip_8, factor=factor).std.Crop(left=clip.width, right=0, top=0, bottom=clip.height - 256)
    if clip.format.bits_per_sample != 8:
        levels = mvf.Depth(levels, clip.format.bits_per_sample)
    histogram_h = TurnLeft(core.hist.Classic(clip.std.Transpose()).std.Crop(left=clip.height))

    bottom = core.std.StackHorizontal([histogram_h, levels])

    return core.std.StackVertical([histogram_v, bottom])


def GuidedFilter(input, guidance=None, radius=4, regulation=0.01, regulation_mode=0, use_gauss=False, fast=None, subsampling_ratio=4, 
    use_fmtc1=False, kernel1='point', kernel1_args=None, use_fmtc2=False, kernel2='bilinear', kernel2_args=None, **depth_args):
    """Guided Filter - fast edge-preserving smoothing algorithm

    Author: Kaiming He et al. (http://kaiminghe.com/eccv10/)

    The guided filter computes the filtering output by considering the content of a guidance image.

    It can be used as an edge-preserving smoothing operator like the popular bilateral filter,
    but it has better behaviors near edges.

    The guided filter is also a more generic concept beyond smoothing:
    It can transfer the structures of the guidance image to the filtering output,
    enabling new filtering applications like detail enhancement, HDR compression,
    image matting/feathering, dehazing, joint upsampling, etc.

    All the internal calculations are done at 32-bit float.

    Args:
        input: Input clip.

        guidance: (clip) Guidance clip used to compute the coefficient of the linear translation on 'input'.
            It must has the same clip properties as 'input'.
            If it is None, it will be set to input, with duplicate calculations being omitted.
            Default is None.

        radius: (int) Box / Gaussian filter's radius.
            If box filter is used, the range of radius is 1 ~ 12(fast=False) or 1 ~ 12*subsampling_ratio in VapourSynth R38 or older 
                because of the limitation of std.Convolution().
            For gaussian filter, the radius can be much larger, even reaching the width/height of the clip.
            Default is 4.

        regulation: (float) A criterion for judging whether a patch has high variance and should be preserved, or is flat and should be smoothed.
            Similar to the range variance in the bilateral filter.
            Default is 0.01.

        regulation_mode: (int) Tweak on regulation.
            It was mentioned in [1] that the local filters such as the Bilateral Filter (BF) or Guided Image Filter (GIF)
            would concentrate the blurring near these edges and introduce halos.

            The author of Weighted Guided Image Filter (WGIF) [3] argued that,
            the Lagrangian factor (regulation) in the GIF is fixed could be another major reason that the GIF produces halo artifacts.

            In [3], a WGIF was proposed to reduce the halo artifacts of the GIF.
            An edge aware factor was introduced to the constraint term of the GIF,
            the factor makes the edges preserved better in the result images and thus reduces the halo artifacts.

            In [4], a gradient domain guided image filter is proposed by incorporating an explicit first-order edge-aware constraint.
            The proposed filter is based on local optimization
            and the cost function is composed of a zeroth order data fidelity term and a first order regularization term.
            So the factors in the new local linear model can represent the images more accurately near edges.
            In addition, the edge-aware factor is multi-scale, which can separate edges of an image from fine details of the image better.

            0: Guided Filter [1]
            1: Weighted Guided Image Filter [3]
            2: Gradient Domain Guided Image Filter [4]
            Default is 0.

        use_gauss: (bool) Whether to use gaussian guided filter [1]. This replaces mean filter with gaussian filter.
            Guided filter is rotationally asymmetric and slightly biases to the x/y-axis because a box window is used in the filter design.
            The problem can be solved by using a gaussian weighted window instead. The resulting kernels are rotationally symmetric.
            The authors of [1] suggest that in practice the original guided filter is always good enough.
            Gaussian is performed by core.tcanny.TCanny(mode=-1).
            The sigma is set to r/sqrt(2).
            Default is False.

        fast: (bool) Whether to use fast guided filter [2].
            This method subsamples the filtering input image and the guidance image,
            computes the local linear coefficients, and upsamples these coefficients.
            The upsampled coefficients are adopted on the original guidance image to produce the output.
            This method reduces the time complexity from O(N) to O(N/s^2) for a subsampling ratio s.
            Default is True if the version number of VapourSynth is less than 39, otherwise is False.

        subsampling_ratio: (float) Only works when fast=True.
            Generally should be no less than 'radius'.
            Default is 4.

        use_fmtc1, use_fmtc2: (bool) Whether to use fmtconv in subsampling / upsampling.
            Default is False.
            Note that fmtconv's point subsampling may causes pixel shift.

        kernel1, kernel2: (string) Subsampling/upsampling kernels.
            Default is 'point'and 'bilinear'.

        kernel1_args, kernel2_args: (dict) Additional parameters passed to resizers in the form of dict.
            Default is {}.

        depth_args: (dict) Additional arguments passed to mvf.Depth() in the form of keyword arguments.
            Default is {}.

    Ref:
        [1] He, K., Sun, J., & Tang, X. (2013). Guided image filtering. 
            IEEE transactions on pattern analysis and machine intelligence, 35(6), 1397-1409.
        [2] He, K., & Sun, J. (2015). Fast guided filter. arXiv preprint arXiv:1505.00996.
        [3] http://kaiminghe.com/eccv10/index.html
        [4] Li, Z., Zheng, J., Zhu, Z., Yao, W., & Wu, S. (2015). Weighted guided image filtering. 
            IEEE Transactions on Image Processing, 24(1), 120-129.
        [5] Kou, F., Chen, W., Wen, C., & Li, Z. (2015). Gradient domain guided image filtering. 
            IEEE Transactions on Image Processing, 24(11), 4528-4539.
        [6] http://koufei.weebly.com/

    """

    funcName = 'GuidedFilter'

    if not isinstance(input, vs.VideoNode):
        raise TypeError(funcName + ': \"input\" must be a clip!')

    # Get clip's properties
    bits = input.format.bits_per_sample
    sampleType = input.format.sample_type
    width = input.width
    height = input.height

    if guidance is not None:
        if not isinstance(guidance, vs.VideoNode):
            raise TypeError(funcName + ': \"guidance\" must be a clip!')
        if input.format.id != guidance.format.id:
            raise TypeError(funcName + ': \"guidance\" must be of the same format as \"input\"!')
        if input.width != guidance.width or input.height != guidance.height:
            raise TypeError(funcName + ': \"guidance\" must be of the same size as \"input\"!')
        if input == guidance: # Remove redundant computation
            guidance = None

    if fast is None:
        fast = False if core.version_number() >= 39 else True

    if kernel1_args is None:
        kernel1_args = {}
    if kernel2_args is None:
        kernel2_args = {}

    # Bitdepth conversion and variable names modification to correspond to the paper
    p = mvf.Depth(input, depth=32, sample=vs.FLOAT, **depth_args)
    I = mvf.Depth(guidance, depth=32, sample=vs.FLOAT, **depth_args) if guidance is not None else p
    r = radius
    eps = regulation
    s = subsampling_ratio

    # Back up guidance image
    I_src = I

    # Fast guided filter's subsampling
    if fast:
        down_w = math.floor(width / s)
        down_h = math.floor(height / s)
        if use_fmtc1:
            p = core.fmtc.resample(p, down_w, down_h, kernel=kernel1, **kernel1_args)
            I = core.fmtc.resample(I, down_w, down_h, kernel=kernel1, **kernel1_args) if guidance is not None else p
        else: # use zimg
            p = eval('core.resize.{kernel}(p, down_w, down_h, **kernel1_args)'.format(kernel=kernel1.capitalize()))
            I = eval('core.resize.{kernel}(I, down_w, down_h, **kernel1_args)'.format(kernel=kernel1.capitalize())) if guidance is not None else p

        r = math.floor(r / s)

    # Select the shape of the kernel. As the width of BoxFilter in this module is (radius*2-1) rather than (radius*2+1), radius should be increased by one.
    Filter = functools.partial(core.tcanny.TCanny, sigma=r/2 * math.sqrt(2), mode=-1) if use_gauss else functools.partial(BoxFilter, radius=r+1)
    Filter_r1 = functools.partial(core.tcanny.TCanny, sigma=1/2 * math.sqrt(2), mode=-1) if use_gauss else functools.partial(BoxFilter, radius=1+1)


    # Edge-Aware Weighting, equation (5) in [3], or equation (9) in [4].
    def FLT(n, f, clip, core, eps0):
        frameMean = f.props['PlaneStatsAverage']

        return core.std.Expr(clip, ['x {eps0} + {avg} *'.format(avg=frameMean, eps0=eps0)])


    # Compute the optimal value of a of Gradient Domain Guided Image Filter, equation (12) in [4]
    def FLT2(n, f, cov_Ip, weight_in, weight, var_I, core, eps):
        frameMean = f.props['PlaneStatsAverage']
        frameMin = f.props['PlaneStatsMin']

        alpha = frameMean
        kk = -4 / (frameMin - alpha - 1e-6) # Add a small num to prevent divided by 0

        return core.std.Expr([cov_Ip, weight_in, weight, var_I], 
            ['x {eps} 1 1 1 {kk} y {alpha} - * exp + / - * z / + a {eps} z / + /'.format(eps=eps, kk=kk, alpha=alpha)])

    # Compute local linear coefficients.
    mean_p = Filter(p)
    mean_I = Filter(I) if guidance is not None else mean_p
    I_square = core.std.Expr([I], ['x dup *'])
    corr_I = Filter(I_square)
    corr_Ip = Filter(core.std.Expr([I, p], ['x y *'])) if guidance is not None else corr_I

    var_I = core.std.Expr([corr_I, mean_I], ['x y dup * -'])
    cov_Ip = core.std.Expr([corr_Ip, mean_I, mean_p], ['x y z * -']) if guidance is not None else var_I

    if regulation_mode: # 0: Original Guided Filter, 1: Weighted Guided Image Filter, 2: Gradient Domain Guided Image Filter
        if r != 1:
            mean_I_1 = Filter_r1(I)
            corr_I_1 = Filter_r1(I_square)
            var_I_1 = core.std.Expr([corr_I_1, mean_I_1], ['x y dup * -'])
        else: # r == 1
            var_I_1 = var_I

        if regulation_mode == 1: # Weighted Guided Image Filter
            weight_in = var_I_1
        else: # regulation_mode == 2, Gradient Domain Guided Image Filter
            weight_in = core.std.Expr([var_I, var_I_1], ['x y * sqrt'])

        eps0 = 0.001 ** 2 # Epsilon in [3] and [4]
        denominator = core.std.Expr([weight_in], ['1 x {} + /'.format(eps0)])

        denominator = core.std.PlaneStats(denominator, plane=[0])
        # equation (5) in [3], or equation (9) in [4]
        weight = core.std.FrameEval(denominator, functools.partial(FLT, clip=weight_in, core=core, eps0=eps0), prop_src=[denominator])

        if regulation_mode == 1: # Weighted Guided Image Filter
            a = core.std.Expr([cov_Ip, var_I, weight], ['x y {eps} z / + /'.format(eps=eps)])
        else: # regulation_mode == 2, Gradient Domain Guided Image Filter
            weight_in = core.std.PlaneStats(weight_in, plane=[0])
            a = core.std.FrameEval(weight, functools.partial(FLT2, cov_Ip=cov_Ip, weight_in=weight_in, weight=weight, 
                var_I=var_I, core=core, eps=eps), prop_src=[weight_in])
    else: # regulation_mode == 0, Original Guided Filter
        a = core.std.Expr([cov_Ip, var_I], ['x y {} + /'.format(eps)])

    b = core.std.Expr([mean_p, a, mean_I], ['x y z * -'])

    mean_a = Filter(a)
    mean_b = Filter(b)

    # Fast guided filter's upsampling
    if fast:
        if use_fmtc2:
            mean_a = core.fmtc.resample(mean_a, width, height, kernel=kernel2, **kernel2_args)
            mean_b = core.fmtc.resample(mean_b, width, height, kernel=kernel2, **kernel2_args)
        else: # use zimg
            mean_a = eval('core.resize.{kernel}(mean_a, {w}, {h}, **kernel2_args)'.format(kernel=kernel2.capitalize(), w=width, h=height))
            mean_b = eval('core.resize.{kernel}(mean_b, {w}, {h}, **kernel2_args)'.format(kernel=kernel2.capitalize(), w=width, h=height))

    # Linear translation
    q = core.std.Expr([mean_a, I_src, mean_b], ['x y * z +'])

    # Final bitdepth conversion
    return mvf.Depth(q, depth=bits, sample=sampleType, **depth_args)


def GuidedFilterColor(input, guidance, radius=4, regulation=0.01, use_gauss=False, fast=None, subsampling_ratio=4, use_fmtc1=False, kernel1='point', 
    kernel1_args=None, use_fmtc2=False, kernel2='bilinear', kernel2_args=None, **depth_args):
    """Guided Filter Color - fast edge-preserving smoothing algorithm using a color image as the guidance

    Author: Kaiming He et al. (http://kaiminghe.com/eccv10/)

    Most of the description of the guided filter can be found in the documentation of native guided filter above.
    Only the native guided filter is implemented.

    A color guidance image can better preserve the edges that are not distinguishable in gray-scale.

    It is also essential in the matting/feathering and dehazing applications,
    because the local linear model is more likely to be valid in the RGB color space than in gray-scale.

    Args:
        input: Input clip. It should be a gray-scale/single channel image.

        guidance: Guidance clip used to compute the coefficient of the linear translation on 'input'.
            It must has no subsampling for the second and third plane in horizontal/vertical direction, e.g. RGB or YUV444.

        Descriptions of other parameter can be found in the documentation of native guided filter above.

    Ref:
        [1] He, K., Sun, J., & Tang, X. (2013). Guided image filtering. IEEE transactions on pattern analysis and machine intelligence, 35(6), 1397-1409.
        [2] He, K., & Sun, J. (2015). Fast guided filter. arXiv preprint arXiv:1505.00996.
        [3] http://kaiminghe.com/eccv10/index.html

    """

    funcName = 'GuidedFilterColor'

    if not isinstance(input, vs.VideoNode) or input.format.num_planes > 1:
        raise TypeError(funcName + ': \"input\" must be a gray-scale/single channel clip!')

    # Get clip's properties
    bits = input.format.bits_per_sample
    sampleType = input.format.sample_type
    width = input.width
    height = input.height

    if not isinstance(guidance, vs.VideoNode) or guidance.format.subsampling_w != 0 or guidance.format.subsampling_h != 0:
        raise TypeError(funcName + ': \"guidance\" must be a RGB or YUV444 clip!')
    if input.width != guidance.width or input.height != guidance.height:
        raise ValueError(funcName + ': \"guidance\" must be of the same size as \"input\"!')

    if fast is None:
        fast = False if core.version_number() >= 39 else True

    if kernel1_args is None:
        kernel1_args = {}
    if kernel2_args is None:
        kernel2_args = {}

    # Bitdepth conversion and variable names modification to correspond to the paper
    p = mvf.Depth(input, depth=32, sample=vs.FLOAT, **depth_args)
    I = mvf.Depth(guidance, depth=32, sample=vs.FLOAT, **depth_args)
    r = radius
    eps = regulation
    s = subsampling_ratio

    # Back up guidance image
    I_src_r = mvf.GetPlane(I, 0)
    I_src_g = mvf.GetPlane(I, 1)
    I_src_b = mvf.GetPlane(I, 2)

    # Fast guided filter's subsampling
    if fast:
        down_w = math.floor(width / s)
        down_h = math.floor(height / s)
        if use_fmtc1:
            p = core.fmtc.resample(p, down_w, down_h, kernel=kernel1, **kernel1_args)
            I = core.fmtc.resample(I, down_w, down_h, kernel=kernel1, **kernel1_args)
        else: # use zimg
            p = eval('core.resize.{kernel}(p, {w}, {h}, **kernel1_args)'.format(kernel=kernel1.capitalize(), w=down_w, h=down_h))
            I = eval('core.resize.{kernel}(I, {w}, {h}, **kernel1_args)'.format(kernel=kernel1.capitalize(), w=down_w, h=down_h)) if guidance is not None else p

        r = math.floor(r / s)

    # Select kernel shape. As the width of BoxFilter in this module is (radius*2-1) rather than (radius*2+1), radius should be be incremented by one.
    Filter = functools.partial(core.tcanny.TCanny, sigma=r/2 * math.sqrt(2), mode=-1) if use_gauss else functools.partial(BoxFilter, radius=r+1)

    # Seperate planes
    I_r = mvf.GetPlane(I, 0)
    I_g = mvf.GetPlane(I, 1)
    I_b = mvf.GetPlane(I, 2)

    # Compute local linear coefficients.
    mean_p = Filter(p)

    mean_I_r = Filter(I_r)
    mean_I_g = Filter(I_g)
    mean_I_b = Filter(I_b)

    corr_I_rr = Filter(core.std.Expr([I_r], ['x dup *']))
    corr_I_rg = Filter(core.std.Expr([I_r, I_g], ['x y *']))
    corr_I_rb = Filter(core.std.Expr([I_r, I_b], ['x y *']))
    corr_I_gg = Filter(core.std.Expr([I_g], ['x dup *']))
    corr_I_gb = Filter(core.std.Expr([I_g, I_b], ['x y *']))
    corr_I_bb = Filter(core.std.Expr([I_b], ['x dup *']))

    corr_Ip_r = Filter(core.std.Expr([I_r, p], ['x y *']))
    corr_Ip_g = Filter(core.std.Expr([I_g, p], ['x y *']))
    corr_Ip_b = Filter(core.std.Expr([I_b, p], ['x y *']))

    var_I_rr = core.std.Expr([corr_I_rr, mean_I_r], ['x y dup * - {} +'.format(eps)])
    var_I_gg = core.std.Expr([corr_I_gg, mean_I_g], ['x y dup * - {} +'.format(eps)])
    var_I_bb = core.std.Expr([corr_I_bb, mean_I_b], ['x y dup * - {} +'.format(eps)])

    cov_I_rg = core.std.Expr([corr_I_rg, mean_I_r, mean_I_g], ['x y z * -'])
    cov_I_rb = core.std.Expr([corr_I_rb, mean_I_r, mean_I_b], ['x y z * -'])
    cov_I_gb = core.std.Expr([corr_I_gb, mean_I_g, mean_I_b], ['x y z * -'])

    cov_Ip_r = core.std.Expr([corr_Ip_r, mean_I_r, mean_p], ['x y z * -'])
    cov_Ip_g = core.std.Expr([corr_Ip_g, mean_I_g, mean_p], ['x y z * -'])
    cov_Ip_b = core.std.Expr([corr_Ip_b, mean_I_b, mean_p], ['x y z * -'])

    # Inverse of Sigma + eps * I
    inv_rr = core.std.Expr([var_I_gg, var_I_bb, cov_I_gb], ['x y * z dup * -'])
    inv_rg = core.std.Expr([cov_I_gb, cov_I_rb, cov_I_rg, var_I_bb], ['x y * z a * -'])
    inv_rb = core.std.Expr([cov_I_rg, cov_I_gb, var_I_gg, cov_I_rb], ['x y * z a * -'])
    inv_gg = core.std.Expr([var_I_rr, var_I_bb, cov_I_rb], ['x y * z dup * -'])
    inv_gb = core.std.Expr([cov_I_rb, cov_I_rg, var_I_rr, cov_I_gb], ['x y * z a * -'])
    inv_bb = core.std.Expr([var_I_rr, var_I_gg, cov_I_rg], ['x y * z dup * -'])

    covDet = core.std.Expr([inv_rr, var_I_rr, inv_rg, cov_I_rg, inv_rb, cov_I_rb], ['x y * z a * + b c * +'])

    inv_rr = core.std.Expr([inv_rr, covDet], ['x y /'])
    inv_rg = core.std.Expr([inv_rg, covDet], ['x y /'])
    inv_rb = core.std.Expr([inv_rb, covDet], ['x y /'])
    inv_gg = core.std.Expr([inv_gg, covDet], ['x y /'])
    inv_gb = core.std.Expr([inv_gb, covDet], ['x y /'])
    inv_bb = core.std.Expr([inv_bb, covDet], ['x y /'])

    a_r = core.std.Expr([inv_rr, cov_Ip_r, inv_rg, cov_Ip_g, inv_rb, cov_Ip_b], ['x y * z a * + b c * +'])
    a_g = core.std.Expr([inv_rg, cov_Ip_r, inv_gg, cov_Ip_g, inv_gb, cov_Ip_b], ['x y * z a * + b c * +'])
    a_b = core.std.Expr([inv_rb, cov_Ip_r, inv_gb, cov_Ip_g, inv_bb, cov_Ip_b], ['x y * z a * + b c * +'])

    b = core.std.Expr([mean_p, a_r, mean_I_r, a_g, mean_I_g, a_b, mean_I_b], ['x y z * - a b * - c d * -'])

    mean_a_r = Filter(a_r)
    mean_a_g = Filter(a_g)
    mean_a_b = Filter(a_b)
    mean_b = Filter(b)

    # Fast guided filter's upsampling
    if fast:
        if use_fmtc2:
            mean_a_r = core.fmtc.resample(mean_a_r, width, height, kernel=kernel2, **kernel2_args)
            mean_a_g = core.fmtc.resample(mean_a_g, width, height, kernel=kernel2, **kernel2_args)
            mean_a_b = core.fmtc.resample(mean_a_b, width, height, kernel=kernel2, **kernel2_args)
            mean_b = core.fmtc.resample(mean_b, width, height, kernel=kernel2, **kernel2_args)
        else: # use zimg
            mean_a_r = eval('core.resize.{kernel}(mean_a_r, {w}, {h}, **kernel2_args)'.format(kernel=kernel2.capitalize(), w=width, h=height))
            mean_a_g = eval('core.resize.{kernel}(mean_a_g, {w}, {h}, **kernel2_args)'.format(kernel=kernel2.capitalize(), w=width, h=height))
            mean_a_b = eval('core.resize.{kernel}(mean_a_b, {w}, {h}, **kernel2_args)'.format(kernel=kernel2.capitalize(), w=width, h=height))
            mean_b = eval('core.resize.{kernel}(mean_b, {w}, {h}, **kernel2_args)'.format(kernel=kernel2.capitalize(), w=width, h=height))

    # Linear translation
    q = core.std.Expr([mean_a_r, I_src_r, mean_a_g, I_src_g, mean_a_b, I_src_b, mean_b], ['x y * z a * + b c * + d +'])

    # Final bitdepth conversion
    return mvf.Depth(q, depth=bits, sample=sampleType, **depth_args)


def GMSD(clip1, clip2, plane=None, downsample=True, c=0.0026, show_map=False, **depth_args):
    """Gradient Magnitude Similarity Deviation Calculator

    GMSD is a new effective and efficient image quality assessment (IQA) model, which utilizes the pixel-wise gradient magnitude similarity (GMS)
    between the reference and distorted images combined with standard deviation of the GMS map to predict perceptual image quality.

    The distortion degree of the distorted image will be stored as frame property 'PlaneGMSD' in the output clip.

    The value of GMSD reflects the range of distortion severities in an image.
    The lowerer the GMSD score, the higher the image perceptual quality.
    If "clip1" == "clip2", GMSD = 0.

    All the internal calculations are done at 32-bit float, only one channel of the image will be processed.

    Args:
        clip1: The distorted clip, will be copied to output if "show_map" is False.

        clip2: Reference clip, must be of the same format and dimension as the "clip1".

        plane: (int) Specify which plane to be processed. Default is None.

        downsample: (bool) Whether to average the clips over local 2x2 window and downsample by a factor of 2 before calculation.
            Default is True.

        c: (float) A positive constant that supplies numerical stability.
            According to the paper, for all the test databases, GMSD shows similar preference to the value of c.
            Default is 0.0026.

        show_map: (bool) Whether to return GMS map. If not, "clip1" will be returned. Default is False.

        depth_args: (dict) Additional arguments passed to mvf.Depth() in the form of keyword arguments.
            Default is {}.

    Ref:
        [1] Xue, W., Zhang, L., Mou, X., & Bovik, A. C. (2014). Gradient magnitude similarity deviation: 
            A highly efficient perceptual image quality index. IEEE Transactions on Image Processing, 23(2), 684-695.
        [2] http://www4.comp.polyu.edu.hk/~cslzhang/IQA/GMSD/GMSD.htm.

    """

    funcName = 'GMSD'

    if not isinstance(clip1, vs.VideoNode):
        raise TypeError(funcName + ': \"clip1\" must be a clip!')
    if not isinstance(clip2, vs.VideoNode):
        raise TypeError(funcName + ': \"clip2\" must be a clip!')

    if clip1.format.id != clip2.format.id:
        raise ValueError(funcName + ': \"clip1\" and \"clip2\" must be of the same format!')
    if clip1.width != clip2.width or clip1.height != clip2.height:
        raise ValueError(funcName + ': \"clip1\" and \"clip2\" must be of the same width and height!')

    # Store the "clip1"
    clip1_src = clip1

    # Convert to float type grayscale image
    clip1 = mvf.GetPlane(clip1, plane)
    clip2 = mvf.GetPlane(clip2, plane)
    clip1 = mvf.Depth(clip1, depth=32, sample=vs.FLOAT, **depth_args)
    clip2 = mvf.Depth(clip2, depth=32, sample=vs.FLOAT, **depth_args)

    # Filtered by a 2x2 average filter and then down-sampled by a factor of 2, as in the implementation of SSIM
    if downsample:
        clip1 = _IQA_downsample(clip1)
        clip2 = _IQA_downsample(clip2)

    # Calculate gradients based on Prewitt filter
    clip1_dx = core.std.Convolution(clip1, [1, 0, -1, 1, 0, -1, 1, 0, -1], divisor=1, saturate=False)
    clip1_dy = core.std.Convolution(clip1, [1, 1, 1, 0, 0, 0, -1, -1, -1], divisor=1, saturate=False)
    clip1_grad_squared = core.std.Expr([clip1_dx, clip1_dy], ['x dup * y dup * +'])

    clip2_dx = core.std.Convolution(clip2, [1, 0, -1, 1, 0, -1, 1, 0, -1], divisor=1, saturate=False)
    clip2_dy = core.std.Convolution(clip2, [1, 1, 1, 0, 0, 0, -1, -1, -1], divisor=1, saturate=False)
    clip2_grad_squared = core.std.Expr([clip2_dx, clip2_dy], ['x dup * y dup * +'])

    # Compute the gradient magnitude similarity (GMS) map
    quality_map = core.std.Expr([clip1_grad_squared, clip2_grad_squared], ['2 x y * sqrt * {c} + x y + {c} + /'.format(c=c)])

    # The following code is modified from mvf.PlaneStatistics(), which is used to compute the standard deviation of the GMS map as GMSD
    if core.std.get_functions().__contains__('PlaneStats'):
        map_mean = core.std.PlaneStats(quality_map, plane=[0], prop='PlaneStats')
    else:
        map_mean = core.std.PlaneAverage(quality_map, plane=[0], prop='PlaneStatsAverage')

    def _PlaneSDFrame(n, f, clip):
        mean = f.props['PlaneStatsAverage']
        expr = "x {mean} - dup *".format(mean=mean)
        return core.std.Expr(clip, expr)
    SDclip = core.std.FrameEval(quality_map, functools.partial(_PlaneSDFrame, clip=quality_map), map_mean)

    if core.std.get_functions().__contains__('PlaneStats'):
        SDclip = core.std.PlaneStats(SDclip, plane=[0], prop='PlaneStats')
    else:
        SDclip = core.std.PlaneAverage(SDclip, plane=[0], prop='PlaneStatsAverage')

    def _PlaneGMSDTransfer(n, f):
        fout = f[0].copy()
        fout.props['PlaneGMSD'] = math.sqrt(f[1].props['PlaneStatsAverage'])
        return fout
    output_clip = quality_map if show_map else clip1_src
    output_clip = core.std.ModifyFrame(output_clip, [output_clip, SDclip], selector=_PlaneGMSDTransfer)

    return output_clip


def SSIM(clip1, clip2, plane=None, downsample=True, k1=0.01, k2=0.03, fun=None, dynamic_range=1, show_map=False, **depth_args):
    """Structural SIMilarity Index Calculator

    The Structural SIMilarity (SSIM) index is a method for measuring the similarity between two images.
    It is based on the hypothesis that the HVS is highly adapted for extracting structural information,
    which compares local patterns of pixel intensities that have been normalized for luminance and contrast.

    The mean SSIM (MSSIM) index value of the distorted image will be stored as frame property 'PlaneSSIM' in the output clip.

    The value of SSIM measures the structural similarity in an image.
    The higher the SSIM score, the higher the image perceptual quality.
    If "clip1" == "clip2", SSIM = 1.

    All the internal calculations are done at 32-bit float, only one channel of the image will be processed.

    Args:
        clip1: The distorted clip, will be copied to output if "show_map" is False.

        clip2: Reference clip, must be of the same format and dimension as the "clip1".

        plane: (int) Specify which plane to be processed. Default is None.

        downsample: (bool) Whether to average the clips over local 2x2 window and downsample by a factor of 2 before calculation.
            Default is True.

        k1, k2: (float) Constants in the SSIM index formula.
            According to the paper, the performance of the SSIM index algorithm is fairly insensitive to variations of these values.
            Default are 0.01 and 0.03.

        fun: (function or float) The function of how the clips are filtered.
            If it is None, it will be set to a gaussian filter whose standard deviation is 1.5.
            Note that the size of gaussian kernel is different from the one in MATLAB.
            If it is a float, it specifies the standard deviation of the gaussian filter. (sigma in core.tcanny.TCanny)
            According to the paper, the quality map calculated from gaussian filter exhibits a locally isotropic property,
            which prevents the present of undesirable “blocking” artifacts in the resulting SSIM index map.
            Default is None.

        dynamic_range: (float) Dynamic range of the internal float point clip. Default is 1.

        show_map: (bool) Whether to return SSIM index map. If not, "clip1" will be returned. Default is False.

        depth_args: (dict) Additional arguments passed to mvf.Depth() in the form of keyword arguments.
            Default is {}.

    Ref:
        [1] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality assessment: from error visibility to structural similarity.
            IEEE transactions on image processing, 13(4), 600-612.
        [2] https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    """

    funcName = 'SSIM'

    if not isinstance(clip1, vs.VideoNode):
        raise TypeError(funcName + ': \"clip1\" must be a clip!')
    if not isinstance(clip2, vs.VideoNode):
        raise TypeError(funcName + ': \"clip2\" must be a clip!')

    if clip1.format.id != clip2.format.id:
        raise ValueError(funcName + ': \"clip1\" and \"clip2\" must be of the same format!')
    if clip1.width != clip2.width or clip1.height != clip2.height:
        raise ValueError(funcName + ': \"clip1\" and \"clip2\" must be of the same width and height!')

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    if fun is None:
        fun = functools.partial(core.tcanny.TCanny, sigma=1.5, mode=-1)
    elif isinstance(fun, (int, float)):
        fun = functools.partial(core.tcanny.TCanny, sigma=fun, mode=-1)
    elif not callable(fun):
        raise TypeError(funcName + ': \"fun\" must be a function or a float!')

    # Store the "clip1"
    clip1_src = clip1

    # Convert to float type grayscale image
    clip1 = mvf.GetPlane(clip1, plane)
    clip2 = mvf.GetPlane(clip2, plane)
    clip1 = mvf.Depth(clip1, depth=32, sample=vs.FLOAT, **depth_args)
    clip2 = mvf.Depth(clip2, depth=32, sample=vs.FLOAT, **depth_args)

    # Filtered by a 2x2 average filter and then down-sampled by a factor of 2
    if downsample:
        clip1 = _IQA_downsample(clip1)
        clip2 = _IQA_downsample(clip2)

    # Core algorithm
    mu1 = fun(clip1)
    mu2 = fun(clip2)
    mu1_sq = core.std.Expr([mu1], ['x dup *'])
    mu2_sq = core.std.Expr([mu2], ['x dup *'])
    mu1_mu2 = core.std.Expr([mu1, mu2], ['x y *'])
    sigma1_sq_pls_mu1_sq = fun(core.std.Expr([clip1], ['x dup *']))
    sigma2_sq_pls_mu2_sq = fun(core.std.Expr([clip2], ['x dup *']))
    sigma12_pls_mu1_mu2 = fun(core.std.Expr([clip1, clip2], ['x y *']))

    if c1 > 0 and c2 > 0:
        expr = '2 x * {c1} + 2 y x - * {c2} + * z a + {c1} + b c - d e - + {c2} + * /'.format(c1=c1, c2=c2)
        expr_clips = [mu1_mu2, sigma12_pls_mu1_mu2, mu1_sq, mu2_sq, sigma1_sq_pls_mu1_sq, mu1_sq, sigma2_sq_pls_mu2_sq, mu2_sq]
        ssim_map = core.std.Expr(expr_clips, [expr])
    else:
        denominator1 = core.std.Expr([mu1_sq, mu2_sq], ['x y + {c1} +'.format(c1=c1)])
        denominator2 = core.std.Expr([sigma1_sq_pls_mu1_sq, mu1_sq, sigma2_sq_pls_mu2_sq, mu2_sq], ['x y - z a - + {c2} +'.format(c2=c2)])

        numerator1_expr = '2 z * {c1} +'.format(c1=c1)
        numerator2_expr = '2 a z - * {c2} +'.format(c2=c2)
        expr = 'x y * 0 > {numerator1} {numerator2} * x y * / x 0 = not y 0 = and {numerator1} x / {i} ? ?'.format(numerator1=numerator1_expr, 
            numerator2=numerator2_expr, i=1)
        ssim_map = core.std.Expr([denominator1, denominator2, mu1_mu2, sigma12_pls_mu1_mu2], [expr])

    # The following code is modified from mvf.PlaneStatistics(), which is used to compute the mean of the SSIM index map as MSSIM
    if core.std.get_functions().__contains__('PlaneStats'):
        map_mean = core.std.PlaneStats(ssim_map, plane=[0], prop='PlaneStats')
    else:
        map_mean = core.std.PlaneAverage(ssim_map, plane=[0], prop='PlaneStatsAverage')

    def _PlaneSSIMTransfer(n, f):
        fout = f[0].copy()
        fout.props['PlaneSSIM'] = f[1].props['PlaneStatsAverage']
        return fout
    output_clip = ssim_map if show_map else clip1_src
    output_clip = core.std.ModifyFrame(output_clip, [output_clip, map_mean], selector=_PlaneSSIMTransfer)

    return output_clip


def _IQA_downsample(clip):
    """Downsampler for image quality assessment model.

    The “clip” is first filtered by a 2x2 average filter, and then down-sampled by a factor of 2.
    """

    return core.std.Convolution(clip, [1, 1, 0, 1, 1, 0, 0, 0, 0]).resize.Point(clip.width // 2, clip.height // 2, src_left=-1, src_top=-1)


def SSIM_downsample(clip, w, h, smooth=1, kernel=None, use_fmtc=False, gamma=False, fulls=False, fulld=False, curve='709', sigmoid=False, 
    epsilon=1e-6, depth_args=None, **resample_args):
    """SSIM downsampler

    SSIM downsampler is an image downscaling technique that aims to optimize for the perceptual quality of the downscaled results.
    Image downscaling is considered as an optimization problem
    where the difference between the input and output images is measured using famous Structural SIMilarity (SSIM) index.
    The solution is derived in closed-form, which leads to the simple, efficient implementation.
    The downscaled images retain perceptually important features and details,
    resulting in an accurate and spatio-temporally consistent representation of the high resolution input.

    This is an pseudo-implementation of SSIM downsampler with slight modification.
    The pre-downsampling is performed by vszimg/fmtconv, and the behaviour of convolution at the border is uniform.

    All the internal calculations are done at 32-bit float, except gamma correction is done at integer.

    Args:
        clip: The input clip.

        w, h: The size of the output clip.

        smooth: (int, float or function) The method to smooth the image.
            If it's an int, it specifies the "radius" of the internel used boxfilter, i.e. the window has a size of (2*smooth+1)x(2*smooth+1).
            If it's a float, it specifies the "sigma" of core.tcanny.TCanny, i.e. the standard deviation of gaussian blur.
            If it's a function, it acs as a general smoother.
            Default is 1. The 3x3 boxfilter will be performed.

        kernel: (string) Resample kernel of vszimg/fmtconv.
            Default is 'Bicubic'.

        use_fmtc: (bool) Whether to use fmtconv for downsampling. If not, vszimg (core.resize.*) will be used.
            Default is False.

        depth_args: (dict) Additional arguments passed to mvf.Depth().
            Default is {}.

        gamma: (bool) Default is False.
            Set to true to turn on gamma correction for the y channel.

        fulls: (bool) Default is False.
            Specifies if the luma is limited range (False) or full range (True) 

        fulld: (bool) Default is False.
            Same as fulls, but for output.
        
        curve: (string) Default is '709'.
            Type of gamma mapping.

        sigmoid: (bool) Default is False.
            When True, applies a sigmoidal curve after the power-like curve (or before when converting from linear to gamma-corrected). 
            This helps reducing the dark halo artefacts around sharp edges caused by resizing in linear luminance.

        resample_args: (dict) Additional arguments passed to vszimg/fmtconv in the form of keyword arguments.
            Default is {}.

    Ref:
        [1] Oeztireli, A. C., & Gross, M. (2015). Perceptually based downscaling of images. ACM Transactions on Graphics (TOG), 34(4), 77.

    """

    funcName = 'SSIM_downsample'

    if not isinstance(clip, vs.VideoNode):
        raise TypeError(funcName + ': \"clip\" must be a clip!')

    import nnedi3_resample as nnrs

    if depth_args is None:
        depth_args = {}

    if isinstance(smooth, int):
        Filter = functools.partial(BoxFilter, radius=smooth+1)
    elif isinstance(smooth, float):
        Filter = functools.partial(core.tcanny.TCanny, sigma=smooth, mode=-1)
    elif callable(smooth):
        Filter = smooth
    else:
        raise TypeError(funcName + ': \"smooth\" must be a int, float or a function!')

    if kernel is None:
        kernel = 'Bicubic'

    if gamma:
        clip = nnrs.GammaToLinear(mvf.Depth(clip, 16), fulls=fulls, fulld=fulld, curve=curve, sigmoid=sigmoid, planes=[0])
    
    clip = mvf.Depth(clip, depth=32, sample=vs.FLOAT, **depth_args)

    if use_fmtc:
        l = core.fmtc.resample(clip, w, h, kernel=kernel, **resample_args)
        l2 = core.fmtc.resample(core.std.Expr([clip], ['x dup *']), w, h, kernel=kernel, **resample_args)
    else: # use vszimg
        l = eval('core.resize.{kernel}(clip, w, h, **resample_args)'.format(kernel=kernel.capitalize()))
        l2 = eval('core.resize.{kernel}(core.std.Expr([clip], ["x dup *"]), w, h, **resample_args)'.format(kernel=kernel.capitalize()))

    m = Filter(l)
    sl_plus_m_square = Filter(core.std.Expr([l], ['x dup *']))
    sh_plus_m_square = Filter(l2)
    m_square = core.std.Expr([m], ['x dup *'])
    r = core.std.Expr([sl_plus_m_square, sh_plus_m_square, m_square], ['x z - {eps} < 0 y z - x z - / sqrt ?'.format(eps=epsilon)])
    t = Filter(core.std.Expr([r, m], ['x y *']))
    m = Filter(m)
    r = Filter(r)
    d = core.std.Expr([m, r, l, t], ['x y z * + a -'])

    if gamma:
        d = nnrs.LinearToGamma(mvf.Depth(d, 16), fulls=fulls, fulld=fulld, curve=curve, sigmoid=sigmoid, planes=[0])

    return d


def LocalStatisticsMatching(src, ref, radius=1, return_all=False, **depth_args):
    """Local statistics matcher

    Match the local statistics (mean, variance) of "src" with "ref".

    All the internal calculations are done at 32-bit float.

    Args:
        src, ref: Inputs.

        radius: (int or function) If it is an integer, it specifies the radius of mean filter.
            It can also be a custom function.
            Default is 1.

        depth_args: (dict) Additional arguments passed to mvf.Depth().
            Default is {}.

    """

    funcName = 'LocalStatisticsMatching'

    if not isinstance(src, vs.VideoNode):
        raise TypeError(funcName + ': \"src\" must be a clip!')
    if not isinstance(ref, vs.VideoNode):
        raise TypeError(funcName + ': \"ref\" must be a clip!')

    bits = src.format.bits_per_sample
    sampleType = src.format.sample_type
    epsilon = 1e-7 # small positive number to avoid dividing by 0

    src, src_mean, src_var = LocalStatistics(src, radius=radius, **depth_args)
    _, ref_mean, ref_var = LocalStatistics(ref, radius=radius, **depth_args)

    flt = core.std.Expr([src, src_mean, src_var, ref_mean, ref_var], ['x y - z sqrt {} + / b sqrt * a +'.format(epsilon)])

    flt = mvf.Depth(flt, depth=bits, sample=sampleType, **depth_args)

    if return_all:
        return flt, src_mean, src_var, ref_mean, ref_var
    else:
        return flt


def LocalStatistics(clip, radius=1, **depth_args):
    """Local statistics calculator

    The local mean and variance will be returned.

    All the internal calculations are done at 32-bit float.

    Args:
        clip: Inputs.

        radius: (int or function) If it is an integer, it specifies the radius of mean filter.
            It can also be a custom function.
            Default is 1.

        depth_args: (dict) Additional arguments passed to mvf.Depth().
            Default is {}.

    Returns:
        A list containing three clips (source, mean, variance) in 32bit float.

    """

    funcName = 'LocalStatistics'

    if not isinstance(clip, vs.VideoNode):
        raise TypeError(funcName + ': \"clip\" must be a clip!')

    Expectation = radius if callable(radius) else functools.partial(BoxFilter, radius=radius+1)

    clip = mvf.Depth(clip, depth=32, sample=vs.FLOAT, **depth_args)

    mean = Expectation(clip)
    squared = Expectation(core.std.Expr(clip, 'x dup *'))
    var = core.std.Expr([squared, mean], 'x y dup * -')

    return clip, mean, var


def TextSub16(src, file, mod=False, tv_range=True, matrix=None, dither=None, **vsfilter_args):
    """TextSub16 for VapourSynth

    Author: mawen1250 (http://nmm.me/109)

    Unofficial description:
        Generate mask in YUV and use it to mask high-precision subtitles overlayed in RGB.

    Args:
        src: Input clip, must be of YUV color family.

        file: Path to subtitle.

        mod: (bool) Whether to use VSFilterMod. If not, VSFilter will be used.
            Default is False.

        tv_range: (bool) Define if input clip is of tv range(limited range).
            Default is True.

        matrix: (int|str) Color matrix of input clip.
            Default is None, guessed according to the color family and size of input clip.

        dither: (str) Dithering method of vszimg.
            The following dithering methods are available: "none", "ordered", "random", "error_diffusion".
            Default is "error_diffusion".

        vsfilter_args: (dict) Additional arguments passed to subtitle plugin.
            Default is {}.

    Requirments:
        1. VSFilter (https://github.com/HomeOfVapourSynthEvolution/VSFilter)
        2. VSFilterMod (https://github.com/sorayuki/VSFilterMod)

    """

    funcName = 'TextSub16'

    if not isinstance(src, vs.VideoNode) or src.format.color_family != vs.YUV:
        raise TypeError(funcName + ': \"src\" must be a YUV clip!')

    matrix = mvf.GetMatrix(src, matrix, True)
    css = src.format.name[3:6]
    sw = src.width
    sh = src.height

    if dither is None:
        dither = 'error_diffusion'

    if src.format.id != vs.YUV420P8:
        src8 = core.resize.Bicubic(src, format=vs.YUV420P8, range_in=tv_range)
    else:
        src8 = src

    src16 = mvf.Depth(src, depth=16, sample=vs.INTEGER, fulls=tv_range, dither='none')

    if mod:
        src8sub = core.vsfm.TextSubMod(src8, file=file, **vsfilter_args)
    else:
        src8sub = core.vsf.TextSub(src8, file=file, **vsfilter_args)

    submask = core.std.Expr([src8, src8sub], ['x y = 0 255 ?']).resize.Bilinear(format=vs.YUV444P8, range=True, range_in=True)
    submaskY = mvf.GetPlane(submask, 0)
    submaskU = mvf.GetPlane(submask, 1)
    submaskV = mvf.GetPlane(submask, 2)
    submask = mvf.Max(mvf.Max(submaskY, submaskU), submaskV).std.Inflate()
    submaskY = core.resize.Bilinear(submask, format=vs.GRAY16, range_in=True, range=True)
    if css == '444':
        submaskC = submaskY
    elif css == '422':
        submaskC = core.resize.Bilinear(submask, sw // 2, sh, format=vs.GRAY16, range_in=True, range=True, src_left=-0.5)
    elif css == '420':
        submaskC = core.resize.Bilinear(submask, sw // 2, sh // 2, format=vs.GRAY16, range_in=True, range=True, src_left=-0.5)
    else:
        raise TypeError(funcName + 'the subsampling of \"src\" must be 444/422/420!')

    submask = core.std.ShufflePlanes([submaskY, submaskC], [0, 0, 0], vs.YUV)

    last = core.resize.Bicubic(src16, format=vs.RGB24, matrix_in_s=matrix, range=tv_range, dither_type=dither)

    if mod:
        last = core.vsfm.TextSubMod(last, file=file, **vsfilter_args)
    else:
        last = core.vsf.TextSub(last, file=file, **vsfilter_args)

    sub16 = core.resize.Bicubic(last, format=src16.format.id, matrix_s=matrix, range=tv_range, dither_type=dither)

    return core.std.MaskedMerge(src16, sub16, submask, planes=[0, 1, 2])


def TMinBlur(clip, r=1, thr=2):
    """Thresholded MinBlur

    Use another MinBlur with larger radius to guide the smoothing effect of current MinBlur.

    For detailed motivation and description (in Chinese), see:
    https://gist.github.com/WolframRhodium/1e3ae9276d70aa1ddc93ea833cdce9c6#file-05-minblurmod-md

    Args:
        clip: Input clip.

        r: (int) Radius of MinBlur() filter.
            Default is 1.

        thr: (float) Threshold in 8 bits scale.
            If it is larger than 255, the output will be identical to original MinBlur().
            Default is 2.

    """

    funcName = 'TMinBlur'

    if not isinstance(clip, vs.VideoNode) or clip.format.sample_type != vs.INTEGER:
        raise TypeError(funcName + ': \"clip\" must be an integer clip!')

    thr = scale(thr, clip.format.bits_per_sample)

    pre1 = haf.MinBlur(clip, r=r)
    pre2 = haf.MinBlur(clip, r=r+1)

    return core.std.Expr([clip, pre1, pre2], ['y z - abs {thr} <= y x ?'.format(thr=thr)])


def mdering(clip, thr=2):
    """A simple light and bright DCT ringing remover

    It is a special instance of TMinBlur (r=1 and only filter the bright part) for higher performance.
    Post-processing is needed to reduce degradation of flat and texture areas.

    Args:
        clip: Input clip.

        thr: (float) Threshold in 8 bits scale.
            Default is 2.

    """

    funcName = 'mdering'

    if not isinstance(clip, vs.VideoNode) or clip.format.sample_type != vs.INTEGER:
        raise TypeError(funcName + ': \"clip\" must be an integer clip!')

    bits = clip.format.bits_per_sample
    thr = scale(thr, bits)

    rg11_1 = core.std.Convolution(clip, matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1])
    rg11_2 = core.std.Convolution(rg11_1, [1]*9)
    rg4_1 = core.std.Median(clip)

    if bits <= 12:
        rg4_2 = core.ctmf.CTMF(clip, radius=2)
    else:
        rg4_2 = core.fmtc.bitdepth(clip, bits=12, dmode=1).ctmf.CTMF(radius=2).fmtc.bitdepth(bits=bits)
        rg4_2 = mvf.LimitFilter(clip, rg4_2, thr=0.0625, elast=2)

    minblur_1 = core.std.Expr([clip, rg11_1, rg4_1], ['x y - x z - xor x x y - abs x z - abs < y z ? ?'])
    minblur_2 = core.std.Expr([clip, rg11_2, rg4_2], ['x y - x z - xor x x y - abs x z - abs < y z ? ?'])
    dering = core.std.Expr([clip, minblur_1, minblur_2], ['y z - abs {thr} <= y x <= and y x ?'.format(thr=thr)])

    return dering


def BMAFilter(clip, guidance=None, radius=1, lamda=1e-2, epsilon=1e-5, mode=3, **depth_args):
    """Edge-Aware BMA Filter

    Edge-aware BMA filter is a family of edge-aware filters proposed based on optimal parameter estimation and Bayesian model averaging (BMA).
    The problem of filtering a pixel in a local pixel patch is formulated as an optimal estimation problem,
    and multiple estimates of the same pixel are combined using BMA.
    Filters in this family differs from different settings of cost functions and log-likelihood and log-prior functions.

    However, only four of six BMA filters are implemented.
    The implementation is modified to allow the filtering to be guided by another source, like GuidedFilter().

    Most of the internal calculations are done at 32-bit float, except median filtering with radius larger than 1 is done at integer.

    Args:
        clip: Input clip.

        guidance: (clip) Guidance clip used to compute the coefficient of the translation on "clip".
            It must has the same clip properties as 'clip'.
            If it is None, it will be set to 'clip', with duplicate calculations being omitted.
            Default is None.

        radius: (int) The radius of box filter and median filter.
            Default is 1.

        lamda: (float) A criterion for judging whether a patch has high variance and should be preserved, or is flat and should be smoothed.
            It only takes effects when `mode` is 3 or 4.
            The limit of filter of `mode` 3 [resp. 4] as `lamda` approaches infinity is filter of `mode` 1 [resp. 2].
            Default is 0.01.

        epsilon: (float) Small number to avoid divide by 0.
            Default is 0.00001.

        mode: (1~4): Number of different BMA filters.
            1: l2-norm based cost function, constant prior and gaussian likelihood.
            2: l1-norm based cost function, constant prior and laplacian likelihood.
            3: 'hit-or-miss' cost function, gaussian prior and gaussian likelihood.
            4: 'hit-or-miss' cost function, gaussian prior and laplacian likelihood.
            Default is 3.

        depth_args: (dict) Additional arguments passed to mvf.Depth().
            Default is {}.

    Ref:
        [1] Deng, G. (2016). Edge-aware BMA filters. IEEE Transactions on Image Processing, 25(1), 439-454.
        [2] https://www.researchgate.net/publication/284391731_Edge-aware_BMA_filters

    """

    funcName = 'BMAFilter'

    if guidance is None:
        guidance = clip
    else:
        if not isinstance(guidance, vs.VideoNode):
            raise TypeError(funcName + ': \"guidance\" must be a clip!')
        if clip.format.id != guidance.format.id:
            raise TypeError(funcName + ': \"guidance\" must be of the same format as \"clip\"!')
        if clip.width != guidance.width or clip.height != guidance.height:
            raise TypeError(funcName + ': \"guidance\" must be of the same size as \"clip\"!')

    bits = clip.format.bits_per_sample
    sampleType = clip.format.sample_type
    clip_src = clip
    clip = mvf.Depth(clip, depth=32, **depth_args)
    guidance = mvf.Depth(guidance, depth=32, **depth_args) if guidance != clip_src else clip

    if mode in (1, 3):
        Filter = functools.partial(BoxFilter, radius=radius+1)
    elif mode in (2, 4):
        def Filter(clip):
            if radius == 1:
                clip = core.std.Median(clip)
            else:
                clip = mvf.Depth(clip, 12, **depth_args)
                clip = core.ctmf.CTMF(clip, radius=radius)
            return mvf.Depth(clip, 32, **depth_args)

    Expectation = functools.partial(BoxFilter, radius=radius+1)

    if mode in (1, 2):
        mean_guidance = Expectation(guidance)
        corr_guidance = Expectation(core.std.Expr([guidance], ['x dup *']))
        unscaled_alpha = core.std.Expr([corr_guidance, mean_guidance], ['1 x y dup * - {epsilon} + /'.format(epsilon=epsilon)]) # Eqn. 10
        alpha_scale = Expectation(unscaled_alpha)

        if mode == 1:
            mean_clip = Filter(clip) if clip != guidance else mean_guidance
            res = Expectation(core.std.Expr([unscaled_alpha, mean_clip], ['x y *'])) # Eqn. 11
        else: # mode == 2
            median_clip = Filter(clip_src)
            res = Expectation(core.std.Expr([unscaled_alpha, median_clip], ['x y *'])) # Eqn. 12

        res = core.std.Expr([res, alpha_scale], ['x y /'])
    elif mode in (3, 4):
        mean_guidance = Expectation(guidance)

        guidance_square = core.std.Expr([guidance], ['x dup *'])
        var_guidance = core.std.Expr([Expectation(guidance_square), mean_guidance], ['x y dup * -'])
        unscaled_alpha = core.std.Expr([var_guidance], ['1 x {epsilon} + /'.format(epsilon=epsilon)]) # Eqn. 10
        alpha_scale = Expectation(unscaled_alpha)
        beta = core.std.Expr([var_guidance], ['1 x {epsilon} + {lamda} * 1 + /'.format(epsilon=epsilon, lamda=1/lamda)]) # Eqn. 18
        tmp1 = core.std.Expr([unscaled_alpha, beta], ['x y *'])

        if mode == 3:
            mean_clip = Filter(clip) if clip != guidance else mean_guidance
            tmp2 = Expectation(core.std.Expr([tmp1, mean_clip], ['x y *'])) # Eqn. 19, left
        else: # mode == 4
            median_clip = Filter(clip_src)
            tmp2 = Expectation(core.std.Expr([tmp1, median_clip], ['x y *'])) # Eqn. 25, left

        tmp3 = Expectation(tmp1) # Eqn. 19 / 25, right
        res = core.std.Expr([tmp2, alpha_scale, tmp3, clip], ['x y / 1 z y / - a * +']) # Eqn.19 / 25
    else:
        raise ValueError(funcName + '\"mode\" must be in [1, 2, 3, 4]!')

    return mvf.Depth(res, depth=bits, sample=sampleType, **depth_args)


def LLSURE(clip, guidance=None, radius=2, sigma=0, epsilon=1e-5, **depth_args):
    """Local Linear SURE-Based Edge-Preserving Image Filtering

    LLSURE is based on a local linear model and using the principle of Stein’s unbiased risk estimate (SURE) 
    as an estimator for the mean squared error from the noisy image.
    Multiple estimates are aggregated using Variance-based Weighted Average (WAV).

    Most of the internal calculations are done at 32-bit float, except median filtering with radius larger than 1 is done at integer.

    Args:
        clip: Input clip.

        guidance: (clip) Guidance clip used to compute the coefficient of the translation on "clip".
            It must has the same clip properties as 'clip'.
            If it is None, it will be set to 'clip', with duplicate calculations being omitted.
            It is not recommended to use such feature since there might be severe numerical precision problem in this implementation.
            Default is None.

        radius: (int) The radius of box filter and median filter.
            Default is 2.

        sigma: (float or clip) Estimation of noise variance.
            If it is 0, it is automatically calculated using MAD (median absolute deviation).
            If it is smaller than 0, the result is MAD multiplied by the absolute value of "sigma".
            Default is 0.

        epsilon: (float) Small number to avoid divide by 0.
            Default is 0.00001.

        depth_args: (dict) Additional arguments passed to mvf.Depth().
            Default is {}.

    Ref:
        [1] Qiu, T., Wang, A., Yu, N., & Song, A. (2013). LLSURE: local linear SURE-based edge-preserving image filtering. 
            IEEE Transactions on Image Processing, 22(1), 80-90.

    """

    funcName = 'LLSURE'

    if guidance is None:
        guidance = clip
    else:
        if not isinstance(guidance, vs.VideoNode):
            raise TypeError(funcName + ': \"guidance\" must be a clip!')
        if clip.format.id != guidance.format.id:
            raise TypeError(funcName + ': \"guidance\" must be of the same format as \"clip\"!')
        if clip.width != guidance.width or clip.height != guidance.height:
            raise TypeError(funcName + ': \"guidance\" must be of the same size as \"clip\"!')

    bits = clip.format.bits_per_sample
    sampleType = clip.format.sample_type

    Expectation = functools.partial(BoxFilter, radius=radius+1)

    clip_src = clip
    clip = mvf.Depth(clip, depth=32, sample=vs.FLOAT, **depth_args)
    guidance = mvf.Depth(guidance, depth=32, **depth_args) if guidance != clip_src else clip

    mean_guidance = Expectation(guidance)
    guidance_square = core.std.Expr([guidance], ['x dup *'])
    var_guidance = core.std.Expr([Expectation(guidance_square), mean_guidance], ['x y dup * -'])
    inv_var = core.std.Expr([var_guidance], ['1 x {epsilon} + /'.format(epsilon=epsilon)])
    normalized_w = Expectation(inv_var)

    if not isinstance(sigma, vs.VideoNode):
        if sigma <= 0:
            absolute_deviation = core.std.Expr([guidance, mean_guidance], ['x y - abs'])

            if radius == 1:
                sigma_tmp = core.std.Median(absolute_deviation)
            else:
                absolute_deviation = core.fmtc.bitdepth(absolute_deviation, bits=12, dmode=1, fulls=True, fulld=True)
                sigma_tmp = core.ctmf.CTMF(absolute_deviation, radius=radius)
                sigma_tmp = mvf.Depth(sigma_tmp, depth=32, sample=vs.FLOAT, fulls=True, fulld=True)

            sigma = sigma_tmp if sigma == 0 else core.std.Expr([sigma_tmp], ['x {sigma} *'.format(sigma=-sigma)])
        else:
            sigma = core.std.BlankClip(clip, color=[sigma**2] * clip.format.num_planes)

    if guidance == clip:
        a_star = core.std.Expr([var_guidance, sigma, inv_var], ['x y - 0 max z *']) # Eqn. 10 (a)
        b_star = core.std.Expr([a_star, mean_guidance], ['1 x - y *']) # Eqn. 10 (b)
    else: # Joint LLSURE
        mean_clip = Expectation(clip)
        corr_clip_guidance = Expectation(core.std.Expr([clip, guidance], ['x y *']))
        cov_clip_guidance = core.std.Expr([corr_clip_guidance, mean_clip, mean_guidance], ['x y z * -'])
        a_star = core.std.Expr([cov_clip_guidance, sigma, inv_var], ['x 0 > 1 -1 ? x abs y - 0 max * z *']) # Eqn. 20 (a)
        b_star = core.std.Expr([mean_clip, cov_clip_guidance, sigma, inv_var, mean_guidance], ['x y z - a * b * -']) # Eqn. 20 (b)

    bar_a = Expectation(core.std.Expr([a_star, inv_var], ['x y *']))
    bar_b = Expectation(core.std.Expr([b_star, inv_var], ['x y *']))
    res = core.std.Expr([bar_a, guidance, bar_b, normalized_w], ['x y * z + a {epsilon} + /'.format(epsilon=epsilon)]) # Eqn. 17 / 21

    return mvf.Depth(res, depth=bits, sample=sampleType, **depth_args)


def YAHRmod(clp, blur=2, depth=32, **limit_filter_args):
    """Modification of YAHR with better texture preserving property

    The YAHR() is a simple and powerful script to reduce halos from over enhanced edges.
    It simply creates two versions of ringing-free result and uses the difference of the source-deringed
    pairs to restore the texture.
    However, it still suffers from texture degradation due to the unconstrained use of MinBlur() in texture area.
    Inspired by the observation that the Repair(13) used in YAHR() has the characteristics of preserving the
    source signal if it is closed to the reference in the same location, i.e. the source signal will be output
    if the two filtered results are closed, we simply add an LimitFilter() before the repair procedure to utilize
    this property to preserve the texture.

    Experiment can denmonstrate its better texture preserving performance over the original version.

    The source code is modified from 
    havsfunc(https://github.com/HomeOfVapourSynthEvolution/havsfunc/blob/2048fcb320ef8121c842d087191708d61f39416b/havsfunc.py#L644-L671).

    Args:
        clp: Input clip.

        blur: (int) "blur" parameter of AWarpSharp2.
            Default is 2.

        depth: (int) "depth" parameter of AWarpSharp2.
            Default is 32.

        limit_filter_args: (dict) Additional arguments passed to mvf.LimitFilter in the form of keyword arguments.

    """

    funcName = 'YAHRmod'

    if not isinstance(clp, vs.VideoNode):
        raise TypeError(funcName + ': \"clp\" must be a clip!')

    if clp.format.color_family != vs.GRAY:
        clp_orig = clp
        clp = mvf.GetPlane(clp, 0)
    else:
        clp_orig = None

    b1 = core.std.Convolution(haf.MinBlur(clp, 2), matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1])
    b1D = core.std.MakeDiff(clp, b1)
    w1 = haf.Padding(clp, 6, 6, 6, 6).warp.AWarpSharp2(blur=blur, depth=depth).std.Crop(6, 6, 6, 6)
    w1b1 = core.std.Convolution(haf.MinBlur(w1, 2), matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1])
    w1b1D = core.std.MakeDiff(w1, w1b1)
    w1b1D = mvf.LimitFilter(b1D, w1b1D, **limit_filter_args) # The only modification
    DD = core.rgvs.Repair(b1D, w1b1D, 13)
    DD2 = core.std.MakeDiff(b1D, DD)
    last = core.std.MakeDiff(clp, DD2)

    if clp_orig is not None:
        return core.std.ShufflePlanes([last, clp_orig], planes=[0, 1, 2], colorfamily=clp_orig.format.color_family)
    else:
        return last


def RandomInterleave(clips, seed=None, rand_list=None):
    """Returns a clip with the frames from all clips randomly interleaved

    Useful for blinded-experiment.

    Args:
        clips: Input clips with same formats.

        seed: (int) Random number generator initializer.
            Default is None.

        rand_list: (list) A list containing frame mappings of the interleaved clip.
            For example, [0, 0, 1] stats that the first two frames of the output clip
                are obtained from the first clip in "clips", while the third frame is
                obtained from the second clip in "clips".
            Default is None.

    """

    funcName = 'RandomInterleave'

    if not isinstance(clips, list):
        raise TypeError(funcName + ': \"clips\" must be a list of clips!')

    length = len(clips)

    if rand_list is None:
        import random
        random.seed(seed)

        tmp = list(range(length))

        rand_list = []

        for i in range(clips[0].num_frames):
            random.shuffle(tmp)
            rand_list += tmp

    for i in range(length):
        clips[i] = core.std.Interleave([clips[i]] * length)

    def selector(n, f):
        return f[rand_list[n]]

    return core.std.ModifyFrame(clips[0], clips=clips, selector=selector)


def super_resolution(clip, model_filename, epoch=0, up_scale=2, block_w=128, block_h=None, is_rgb_model=True, pad=None, crop=None, 
    pre_upscale=False, upscale_uv=False, merge_source=False, use_fmtc=False, resample_kernel=None, resample_args=None, pad_mode=None, 
    framework=None, data_format=None, device_id=0, use_plugins_padding=False):
    ''' Use MXNet to accelerate Image-Processing in VapourSynth using C++ interface

    Drop-in replacement of muvsfunc_numpy's counterpart using core.mx.Predict().
    The plugin can be downloaded from https://github.com/kice/vs_mxnet

    The results from two versions of the functinos may not identical when the size of block is smaller than the frame
        or padding is used, due to different implementation.

    Currently only MXNet backend is supported. Multi-GPU data parallelism is supported.

    The color space and bit depth of the output depends on the super resolution algorithm.
    Currently only RGB and GRAY models are supported.

    All the internal calculations are done at 32-bit float.

    Demo:
        https://github.com/WolframRhodium/muvsfunc/blob/master/Collections/examples/super_resolution_mxnet.vpy

    Args:
        clip: Input clip.
            The color space will be automatically converted by mvf.ToRGB/YUV if it is not
            compatiable with the super resolution algorithm.

        model_filename: Path to the pre-trained model.
            This specifies the path prefix of saved model files.
            You should have "model_filename-symbol.json", "model_filename-xxxx.params", where xxxx is the epoch number.

        epoch: (int) Epoch to load of MXNet model file.
            Default is 0.

        up_scale: (int) Upscaling factor.
            Should be compatiable with the model.
            Default is 2.

        block_w, block_h: (int) The horizontal/vertical block size for dividing the image during processing.
            The optimal value may vary according to different graphics card and image size.
            Default is 128.

        is_rgb_model: (bool) Whether the model is RGB model.
            If not, it is assumed to be Y model, and RGB input will be converted to YUV before feeding to the network
            Default is True.

        pad: (list of four ints) Patch-wise padding before upscaling.
            The four values indicate padding at top, bottom, left, right of each patch respectively.
            Default is None.

        crop: (list of four ints) Patch-wise cropping after upscaling.
            The four values indicate cropping at top, bottom, left, right of each patch respectively.
            Moreover, due to the implementation of vs_mxnet, the values at top and left should be zero.
            Default is None.

        pre_upscale: (bool) Whether to upscale the image before feed to the network.
            If true, currently the function will only upscale the whole image directly rather than upscale
                the patches separately, which may results in blocking artifacts on some algorithms.
            Default is False.

        upscale_uv: (bool) Whether to upscale UV channels when using Y model.
            If not, the UV channels will be discarded.
            Only works when "is_rgb_model" is False.
            Default is False.

        merge_source: (bool) Whether to merge the output of the network to the (nearest/bilinear/bicubic) enlarged source image.
            Default is False.

        use_fmtc: (bool) Whether to use fmtconv for enlarging. If not, vszimg (core.resize.*) will be used.
            Only works when "pre_upscale" is True.
            Default is False.

        resample_kernel: (str) Resample kernel.
            If can be 'Catmull-Rom', i.e. BicubicResize with b=0 and c=0.5.
            Only works when "pre_upscale" is True.
            Default is 'Catmull-Rom'.

        resample_args: (dict) Additional arguments passed to vszimg/fmtconv resample kernel.
            Only works when "pre_upscale" is True.
            Default is {}.

        pad_mode: (str) Padding type to use.
            If set to "source", the pixels in the source image will be used.
            Only "source" is supported. Please switch to muvsfunc_numpy's implementation for other modes.
            Default is "source"

        framework: INVALID. Please switch to muvsfunc_numpy's implementation.

        data_format: INVALID. Please switch to muvsfunc_numpy's implementation.

        device_id: (int or list of ints) Which device(s) to use. 
            Starting with 0. If it is smaller than 0, CPU will be used.
            It can be a list of integers, indicating devices for multi-GPU data parallelism.
            Default is 0.

        use_plugins_padding: (bool) Whether to use core.mx.Predict()'s built-in OpenCV based inplace padding'.
            If not, vszimg (core.resize.Point) will be used.
            Default is False.

    '''

    funcName = 'super_resolution'

    isGray = clip.format.color_family == vs.GRAY
    isRGB = clip.format.color_family == vs.RGB

    symbol_filename = model_filename + '-symbol.json'
    param_filename = model_filename + '-{:04d}.params'.format(epoch)

    if block_h is None:
        block_h = block_w

    if pad is None:
        pad = (0, 0, 0, 0)

    if crop is None:
        crop = (0, 0, 0, 0)
    else:
        if crop[0] != 0 or crop[2] != 0:
            raise ValueError(funcName + ': Cropping at left or top should be zero! Please switch to muvsfunc_numpy\'s implementation.')

    if resample_kernel is None:
        resample_kernel = 'Bicubic'

    if resample_args is None:
        resample_args = {}

    if pad_mode is not None and pad_mode.lower() != 'source':
        raise ValueError(funcName + ': Only source padding mode is supported! Please switch to muvsfunc_numpy\'s implementation.')

    if framework is None:
        framework = 'MXNet'
    framework = framework.lower()
    if framework.lower() != 'mxnet':
        raise ValueError(funcName + ': Only MXNet framework is supported! Please switch to muvsfunc_numpy\'s implementation.')
    else:
        import mxnet as mx

        if not hasattr(core, 'mx'):
            core.std.LoadPlugin(r'vs_mxnet.dll', altsearchpath=True)        

    if data_format is None:
        data_format = 'NCHW'
    else:
        data_format = data_format.upper()
        if data_format != 'NCHW':
            raise ValueError(funcName + ': Only NCHW data format is supported! Please switch to muvsfunc_numpy\'s implementation.')

    if not isinstance(device_id, list) or not isinstance(device_id, tuple):
        device_id = [device_id]

    if use_plugins_padding and not pad[0] == pad[1] == pad[2] == pad[3]:
        raise ValueError(funcName + ': \'use_plugins_padding\' only allows symmetric padding! Please set its value to False!')

    # color space conversion
    if is_rgb_model and not isRGB:
        clip = mvf.ToRGB(clip, depth=32)

    elif not is_rgb_model:
        if isRGB:
            clip = mvf.ToYUV(clip, depth=32)

        if not isGray:
            if not upscale_uv: # isYUV/RGB and only upscale Y
                clip = mvf.GetPlane(clip)
            else:
                clip = core.std.Expr([clip], ['', 'x 0.5 +']) # change the range of UV from [-0.5, 0.5] to [0, 1]

    # bit depth conversion
    clip = mvf.Depth(clip, depth=32, sample=vs.FLOAT)

    # pre-upscaling
    if pre_upscale:
        if up_scale != 1:
            if use_fmtc:
                if resample_kernel.lower() == 'catmull-rom':
                    clip = core.fmtc.resample(clip, clip.width*up_scale, clip.height*up_scale, kernel='bicubic', a1=0, a2=0.5, **resample_args)
                else:
                    clip = core.fmtc.resample(clip, clip.width*up_scale, clip.height*up_scale, kernel=resample_kernel, **resample_args)
            else: # use vszimg
                if resample_kernel.lower() == 'catmull-rom':
                    clip = core.resize.Bicubic(clip, clip.width*up_scale, clip.height*up_scale, filter_param_a=0, filter_param_b=0.5, **resample_args)
                else:
                    clip = eval('core.resize.{kernel}(clip, clip.width*up_scale, clip.height*up_scale, **resample_args)'.format(
                        kernel=resample_kernel.capitalize()))

            up_scale = 1

    # inference
    def inference(clip, dev_id):
        '''wrapper function for inference'''

        if is_rgb_model or not upscale_uv:
            w, h = clip.width, clip.height

            if not use_plugins_padding and (pad[0]-crop[0]//up_scale > 0 or pad[1]-crop[1]//up_scale > 0 or 
                pad[2]-crop[2]//up_scale > 0 or pad[3]-crop[3]//up_scale > 0):

                clip = haf.Padding(clip, pad[2]-crop[2]//up_scale, pad[3]-crop[3]//up_scale, 
                    pad[0]-crop[0]//up_scale, pad[1]-crop[1]//up_scale)

            super_res = core.mx.Predict(clip, symbol=symbol_filename, param=param_filename, 
                patch_w=block_w+pad[2]+pad[3], patch_h=block_h+pad[0]+pad[1], scale=up_scale, 
                output_w=block_w*up_scale+crop[2]+crop[3], output_h=block_h*up_scale+crop[0]+crop[1], # crop[0] == crop[2] == 0
                frame_w=w*up_scale, frame_h=h*up_scale, step_w=block_w, step_h=block_h, 
                outstep_w=block_w*up_scale, outstep_h=block_h*up_scale, 
                padding=pad[0]-crop[0]//up_scale if use_plugins_padding else 0, 
                ctx=2 if dev_id >= 0 else 1, dev_id=max(dev_id, 0))

        else: # Y model, YUV input that may have subsampling, need to upscale uv
            num_planes = clip.format.num_planes
            yuv_list = [mvf.GetPlane(clip, i) for i in range(num_planes)]

            for i in range(num_planes):
                w, h = yuv_list[i].width, yuv_list[i].height

                if not use_plugins_padding and (pad[0]-crop[0]//up_scale > 0 or pad[1]-crop[1]//up_scale > 0 or 
                    pad[2]-crop[2]//up_scale > 0 or pad[3]-crop[3]//up_scale > 0):

                    yuv_list[i] = haf.Padding(yuv_list[i], pad[2]-crop[2]//up_scale, pad[3]-crop[3]//up_scale, 
                        pad[0]-crop[0]//up_scale, pad[1]-crop[1]//up_scale)

                yuv_list[i] = core.mx.Predict(yuv_list[i], symbol=symbol_filename, param=param_filename, 
                    patch_w=block_w+pad[2]+pad[3], patch_h=block_h+pad[0]+pad[1], scale=up_scale, 
                    output_w=block_w*up_scale+crop[2]+crop[3], output_h=block_h*up_scale+crop[0]+crop[1], # crop[0] == crop[2] == 0
                    frame_w=w*up_scale, frame_h=h*up_scale, step_w=block_w, step_h=block_h, 
                    outstep_w=block_w*up_scale, outstep_h=block_h*up_scale, 
                    padding=pad[0]-crop[0]//up_scale if use_plugins_padding else 0, 
                    ctx=2 if dev_id >= 0 else 1, dev_id=max(dev_id, 0))

            super_res = core.std.ShufflePlanes(yuv_list, [0] * num_planes, clip.format.color_family)

        return super_res

    if len(device_id) == 1:
        super_res = inference(clip, device_id[0])

    else: # multi-GPU data parallelism
        workers = len(device_id)
        super_res_list = [inference(clip[i::workers], device_id[i]) for i in range(workers)]
        super_res = core.std.Interleave(super_res_list)

    # post-processing
    if not is_rgb_model and not isGray and upscale_uv:
        super_res = core.std.Expr([super_res], ['', 'x 0.5 -']) # restore the range of UV

    if merge_source:
        if up_scale != 1:
            if use_fmtc:
                if resample_kernel.lower() == 'catmull-rom':
                    low_res = core.fmtc.resample(clip, super_res.width, super_res.height, kernel='bicubic', a1=0, a2=0.5, **resample_args)
                else:
                    low_res = core.fmtc.resample(clip, super_res.width, super_res.height, kernel=resample_kernel, **resample_args)
            else: # use vszimg
                if resample_kernel.lower() == 'catmull-rom':
                    low_res = core.resize.Bicubic(clip, super_res.width, super_res.height, filter_param_a=0, filter_param_b=0.5, **resample_args)
                else:
                    low_res = eval('core.resize.{kernel}(clip, {w}, {h}, **resample_args)'.format(w=super_res.width, h=super_res.height, 
                        kernel=resample_kernel.capitalize()))
        else:
            low_res = clip

        super_res = core.std.Expr([super_res, low_res], ['x y +'])

    return super_res
