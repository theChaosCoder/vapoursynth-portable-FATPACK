import vapoursynth as vs
import re
from functools import partial
import havsfunc as haf  # https://github.com/HomeOfVapourSynthEvolution/havsfunc
import mvsfunc as mvf  # https://github.com/HomeOfVapourSynthEvolution/mvsfunc
import muvsfunc as muf  # https://github.com/WolframRhodium/muvsfunc
import nnedi3_rpow2  # https://gist.github.com/4re/342624c9e1a144a696c6

# Small collection of VapourSynth functions I used at least once.
# Most are simple wrappers or ports of AviSynth functions.

# Included functions:
#
#       GradFun3mod
#       DescaleM (DebilinearM, DebicubicM etc.)
#       Downscale444
#       JIVTC
#       OverlayInter
#       AutoDeblock
#       ReplaceFrames (ReplaceFramesSimple)
#       maa
#       TemporalDegrain
#       DescaleAA
#       InsertSign


core = vs.core


"""
VapourSynth port of Gebbi's GradFun3mod

Based on Muonium's GradFun3 port:
https://github.com/WolframRhodium/muvsfunc

If you don't use any of the newly added arguments
it will behave just like unmodified GradFun3.

Differences:

 - added smode=5 that uses a bilateral filter on the GPU (CUDA)
   output should be very similar to smode=2
 - fixed the strength of the bilateral filter when using 
   smode=2 to match the AviSynth version
 - changed argument lsb to bits (default is input bitdepth)
 - case of the resizer doesn't matter anymore
 - every resizer supported by fmtconv.resample can be specified
 - yuv444 can now be used with any output resolution
 - removed fh and fv arguments for all resizers

Requirements:

 - muvsfunc  https://github.com/WolframRhodium/muvsfunc
 - havsfunc  https://github.com/HomeOfVapourSynthEvolution/havsfunc
 - mvsfunc  https://github.com/HomeOfVapourSynthEvolution/mvsfunc
 - Bilateral  https://github.com/HomeOfVapourSynthEvolution/VapourSynth-Bilateral
 - BilateralGPU (optional, needs OpenCV 3.2 with CUDA module)  https://github.com/WolframRhodium/VapourSynth-BilateralGPU
 - fmtconv  https://github.com/EleonoreMizo/fmtconv
 - Descale (optional)  https://github.com/Frechdachs/vapoursynth-descale
 - dfttest  https://github.com/HomeOfVapourSynthEvolution/VapourSynth-DFTTest
 - nnedi3  https://github.com/dubhater/vapoursynth-nnedi3
 - nnedi3_rpow2  https://gist.github.com/4re/342624c9e1a144a696c6

Original header:

##################################################################################################################
#
#   High bitdepth tools for Avisynth - GradFun3mod r6
#       based on Dither v1.27.2
#   Author: Firesledge, slightly modified by Gebbi
#
#  What?
#       - This is a slightly modified version of the original GradFun3.
#       - It combines the usual color banding removal stuff with resizers during the process
#         for sexier results (less detail loss, especially for downscales of cartoons).
#       - This is a starter script, not everything is covered through parameters. Modify it to your needs.
#
#   Requirements (in addition to the Dither requirements):
#       - AviSynth 2.6.x
#       - Debilinear, Debicubic, DebilinearM
#       - NNEDI3 + nnedi3_resize16
#
#  Changes from the original GradFun3:
#       - yuv444 = true
#         (4:2:0 -> 4:4:4 colorspace conversion, needs 1920x1080 input)
#       - resizer = [ "none", "Debilinear", "DebilinearM", "Debicubic", "DebicubicM", "Spline16",
#         "Spline36", "Spline64", "lineart_rpow2", "lineart_rpow2_bicubic" ] 
#         (use it only for downscales)
#           NOTE: As of r2 Debicubic doesn't have 16-bit precision, so a Y (luma) plane fix by torch is used here,
#                 more info: https://mechaweaponsvidya.wordpress.com/2015/07/07/a-precise-debicubic/
#                 Without yuv444=true Dither_resize16 is used with an inverse bicubic kernel.
#       - w = 1280, h = 720
#         (output width & height for the resizers; or production resolution for resizer="lineart_rpow2")
#       - smode = 4
#         (the old GradFun3mod behaviour for legacy reasons; based on smode = 1 (dfttest);
#         not useful anymore in most cases, use smode = 2 instead (less detail loss))
#       - deb = true
#         (legacy parameter; same as resizer = "DebilinearM")
#
#  Usage examples:
#       - Source is bilinear 720p->1080p upscale (BD) with 1080p credits overlayed,
#         revert the upscale without fucking up the credits:
#               lwlibavvideosource("lol.m2ts")
#               GradFun3mod(smode=1, yuv444=true, resizer="DebilinearM")
#
#       - same as above, but bicubic Catmull-Rom upscale (outlines are kind of "blocky" and oversharped):
#               GradFun3mod(smode=1, yuv444=true, resizer="DebicubicM", b=0, c=1)
#               (you may try any value between 0 and 0.2 for b, and between 0.7 and 1 for c)
#
#       - You just want to get rid off the banding without changing the resolution:
#               GradFun3(smode=2)
#
#       - Source is 1080p production (BD), downscale to 720p:
#               GradFun3mod(smode=2, yuv444=true, resizer="Spline36")
#
#       - Source is a HDTV transportstream (or CR or whatever), downscale to 720p:
#               GradFun3mod(smode=2, resizer="Spline36")
#
#       - Source is anime, 720p->1080p upscale, keep the resolution
#         but with smoother lineart instead of bilinear upscaled shit:
#               GradFun3mod(smode=2, resizer="lineart_rpow2")
#         This won't actually resize the video but instead mask the lineart and re-upscale it using
#         nnedi3_rpow2 which often results in much better looking lineart (script mostly by Daiz).
#
#       Note: Those examples don't include parameters like thr, radius, elast, mode, ampo, ampn, staticnoise.
#             You probably don't want to use the default values.
#             For 16-bit output use:
#              GradFun3mod(lsb=true).Dither_out()
#
#  What's the production resolution of my korean cartoon?
#       - Use your eyes combined with Debilinear(1280,720) - if it looks like oversharped shit,
#         it was probably produced in a higher resolution.
#       - Use Debilinear(1280,720).BilinearResize(1920,1080) for detail loss search.
#       - Alternatively you can lookup the (estimated) production resolution at
#         http://anibin.blogspot.com  (but don't blindly trust those results)
#
#   This program is free software. It comes without any warranty, to
#   the extent permitted by applicable law. You can redistribute it
#   and/or modify it under the terms of the Do What The Fuck You Want
#   To Public License, Version 2, as published by Sam Hocevar. See
#   http://sam.zoy.org/wtfpl/COPYING for more details.
#
##################################################################################################################

"""
def GradFun3(src, thr=None, radius=None, elast=None, mask=None, mode=None, ampo=None,
                ampn=None, pat=None, dyn=None, staticnoise=None, smode=None, thr_det=None,
                debug=None, thrc=None, radiusc=None, elastc=None, planes=None, ref=None,
                yuv444=None, w=None, h=None, resizer=None, b=None, c=None, bits=None):

    def smooth_mod(src_16, ref_16, smode, radius, thr, elast, planes):
        if smode == 0:
            return muf._GF3_smoothgrad_multistage(src_16, ref_16, radius, thr, elast, planes)
        elif smode == 1:
            return muf._GF3_dfttest(src_16, ref_16, radius, thr, elast, planes)
        elif smode == 2:
            return bilateral(src_16, ref_16, radius, thr, elast, planes)
        elif smode == 3:
            return muf._GF3_smoothgrad_multistage_3(src_16, radius, thr, elast, planes)
        elif smode == 4:
            return dfttest_mod(src_16, ref_16, radius, thr, elast, planes)
        elif smode == 5:
            return bilateral_gpu(src_16, ref_16, radius, thr, elast, planes)
        else:
            raise ValueError(funcname + ': wrong smode value!')

    def dfttest_mod(src, ref, radius, thr, elast, planes):
        hrad = max(radius * 3 // 4, 1)
        last = core.dfttest.DFTTest(src, sigma=thr * 12, sbsize=hrad * 4,
                                    sosize=hrad * 3, tbsize=1, planes=planes)
        last = mvf.LimitFilter(last, ref, thr=thr, elast=elast, planes=planes)
        return last

    def bilateral(src, ref, radius, thr, elast, planes):
        thr_1 = max(thr * 4.5, 1.25)
        thr_2 = max(thr * 9, 5.0)
        r4 = max(radius * 4 / 3, 4.0)
        r2 = max(radius * 2 / 3, 3.0)
        r1 = max(radius * 1 / 3, 2.0)
        last = src
        last = core.bilateral.Bilateral(last, ref=ref, sigmaS=r4 / 2, sigmaR=thr_1 / 255,
                                        planes=planes, algorithm=0)
        # NOTE: I get much better results if I just call Bilateral once
        #last = core.bilateral.Bilateral(last, ref=ref, sigmaS=r2 / 2, sigmaR=thr_2 / 255,
        #                                planes=planes, algorithm=0)
        #last = core.bilateral.Bilateral(last, ref=ref, sigmaS=r1 / 2, sigmaR=thr_2 / 255,
        #                                planes=planes, algorithm=0)
        last = mvf.LimitFilter(last, src, thr=thr, elast=elast, planes=planes)
        return last

    def bilateral_gpu(src, ref, radius, thr, elast, planes):
        t = max(thr * 4.5, 1.25)
        r = max(radius * 4 / 3, 4.0)
        last = core.bilateralgpu.Bilateral(src, sigma_spatial=r / 2, sigma_color=t,
                                        planes=planes, kernel_size=0, borderMode=4)
        last = mvf.LimitFilter(last, ref, thr=thr, elast=elast, planes=planes)
        return last

    funcname = 'GradFun3'

    # Type checking
    kwargsdict = {'src': [src, (vs.VideoNode,)], 'thr': [thr, (int, float)], 'radius': [radius, (int,)],
                  'elast': [elast, (int, float)], 'mask': [mask, (int,)], 'mode': [mode, (int,)],
                  'ampo': [ampo, (int, float)], 'ampn': [ampn, (int, float)], 'pat': [pat, (int,)],
                  'dyn': [dyn, (bool,)], 'staticnoise': [staticnoise, (bool,)], 'smode': [smode, (int,)],
                  'thr_det': [thr_det, (int, float)], 'debug': [debug, (bool, int)], 'thrc': [thrc, (int, float)],
                  'radiusc': [radiusc, (int,)], 'elastc': [elastc, (int, float)], 'planes': [planes, (int, list)],
                  'ref': [ref, (vs.VideoNode,)], 'yuv444': [yuv444, (bool,)], 'w': [w, (int,)], 'h': [h, (int,)],
                  'resizer': [resizer, (str,)], 'b': [b, (int, float)], 'c': [c, (int, float)], 'bits': [bits, (int,)]}

    for k, v in kwargsdict.items():
        if v[0] is not None and not isinstance(v[0], v[1]):
            raise TypeError('{funcname}: "{variable}" must be {types}!'
                            .format(funcname=funcname, variable=k, types=' or '.join([TYPEDICT[t] for t in v[1]])))

    # Set defaults
    if smode is None:
        smode = 2
    if thr is None:
        thr = 0.35
    if radius is None:
        radius = 12 if smode not in [0, 3] else 9
    if elast is None:
        elast = 3.0
    if mask is None:
        mask = 2
    if thr_det is None:
        thr_det = 2 + round(max(thr - 0.35, 0) / 0.3)
    if debug is None:
        debug = False
    if thrc is None:
        thrc = thr
    if radiusc is None:
        radiusc = radius
    if elastc is None:
        elastc = elast
    if planes is None:
        planes = list(range(src.format.num_planes))
    if ref is None:
        ref = src
    if yuv444 is None:
        yuv444 = False
    if w is None:
        w = 1280
    if h is None:
        h = 720
    if resizer is None:
        resizer = ''
    if yuv444 and not resizer:
        resizer = 'spline36'
    if b is None:
        b = 1/3
    if c is None:
        c = 1/3
    if bits is None:
        bits = src.format.bits_per_sample

    # Value checking
    if src.format.color_family not in [vs.YUV, vs.GRAY, vs.YCOCG]:
        raise TypeError(funcname + ': "src" must be YUV, GRAY or YCOCG color family!')
    if ref.format.color_family not in [vs.YUV, vs.GRAY, vs.YCOCG]:
        raise TypeError(funcname + ': "ref" must be YUV, GRAY or YCOCG color family!')
    if thr < 0.1 or thr > 10.0:
        raise ValueError(funcname + ': "thr" must be in [0.1, 10.0]!')
    if thrc < 0.1 or thrc > 10.0:
        raise ValueError(funcname + ': "thrc" must be in [0.1, 10.0]!')
    if radius <= 0:
        raise ValueError(funcname + ': "radius" must be positive.')
    if radiusc <= 0:
        raise ValueError(funcname + ': "radiusc" must be positive.')
    if elast < 1:
        raise ValueError(funcname + ': Valid range of "elast" is [1, +inf)!')
    if elastc < 1:
        raise ValueError(funcname + ': Valid range of "elastc" is [1, +inf)!')
    if smode not in [0, 1, 2, 3, 4, 5]:
        raise ValueError(funcname + ': "smode" must be in [0, 1, 2, 3, 4, 5]!')
    if smode in [0, 3]:
        if radius not in list(range(2, 10)):
            raise ValueError(funcname + ': "radius" must be in 2-9 for smode=0 or 3 !')
        if radiusc not in list(range(2, 10)):
            raise ValueError(funcname + ': "radiusc" must be in 2-9 for smode=0 or 3 !')
    elif smode in [1, 4]:
        if radius not in list(range(1, 129)):
            raise ValueError(funcname + ': "radius" must be in 1-128 for smode=1 or smode=4 !')
        if radiusc not in list(range(1, 129)):
            raise ValueError(funcname + ': "radiusc" must be in 1-128 for smode=1 or smode=4 !')
    if thr_det <= 0.0:
        raise ValueError(funcname + ': "thr_det" must be positive!')

    ow = src.width
    oh = src.height

    src_16 = core.fmtc.bitdepth(src, bits=16, planes=planes) if src.format.bits_per_sample < 16 else src
    src_8 = core.fmtc.bitdepth(src, bits=8, dmode=1, planes=[0]) if src.format.bits_per_sample != 8 else src
    ref_16 = core.fmtc.bitdepth(ref, bits=16, planes=planes) if ref.format.bits_per_sample < 16 else ref

    # Do lineart smoothing first for sharper results
    if resizer.lower() == 'lineart_rpow2':
        src_16 = ProtectedDebiXAA(src_16, w, h, bicubic=False)
    elif resizer.lower() == 'lineart_rpow2_bicubic':
        src_16 = ProtectedDebiXAA(src_16, w, h, bicubic=True, b=b, c=c)

    # Main debanding
    chroma_flag = (thrc != thr or radiusc != radius or
                   elastc != elast) and 0 in planes and (1 in planes or 2 in planes)

    if chroma_flag:
        planes2 = [0] if 0 in planes else []
    else:
        planes2 = planes

    if not planes2:
        raise ValueError(funcname + ': no plane is processed')

    flt_y = smooth_mod(src_16, ref_16, smode, radius, thr, elast, planes2)
    if chroma_flag:
        flt_c = smooth_mod(src_16, ref_16, smode, radiusc, thrc, elastc, [x for x in planes if x != 0])
        flt = core.std.ShufflePlanes([flt_y,flt_c], [0,1,2], src.format.color_family)
    else:
        flt = flt_y

    # Edge/detail mask
    td_lo = max(thr_det * 0.75, 1.0)
    td_hi = max(thr_det, 1.0)
    mexpr = 'x {tl} - {th} {tl} - / 255 *'.format(tl=td_lo - 0.0001, th=td_hi + 0.0001)

    if mask > 0:
        dmask = mvf.GetPlane(src_8, 0)
        dmask = muf._Build_gf3_range_mask(dmask, mask)
        dmask = core.std.Expr([dmask], [mexpr])
        dmask = core.rgvs.RemoveGrain(dmask, [22])
        if mask > 1:
            dmask = core.std.Convolution(dmask, matrix=[1,2,1,2,4,2,1,2,1])
            if mask > 2:
                dmask = core.std.Convolution(dmask, matrix=[1,1,1,1,1,1,1,1,1])
        dmask = core.fmtc.bitdepth(dmask, bits=16)
        res_16 = core.std.MaskedMerge(flt, src_16, dmask, planes=planes, first_plane=True)
    else:
        res_16 = flt

    # Resizing / colorspace conversion (GradFun3mod)
    res_16_y = core.std.ShufflePlanes(res_16, planes=0, colorfamily=vs.GRAY)
    if resizer.lower() == 'debilinear':
        rkernel = Resize(res_16_y if yuv444 else res_16, w, h, kernel='bilinear', invks=True)
    elif resizer.lower() == 'debicubic':
        rkernel = Resize(res_16_y if yuv444 else res_16, w, h, kernel='bicubic', a1=b, a2=c, invks=True)
    elif resizer.lower() == 'debilinearm':
        rkernel = DebilinearM(res_16_y if yuv444 else res_16, w, h, chroma=not yuv444)
    elif resizer.lower() == 'debicubicm':
        rkernel = DebicubicM(res_16_y if yuv444 else res_16, w, h, b=b, c=c, chroma=not yuv444)
    elif resizer.lower() in ('lineart_rpow2', 'lineart_rpow2_bicubic'):
        if yuv444:
            rkernel = Resize(res_16_y, w, h, kernel='spline36')
        else:
            rkernel = res_16
    elif not resizer:
        rkernel = res_16
    else:
       rkernel = Resize(res_16_y if yuv444 else res_16, w, h, kernel=resizer.lower())

    if yuv444:
        ly = rkernel
        lu = core.std.ShufflePlanes(res_16, planes=1, colorfamily=vs.GRAY)
        lv = core.std.ShufflePlanes(res_16, planes=2, colorfamily=vs.GRAY)
        lu = Resize(lu, w, h, kernel='spline16', sx=0.25)
        lv = Resize(lv, w, h, kernel='spline16', sx=0.25)
        rkernel = core.std.ShufflePlanes([ly,lu,lv], planes=[0,0,0], colorfamily=vs.YUV)
    res_16 = rkernel

    # Dithering
    result = res_16 if bits == 16 else core.fmtc.bitdepth(res_16, bits=bits, planes=planes, dmode=mode, ampo=ampo,
                                                          ampn=ampn, dyn=dyn, staticnoise=staticnoise, patsize=pat)

    if debug:
        last = dmask
        if bits != 16:
            last = core.fmtc.bitdepth(last, bits=bits)
    else:
        last = result
    return last


# GradFun3 alias
GradFun3mod = GradFun3

# GradFun3 alias
gf3 = GradFun3


"""
VapourSynth port of DebilinearM

Currently only YUV420 and YUV422 input makes sense

Differences:

 - changed the cubic argument to descale_kernel,
   so that this function is not limited to bilinear or bicubic
 - chroma is never scaled with an inverted kernel
 - added yuv444 argument to convert to yuv444
 - added arguments to fine tune the resizers

Usage:

  It is recommended to use the function alias for the desired kernel:

    DebilinearM(clip, 1280, 720)
    DebicubicM(clip, 1280, 720, b=0, c=0.5)
    DelanczosM(clip, 1280, 720, taps=3)
    Despline16M(clip, 1280, 720)
    Despline36M(clip, 1280, 720)

Original header:

DebilinearM is a wrapper function for the Debilinear and Debicubic plugins that masks parts of the frame that aren't upscaled,
such as text overlays, and uses a regular resize kernel to downscale those areas. It works by downscaling the input
clip to the target resolution with Debilinear or Debicubic, upscaling it again, comparing it to the original clip,
and masking pixels that have a difference greater than the specified threshold.

"""
def DescaleM(src, w, h, thr=None, expand=None, inflate=None, descale_kernel=None, kernel=None, kernely=None, kerneluv=None,
             taps=None, tapsy=None, tapsuv=None, a1=None, a2=None, a1y=None, a2y=None, a1uv=None, a2uv=None, b=None, c=None,
             chroma=None, yuv444=None, showmask=None, ow=None, oh=None):

    # Type checking
    kwargsdict = {'src': [src, (vs.VideoNode,)], 'w': [w, (int,)], 'h': [h, (int,)], 'thr': [thr, (int,)],
                  'expand': [expand, (int,)], 'inflate': [inflate, (int,)],'descale_kernel': [descale_kernel, (str,)],
                  'kernel': [kernel, (str,)], 'kernely': [kernely, (str,)], 'kerneluv': [kerneluv, (str,)],
                  'taps': [taps, (int,)], 'tapsy': [tapsy, (int,)], 'tapsuv': [tapsuv, (int,)], 'a1': [a1, (int, float)],
                  'a2': [a2, (int, float)], 'a1y': [a1y, (int, float)], 'a2y': [a2y, (int, float)],
                  'a1uv': [a1uv, (int, float)], 'a2uv': [a2uv, (int, float)], 'b': [b, (int, float)],
                  'c': [c, (int, float)], 'chroma': [chroma, (bool,)], 'yuv444': [yuv444, (bool,)],
                  'showmask': [showmask, (int,)], 'ow': [ow, (int,)], 'oh': [oh, (int,)]}

    for k, v in kwargsdict.items():
        if v[0] is not None and not isinstance(v[0], v[1]):
            raise TypeError('DescaleM: "{variable}" must be {types}!'
                            .format(variable=k, types=' or '.join([TYPEDICT[t] for t in v[1]])))

    # Set defaults
    if thr is None:
        thr = 10
    if expand is None:
        expand = 1
    if inflate is None:
        inflate = 2
    if chroma is None:
        chroma = True
    if src.format.num_planes == 1:
        chroma = False
    if yuv444 is None:
        yuv444 = False
    if showmask is None:
        showmask = 0
    if descale_kernel is None:
        descale_kernel = 'bilinear'
    elif descale_kernel.lower().startswith('de'):
        descale_kernel = descale_kernel[2:]
    if kernely is None:
        kernely = kernel
    if kerneluv is None:
        kerneluv = kernel
    if tapsy is None:
        tapsy = taps
    if tapsuv is None:
        tapsuv = taps
    if a1y is None:
        a1y = a1
    if a2y is None:
        a2y = a2
    if a1uv is None:
        a1uv = a1
    if a2uv is None:
        a2uv = a2
    if ow is None:
        ow = w
    if oh is None:
        oh = h

    # Value checking
    if thr < 0 or thr > 0xFF:
        raise ValueError('DebilinearM: "thr" must be in the range of 0 and 255!')
    if showmask < 0 or showmask > 2:
        raise ValueError('DebilinearM: "showmask" must be 0, 1 or 2!')
    if yuv444 and not chroma:
        raise ValueError('DebilinearM: "yuv444=True" and "chroma=False" cannot be used at the same time!')

    src_w = src.width
    src_h = src.height

    bits = src.format.bits_per_sample
    sample_type = src.format.sample_type
    
    if sample_type == vs.INTEGER:
        maxvalue = (1 << bits) - 1
        thr = thr * maxvalue // 0xFF
    else:
        thr /= (235 - 16)

    # Resizing
    src_y = core.std.ShufflePlanes(src, planes=0, colorfamily=vs.GRAY)
    if chroma:
        src_u = core.std.ShufflePlanes(src, planes=1, colorfamily=vs.GRAY)
        src_v = core.std.ShufflePlanes(src, planes=2, colorfamily=vs.GRAY)

    dbi = Resize(src_y, w, h, kernel=descale_kernel, a1=b, a2=c, taps=taps, invks=True)
    dbi2 = Resize(dbi, src_w, src_h, kernel=descale_kernel, a1=b, a2=c, taps=taps)
    if (w, h) != (ow, oh):
        dbi = Resize(dbi, ow, oh, kernel=kernely, taps=tapsy, a1=a1y, a2=a2y)

    if chroma and yuv444:
        rs = Resize(src_y, ow, oh, kernel=kernely, taps=tapsy, a1=a1y, a2=a2y)
        rs_u = Resize(src_u, ow, oh, kernel=kerneluv, taps=tapsuv, a1=a1uv, a2=a2uv, sx=0.25)
        rs_v = Resize(src_v, ow, oh, kernel=kerneluv, taps=tapsuv, a1=a1uv, a2=a2uv, sx=0.25)
    else:
        rs = Resize(src if chroma else src_y, ow, oh, kernel=kernely, taps=tapsy, a1=a1y, a2=a2y)

    # Masking
    diffmask = core.std.Expr([src_y, dbi2], 'x y - abs')
    if showmask != 2:
        diffmask = Resize(diffmask, ow, oh, kernel='bilinear')
        diffmask = core.std.Binarize(diffmask, threshold=thr)
    for _ in range(expand):
        diffmask = core.std.Maximum(diffmask, planes=0)
    for _ in range(inflate):
        diffmask = core.std.Inflate(diffmask, planes=0)

    if chroma:
        merged = core.std.ShufflePlanes([dbi,rs_u,rs_v] if yuv444 else [dbi,rs], planes=[0,0,0] if yuv444 else [0,1,2], colorfamily=vs.YUV)
    else:
        merged = dbi

    if showmask > 0:
        out = diffmask
    else:
        if yuv444:
            rs = core.std.ShufflePlanes([rs,merged], planes=[0,1,2], colorfamily=vs.YUV)
        out = core.std.MaskedMerge(merged, rs, diffmask, planes=0)

    return out


# DescaleM alias
DebilinearM = partial(DescaleM, descale_kernel='bilinear')

# DescaleM alias
DebicubicM = partial(DescaleM, descale_kernel='bicubic')

# DescaleM alias
DelanczosM = partial(DescaleM, descale_kernel='lanczos')

# DescaleM alias
Despline16M = partial(DescaleM, descale_kernel='spline16')

# DescaleM alias
Despline36M = partial(DescaleM, descale_kernel='spline36')


"""
Wrapper for fmtconv to scale each plane individually to the same size and fix chroma shift

Will only produce correct results if input is YUV420 or YUV422 with left aligned chroma

"""
def Downscale444(clip, w=1280, h=720, kernely="spline36", kerneluv="spline16", tapsy=3, tapsuv=3, a1y=None,
                 a1uv=None, a2y=None, a2uv=None, a3y=None, a3uv=None, invks=False, invkstaps=None):
    y = core.std.ShufflePlanes(clip, planes=0, colorfamily=vs.GRAY)
    u = core.std.ShufflePlanes(clip, planes=1, colorfamily=vs.GRAY)
    v = core.std.ShufflePlanes(clip, planes=2, colorfamily=vs.GRAY)
    y = Resize(y, w=w, h=h, kernel=kernely, taps=tapsy, a1=a1y, a2=a2y, a3=a3y, invks=invks, invkstaps=invkstaps)
    u = Resize(u, w=w, h=h, kernel=kerneluv, taps=tapsuv, a1=a1uv, a2=a2uv, a3=a3uv, sx=0.25)
    v = Resize(v, w=w, h=h, kernel=kerneluv, taps=tapsuv, a1=a1uv, a2=a2uv, a3=a3uv, sx=0.25)
    out = core.std.ShufflePlanes(clips=[y,u,v], planes=[0,0,0], colorfamily=vs.YUV)
    return out


"""
VapourSynth port of JIVTC.
Original script by lovesyk (https://github.com/lovesyk/avisynth-scripts/blob/master/JIVTC.avsi)

JIVTC applies inverse telecine in a way to minimize artifacts often seen on Japanese
TV broadcasts followed by recalculating the fields that might still contain some.

Dependencies: yadifmod, nnedi3

clip   src:            Source clip. Has to be 60i (30000/1001).
int    pattern:        First frame of any clean-combed-combed-clean-clean sequence.
int    threshold (10): This setting controls with how much probability one field has to
                       look better than the other to recalculate the other one using it.
                       Since there is no point dropping a field on a still (detail loss)
                       or an action (both results will look bad) scene, keep this above 0.
bool   draft (false):  If set to true, skip recalculate step (which means keep 50% of bad fields).
clip   ivtced:         Can be used to supply a custom IVTCed clip.
                       Keep in mind that the default IVTC process gets rid of 50% of
                       bad fields which might be "restored" depending on your supplied clip.
string bobber:         Can be used to supply a custom bobber.
                       The less information the bobber uses from the other field,
                       the better the result will be.
bool show (false):     If set to true, mark those frames that were recalculated.

"""
def JIVTC(src, pattern, thr=10, draft=False, ivtced=None, bobber=None, show=False, tff=None):

    def calculate(n, f, ivtced, bobbed):
        diffprev = f[0].props.EvenDiff
        diffnext = f[1].props.OddDiff
        if diffnext > diffprev:
            prerecalc = core.std.SelectEvery(bobbed, 2, 0)
        else:
            prerecalc = core.std.SelectEvery(bobbed, 2, 1)
        if abs(diffprev - diffnext) * 0xFF < thr:
            return ivtced
        if show:
            prerecalc = core.text.Text(prerecalc, 'Recalculated')
        return prerecalc

    pattern = pattern % 5

    defivtc = core.std.SeparateFields(src, tff=tff).std.DoubleWeave()
    selectlist = [[0,3,6,8], [0,2,5,8], [0,2,4,7], [2,4,6,9], [1,4,6,8]]
    defivtc = core.std.SelectEvery(defivtc, 10, selectlist[pattern])

    ivtced = defivtc if ivtced is None else ivtced
    if bobber is None:
        bobbed = core.yadifmod.Yadifmod(ivtced, edeint=core.nnedi3.nnedi3(ivtced, 2), order=0, mode=1)
    else:
        bobbed = bobber(ivtced)

    if src.fps_num != 30000 or src.fps_den != 1001:
        raise ValueError('JIVTC: This filter can only be used with 60i clips.')
    if bobbed.fps_num != 48000 or bobbed.fps_den != 1001:
        raise ValueError('JIVTC: The bobber you specified does not double the frame rate.')

    sep = core.std.SeparateFields(ivtced)
    even = core.std.SelectEvery(sep, 2, 0)
    odd = core.std.SelectEvery(sep, 2, 1)
    diffeven = core.std.PlaneStats(even, even.std.DuplicateFrames([0]), prop='Even')
    diffodd = core.std.PlaneStats(odd, odd.std.DeleteFrames([0]), prop='Odd')
    recalc = core.std.FrameEval(ivtced, partial(calculate, ivtced=ivtced, bobbed=bobbed),
                                prop_src=[diffeven,diffodd])

    inter = core.std.Interleave([ivtced, recalc])
    selectlist = [[0,3,4,6], [0,2,5,6], [0,2,4,7], [0,2,4,7], [1,2,4,6]]
    final = core.std.SelectEvery(inter, 8, selectlist[pattern])

    out = ivtced if draft else final
    out = core.std.SetFrameProp(out, prop='_FieldBased', intval=0)
    return out


"""
VapourSynth port of OverlayInter

Based on the AviSynth script by Majin3 and the already ported
ivtc_txt60mc by Firesledge that can be found inside havsfunc

It's much faster than ivtc_txt60mc because you can limit processing
to a small part of the clip.

Original Header:
# OverlayInter 0.1 by Majin3 (06.09.2012)
# Converts 60i overlays (like scrolling credits) on top of telecined 24p video to 24p using motion interpolation.
# Required: MVTools2, QTGMC (if not using a custom bobber)
# int		pattern:			First frame of a clean-combed-combed-clean-clean sequence
# int		pos (0):			Overlay position: 0: whole screen - 1: left - 2: top - 3: right - 4: bottom
# int		size (0):			Overlay size in px from the corresponding position
# bool		show (false):		Enable this to show the area selected by "pos" and "size"
# bool		draft (false):		Enable this to speed up processing by using low-quality bobbing and motion interpolation
# string	bobber:				A custom bobber if you do not wish to use "QTGMC(Preset="Very Slow", SourceMatch=2, Lossless=2)"
# string	ivtc:				A custom IVTC if you do not wish to use simple IVTC based on "pattern"
# Based on ivtc_txt60mc 1.1 by Firesledge

"""
def OverlayInter(src, pattern, pos=0, size=0, show=False, draft=False, bobber=None, ivtc=None, tff=None):

    if bobber is None and not isinstance(tff, bool):
        raise TypeError('OverlayInter: "tff" must be set. Setting tff to True means top field first. False means bottom field first')

    field_ref = (pattern * 2) % 5
    invpos = (5 - field_ref) % 5
    pattern %= 5

    croplist = [[0,0,0,0], [size,0,0,0], [0,0,size,0], [0,size,0,0], [0,0,0,size]]
    keep = core.std.CropRel(src, *croplist[pos])

    croplist = [[0,0,0,0], [0,src.width-size-4,0,0], [0,0,0,src.height-size-4],
                [src.width-size-4,0,0,0], [0,0,src.height-size-4,0]]
    bobbed = core.std.CropRel(src, *croplist[pos])
    if draft:
        bobbed = haf.Bob(bobbed, tff=tff)
    elif bobber is None:
        bobbed = haf.QTGMC(bobbed, Preset='very slow', SourceMatch=3, Lossless=2, TFF=tff)
    else:
        bobbed = bobber(bobbed)

    if ivtc is None:
        ivtclist = [[0,3,6,8], [0,2,5,8], [0,2,4,7], [2,4,6,9], [1,4,6,8]]
        ivtc = keep.std.SeparateFields(tff=tff).std.DoubleWeave().std.SelectEvery(10, ivtclist[pattern])
    else:
        ivtc = ivtc(keep)

    if invpos > 1:
        clean = core.std.AssumeFPS(bobbed[0] + core.std.SelectEvery(bobbed, 5, [6 - invpos]),
                                   fpsnum=12000, fpsden=1001)
    else:
        clean = core.std.SelectEvery(bobbed, 5, [1 - invpos])
    if invpos > 3:
        jitter = core.std.AssumeFPS(bobbed[0] + core.std.SelectEvery(bobbed, 5, [4 - invpos, 8 - invpos]),
                                    fpsnum=24000, fpsden=1001)
    else:
        jitter = core.std.SelectEvery(bobbed, 5, [3 - invpos, 4 - invpos])

    jsup = core.mv.Super(jitter)
    vecsup = haf.DitherLumaRebuild(jitter, s0=1).mv.Super(rfilter=4)
    vectb = core.mv.Analyse(jsup if draft else vecsup, overlap=0 if draft else 4, blksize=16, isb=True)
    if not draft:
        vectb = core.mv.Recalculate(vecsup, vectb, blksize=8, overlap=2)
    vectf = core.mv.Analyse(jsup if draft else vecsup, overlap=0 if draft else 4, blksize=16, isb=False)
    if not draft:
        vectf = core.mv.Recalculate(vecsup, vectf, blksize=8, overlap=2)
    comp = core.mv.FlowInter(jitter, jsup, vectb, vectf)
    fixed = core.std.SelectEvery(comp, 2, 0)
    fixed = core.std.Interleave([clean,fixed])[invpos // 2:]

    croplist = [[0,0,0,0], [0,4,0,0], [0,0,0,4], [4,0,0,0], [0,0,4,0]]
    fixed = core.std.CropRel(fixed, *croplist[pos])

    if show:
        maxvalue = (1 << src.format.bits_per_sample) - 1
        offset = 32 * maxvalue // 0xFF if fixed.format.sample_type == vs.INTEGER else 32 / 0xFF
        fixed = core.std.Expr(fixed, ['','x {} +'.format(offset),''])

    if pos == 1:
        out = core.std.StackHorizontal([fixed,ivtc])
    elif pos == 2:
        out = core.std.StackVertical([fixed,ivtc])
    elif pos == 3:
        out = core.std.StackHorizontal([ivtc,fixed])
    elif pos == 4:
        out = core.std.StackVertical([ivtc,fixed])
    else:
        out = fixed
    out = core.std.SetFrameProp(out, prop='_FieldBased', intval=0)

    return out


"""
VapourSynth port of AutoDeblock2. Original script by joletb, vinylfreak89, eXmendiC and Gebbi.

The purpose of this script is to automatically remove MPEG2 artifacts.

Only 8-bit input supported currently

"""
def AutoDeblock(src, edgevalue=24, db1=1, db2=6, db3=15, deblocky=True, deblockuv=True, debug=False, redfix=False,
                fastdeblock=False, adb1=3, adb2=4, adb3=8, adb1d=2, adb2d=7, adb3d=11, planes=None):

    def to8bit(f):
        return f * 0xFF

    def eval_deblock_strength(n, f, fastdeblock, unfiltered, fast, weakdeblock,
                              mediumdeblock, strongdeblock):
        out = unfiltered
        if fastdeblock:
            if to8bit(f[0].props.OrigDiff) > adb1 and to8bit(f[1].props.YNextDiff) > adb1d:
                return fast
            else:
                return unfiltered
        if to8bit(f[0].props.OrigDiff) > adb1 and to8bit(f[1].props.YNextDiff) > adb1d:
            out = weakdeblock
        if to8bit(f[0].props.OrigDiff) > adb2 and to8bit(f[1].props.YNextDiff) > adb2d:
            out = mediumdeblock
        if to8bit(f[0].props.OrigDiff) > adb3 and to8bit(f[1].props.YNextDiff) > adb3d:
            out = strongdeblock
        return out

    def fix_red(n, f, unfiltered, autodeblock):
        if (to8bit(f[0].props.YAverage) > 50 and to8bit(f[0].props.YAverage) < 130
                and to8bit(f[1].props.UAverage) > 95 and to8bit(f[1].props.UAverage) < 130
                and to8bit(f[2].props.VAverage) > 130 and to8bit(f[2].props.YAverage) < 155):
            return unfiltered
        return autodeblock

    if redfix and fastdeblock:
        raise ValueError('AutoDeblock: You cannot set both "redfix" and "fastdeblock" to True!')

    if planes is None:
        planes = []
        if deblocky: planes.append(0)
        if deblockuv: planes.extend([1,2])

    maxvalue = (1 << src.format.bits_per_sample) - 1
    orig = core.std.Prewitt(src)
    orig = core.std.Expr(orig, "x {edgevalue} >= {maxvalue} x ?".format(edgevalue=edgevalue, maxvalue=maxvalue))
    orig_d = orig.rgvs.RemoveGrain(4).rgvs.RemoveGrain(4)

    predeblock = haf.Deblock_QED(src.rgvs.RemoveGrain(2).rgvs.RemoveGrain(2))
    fast = core.dfttest.DFTTest(predeblock, tbsize=1)
    fast = core.text.Text(fast, 'deblock') if debug else fast

    unfiltered = src
    unfiltered = core.text.Text(unfiltered, 'unfiltered') if debug else unfiltered
    weakdeblock = core.dfttest.DFTTest(predeblock, sigma=db1, tbsize=1, planes=planes)
    weakdeblock = core.text.Text(weakdeblock, 'weakdeblock') if debug else weakdeblock
    mediumdeblock = core.dfttest.DFTTest(predeblock, sigma=db2, tbsize=1, planes=planes)
    mediumdeblock = core.text.Text(mediumdeblock, 'mediumdeblock') if debug else mediumdeblock
    strongdeblock = core.dfttest.DFTTest(predeblock, sigma=db3, tbsize=1, planes=planes)
    strongdeblock = core.text.Text(strongdeblock, 'strongdeblock') if debug else strongdeblock

    difforig = core.std.PlaneStats(orig, orig_d, prop='Orig')
    diffnext = core.std.PlaneStats(src, src.std.DeleteFrames([0]), prop='YNext')
    autodeblock = core.std.FrameEval(unfiltered, partial(eval_deblock_strength, fastdeblock=fastdeblock,
                                     unfiltered=unfiltered, fast=fast, weakdeblock=weakdeblock,
                                     mediumdeblock=mediumdeblock, strongdeblock=strongdeblock),
                                     prop_src=[difforig,diffnext])

    if redfix:
        src = core.std.PlaneStats(src, prop='Y')
        src_u = core.std.PlaneStats(src, plane=1, prop='U')
        src_v = core.std.PlaneStats(src, plane=2, prop='V')
        autodeblock = core.std.FrameEval(unfiltered, partial(fix_red, unfiltered=unfiltered,
                                         autodeblock=autodeblock), prop_src=[src,src_u,src_v])

    return autodeblock


"""
Basically a wrapper for std.Trim and std.Splice that recreates the functionality of
AviSynth's ReplaceFramesSimple (http://avisynth.nl/index.php/RemapFrames)
that was part of the plugin RemapFrames by James D. Lin

Usage: ReplaceFrames(clipa, clipb, mappings="[200 300] [1100 1150] 400 1234")

This will replace frames 200..300, 1100..1150, 400 and 1234 from clipa with
the corresponding frames from clipb.

"""
def ReplaceFrames(clipa, clipb, mappings=None, filename=None):

    if not isinstance(clipa, vs.VideoNode):
        raise TypeError('ReplaceFrames: "clipa" must be a clip!')
    if not isinstance(clipb, vs.VideoNode):
        raise TypeError('ReplaceFrames: "clipb" must be a clip!')
    if clipa.format.id != clipb.format.id:
        raise TypeError('ReplaceFrames: "clipa" and "clipb" must have the same format!')
    if filename is not None and not isinstance(filename, str):
        raise TypeError('ReplaceFrames: "filename" must be a string!')
    if mappings is not None and not isinstance(mappings, str):
        raise TypeError('ReplaceFrames: "mappings" must be a string!')
    if mappings is None:
        mappings = ''

    if filename:
        with open(filename, 'r') as mf:
            mappings += '\n{}'.format(mf.read())
    # Some people used this as separators and wondered why it wasn't working
    mappings = mappings.replace(',', ' ').replace(':', ' ')

    frames = re.findall('\d+(?!\d*\s*\d*\s*\d*\])', mappings)
    ranges = re.findall('\[\s*\d+\s+\d+\s*\]', mappings)
    maps = []
    for range_ in ranges:
        maps.append([int(x) for x in range_.strip('[ ]').split()])
    for frame in frames:
        maps.append([int(frame), int(frame)])

    for start, end in maps:
        if start > end:
            raise ValueError('ReplaceFrames: Start frame is bigger than end frame: [{} {}]'.format(start, end))
        if end >= clipa.num_frames or end >= clipb.num_frames:
            raise ValueError('ReplaceFrames: End frame too big, one of the clips has less frames: {}'.format(end)) 

    out = clipa
    for start, end in maps:
        temp = clipb[start:end+1] 
        if start != 0:
            temp = out[:start] + temp
        if end < out.num_frames - 1:
            temp = temp + out[end+1:]
        out = temp
    return out


# ReplaceFrames alias
ReplaceFramesSimple = ReplaceFrames

# ReplaceFrames alias
rfs = ReplaceFrames


"""
This overlays a clip onto another.
Default matrix for RGB -> YUV conversion is 601 to match AviSynth's Overlay()
overlay should be a list of [video, mask] or a path string to an RGBA file
If you specifiy a clip instead it will just be spliced into the source
(RGBA videos opened by ffms2 are already such a list)
"""
def InsertSign(clip, overlay, start, end=None, matrix='601'):

    if isinstance(overlay, str):
        overlay = core.ffms2.Source(overlay, alpha=True)
    if not isinstance(overlay, list):
        overlay = [overlay, None]
    if end is None:
        end = start + overlay[0].num_frames
    else:
        end += 1
    if end > clip.num_frames:
        end = clip.num_frames
    if start >= end:
        raise ValueError('InsertSign: "start" must be smaller than or equal to "end"!')
    clip_cf = clip.format.color_family
    overlay_cf = overlay[0].format.color_family

    before = clip[:start] if start != 0 else None
    middle = clip[start:end]
    after = clip[end:] if end != clip.num_frames else None

    if clip_cf != overlay_cf and (clip_cf == vs.YUV or overlay_cf == vs.YUV):
        sign = core.fmtc.matrix(overlay[0], mat=matrix)
    else:
        sign = overlay[0]
    sign = core.resize.Spline36(sign, clip.width, clip.height, format=clip.format.id,
                                dither_type='error_diffusion')

    if overlay[1] is None:
        if clip.format.sample_type == vs.INTEGER:
            overlay[1] = sign.std.Binarize(0)
        else:
            overlay[1] = sign.std.Expr("1.0")
    mask = core.resize.Bicubic(overlay[1], clip.width, clip.height)
    mask = Depth(mask, bits=clip.format.bits_per_sample)

    middle = core.std.MaskedMerge(middle, sign, mask)

    out = middle
    if before is not None:
        out = before + out
    if after is not None:
        out = out + after
    return out


"""
Downscale only lineart with an inverted kernel and interpolate
it back to its original resolution with NNEDI3.

Parts of higher resolution like credits are protected by a mask.

Basic idea stolen from a script made by Daiz.

"""
def DescaleAA(src, w=1280, h=720, thr=10, kernel='bilinear', b=1/3, c=1/3, taps=3,
              expand=3, inflate=3, showmask=False):

    if kernel.lower().startswith('de'):
        kernel = kernel[2:]

    ow = src.width
    oh = src.height

    bits = src.format.bits_per_sample
    sample_type = src.format.sample_type
    
    if sample_type == vs.INTEGER:
        maxvalue = (1 << bits) - 1
        thr = thr * maxvalue // 0xFF
    else:
        maxvalue = 1
        thr /= (235 - 16)

    # Fix lineart
    src_y = core.std.ShufflePlanes(src, planes=0, colorfamily=vs.GRAY)
    deb = Resize(src_y, w, h, kernel=kernel, a1=b, a2=c, taps=taps, invks=True)
    sharp = nnedi3_rpow2.nnedi3_rpow2(deb, 2, ow, oh)
    thrlow = 4 * maxvalue // 0xFF if sample_type == vs.INTEGER else 4 / 0xFF
    thrhigh = 24 * maxvalue // 0xFF if sample_type == vs.INTEGER else 24 / 0xFF
    edgemask = core.std.Prewitt(sharp, planes=0)
    edgemask = core.std.Expr(edgemask, "x {thrhigh} >= {maxvalue} x {thrlow} <= 0 x ? ?"
                                       .format(thrhigh=thrhigh, maxvalue=maxvalue, thrlow=thrlow))
    if kernel == "bicubic" and c >= 0.7:
        edgemask = core.std.Maximum(edgemask, planes=0)
    sharp = core.resize.Point(sharp, format=src.format.id)

    # Restore true 1080p
    deb_upscale = Resize(deb, ow, oh, kernel=kernel, a1=b, a2=c, taps=taps)
    diffmask = core.std.Expr([src_y, deb_upscale], 'x y - abs')
    for _ in range(expand):
        diffmask = core.std.Maximum(diffmask, planes=0)
    for _ in range(inflate):
        diffmask = core.std.Inflate(diffmask, planes=0)

    mask = core.std.Expr([diffmask,edgemask], 'x {thr} >= 0 y ?'.format(thr=thr))
    mask = mask.std.Inflate().std.Deflate()
    out = core.std.MaskedMerge(src, sharp, mask, planes=0)

    if showmask:
        out = mask

    return out


# Legacy DescaleAA alias
def ProtectedDebiXAA(src, w=1280, h=720, thr=10, expand=3, inflate=3,
                     bicubic=False, b=1/3, c=1/3, showmask=False, bits=None):

    if bicubic:
        return DescaleAA(src, w=w, h=h, thr=thr, kernel='bicubic', b=b, c=c, taps=None,
                         expand=expand, inflate=inflate, showmask=showmask)
    else:
        return DescaleAA(src, w=w, h=h, thr=thr, kernel='bilinear', b=None, c=None, taps=None,
                         expand=expand, inflate=inflate, showmask=showmask)


"""
VapourSynth port of AviSynth's maa2 (https://github.com/AviSynth/avs-scripts)

Works on any bitdepth

"""
def maa(src, mask=None, chroma=None, ss=None, aa=None, aac=None, show=None):

    def SangNomAA(src, ss=2.0, aa=48, aac=None):
        ss_w = round(src.width * ss / 4) * 4
        ss_h = round(src.height * ss / 4) * 4
        out = core.resize.Spline36(src, ss_w, ss_h).std.Transpose()
        out = core.sangnom.SangNom(out, aa=aa if aac is None else [aa, aac, aac]).std.Transpose()
        out = core.sangnom.SangNom(out, aa=aa if aac is None else [aa, aac, aac])
        out = core.resize.Spline36(out, src.width, src.height)
        return out

    # Type checking
    kwargsdict = {'src': [src, (vs.VideoNode,)], 'mask': [mask, (int,)], 'ss': [ss, (int, float)],
                  'aa': [aa, (int,)], 'aac': [aac, (int,)], 'show': [show, (bool,)]}

    for k, v in kwargsdict.items():
        if v[0] is not None and not isinstance(v[0], v[1]):
            raise TypeError('maa: "{variable}" must be {types}!'
                            .format(variable=k, types=' or '.join([TYPEDICT[t] for t in v[1]])))

    # Set defaults
    if mask is None:
        mask = 1
    if chroma is None:
        chroma = False
    if ss is None:
        ss = 2.0
    if aa is None:
        aa = 48
    if aac is None:
        aac = aa - 8
    if show is None:
        show = False

    # Value checking
    if mask < -0xFF or mask > 1:
        raise ValueError('maa: "mask" must be between -255 and 1!')
    if ss <= 0:
        raise ValueError('maa: "ss" must be > 0!')

    bits = src.format.bits_per_sample
    sample_type = src.format.sample_type
    maxvalue = (1 << bits) - 1
    
    if sample_type == vs.INTEGER:
        mthresh = -mask * maxvalue // 0xFF if mask < 0 else 7 * maxvalue // 0xFF
        mthreshc = mthresh - 6 * maxvalue // 0xFF
    else:
        mthresh = -mask / 0xFF if mask < 0 else 7 / 0xFF
        mthreshc = mthresh - 6 / 0xFF

    if src.format.num_planes == 1:
        chroma = False

    if mask != 0:
        m = core.std.Sobel(mvf.GetPlane(src, 0))
        m = core.std.Binarize(m, mthresh)
        if chroma:
            mu = core.std.Sobel(mvf.GetPlane(src, 1))
            mu = core.std.Binarize(mu, mthreshc)
            mv = core.std.Sobel(mvf.GetPlane(src, 2))
            mv = core.std.Binarize(mv, mthreshc)
            m = core.std.ShufflePlanes([m,mu,mv], planes=[0,0,0], colorfamily=vs.YUV)
    if not chroma:
        c_aa = SangNomAA(mvf.GetPlane(src, 0), ss, aa)
    else:
        c_aa = SangNomAA(src, ss, aa, aac)

    if not chroma and src.format.num_planes != 1:
        c_aa = core.std.ShufflePlanes([c_aa, mvf.GetPlane(src, 1), mvf.GetPlane(src, 2)], planes=[0,0,0], colorfamily=vs.YUV)
    if mask == 0:
        out = c_aa
    elif show:
        out = m
    else:
        out = core.std.MaskedMerge(src, c_aa, m)
    return out


"""
VapourSynth port of TemporalDegrain (http://avisynth.nl/index.php/Temporal_Degrain)

(only 8 bit YUV input)

Differences:

- all keyword arguments are now lowercase
- hq > 0 is not implemented
- gpu=True is not implemented

"""
def TemporalDegrain(input_, denoise=None, gpu=False, sigma=16, bw=16, bh=16, pel=2,
                    blksize=8, ov=None, degrain=2, limit=255, sad1=400, sad2=300, hq=0):

    if not isinstance(input_, vs.VideoNode):
        raise TypeError('TemporalDegrain: "input_" must be a clip!')
    if denoise is not None and not isinstance(denoise, vs.VideoNode):
        raise TypeError('TemporalDegrain: "denoise" must be a clip!')
    if not isinstance(gpu, bool):
        raise TypeError('TemporalDegrain: "gpu" must be a bool!')
    if gpu:
        raise NotImplementedError('TemporalDegrain: "gpu=True" is not implemented!')
    if not isinstance(sigma, int):
        raise TypeError('TemporalDegrain: "sigma" must be an int!')
    if not isinstance(bw, int):
        raise TypeError('TemporalDegrain: "bw" must be an int!')
    if not isinstance(bh, int):
        raise TypeError('TemporalDegrain: "bh" must be an int!')
    if not isinstance(pel, int):
        raise TypeError('TemporalDegrain: "pel" must be an int!')
    if not isinstance(blksize, int):
        raise TypeError('TemporalDegrain: "blksize" must be an int!')
    if ov is not None and not isinstance(ov, int):
        raise TypeError('TemporalDegrain: "ov" must be an int!')
    if not isinstance(degrain, int):
        raise TypeError('TemporalDegrain: "degrain" must be an int!')
    if not isinstance(limit, int):
        raise TypeError('TemporalDegrain: "limit" must be an int!')
    if not isinstance(sad1, int):
        raise TypeError('TemporalDegrain: "sad1" must be an int!')
    if not isinstance(sad2, int):
        raise TypeError('TemporalDegrain: "sad2" must be an int!')
    if not isinstance(hq, int):
        raise TypeError('TemporalDegrain: "hq" must be an int!')
    if hq > 0:
        raise NotImplementedError('TemporalDegrain: "hq" > 0 is not implemented!')

    o = input_
    s2 = int(sigma * 0.625)
    s3 = int(sigma * 0.375)
    s4 = int(sigma * 0.250)
    ow = int(bw / 2)
    oh = int(bh / 2)
    ov = int(blksize / 2) if not ov or ov*2 > blksize else ov

    if denoise:
        filter_ = denoise
    elif gpu:
        filter_ = o.FFT3DGPU(sigma=sigma, sigma2=s2 , sigma3=s3, sigma4=s4, bt=4, bw=bw, bh=bh, ow=ow, oh=oh)  # not implemented
    else:
        filter_ = core.fft3dfilter.FFT3DFilter(o, sigma=sigma, sigma2=s2, sigma3=s3, sigma4=s4, bt=4, bw=bw, bh=bh, ow=ow, oh=oh)
    if hq >= 1:
        filter_ = filter_.HQdn3D(4,3,6,3)  # not implemented

    spat = filter_
    spatd = core.std.MakeDiff(o, spat)

    srch = filter_
    srch_super = core.mv.Super(filter_, pel=pel)

    if degrain == 3:
        bvec3 = core.mv.Analyse(srch_super, isb=True, delta=3, blksize=blksize, overlap=ov)
    else:
        bvec3 = core.std.BlankClip()
    if degrain >= 2:
        bvec2 = core.mv.Analyse(srch_super, isb=True, delta=2, blksize=blksize, overlap=ov)
    else:
        bvec2 = core.std.BlankClip()
    bvec1 = core.mv.Analyse(srch_super, isb=True,  delta=1, blksize=blksize, overlap=ov)
    fvec1  = core.mv.Analyse(srch_super, isb=False, delta=1, blksize=blksize, overlap=ov)
    if degrain >= 2:
        fvec2 = core.mv.Analyse(srch_super, isb=False, delta=2, blksize=blksize, overlap=ov)
    else:
        fvec2 = core.std.BlankClip()
    if degrain == 3:
        fvec3 = core.mv.Analyse(srch_super, isb=False, delta=3, blksize=blksize, overlap=ov)
    else:
        fvec3 = core.std.BlankClip()

    o_super = core.mv.Super(o, pel=2, levels=1)

    if degrain == 3:
        nr1 = core.mv.Degrain3(o, o_super, bvec1, fvec1, bvec2, fvec2, bvec3, fvec3, thsad=sad1, limit=limit)
    elif degrain == 2:
        nr1 = core.mv.Degrain2(o, o_super, bvec1, fvec1, bvec2, fvec2, thsad=sad1, limit=limit)
    else:
        nr1 = core.mv.Degrain1(o, o_super, bvec1, fvec1, thsad=sad1, limit=limit)
    nr1d = core.std.MakeDiff(o, nr1)

    dd = core.std.Expr([spatd, nr1d], 'x 128 - abs y 128 - abs < x y ?')
    nr1x = core.std.MakeDiff(o, dd, planes=0)

    nr1x_super = core.mv.Super(nr1x, pel=2, levels=1)

    if degrain == 3:
        nr2 = core.mv.Degrain3(nr1x, nr1x_super, bvec1, fvec1, bvec2, fvec2, bvec3, fvec3, thsad=sad2, limit=limit)
    elif degrain == 2:
        nr2 = core.mv.Degrain2(nr1x, nr1x_super, bvec1, fvec1, bvec2, fvec2, thsad=sad2, limit=limit)
    else:
        nr2 = core.mv.Degrain1(nr1x, nr1x_super, bvec1, fvec1, thsad=sad2, limit=limit)

    if hq >= 2:
        nr2.HQDn3D(0,0,4,1)  # not implemented

    s = haf.MinBlur(nr2, 1, 0)
    alld = core.std.MakeDiff(o, nr2)
    temp = core.rgvs.RemoveGrain(s, [11, 0])
    ssd = core.std.MakeDiff(s, temp)
    ssdd = core.rgvs.Repair(ssd, alld, 1)
    ssdd = core.std.Expr([ssdd, ssd], 'x 128 - abs y 128 - abs < x y ?')

    output = core.std.MergeDiff(nr2, ssdd, planes=0)
    return output


# Helpers

# Wrapper with fmtconv syntax that tries to use the internal resizers whenever it is possible
def Resize(src, w, h, sx=None, sy=None, sw=None, sh=None, kernel='spline36', taps=None, a1=None,
             a2=None, a3=None, invks=None, invkstaps=None, fulls=None, fulld=None):

    bits = src.format.bits_per_sample

    if (src.width, src.height, fulls) == (w, h, fulld):
        return src

    if kernel is None:
        kernel = 'spline36'
    kernel = kernel.lower()

    if invks and kernel == 'bilinear' and hasattr(core, 'unresize') and invkstaps is None:
        return core.unresize.Unresize(src, w, h, src_left=sx, src_top=sy)
    if invks and kernel in ['bilinear', 'bicubic', 'lanczos', 'spline16', 'spline36'] and hasattr(core, 'descale') and invkstaps is None:
        return Descale(src, w, h, kernel=kernel, b=a1, c=a2, taps=taps)
    if not invks:
        if kernel == 'bilinear':
            return core.resize.Bilinear(src, w, h, range=fulld, range_in=fulls, src_left=sx, src_top=sy,
                                        src_width=sw, src_height=sh)
        if kernel == 'bicubic':
            return core.resize.Bicubic(src, w, h, range=fulld, range_in=fulls, filter_param_a=a1, filter_param_b=a2,
                                       src_left=sx, src_top=sy, src_width=sw, src_height=sh)
        if kernel == 'spline16':
            return core.resize.Spline16(src, w, h, range=fulld, range_in=fulls, src_left=sx, src_top=sy,
                                        src_width=sw, src_height=sh)
        if kernel == 'spline36':
            return core.resize.Spline36(src, w, h, range=fulld, range_in=fulls, src_left=sx, src_top=sy,
                                        src_width=sw, src_height=sh)
        if kernel == 'lanczos':
            return core.resize.Lanczos(src, w, h, range=fulld, range_in=fulls, filter_param_a=taps,
                                       src_left=sx, src_top=sy, src_width=sw, src_height=sh)
    return Depth(core.fmtc.resample(src, w, h, sx=sx, sy=sy, sw=sw, sh=sh, kernel=kernel, taps=taps,
                              a1=a1, a2=a2, a3=a3, invks=invks, invkstaps=invkstaps, fulls=fulls, fulld=fulld), bits)


def Debilinear(src, width, height, yuv444=False, gray=False, chromaloc=None):
    return Descale(src, width, height, kernel='bilinear', b=None, c=None, taps=None, yuv444=yuv444, gray=gray, chromaloc=chromaloc)

def Debicubic(src, width, height, b=1/3, c=1/3, yuv444=False, gray=False, chromaloc=None):
    return Descale(src, width, height, kernel='bicubic', b=b, c=c, taps=None, yuv444=yuv444, gray=gray, chromaloc=chromaloc)

def Delanczos(src, width, height, taps=3, yuv444=False, gray=False, chromaloc=None):
    return Descale(src, width, height, kernel='lanczos', b=None, c=None, taps=taps, yuv444=yuv444, gray=gray, chromaloc=chromaloc)

def Despline16(src, width, height, yuv444=False, gray=False, chromaloc=None):
    return Descale(src, width, height, kernel='spline16', b=None, c=None, taps=None, yuv444=yuv444, gray=gray, chromaloc=chromaloc)

def Despline36(src, width, height, yuv444=False, gray=False, chromaloc=None):
    return Descale(src, width, height, kernel='spline36', b=None, c=None, taps=None, yuv444=yuv444, gray=gray, chromaloc=chromaloc)


def Descale(src, width, height, kernel='bilinear', b=1/3, c=1/3, taps=3, yuv444=False, gray=False, chromaloc=None):
    src_f = src.format
    src_cf = src_f.color_family
    src_st = src_f.sample_type
    src_bits = src_f.bits_per_sample
    src_sw = src_f.subsampling_w
    src_sh = src_f.subsampling_h

    descale_filter = get_descale_filter(b, c, taps, kernel)

    if src_cf == vs.RGB and not gray:
        rgb = descale_filter(to_rgbs(src), width, height)
        return rgb.resize.Point(format=src_f.id)

    y = descale_filter(to_grays(src), width, height)
    y_f = core.register_format(vs.GRAY, src_st, src_bits, 0, 0)
    y = y.resize.Point(format=y_f.id)

    if src_cf == vs.GRAY or gray:
        return y

    if not yuv444 and ((width % 2 and src_sw) or (height % 2 and src_sh)):
        raise ValueError('Descale: The output dimension and the subsampling are incompatible.')

    uv_f = core.register_format(src_cf, src_st, src_bits, 0 if yuv444 else src_sw, 0 if yuv444 else src_sh)
    uv = src.resize.Spline36(width, height, format=uv_f.id, chromaloc_s=chromaloc)

    return core.std.ShufflePlanes([y,uv], [0,1,2], vs.YUV)


def to_grays(src):
    return src.resize.Point(format=vs.GRAYS)


def to_rgbs(src):
    return src.resize.Point(format=vs.RGBS)


def get_plane(src, plane):
    return core.std.ShufflePlanes(src, plane, vs.GRAY)


def get_descale_filter(b, c, taps, kernel):
    kernel = kernel.lower()
    if kernel == 'bilinear':
        return core.descale.Debilinear
    elif kernel == 'bicubic':
        return partial(core.descale.Debicubic, b=b, c=c)
    elif kernel == 'lanczos':
        return partial(core.descale.Delanczos, taps=taps)
    elif kernel == 'spline16':
        return core.descale.Despline16
    elif kernel == 'spline36':
        return core.descale.Despline36
    else:
        raise ValueError('Descale: Invalid kernel specified.')


def Depth(src, bits, dither_type='error_diffusion', range=None, range_in=None):
    src_f = src.format
    src_cf = src_f.color_family
    src_st = src_f.sample_type
    src_bits = src_f.bits_per_sample
    src_sw = src_f.subsampling_w
    src_sh = src_f.subsampling_h
    dst_st = vs.INTEGER if bits < 32 else vs.FLOAT

    if isinstance(range, str):
        range = RANGEDICT[range]

    if isinstance(range_in, str):
        range_in = RANGEDICT[range_in]

    if (src_bits, range_in) == (bits, range):
        return src
    out_f = core.register_format(src_cf, dst_st, bits, src_sw, src_sh)
    return core.resize.Point(src, format=out_f.id, dither_type=dither_type, range=range, range_in=range_in)


TYPEDICT = {vs.VideoNode: 'a clip', int: 'an int', float: 'a float', bool: 'a bool', str: 'a str', list: 'a list'}

RANGEDICT = {'limited': 0, 'full': 1}
