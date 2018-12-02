import vapoursynth as vs
import havsfunc as haf
import mvsfunc as mvf
import descale as dsc

core = vs.core

# InsaneAA Anti-Aliasing Script (VS port) v0.3 (30.11.2017)
# 
# Original idea by tonik & tophf, edited and ported by DJATOM.
# Use this script to fix ugly upscaled anime BDs.
# 
# Processing chain: 
#   1) extract luma from clip;
#   2) apply Descale to it;
#   3) resize luma with Spline36 for smooth edges;
#   4) merge "smooth" clip with Descale clip according to descale_str;
#   5) re-upscale it back to 1080p (or clip's original resolution) using eedi3+nnedi3 method;
#   6) merge rescaled clip with source clip using lines mask. This should prevent noise and textures distortion;
#   7) combine merged clip with color planes. 
# 
# Prerequisites:
#   eedi3/eedi3cl: https://github.com/HomeOfVapourSynthEvolution/VapourSynth-EEDI3 (you should build it from source, r1 is not working due to wrong namespace)
#   nnedi3: https://github.com/dubhater/vapoursynth-nnedi3
#   nnedi3cl: https://github.com/HomeOfVapourSynthEvolution/VapourSynth-NNEDI3CL
#   descale: https://github.com/Irrational-Encoding-Wizardry/vapoursynth-descale
#   havsfunc: https://github.com/HomeOfVapourSynthEvolution/havsfunc
#   mvsfunc: https://github.com/HomeOfVapourSynthEvolution/mvsfunc
# 
# Basic usage:
#   import insaneAA
#   insaneAA.insaneAA(clip, eedi3Cl1=False, eedi3Cl2=False, nnedi3Cl=False, descale_str=0.3, kernel='bilinear', descale_h=720, descale_w=None, pscrn=1, alpha=0.2, beta=0.25, gamma=1000.0, outputMode=0)
#     eedi3Mode: defaults - dict(first=dict(mode='cpu', device=-1), second=dict(mode='cpu', device=-1)). 'first' refers to 1st instance of filter call, 'second' - for 2nd call. 
#       Each call should have at least 'mode' key. You can specify 'device' if you need to use non-default GPU card.
#       Valid values for 'mode' key: 'cpu', 'opencl'.
#     nnedi3Mode: defaults - dict(first=dict(mode='cpu', device=-1), second=dict(mode='cpu', device=-1)). 'first' refers to 1st instance of filter call, 'second' - for 2nd call. 
#       Each call should have at least 'mode' key. You can specify 'device' if you need to use non-default GPU card.
#       Valid values for 'mode' key: 'cpu', 'znedi3', 'opencl'.
#     descale_str: strengh of mixing between descaled clip and Spline36 clip (for AA purposes). More strengh means more haloes, keep that in mind.
#     kernel: descaling kernel. Use getnative.py for determining native resolution and try various kernels to find the best suitable.
#     descale_h/descale_w: once you know native resolution, set descale_h. descale_w is almost useless, script will guess descaling width automaticaly. But you can set it, lol.
#     pscrn: nnedi3's prescreener for faster operation. Does nothing if nnedi3Cl is True.
#     alpha: eedi3's alpha.
#     beta: eedi3's beta.
#     gamma: eedi3's gamma.
#     outputMode: 1 - only rescale (GRAY), 2 - linemasked rescale (GRAY), 0 - linemasked rescale + untouched colors. This option useful for, say, processing all clip into lossless file and masking high resolution details later or for importing filtered luma into avisynth.
#   Please do something with FullHD details! At least mask them or somehow exclude from processing.
# 
# Changelog:
#   version 0.3
#     Major change in eedi3/nnedi3 options: use dict(first=dict(mode='cpu', device=-1), second=dict(mode='cpu', device=-1)) for eedi3Mode/nnedi3Mode. More in usage.
#     Now you can pick znedi3 for sclip. The fastest nnedi3 option on my observation, but in complex scripts it might be better to use opencl nnedi3 for saving cpu cycles for other stuff.
#   version 0.2
#     Turn off OpenCL plugins by default.
#     Split eedi3Cl for every eedi3 call, may improve performance on cheap GPUs.
#   version 0.1
#     Initial release.

def insaneAA(c, eedi3Mode=None, nnedi3Mode=None, descale_str=0.3, kernel='bilinear', descale_h=720, descale_w=None, pscrn=1, alpha=0.2, beta=0.25, gamma=1000.0, outputMode=0):
    if not isinstance(c, vs.VideoNode):
        raise TypeError('insaneAA: This is not a clip')

    if not all (k in eedi3Mode for k in ('first', 'second')):
        if all (k in eedi3Mode for k in ('mode', 'device')):
            eedi3Mode = dict(first=dict(mode=eedi3Mode['mode'], device=eedi3Mode['device']), second=dict(mode=eedi3Mode['mode'], device=eedi3Mode['device']))
        elif 'mode' in eedi3Mode:
            eedi3Mode = dict(first=dict(mode=eedi3Mode['mode'], device=-1), second=dict(mode=eedi3Mode['mode'], device=-1))
        else:
            eedi3Mode = dict(first=dict(mode='cpu', device=-1), second=dict(mode='cpu', device=-1))
    elif not all (k in eedi3Mode['first'] for k in ('mode', 'device')):
        if 'device' not in eedi3Mode['first']:
            eedi3Mode['first']['device'] = -1
        elif 'mode' not in eedi3Mode['first']:
            raise ValueError('insaneAA: first instance of eedi3Mode lacks "mode" key')
    elif not all (k in eedi3Mode['second'] for k in ('mode', 'device')):
        if 'device' not in eedi3Mode['first']:
            eedi3Mode['second']['device'] = -1
        elif 'mode' not in eedi3Mode['second']:
            raise ValueError('insaneAA: second instance of eedi3Mode lacks "mode" key')

    if not all (k in nnedi3Mode for k in ('first', 'second')):
        if all (k in nnedi3Mode for k in ('mode', 'device')):
            nnedi3Mode = dict(first=dict(mode=nnedi3Mode['mode'], device=nnedi3Mode['device']), second=dict(mode=nnedi3Mode['mode'], device=nnedi3Mode['device']))
        elif 'mode' in nnedi3Mode:
            nnedi3Mode = dict(first=dict(mode=nnedi3Mode['mode'], device=-1), second=dict(mode=nnedi3Mode['mode'], device=-1))
        else:
            nnedi3Mode = dict(first=dict(mode='cpu', device=-1), second=dict(mode='cpu', device=-1))
    elif not all (k in nnedi3Mode['first'] for k in ('mode', 'device')):
        if 'device' not in nnedi3Mode['first']:
            nnedi3Mode['first']['device'] = -1
        elif 'mode' not in nnedi3Mode['first']:
            raise ValueError('insaneAA: first instance of nnedi3Mode lacks "mode" key')
    elif not all (k in nnedi3Mode['second'] for k in ('mode', 'device')):
        if 'device' not in nnedi3Mode['first']:
            nnedi3Mode['second']['device'] = -1
        elif 'mode' not in nnedi3Mode['second']:
            raise ValueError('insaneAA: second instance of nnedi3Mode lacks "mode" key')

    w = c.width
    h = c.height
    gray_c = mvf.GetPlane(c, 0)
    descale_clp = revert_upscale(gray_c, descale_str, kernel, descale_h, descale_w)
    upscale = rescale(descale_clp, eedi3Mode, nnedi3Mode, w, h, pscrn, alpha, beta, gamma)
    if outputMode is 1:
        return upscale

    linemask = core.std.Maximum(core.std.Expr(core.std.Sobel(gray_c), "x 2 *"))
    merged_aa = core.std.MaskedMerge(gray_c, upscale, linemask)
    if outputMode is 2:
        return merged_aa

    return core.std.ShufflePlanes([merged_aa, mvf.GetPlane(c, 1), mvf.GetPlane(c, 2)], planes=[0, 0, 0], colorfamily=c.format.color_family)

def revert_upscale(c, descale_str=0.3, kernel='bilinear', descale_h=720, descale_w=None):
    w = c.width
    h = c.height
    descale_w = haf.m4((w * descale_h) / h) if descale_w == None else descale_w
    descale_natural = dsc.Descale(c, descale_w, descale_h, kernel=kernel)
    descale_aa = core.resize.Spline36(c, descale_w, descale_h)
    descaled = core.std.Merge(clipb=descale_natural, clipa=descale_aa, weight=descale_str)
    return descaled

def rescale(c, eedi3Mode=None, nnedi3Mode=None, dx=None, dy=None, pscrn=1, alpha=0.2, beta=0.25, gamma=1000.0):
    ux = c.width * 2
    uy = c.height * 2

    if dx is None:
        raise ValueError('insaneAA: rescale lacks "dx" parameter')
    if dy is None:
        raise ValueError('insaneAA: rescale lacks "dy" parameter')

    c = eedi3_instance(c, eedi3Mode['first'], nnedi3Mode['first'], pscrn, alpha, beta, gamma)
    c = core.std.Transpose(clip=c)
    c = eedi3_instance(c, eedi3Mode['second'], nnedi3Mode['second'], pscrn, alpha, beta, gamma)
    c = core.std.Transpose(clip=c)
    return core.resize.Spline36(c, dx, dy, src_left=-0.5, src_top=-0.5, src_width=ux, src_height=uy)

def eedi3_instance(c, eedi3Mode=None, nnedi3Mode=None, pscrn=1, alpha=0.2, beta=0.25, gamma=1000.0):
    if 'mode' not in eedi3Mode:
        eedi3Mode['mode'] = 'cpu'

    if 'device' not in eedi3Mode:
        eedi3Mode['device'] = -1

    if 'mode' not in nnedi3Mode:
        nnedi3Mode['mode'] = 'cpu'

    if 'device' not in nnedi3Mode:
        nnedi3Mode['device'] = -1

    if eedi3Mode['mode'] == 'opencl':
        return core.eedi3m.EEDI3CL(c, field=1, dh=True, alpha=alpha, beta=beta, gamma=gamma, vcheck=3, sclip=nnedi3_superclip(c, nnedi3Mode, pscrn), device=eedi3Mode['device'])
    else:
        return core.eedi3m.EEDI3(c, field=1, dh=True, alpha=alpha, beta=beta, gamma=gamma, vcheck=3, sclip=nnedi3_superclip(c, nnedi3Mode, pscrn))

def nnedi3_superclip(c, nnedi3Mode=None, pscrn=1):
    if nnedi3Mode['mode'] == 'opencl':
        return core.nnedi3cl.NNEDI3CL(c, field=1, dh=True, nsize=0, nns=4, pscrn=pscrn, device=nnedi3Mode['device'])
    elif nnedi3Mode['mode'] == 'znedi3':
        return core.znedi3.nnedi3(c, field=1, dh=True, nsize=0, nns=4, pscrn=pscrn)
    else:
        return core.nnedi3.nnedi3(c, field=1, dh=True, nsize=0, nns=4, pscrn=pscrn)
