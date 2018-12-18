from vapoursynth import core
from functools import partial
import vapoursynth as vs
import havsfunc as haf
import mvsfunc as mvf
import muvsfunc as muf
import nnedi3_resample as nnrs
import mvmulti, math

# hnwvsfunc - Helenerineium (Wolfberry)'s VapourSynth functions.

"""
Main functions:
FineSharp
psharpen
MCDegrainSharp
LSFmod
SeeSaw
QTGMC
SMDegrain
TemporalDegrain2
mClean
STPressoHD
MLDegrain
NonlinUSM
DetailSharpen
Hysteria
SuperToon
EdgeDetect
JohnFPS
SpotLess
HQDeringmod
MaskedDHA
daamod
LUSM
"""
### Main functions below
def FineSharp(clip, mode=1, sstr=2.5, cstr=None, xstr=0, lstr=1.5, pstr=1.28, ldmp=None, hdmp=0.01, rep=12):
    """
    Original author: Didée (https://forum.doom9.org/showthread.php?t=166082)
    Small and relatively fast realtime-sharpening function, for 1080p,
    or after scaling 720p → 1080p during playback.
    (to make 720p look more like being 1080p)
    It's a generic sharpener. Only for good quality sources!
    (If the source is crap, FineSharp will happily sharpen the crap) :)
    Noise/grain will be enhanced, too. The method is GENERIC.

    Modus operandi: A basic nonlinear sharpening method is performed,
    then the *blurred* sharp-difference gets subtracted again.
    
    Args:
        mode  (int)  - 1 to 3, weakest to strongest. When negative -1 to -3,
                       a broader kernel for equalisation is used.
        sstr (float) - strength of sharpening.
        cstr (float) - strength of equalisation (recommended 0.5 to 1.25)
        xstr (float) - strength of XSharpen-style final sharpening, 0.0 to 1.0.
                       (but, better don't go beyond 0.25...)
        lstr (float) - modifier for non-linear sharpening.
        pstr (float) - exponent for non-linear sharpening.
        ldmp (float) - "low damp", to not over-enhance very small differences.
                       (noise coming out of flat areas)
        hdmp (float) - "high damp", this damping term has a larger effect than ldmp
                        when the sharp-difference is larger than 1, vice versa.
        rep   (int)  - repair mode used in final sharpening, recommended modes are 1/12/13.
    """

    color = clip.format.color_family
    bd = clip.format.bits_per_sample
    isFLOAT = clip.format.sample_type == vs.FLOAT
    mid = 0 if isFLOAT else 1 << (bd - 1)
    i = 0.00392 if isFLOAT else 1 << (bd - 8)
    xy = 'x y - {} /'.format(i) if bd != 8 else 'x y -'
    R = core.rgsf.Repair if isFLOAT else core.rgvs.Repair
    mat1 = [1, 2, 1, 2, 4, 2, 1, 2, 1]
    mat2 = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    
    if not isinstance(clip, vs.VideoNode):
        raise TypeError("FineSharp: This is not a clip!")

    if clip.format.color_family == vs.COMPAT:
        raise TypeError("FineSharp: COMPAT color family is not supported!")

    if cstr is None:
        cstr = spline(sstr, {0: 0, 0.5: 0.1, 1: 0.6, 2: 0.9, 2.5: 1, 3: 1.1, 3.5: 1.15, 4: 1.2, 8: 1.25, 255: 1.5})
        cstr **= 0.8 if mode > 0 else cstr

    if ldmp is None:
        ldmp = sstr

    sstr = max(sstr, 0)
    cstr = max(cstr, 0)
    xstr = min(max(xstr, 0), 1)
    ldmp = max(ldmp, 0)
    hdmp = max(hdmp, 0)

    if sstr < 0.01 and cstr < 0.01 and xstr < 0.01:
        return clip

    tmp = core.std.ShufflePlanes(clip, [0], vs.GRAY) if color in [vs.YUV, vs.YCOCG] else clip

    if abs(mode) == 1:
        c2 = core.std.Convolution(tmp, matrix=mat1).std.Median()
    else:
        c2 = core.std.Median(tmp).std.Convolution(matrix=mat1)
    if abs(mode) == 3:
        c2 = c2.std.Median()
    
    if sstr >= 0.01:
        expr = 'x y = x dup {} dup dup dup abs {} / {} pow swap3 abs {} + / swap dup * dup {} + / * * {} * + ?'
        shrp = core.std.Expr([tmp, c2], [expr.format(xy, lstr, 1/pstr, hdmp, ldmp, sstr*i)])

        if cstr >= 0.01:
            diff = core.std.MakeDiff(shrp, tmp)
            if cstr != 1:
                expr = 'x {} *'.format(cstr) if isFLOAT else 'x {} - {} * {} +'.format(mid, cstr, mid)
                diff = core.std.Expr([diff], [expr])
            diff = core.std.Convolution(diff, matrix=mat1) if mode > 0 else core.std.Convolution(diff, matrix=mat2)
            shrp = core.std.MakeDiff(shrp, diff)

    if xstr >= 0.01:
        xyshrp = core.std.Expr([shrp, core.std.Convolution(shrp, matrix=mat2)], ['x dup y - 9.69 * +'])
        rpshrp = R(xyshrp, shrp, [rep])
        shrp = core.std.Merge(shrp, rpshrp, [xstr])

    return core.std.ShufflePlanes([shrp, clip], [0, 1, 2], color) if color in [vs.YUV, vs.YCOCG] else shrp


def psharpen(clip, strength=25, threshold=75, ssx=1, ssy=1, dw=None, dh=None):
    """
    From https://forum.doom9.org/showthread.php?p=683344 by ilpippo80.
    Sharpeing function similar to LimitedSharpenFaster,
    performs two-point sharpening to avoid overshoot.
    
    Args:
        strength  (float) - Strength of the sharpening, 0 to 100.
        threshold (float) - Controls "how much" to be sharpened, 0 to 100.
        ssx, ssy  (float) - Supersampling factor (reduce aliasing on edges).
        dw, dh     (int)  - Output resolution after sharpening.
    """
    
    if not isinstance(clip, vs.VideoNode):
        raise TypeError("psharpen: This is not a clip!")
    
    if clip.format.sample_type != vs.INTEGER:
        raise TypeError("psharpen: clip must be of integer sample type")

    if clip.format.color_family == vs.COMPAT:
        raise TypeError("psharpen: COMPAT color family is not supported!")
    
    color = clip.format.color_family
    ow = clip.width
    oh = clip.height
    ssx = max(ssx, 1.0)
    ssy = max(ssy, 1.0)
    strength = min(max(strength, 0), 100)
    threshold = min(max(threshold, 0), 100)
    xss = m4(ow * ssx)
    yss = m4(oh * ssy)

    if dw is None:
        dw = ow

    if dh is None:
        dh = oh

    # oversampling
    if ssx > 1 or ssy > 1:
        clip = core.resize.Spline36(clip, xss, yss)
    
    tmp = core.std.ShufflePlanes(clip, [0], vs.GRAY) if color in [vs.YUV, vs.YCOCG] else clip

    # calculating the max and min in every 3*3 square
    maxi = core.std.Maximum(tmp)
    mini = core.std.Minimum(tmp)
    
    # normalizing max and val to values from 0 to (max-min)
    nmax = core.std.Expr([maxi, mini], ['x y -'])
    nval = core.std.Expr([tmp,  mini], ['x y -'])
    
    # initializing expression used to obtain the output luma value
    s = strength / 100
    t = threshold / 100
    st = 1 - (s / (s + t - s * t))
    expr = 'x y / 2 * 1 - abs {} < {} 1 = x y 2 / = 0 y 2 / ? x y / 2 * 1 - abs 1 {} - / ? x y / 2 * 1 - abs 1 {} - * {} + ? x y 2 / > 1 -1 ? * 1 + y * 2 /'
    expr = expr.format(st, s, s, t, t)

    # calculates the new luma value pushing it towards min or max
    nval = core.std.Expr([nval, nmax], [expr])
    
    # normalizing val to values from min to max
    tmp = core.std.Expr([nval, mini], ['x y +'])
    
    # resizing the image to the output resolution
    # applying the new luma value to clip
    if dw != ow or dh != oh:
        if color in [vs.YUV, vs.YCOCG]:
            tmp = core.std.ShufflePlanes([tmp, clip], [0, 1, 2], color)
        return core.resize.Spline36(tmp, dw, dh)
    elif ssx > 1 or ssy > 1:
        if color in [vs.YUV, vs.YCOCG]:
            tmp = core.std.ShufflePlanes([tmp, clip], [0, 1, 2], color)
        return core.resize.Spline36(tmp, dw, dh)
    elif color in [vs.YUV, vs.YCOCG]:
        return core.std.ShufflePlanes([tmp, clip], [0, 1, 2], color)
    else:
        return tmp


def MCDegrainSharp(clip, tr=3, bblur=.69, csharp=.69, thSAD=400, rec=False, chroma=True, analyse_args=None, recalculate_args=None):
    """
    Based on MCDegrain By Didée:
    https://forum.doom9.org/showthread.php?t=161594
    Also based on Didée observations in this thread:
    https://forum.doom9.org/showthread.php?t=161580

    Denoise with MDegrainX, do slight sharpening where motionmatch
    is good, do slight blurring where motionmatch is bad.

    In areas where MAnalyse cannot find good matches,
    the Blur() will be dominant.
    In areas where good matches are found,
    the Sharpen()'ed pixels will overweight the Blur()'ed pixels
    when the pixel averaging is performed.
    
    Args:
        bblur, csharp (int, float or function) The method to smooth/sharpen the image.
        If it's an int or a float, it specifies the amount in Blur() and Sharpen().
        If it's a function, it will use the specified function to smooth/sharpen the image.
        tr     (int)  - Strength of the denoising (1-24).
        thSAD  (int)  - Soft threshold of block sum absolute differences.
                         Low value can result in staggered denoising,
                         High value can result in ghosting and artifacts.
        rec    (bool) - Recalculate the motion vectors to obtain more precision.
        chroma (bool) - Whether to process chroma.
    """
    
    if not isinstance(clip, vs.VideoNode) or clip.format.color_family not in [vs.GRAY, vs.YUV]:
        raise TypeError("MCDegrainSharp: This is not a GRAY or YUV clip!")
    
    isFLOAT = clip.format.sample_type == vs.FLOAT
    isGRAY = clip.format.color_family == vs.GRAY
    chroma = False if isGRAY else chroma
    planes = [0, 1, 2] if chroma else [0]
    plane = 4 if chroma else 0
    bs = 32 if clip.width > 2400 else 16 if clip.width > 960 else 8
    pel = 1 if clip.width > 960 else 2
    truemotion = False if clip.width > 960 else True
    A = core.mvsf.Analyse if isFLOAT else core.mv.Analyse
    S = core.mvsf.Super if isFLOAT else core.mv.Super
    R = core.mvsf.Recalculate if isFLOAT else core.mv.Recalculate
    D1 = core.mvsf.Degrain1 if isFLOAT else core.mv.Degrain1
    D2 = core.mvsf.Degrain2 if isFLOAT else core.mv.Degrain2
    D3 = core.mvsf.Degrain3 if isFLOAT else core.mv.Degrain3

    if isinstance(bblur, (int, float)):
        c2 = muf.Blur(clip, amountH=bblur, planes=planes)
    elif callable(bblur):
        c2 = bblur(clip)
    else:
        raise TypeError("MCDegrainSharp: bblur must be an int, a float or a function!")
        
    if isinstance(csharp, (int, float)):
        c4 = muf.Sharpen(clip, amountH=csharp, planes=planes)
    elif callable(csharp):
        c4 = csharp(clip)
    else:
        raise TypeError("MCDegrainSharp: csharp must be an int, a float or a function!")

    if analyse_args is None:
        analyse_args = dict(blksize=bs, overlap=bs//2, search=5, chroma=chroma, truemotion=truemotion)
        
    if recalculate_args is None:
        recalculate_args = dict(blksize=bs//2, overlap=bs//4, search=5, chroma=chroma, truemotion=truemotion)

    if tr > 3 and not isFLOAT:
        raise TypeError("MCDegrainSharp: DegrainN is only available in float")

    super_b = S(DitherLumaRebuild(c2, 1), hpad=bs, vpad=bs, pel=pel, sharp=1, rfilter=4)
    super_rend = S(c4, hpad=bs, vpad=bs, pel=pel, levels=1, rfilter=1)

    if tr < 4:
        bv1 = A(super_b, isb=True,  delta=1, **analyse_args)
        fv1 = A(super_b, isb=False, delta=1, **analyse_args)
        if tr > 1:
            bv2 = A(super_b, isb=True,  delta=2, **analyse_args)
            fv2 = A(super_b, isb=False, delta=2, **analyse_args)
        if tr > 2:
            bv3 = A(super_b, isb=True,  delta=3, **analyse_args)
            fv3 = A(super_b, isb=False, delta=3, **analyse_args)
    else:
        vec = mvmulti.Analyze(super_b, tr=tr, **analyse_args)
    
    if rec:
        if tr < 4:
            bv1 = R(super_b, bv1, **recalculate_args)
            fv1 = R(super_b, fv1, **recalculate_args)
            if tr > 1:
                bv2 = R(super_b, bv2, **recalculate_args)
                fv2 = R(super_b, fv2, **recalculate_args)
            if tr > 2:
                bv3 = R(super_b, bv3, **recalculate_args)
                fv3 = R(super_b, fv3, **recalculate_args)    
        else:
            vec = mvmulti.Recalculate(super_b, vec, tr=tr, **recalculate_args)
    
    if tr <= 1:
        return D1(c2, super_rend, bv1, fv1, thSAD, plane=plane)
    elif tr == 2:
        return D2(c2, super_rend, bv1, fv1, bv2, fv2, thSAD, plane=plane)
    elif tr == 3:
        return D3(c2, super_rend, bv1, fv1, bv2, fv2, bv3, fv3, thSAD, plane=plane)
    else:
        return mvmulti.DegrainN(c2, super_rend, vec, tr=tr, thsad=thSAD, plane=plane)


def LSFmod(clip, strength=100, Smode=None, Smethod=None, kernel=11, preblur=False, secure=None, source=None,
           Szrp=16, Spwr=None, SdmpLo=None, SdmpHi=None, Lmode=None, overshoot=None, undershoot=None, overshoot2=None, undershoot2=None,
           soft=None, soothe=None, keep=None, edgemode=0, edgemaskHQ=None, ssx=None, ssy=None, dw=None, dh=None, defaults='slow'):
    """                                                                                          
    LimitedSharpenFaster Modded Version by LaTo INV.

    +--------------+
    | DEPENDENCIES |
    +--------------+

     → RGVS / RGSF

    +---------+
    | GENERAL |
    +---------+

    strength [int]
    --------------
    Strength of the sharpening

    Smode [int]
    ----------------------
    Sharpen mode:
        = 1 : Range sharpening
        = 2 : Nonlinear sharpening (corrected version)
        = 3 : Nonlinear sharpening (original version)

    Smethod [int]
    --------------------
    Sharpen method:
        = 1 : 3x3 kernel
        = 2 : Min/Max
        = 3 : Min/Max + 3x3 kernel

    kernel [int]
    -------------------------
    Kernel used in Smethod=1&3
    In strength order: + 19 >> 20 > 11/12 -
    Negative: absolute value specifies the MinBlur radius used.

    +---------+
    | SPECIAL |
    +---------+

    preblur [bool]
    --------------------------------
    Mode to avoid noise sharpening & ringing

    secure [bool]
    -------------
    Mode to avoid banding & oil painting (or face wax) effect of sharpening

    source [clip]
    -------------
    If source is defined, LSFmod doesn't sharp more a denoised clip than this source clip
    In this mode, you can safely set Lmode = 0 & PP = OFF

    +----------------------+
    | NONLINEAR SHARPENING |
    +----------------------+

    Szrp [int]
    ----------
    Zero Point:
        - differences below Szrp are amplified (overdrive sharpening)
        - differences above Szrp are reduced   (reduced sharpening)

    Spwr [int]
    ----------
    Power: exponent for sharpener

    SdmpLo [int]
    ------------
    Damp Low: reduce sharpening for small changes [0:disable]

    SdmpHi [int]
    ------------
    Damp High: reduce sharpening for big changes  [0:disable]

    +----------+
    | LIMITING |
    +----------+

    Lmode [int]
    --------------------------
    Limit mode:
        < 0 : Limit with Repair (ex: Lmode = -1 → Repair(1), Lmode = -5 → Repair(5)...)
        = 0 : No limit
        = 1 : Limit to over/undershoot
        = 2 : Limit to over/undershoot on edges and no limit on not-edges
        = 3 : Limit to zero on edges and to over/undershoot on not-edges
        = 4 : Limit to over/undershoot on edges and to over/undershoot2 on not-edges

    overshoot [int]
    ---------------
    Limit for pixels that get brighter during sharpening

    undershoot [int]
    ----------------
    Limit for pixels that get darker during sharpening

    overshoot2 [int]
    ----------------
    Same as overshoot,  only for Lmode = 4

    undershoot2 [int]
    -----------------
    Same as undershoot, only for Lmode = 4

    +-----------------+
    | POST-PROCESSING |
    +-----------------+

    soft [int]
    -------------------------
    Soft the sharpening effect (Negative: new autocalculate)

    soothe [bool]
    -------------
        = True  : Enable  soothe temporal stabilization
        = False : Disable soothe temporal stabilization

    keep [int]
    -------------------
    Minimum percent of the original sharpening to keep (only with soothe=True)

    +-------+
    | EDGES |
    +-------+

    edgemode [int]
    ------------------------
        =-1 : Show edgemask
        = 0 : Sharpening all
        = 1 : Sharpening only edges
        = 2 : Sharpening only not-edges

    edgemaskHQ [bool]
    -----------------
        = True  : Original edgemask
        = False : Faster edgemask

    +------------+
    | UPSAMPLING |
    +------------+

    ssx ssy [float]
    -------------------
    Supersampling factor (reduce aliasing on edges)

    dw dh [int]
    ---------------------
    Output resolution after sharpening (avoid a resizing step)

    +----------+
    | SETTINGS |
    +----------+

    defaults [string]
    --------------------------------------------
        = "old"  : Reset settings to original version (output will be THE SAME AS LSF)
        = "slow" : Enable SLOW modded version settings
        = "fast" : Enable FAST modded version settings

    defaults = "old" :  - strength    = 100
    ----------------    - Smode       = 1
                        - Smethod     = Smode == 1 ? 2 : 1
                        - kernel      = 11
                        - preblur     = False
                        - secure      = False
                        - source      = undefined
                        - Szrp        = 16
                        - Spwr        = 2
                        - SdmpLo      = strength / 25
                        - SdmpHi      = 0
                        - Lmode       = 1
                        - overshoot   = 1
                        - undershoot  = overshoot
                        - overshoot2  = overshoot * 2
                        - undershoot2 = overshoot2
                        - soft        = 0
                        - soothe      = False
                        - keep        = 25
                        - edgemode    = 0
                        - edgemaskHQ  = True
                        - ssx         = Smode == 1 ? 1.50 : 1.25
                        - ssy         = ssx
                        - dw          = ow
                        - dh          = oh

    defaults = "slow" : - strength    = 100
    -----------------   - Smode       = 2
                        - Smethod     = 3
                        - kernel      = 11
                        - preblur     = False
                        - secure      = True
                        - source      = undefined
                        - Szrp        = 16
                        - Spwr        = 4
                        - SdmpLo      = 4
                        - SdmpHi      = 48
                        - Lmode       = 4
                        - overshoot   = strength / 100
                        - undershoot  = overshoot
                        - overshoot2  = overshoot * 2
                        - undershoot2 = overshoot2
                        - soft        = -2
                        - soothe      = True
                        - keep        = 20
                        - edgemode    = 0
                        - edgemaskHQ  = True
                        - ssx         = 1.50
                        - ssy         = ssx
                        - dw          = ow
                        - dh          = oh

    defaults = "fast" : - strength    = 100
    -----------------   - Smode       = 1
                        - Smethod     = 2
                        - kernel      = 11
                        - preblur     = False
                        - secure      = True
                        - source      = undefined
                        - Szrp        = 16
                        - Spwr        = 4
                        - SdmpLo      = 4
                        - SdmpHi      = 48
                        - Lmode       = 1
                        - overshoot   = strength / 100
                        - undershoot  = overshoot
                        - overshoot2  = overshoot * 2
                        - undershoot2 = overshoot2
                        - soft        = 0
                        - soothe      = True
                        - keep        = 20
                        - edgemode    = 0
                        - edgemaskHQ  = False
                        - ssx         = 1.25
                        - ssy         = ssx
                        - dw          = ow
                        - dh          = oh
    """
    # Modified from havsfunc: https://github.com/HomeOfVapourSynthEvolution/havsfunc/blob/r31/havsfunc.py#L4492

    if not isinstance(clip, vs.VideoNode):
        raise TypeError('LSFmod: This is not a clip!')
    if source is not None and (not isinstance(source, vs.VideoNode) or source.format.id != clip.format.id):
        raise TypeError("LSFmod: source must be the same format as clip")
    if source is not None and (source.width != clip.width or source.height != clip.height):
        raise TypeError("LSFmod: source must be the same size as clip")
    if clip.format.color_family == vs.COMPAT:
        raise TypeError("LSFmod: COMPAT color family is not supported!")
    
    color = clip.format.color_family
    bd = clip.format.bits_per_sample
    isFLOAT = clip.format.sample_type == vs.FLOAT
    mid = 0 if isFLOAT else 1 << (bd - 1)
    i = 0.00392 if isFLOAT else 1 << (bd - 8)
    R = core.rgsf.Repair if isFLOAT else core.rgvs.Repair
    ow = clip.width
    oh = clip.height
    csrc = clip

    ### DEFAULTS
    try:
        num = ['old', 'slow', 'fast'].index(defaults.lower())
    except:
        raise ValueError('LSFmod: Defaults must be "old" or "slow" or "fast"')

    if Smode is None:
        Smode = [1, 2, 1][num]
    if Smethod is None:
        Smethod = [2 if Smode == 1 else 1, 3, 2][num]
    if secure is None:
        secure = [False, True, True][num]
    if Spwr is None:
        Spwr = [2, 4, 4][num]
    if SdmpLo is None:
        SdmpLo = [strength/25, 4, 4][num]
    if SdmpHi is None:
        SdmpHi = [0, 48, 48][num]
    if Lmode is None:
        Lmode = [1, 4, 1][num]
    if overshoot is None:
        overshoot = [1, strength/100, strength/100][num]
    if undershoot is None:
        undershoot = overshoot
    if overshoot2 is None:
        overshoot2 = overshoot * 2
    if undershoot2 is None:
        undershoot2 = overshoot2
    if soft is None:
        soft = [0, -2, 0][num]
    if soothe is None:
        soothe = [False, True, True][num]
    if keep is None:
        keep = [25, 20, 20][num]
    if edgemaskHQ is None:
        edgemaskHQ = [True, True, False][num]
    if ssx is None:
        ssx = [1.5 if Smode == 1 else 1.25, 1.5, 1.25][num]
    if ssy is None:
        ssy = ssx
    if dw is None:
        dw = ow
    if dh is None:
        dh = oh

    if kernel <= 0:
        Filter = partial(MinBlur, r=abs(kernel))
    elif kernel == 4:
        Filter = partial(core.std.Median)
    elif kernel in [11, 12]:
        Filter = partial(core.std.Convolution, matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1])
    elif kernel == 19:
        Filter = partial(core.std.Convolution, matrix=[1, 1, 1, 1, 0, 1, 1, 1, 1])
    elif kernel == 20:
        Filter = partial(core.std.Convolution, matrix=[1, 1, 1, 1, 1, 1, 1, 1, 1])
    else:
        if isFLOAT:
            Filter = partial(core.rgsf.RemoveGrain, mode=[kernel])
        else:
            Filter = partial(core.rgvs.RemoveGrain, mode=[kernel])

    if soft < 0:
        soft = (1 + (2 / (ssx + ssy))) * math.sqrt(strength)

    Spwr = max(Spwr, 1)
    SdmpLo = max(SdmpLo, 0)
    SdmpHi = max(SdmpHi, 0)
    ssx = max(ssx, 1)
    ssy = max(ssy, 1)
    soft = max(min(soft/100, 1), 0)
    keep = max(min(keep/100, 1), 0)
    xss = m4(ow * ssx)
    yss = m4(oh * ssy)
    strength = max(strength, 0)
    Str = strength / 100

    ### SHARP
    if ssx > 1 or ssy > 1:
        clip = core.resize.Spline36(clip, xss, yss)

    tmp = core.std.ShufflePlanes(clip, [0], vs.GRAY) if color in [vs.YUV, vs.YCOCG] else clip
    pre = MinBlur(tmp) if preblur else tmp
    darklimit = core.std.Minimum(pre)
    brightlimit = core.std.Maximum(pre)

    if Smethod <= 1:
        method = Filter(pre)
    elif Smethod == 2:
        method = core.std.Merge(darklimit, brightlimit)
    else:
        method = Filter(core.std.Merge(darklimit, brightlimit))

    if secure:
        method = core.std.Expr([method, pre], ['x y < x {} + x y > x {} - x ? ?'.format(i, i)])

    if preblur:
        method = core.std.Expr([method, tmp, pre], ['x y + z -'])

    if Smode <= 1:
        normsharp = core.std.Expr([tmp, method], ['x dup y - {} * +'.format(Str)])
    elif Smode == 2:
        xy = 'x y - abs {} /'.format(i) if bd != 8 else 'x y - abs'
        Hi = 1 if SdmpHi == 0 else '1 {} {} / 4 pow + 1 {} {} / 4 pow + /'.format(Szrp, SdmpHi, xy, SdmpHi)
        expr1 = 'x y = x dup {} dup {} / {} pow swap dup * dup {} {} + * swap {} + {} * / * {} * {} * x y > 1 -1 ? * + ?'
        normsharp = core.std.Expr([tmp, method], [expr1.format(xy, Szrp, 1/Spwr, Szrp**2, SdmpLo, SdmpLo, Szrp**2, Hi, Szrp*Str*i)])
    else:
        xy = 'x y - abs {} /'.format(i) if bd != 8 else 'x y - abs'
        Hi = 1 if SdmpHi == 0 else '1 {} {} / 4 pow +'.format(xy, SdmpHi)
        expr1 = 'x y = x dup {} dup {} / {} pow swap dup * dup {} + / * {} / {} * x y > 1 -1 ? * + ?'
        normsharp = core.std.Expr([tmp, method], [expr1.format(xy, Szrp, 1/Spwr, SdmpLo, Hi, Szrp*Str*i)])

    ### LIMIT
    normal = haf.Clamp(normsharp, brightlimit, darklimit, max(overshoot, 0)*i, max(undershoot, 0)*i)

    if edgemaskHQ:
        edge = core.std.Expr([core.std.Sobel(tmp, scale=2)], 'x {} / 0.87 pow {} *'.format(128*i, 255*i))
    else:
        edge = core.std.Expr([core.std.Maximum(tmp), core.std.Minimum(tmp)], ['x y - {} / 0.87 pow {} *'.format(32*i, 255*i)])
    
    inflate = core.std.Inflate(edge)

    if Lmode == 0:
        limit1 = normsharp
    elif Lmode == 1:
        limit1 = normal
    elif Lmode == 2:
        limit1 = core.std.MaskedMerge(normsharp, normal, inflate)
    elif Lmode == 3:
        zero = haf.Clamp(normsharp, brightlimit, darklimit, 0, 0)
        limit1 = core.std.MaskedMerge(normal, zero, inflate)
    elif Lmode == 4:
        second = haf.Clamp(normsharp, brightlimit, darklimit, max(overshoot2, 0)*i, max(undershoot2, 0)*i)
        limit1 = core.std.MaskedMerge(second, normal, inflate)
    else:
        limit1 = R(normsharp, tmp, abs(Lmode))

    if edgemode < 0:
        return core.resize.Spline36(edge, dw, dh)
    if edgemode == 0:
        limit2 = limit1
    elif edgemode == 1:
        limit2 = core.std.MaskedMerge(tmp, limit1, inflate.std.Inflate().std.Convolution(matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1]))
    else:
        limit2 = core.std.MaskedMerge(limit1, tmp, inflate.std.Inflate().std.Convolution(matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1]))

    ### SOFT
    if soft == 0:
        PP1 = limit2
    else:
        sharpdiff = core.std.MakeDiff(tmp, limit2)
        if isFLOAT:
            expr2 = 'x abs y abs > y {} * x {} * + x ?'.format(soft, 1-soft)
        else:
            expr2 = 'x {} - abs y {} - abs > y {} * x {} * + x ?'.format(mid, mid, soft, 1-soft)
        sharpdiff = core.std.Expr([sharpdiff, core.std.Convolution(sharpdiff, matrix=[1]*9)],[expr2])
        PP1 = core.std.MakeDiff(tmp, sharpdiff)

    ### SOOTHE
    if soothe:
        diff = core.std.MakeDiff(tmp, PP1)
        if isFLOAT:
            expr3 = 'x y * 0 < x {} * x abs y abs > x {} * y {} * + x ? ?'.format(keep, keep, 1-keep)
        else:
            expr3 = 'x {m} - y {m} - * 0 < x {m} - {k} * {m} + x {m} - abs y {m} - abs > x {k} * y {j} * + x ? ?'.format(m=mid, k=keep, j=1-keep)
        diff = core.std.Expr([diff, haf.AverageFrames(diff, [1, 1, 1], 0.125)],[expr3])
        PP2 = core.std.MakeDiff(tmp, diff)
    else:
        PP2 = PP1

    ### OUTPUT
    if dw != ow or dh != oh:
        if color in [vs.YUV, vs.YCOCG]:
            PP2 = core.std.ShufflePlanes([PP2, clip], [0, 1, 2], color)
        PPP = core.resize.Spline36(PP2, dw, dh)
    elif ssx > 1 or ssy > 1:
        if color in [vs.YUV, vs.YCOCG]:
            PP2 = core.std.ShufflePlanes([PP2, clip], [0, 1, 2], color)
        PPP = core.resize.Spline36(PP2, dw, dh)
    elif color in [vs.YUV, vs.YCOCG]:
        PPP = core.std.ShufflePlanes([PP2, clip], [0, 1, 2], color)
    else:
        PPP = PP2

    if source is not None:
        if dw != ow or dh != oh:
            src = core.resize.Spline36(source, dw, dh)
            In = core.resize.Spline36(csrc, dw, dh)
        else:
            src = source
            In = csrc
        shrpD = core.std.MakeDiff(In, PPP, [0]) if color in [vs.YUV, vs.YCOCG] else core.std.MakeDiff(In, PPP)
        expr4 = 'x abs y abs < x y ?' if isFLOAT else 'x {} - abs y {} - abs < x y ?'.format(mid, mid)
        if color in [vs.YUV, vs.YCOCG]:
            shrpL = core.std.Expr([R(shrpD, core.std.MakeDiff(In, src, [0]), [1, 0]), shrpD], [expr4, ''])
        else:
            shrpL = core.std.Expr([R(shrpD, core.std.MakeDiff(In, src), [1]), shrpD], [expr4])
        return core.std.MakeDiff(In, shrpL, [0]) if color in [vs.YUV, vs.YCOCG] else core.std.MakeDiff(In, shrpL)
    else:
        return PPP


def SeeSaw(clip, denoised=None, NRlimit=2, NRlimit2=None, sstr=2, Slimit=None, Spower=4, SdampLo=None, SdampHi=24, Szp=16, bias=50,
                 Smode=None, NSmode=1, sootheT=50, sootheS=0, ssx=1, ssy=None, diff=False):
    """
    Author: Didée (https://avisynth.nl/images/SeeSaw.avs)
    (Full Name: "Denoiser-and-Sharpener-are-riding-the-SeeSaw")
    This function provides a (simple) implementation of the "crystality sharpen" principle.
    In conjunction with a user-specified denoised clip, the aim is to enhance weak detail, 
    hopefully without oversharpening or creating jaggies on strong detail, 
    and produce a result that is temporally stable without detail shimmering,
    while keeping everything within reasonable bitrate requirements.
    This is done by intermixing source, denoised source and a modified sharpening process, in a seesaw-like manner.
    You're very much encouraged to feed your own custom denoised clip into SeeSaw.
    If the "denoised" clip parameter is omitted, a simple "spatial pressdown" filter is used.

    Args:
        NRlimit  (int)  - Absolute limit for pixel change by denoising.
        NRlimit2 (int)  - Limit for intermediate denoising.
        sstr    (float) - Sharpening strength (don't touch this too much).
        Slimit   (int)  - Positive: absolute limit for pixel change by sharpening.
                          Negative: pixel's sharpening difference is reduced to pow(diff, 1/abs(Slimit)).
        Spower  (float) - Exponent for modified sharpener.
        Szp     (float) - Zero point, below: overdrive sharpening, above: reduced sharpening.
        SdampLo (float) - Reduces overdrive sharpening for very small changes.
        SdampHi (float) - Further reduces sharpening for big sharpening changes. Try 15~30. "0" disables.
        bias    (float) - Bias towards detail (> 50), or towards calm result (< 50).
        Smode    (int)  - RemoveGrain mode used in the modified sharpening function (sharpen2).
                          Negative: absolute value specifies the MinBlur radius used.
        NSmode   (int)  - 0 = corrected, 1 = original nonlinear sharpening.
        sootheT  (int)  - 0 = minimum, 100 = maximum soothing of sharpener's temporal instability.
                          Negative: will chain 2 instances of temporal soothing.
        sootheS  (int)  - 0 = minimum, 100 = maximum smoothing of sharpener's spatial effect.
        ssx, ssy (int)  - SeeSaw doesn't require supersampling urgently, if at all, small values ~1.25 seems to be enough.
        diff     (bool) - When True, limit the sharp-difference instead of the sharpened clip.
                          Relative limiting is more safe, less aliasing, but also less sharpening.
                          This parameter has no effect when Smode <= 0.
    """
    # Modified from muvsfunc: https://github.com/WolframRhodium/muvsfunc/blob/v0.2.0/muvsfunc.py#L2007

    if not isinstance(clip, vs.VideoNode):
        raise TypeError("SeeSaw: This is not a clip!")
    if clip.format.color_family == vs.COMPAT:
        raise TypeError("SeeSaw: COMPAT color family is not supported!")
    if NRlimit2 is None:
        NRlimit2 = NRlimit + 1
    if Slimit is None:
        Slimit = NRlimit + 2
    if SdampLo is None:
        SdampLo = Spower + 1
    if Smode is None:
        if ssx <= 1.25:
            Smode = 11
        elif ssx <= 1.6:
            Smode = 20
        else:
            Smode = 19
    if ssy is None:
        ssy = ssx
    
    color = clip.format.color_family
    Spower = max(Spower, 1)
    SdampLo = max(SdampLo, 0)
    SdampHi = max(SdampHi, 0)
    ssx = max(ssx, 1)
    ssy = max(ssy, 1)
    ow = clip.width
    oh = clip.height
    xss = m4(ow * ssx)
    yss = m4(oh * ssy)
    isFLOAT = clip.format.sample_type == vs.FLOAT
    bd = clip.format.bits_per_sample
    i = 0.00392 if isFLOAT else 1 << (bd - 8)
    mid = 0 if isFLOAT else 1 << (bd - 1)
    peak = 1.0 if isFLOAT else (1 << bd) - 1
    bias = max(min(bias/100, 1), 0)
    NRL = scale(NRlimit, peak)
    NRL2 = scale(NRlimit2, peak)
    NRLL = scale(NRlimit2/bias - 1, peak)
    SLIM = scale(Slimit, peak) if Slimit >= 0 else 1/abs(Slimit)
    R = core.rgsf.Repair if isFLOAT else core.rgvs.Repair
    
    if denoised is None:
        dnexpr = 'x {N} + y < x {N} + x {N} - y > x {N} - y ? ?'.format(N=NRL)
        denoised = core.std.Expr([clip, core.std.Median(clip, [0])], [dnexpr, '']) if color in [vs.YUV, vs.YCOCG] else core.std.Expr([clip, core.std.Median(clip)], [dnexpr])
    else:
        if not isinstance(denoised, vs.VideoNode) or denoised.format.id != clip.format.id:
            raise TypeError("SeeSaw: denoised must be the same format as clip!")
        if denoised.width != clip.width or denoised.height != clip.height:
            raise TypeError("SeeSaw: denoised must be the same size as clip!")

    if color in [vs.YUV, vs.YCOCG]:
        tmp = core.std.ShufflePlanes(clip, [0], vs.GRAY)
        tmp2 = core.std.ShufflePlanes(denoised, [0], vs.GRAY) if clip != denoised else tmp
    else:
        tmp = clip
        tmp2 = denoised

    tameexpr = 'x {N} + y < x {N2} + x {N} - y > x {N2} - x {B} * y {j} * + ? ?'.format(N=NRLL, N2=NRL2, B=bias, j=1-bias)
    tame = core.std.Expr([tmp, tmp2], [tameexpr])

    if Smode > 0:
        head = Sharpen2(tame, sstr, Spower, Szp, SdampLo, SdampHi, 4, NSmode, diff)
        head = core.std.MaskedMerge(head, tame, tame.std.Sobel().std.Maximum().std.Convolution(matrix=[1]*9))

    if ssx == 1 and ssy == 1:
        if Smode <= 0:
            sharp = Sharpen2(tame, sstr, Spower, Szp, SdampLo, SdampHi, Smode, NSmode, diff)
        else:
            sharp = R(Sharpen2(tame, sstr, Spower, Szp, SdampLo, SdampHi, Smode, NSmode, diff), head, [1])
    else:
        if Smode <= 0:
            sharp = Sharpen2(tame.resize.Spline36(xss, yss), sstr, Spower, Szp, SdampLo, SdampHi, Smode, NSmode, diff).resize.Spline36(ow, oh)
        else:
            sharp = R(Sharpen2(tame.resize.Spline36(xss, yss), sstr, Spower, Szp, SdampLo, SdampHi, Smode, NSmode, diff), head.resize.Spline36(xss, yss), [1]).resize.Spline36(ow, oh)
        
    if diff and Smode > 0:
        sharp = core.std.MergeDiff(tame, sharp)
        
    soothed = SootheSS(sharp, tame, sootheT, sootheS)
    sharpdiff = core.std.MakeDiff(tame, soothed)

    if NRlimit == 0 or clip == denoised:
        calm = tmp
    else:
        NRdiff = core.std.MakeDiff(tmp, tmp2)
        if isFLOAT:
            expr = 'y {N} > x {N} - y -{N} < x {N} + x y - ? ?'.format(N=NRL)
        else:
            expr = 'y {m} {N} + > x {N} - y {m} {N} - < x {N} + x y {m} - - ? ?'.format(m=mid, N=NRL)
        calm = core.std.Expr([tmp, NRdiff], [expr])

    if Slimit >= 0:
        if isFLOAT:
            limitexpr = 'y {S} > x {S} - y -{S} < x {S} + x y - ? ?'.format(S=SLIM)
        else:
            limitexpr = 'y {m} {S} + > x {S} - y {m} {S} - < x {S} + x y {m} - - ? ?'.format(m=mid, S=SLIM)
        limited = core.std.Expr([calm, sharpdiff], [limitexpr])
    else:
        if isFLOAT:
            limitexpr = 'y 0 = x dup y abs {i} / {S} pow {i} * y 0 > 1 -1 ? * - ?'.format(S=SLIM, i=i)
        else:
            limitexpr = 'y {m} = x dup y {m} - abs {i} / {S} pow {i} * y {m} > 1 -1 ? * - ?'.format(m=mid, S=SLIM, i=i)
        limited = core.std.Expr([calm, sharpdiff], [limitexpr])

    return core.std.ShufflePlanes([limited, clip], [0, 1, 2], color) if color in [vs.YUV, vs.YCOCG] else limited


def Sharpen2(clip, sstr, power, zp, ldmp, hdmp, rg, mode, diff):
    """Modified sharpening function from SeeSaw()"""

    color = clip.format.color_family
    isFLOAT = clip.format.sample_type == vs.FLOAT
    bd = clip.format.bits_per_sample
    i = 0.00392 if isFLOAT else 1 << (bd - 8)
    R = core.rgsf.RemoveGrain if isFLOAT else core.rgvs.RemoveGrain

    if not isinstance(clip, vs.VideoNode):
        raise TypeError("Sharpen2: This is not a clip!")
    if color == vs.COMPAT:
        raise TypeError("Sharpen2: COMPAT color family is not supported!")
    
    tmp = core.std.ShufflePlanes(clip, [0], vs.GRAY) if color in [vs.YUV, vs.YCOCG] else clip

    if rg <= 0:
        diff = False
        method = MinBlur(tmp, abs(rg))
    elif rg == 4:
        method = tmp.std.Median()
    elif rg in [11, 12]:
        method = tmp.std.Convolution(matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1])
    elif rg == 19:
        method = tmp.std.Convolution(matrix=[1, 1, 1, 1, 0, 1, 1, 1, 1])
    elif rg == 20:
        method = tmp.std.Convolution(matrix=[1, 1, 1, 1, 1, 1, 1, 1, 1])
    else:
        method = R(tmp, mode=[rg])
    
    if mode == 0:
        xy = 'x y - abs {} /'.format(i) if bd != 8 else 'x y - abs'
        Hi = 1 if hdmp == 0 else '1 {} {} / 4 pow + 1 {} {} / 4 pow + /'.format(zp, hdmp, xy, hdmp)
        expr = 'x y = x dup {} dup {} / {} pow swap dup * dup {} {} + * swap {} + {} * / * {} * {} * x y > 1 -1 ? * + ?'
        expr = expr.format(xy, zp, 1/power, zp**2, ldmp, ldmp, zp**2, Hi, zp*sstr*i)
        normsharp = core.std.Expr([tmp, method], [expr])
    else:
        xy = 'x y - abs {} /'.format(i) if bd != 8 else 'x y - abs'
        Hi = 1 if hdmp == 0 else '1 {} {} / 4 pow +'.format(xy, hdmp)
        expr = 'x y = x dup {} dup {} / {} pow swap dup * dup {} + / * {} / {} * x y > 1 -1 ? * + ?'
        normsharp = core.std.Expr([tmp, method], [expr.format(xy, zp, 1/power, ldmp, Hi, zp*sstr*i)])
    
    if color in [vs.YUV, vs.YCOCG]:
        normsharp = core.std.ShufflePlanes([normsharp, clip], [0, 1, 2], color)
    
    return normsharp if not diff else core.std.MakeDiff(normsharp, clip) 


def SootheSS(sharp, src, sootheT=50, sootheS=0):
    """Soothe() function to stabilze sharpening from SeeSaw()"""

    if not isinstance(sharp, vs.VideoNode):
        raise TypeError("SootheSS: sharp must be a clip!")
    if sharp.format.color_family == vs.COMPAT:
        raise TypeError("Sharpen2: COMPAT color family is not supported!")
    if not isinstance(src, vs.VideoNode) or src.format.id != sharp.format.id:
        raise TypeError("SootheSS: src must be of the same format as sharp!") 
    if src.width != sharp.width or src.height != sharp.height:
        raise TypeError("SootheSS: src must be of the same size as sharp!")

    ssrc = src
    ST = min(abs(sootheT)/100, 1)
    SS = min(abs(sootheS)/100, 1)
    color = sharp.format.color_family
    isFLOAT = sharp.format.sample_type == vs.FLOAT
    mid = 0 if isFLOAT else 1 << (sharp.format.bits_per_sample - 1)

    if color in [vs.YUV, vs.YCOCG]:
        sharp = core.std.ShufflePlanes(sharp, [0], vs.GRAY)
        src = core.std.ShufflePlanes(src, [0], vs.GRAY)

    diff = core.std.MakeDiff(src, sharp)
        
    if isFLOAT:
        expr1 = 'x y * 0 < x {S} * x abs y abs > x {S} * y {SS} * + x ? ?'.format(S=1-SS, SS=SS)
    else:
        expr1 = 'x {m} - y {m} - * 0 < x {m} - {SS} * {m} + x {m} - abs y {m} - abs > x {S} * y {SS} * + x ? ?'.format(m=mid, S=1-SS, SS=SS)
    
    if isFLOAT:
        expr2 = 'x y * 0 < x {T} * x abs y abs > x {T} * y {ST} * + x ? ?'.format(T=1-ST, ST=ST)
    else:
        expr2 = 'x {m} - y {m} - * 0 < x {m} - {T} * {m} + x {m} - abs y {m} - abs > x {T} * y {ST} * + x ? ?'.format(m=mid, T=1-ST, ST=ST)

    if SS > 0:
        soothed = core.std.Expr([diff, core.std.Convolution(diff, [1]*9)], [expr1])
    if ST > 0:
        soothed = core.std.Expr([diff, haf.AverageFrames(diff, [1, 1, 1], 0.125)], [expr2])
    if sootheT < 0:
        soothed = core.std.Expr([diff, haf.AverageFrames(diff, [1, 1, 1], 0.125)], [expr2])

    soothed = core.std.MakeDiff(src, soothed)
    
    return core.std.ShufflePlanes([soothed, ssrc], [0, 1, 2], color) if color in [vs.YUV, vs.YCOCG] else soothed


def QTGMC(clip, Preset='Slower', TR0=None, TR1=None, TR2=None, Rep0=None, Rep1=0, Rep2=None, EdiMode=None, RepChroma=True, NNSize=None, NNeurons=None, EdiQual=1, EdiMaxD=None, ChromaEdi='',
          EdiExt=None, Sharpness=None, SMode=None, SLMode=None, SLRad=None, SOvs=0, SVThin=0, Sbb=None, SrchClipPP=None, SubPel=None, SubPelInterp=2, BlockSize=None, Overlap=None, Search=None,
          SearchParam=None, PelSearch=None, ChromaMotion=None, TrueMotion=False, Lambda=None, LSAD=None, PNew=None, PLevel=None, GlobalMotion=True, DCT=0, ThSAD1=625, ThSAD2=256, ThSCD1=180, ThSCD2=100,
          SourceMatch=0, MatchPreset=None, MatchEdi=None, MatchPreset2=None, MatchEdi2=None, MatchTR2=1, MatchEnhance=0.5, Lossless=0, NoiseProcess=None, EZDenoise=None, EZKeepGrain=None,
          NoisePreset='Medium', Denoiser=None, FftThreads=1, DenoiseMC=None, NoiseTR=None, Sigma=None, ChromaNoise=False, ShowNoise=0, GrainRestore=None, NoiseRestore=None, NoiseDeint=None,
          StabilizeNoise=None, InputType=0, ProgSADMask=None, FPSDivisor=1, ShutterBlur=0, ShutterAngleSrc=180, ShutterAngleOut=180, SBlurLimit=4, Border=False, Precise=None, RefineMotion=False,
          ShowSettings=False, ForceTR=0, TFF=None, pscrn=None, int16_prescreener=None, int16_predictor=None, exp=None, alpha=None, beta=None, gamma=None, nrad=None, vcheck=None, opencl=False, device=None):
    """
    QTGMC by Vit
    A high quality deinterlacer using motion-compensated temporal smoothing, with a range of features for quality and convenience
    Originally based on TempGaussMC by Didée

    --- REQUIREMENTS ---

    Core plugins:
      MVTools / MVTools-sf
      Miscellaneous Filters
      znedi3 / nnedi3
      RGVS / RGSF
      fmtconv

    Additional plugins:
      eedi3m - if selected directly or via a source-match preset
      FFT3DFilter - if selected for noise processing
      DFTTest - if selected for noise processing
      KNLMeansCL - if selected for noise processing
      AddGrain - if NoiseDeint = "Generate" selected for noise bypass
      For FFT3DFilter & DFTTest you also need the FFTW3 library (FFTW.org). On Windows the file needed for both is libfftw3f-3.dll.

    --- GETTING STARTED ---

    The "Preset" used selects sensible settings for a given encoding speed. Choose a preset from:
        "Placebo", "Very Slow", "Slower", "Slow", "Medium", "Fast", "Faster", "Very Fast", "Super Fast", "Ultra Fast" & "Draft"

    Don't be obsessed with using slower settings as the differences can be small. HD material benefits little from extreme settings (and will be very slow)
    There are many settings for tweaking the script, full details in the main documentation. You can display settings currently being used with QTGMC(ShowSettings=True)
    """
    # Modified from havsfunc: https://github.com/HomeOfVapourSynthEvolution/havsfunc/blob/r31/havsfunc.py#L846
    
    if not isinstance(clip, vs.VideoNode) or clip.format.color_family not in [vs.GRAY, vs.YUV]:
        raise TypeError("QTGMC: This is not a GRAY or YUV clip!")

    # Presets
    # Select presets / tuning
    presets = ['placebo', 'very slow', 'slower', 'slow', 'medium', 'fast', 'faster', 'very fast', 'super fast', 'ultra fast', 'draft']
    try:
        pNum = presets.index(Preset.lower())
    except:
        raise ValueError("QTGMC: Preset choice is invalid")

    if MatchPreset is None:
        mpNum1 = min(pNum + 2, 9)
        MatchPreset = presets[mpNum1]
    else:
        try:
            mpNum1 = presets[:10].index(MatchPreset.lower())
        except:
            raise ValueError("QTGMC: MatchPreset choice is invalid")

    if MatchPreset2 is None:
        mpNum2 = min(mpNum1 + 2, 9)
        MatchPreset2 = presets[mpNum2]
    else:
        try:
            mpNum2 = presets[:10].index(MatchPreset2.lower())
        except:
            raise ValueError("QTGMC: MatchPreset2 choice is invalid")

    try:
        npNum = presets[2:7].index(NoisePreset.lower())
    except:
        raise ValueError("QTGMC: NoisePreset choice is invalid")

    bs  = 16 if BlockSize is None else BlockSize
    bs2 = 32 if BlockSize is None else BlockSize

    #                                                  Very                                                        Very      Super     Ultra
    # Preset groups:                        Placebo    Slow     Slower     Slow     Medium     Fast     Faster     Fast      Fast      Fast     Draft
    if TR0          is None: TR0          = [ 2,        2,        2,        2,        2,        2,        1,        1,        1,        1,        0     ][pNum]
    if TR1          is None: TR1          = [ 2,        2,        2,        1,        1,        1,        1,        1,        1,        1,        1     ][pNum]
    if TR2          is None: TR2          = [ 3,        2,        1,        1,        1,        0,        0,        0,        0,        0,        0     ][pNum]
    if Rep0         is None: Rep0         = [ 4,        4,        4,        4,        3,        3,        0,        0,        0,        0,        0     ][pNum]
    if Rep2         is None: Rep2         = [ 4,        4,        4,        4,        4,        4,        4,        4,        3,        3,        0     ][pNum]
    if EdiMode      is None: EdiMode      = ['nnedi3', 'nnedi3', 'nnedi3', 'nnedi3', 'nnedi3', 'nnedi3', 'nnedi3', 'nnedi3', 'nnedi3', 'nnedi3', 'bob'  ][pNum]
    if NNSize       is None: NNSize       = [ 1,        1,        1,        1,        5,        5,        4,        4,        4,        4,        4     ][pNum]
    if NNeurons     is None: NNeurons     = [ 3,        2,        1,        1,        1,        0,        0,        0,        0,        0,        0     ][pNum]
    if EdiMaxD      is None: EdiMaxD      = [ 12,       10,       8,        7,        7,        6,        6,        5,        4,        4,        4     ][pNum]
    if SMode        is None: SMode        = [ 2,        2,        2,        2,        2,        2,        2,        2,        2,        2,        0     ][pNum]
    if SLMode       is None: SLMod        = [ 2,        2,        2,        2,        2,        2,        2,        2,        0,        0,        0     ][pNum]
    if SLRad        is None: SLRad        = [ 3,        1,        1,        1,        1,        1,        1,        1,        1,        1,        1     ][pNum]
    if Sbb          is None: Sbb          = [ 3,        1,        1,        0,        0,        0,        0,        0,        0,        0,        0     ][pNum]
    if SrchClipPP   is None: SrchClipPP   = [ 3,        3,        3,        3,        3,        2,        2,        2,        1,        1,        0     ][pNum]
    if SubPel       is None: SubPel       = [ 2,        2,        2,        2,        1,        1,        1,        1,        1,        1,        1     ][pNum]
    if BlockSize    is None: BlockSize    = [ bs,       bs,       bs,       bs,       bs,       bs,       bs2,      bs2,      bs2,      bs2,      bs2   ][pNum]
    if Overlap      is None: Overlap      = [ bs//2,    bs//2,    bs//2,    bs//2,    bs//2,    bs//2,    bs2//2,   bs2//4,   bs2//4,   bs2//4,   bs2//4][pNum]
    if Search       is None: Search       = [ 3,        5,        5,        4,        4,        4,        4,        4,        0,        0,        0     ][pNum]
    if SearchParam  is None: SearchParam  = [ 3,        3,        2,        2,        2,        2,        2,        1,        1,        1,        1     ][pNum]
    if PelSearch    is None: PelSearch    = [ 2,        2,        2,        2,        1,        1,        1,        1,        1,        1,        1     ][pNum]
    if ChromaMotion is None: ChromaMotion = [ True,     True,     True,     False,    False,    False,    False,    False,    False,    False,    False ][pNum]
    if Precise      is None: Precise      = [ True,     True,     False,    False,    False,    False,    False,    False,    False,    False,    False ][pNum]
    if ProgSADMask  is None: ProgSADMask  = [ 10,       10,       10,       10,       10,       0,        0,        0,        0,        0,        0     ][pNum]

    # Noise presets                               Slower      Slow       Medium     Fast      Faster
    if Denoiser       is None: Denoiser       = ['dfttest',  'dfttest', 'dfttest', 'fft3d',  'fft3d'][npNum]
    if DenoiseMC      is None: DenoiseMC      = [ True,       True,      False,     False,    False ][npNum]
    if NoiseTR        is None: NoiseTR        = [ 2,          1,         1,         1,        0     ][npNum]
    if NoiseDeint     is None: NoiseDeint     = ['generate', 'bob',      '',        '',       ''    ][npNum]
    if StabilizeNoise is None: StabilizeNoise = [ True,       True,      True,      False,    False ][npNum]

    # The basic source-match step corrects and re-runs the interpolation of the input clip. So it initialy uses same interpolation settings as the main preset
    MatchEdi = EdiMode
    MatchNNSize = NNSize
    MatchNNeurons = NNeurons
    MatchEdiQual = EdiQual
    MatchEdiMaxD = EdiMaxD

    # However, can use a faster initial interpolation when using source-match allowing the basic source-match step to "correct" it with higher quality settings
    if SourceMatch > 0 and (mpNum1 < pNum or mpNum2 < pNum):
        raise ValueError("QTGMC: MatchPreset cannot use a slower setting than Preset")
    # Basic source-match presets
    # Main interpolation is actually done by basic-source match step when enabled, so a little swap and wriggle is needed
    if SourceMatch > 0:
        #                       Very                                                        Very     Super   Ultra
        #           Placebo     Slow     Slower     Slow     Medium     Fast     Faster     Fast     Fast    Fast
        EdiMode  = ['nnedi3', 'nnedi3', 'nnedi3', 'nnedi3', 'nnedi3', 'nnedi3', 'nnedi3', 'nnedi3', 'nnedi3', ''][mpNum1]
        NNSize   = [ 1,        1,        1,        1,        5,        5,        4,        4,        4,       4 ][mpNum1]
        NNeurons = [ 3,        2,        1,        1,        1,        0,        0,        0,        0,       0 ][mpNum1]
        EdiQual  = [ 1,        1,        1,        1,        1,        1,        1,        1,        1,       1 ][mpNum1]
        EdiMaxD  = [ 12,       10,       8,        7,        7,        6,        6,        5,        4,       4 ][mpNum1]
    
    # Refined source-match presets  Very                                                        Very     Super   Ultra
    #                   Placebo     Slow     Slower     Slow     Medium     Fast     Faster     Fast     Fast    Fast
    MatchEdi2        = ['nnedi3', 'nnedi3', 'nnedi3', 'nnedi3', 'nnedi3', 'nnedi3', 'nnedi3', 'nnedi3', 'nnedi3', ''][mpNum2]
    MatchNNSize2     = [ 1,        1,        1,        1,        5,        5,        4,        4,        4,       4 ][mpNum2]
    MatchNNeurons2   = [ 3,        2,        1,        1,        1,        0,        0,        0,        0,       0 ][mpNum2]
    MatchEdiQual2    = [ 1,        1,        1,        1,        1,        1,        1,        1,        1,       1 ][mpNum2]
    MatchEdiMaxD2    = [ 12,       10,       8,        7,        7,        6,        6,        5,        4,       4 ][mpNum2]

    EdiMode = EdiMode.lower()
    ChromaEdi = ChromaEdi.lower()
    MatchEdi = MatchEdi.lower()
    MatchEdi2 = MatchEdi2.lower()
    Denoiser = Denoiser.lower()
    NoiseDeint = NoiseDeint.lower()

    #---------------------------------------
    # Settings

    if EdiExt is not None and (not isinstance(EdiExt, vs.VideoNode) or EdiExt.format.id != clip.format.id):
        raise TypeError("QTGMC: EdiExt must be the same format as clip")
    if InputType != 1 and not isinstance(TFF, bool):
        raise TypeError("QTGMC: TFF must be set when InputType is not 1. Setting TFF to True means top field first and False means bottom field first")

    bd = clip.format.bits_per_sample
    isFLOAT = clip.format.sample_type == vs.FLOAT
    isGRAY = clip.format.color_family == vs.GRAY
    mid = 0 if isFLOAT else 1 << (bd - 1)
    i = 0.00392 if isFLOAT else 1 << (bd - 8)
    S = core.mvsf.Super if isFLOAT else core.mv.Super
    A = core.mvsf.Analyse if isFLOAT else core.mv.Analyse
    C = core.mvsf.Compensate if isFLOAT else core.mv.Compensate
    R = core.mvsf.Recalculate if isFLOAT else core.mv.Recalculate
    M = core.mvsf.Mask if isFLOAT else core.mv.Mask
    D1 = core.mvsf.Degrain1 if isFLOAT else core.mv.Degrain1
    D2 = core.mvsf.Degrain2 if isFLOAT else core.mv.Degrain2
    D3 = core.mvsf.Degrain3 if isFLOAT else core.mv.Degrain3
    FB = core.mvsf.FlowBlur if isFLOAT else core.mv.FlowBlur
    RE = core.rgsf.Repair if isFLOAT else core.rgvs.Repair
    V = core.rgsf.VerticalCleaner if isFLOAT else core.rgvs.VerticalCleaner
    mat = [1, 2, 1, 2, 4, 2, 1, 2, 1]
    hpad = BlockSize
    vpad = BlockSize

    # Core defaults
    if SourceMatch > 0:
        TR2 = max(TR2, 1) # TR2 defaults always at least 1 when using source-match

    # Source-match defaults
    MatchTR1 = TR1

    # Sharpness defaults. Sharpness default is always 1.0 (0.2 with source-match), but adjusted to give roughly same sharpness for all settings
    if SLMode is None:
        SLMode = SLMod
        if SourceMatch > 0:
            SLMode = 0 # Sharpness limiting disabled by default for source-match
    
    if SLRad <= 0:
        SLMode = 0

    if Sharpness is None:
        Sharpness = 0 if SMode <= 0 else 0.2 if SourceMatch > 0 else 1 # Default sharpness is 1.0, or 0.2 if using source-match

    if Sharpness <= 0:
        SMode = 0

    if SMode <= 0:
        Sbb = 0
    
    sharpMul = 2 if SLMode in [2, 4] else 1.5 if SLMode in [1, 3] else 1 # Adjust sharpness based on other settings
    sharpAdj = Sharpness * (sharpMul * (TR1 + TR2 + 1) / 4 + (0.1 if SMode == 1 else 0)) # [This needs a bit more refinement]

    # Noise processing settings
    if not (EZDenoise is None or EZKeepGrain is None) and EZDenoise > 0 and EZKeepGrain > 0:
        raise ValueError("QTGMC: EZDenoise and EZKeepGrain cannot be used together")
    if NoiseProcess is None:
        if EZDenoise is not None and EZDenoise > 0:
            NoiseProcess = 1
        elif (EZKeepGrain is not None and EZKeepGrain > 0) or Preset in ['placebo', 'very slow']:
            NoiseProcess = 2
        else:
            NoiseProcess = 0
    if GrainRestore is None:
        if EZDenoise is not None and EZDenoise > 0:
            GrainRestore = 0
        elif EZKeepGrain is not None and EZKeepGrain > 0:
            GrainRestore = 0.3 * math.sqrt(EZKeepGrain)
        else:
            GrainRestore = [0, 0.7, 0.3][NoiseProcess]
    if NoiseRestore is None:
        if EZDenoise is not None and EZDenoise > 0:
            NoiseRestore = 0
        elif EZKeepGrain is not None and EZKeepGrain > 0:
            NoiseRestore = 0.1 * math.sqrt(EZKeepGrain)
        else:
            NoiseRestore = [0, 0.3, 0.1][NoiseProcess]
    if Sigma is None:
        if EZDenoise is not None and EZDenoise > 0:
            Sigma = EZDenoise
        elif EZKeepGrain is not None and EZKeepGrain > 0:
            Sigma = 4 * EZKeepGrain
        else:
            Sigma = 2
    if isinstance(ShowNoise, bool):
        ShowNoise = 10 if ShowNoise else 0
    if ShowNoise > 0:
        NoiseProcess = 2
        NoiseRestore = 1
    if NoiseProcess <= 0:
        NoiseTR = 0
        GrainRestore = 0
        NoiseRestore = 0
    totalRestore = GrainRestore + NoiseRestore
    if totalRestore <= 0:
        StabilizeNoise = False
    noiseTD = [1, 3, 5][NoiseTR]
    noiseCentre = mid if Denoiser in ['dfttest', 'knlmeanscl'] else 128.5*i

    # MVTools settings
    if Lambda is None:
        Lambda = (1000 if TrueMotion else 100) * (BlockSize ** 2) // 64
    if LSAD is None:
        LSAD = 1200 if TrueMotion else 400
    if PNew is None:
        PNew = 50 if TrueMotion else 25
    if PLevel is None:
        PLevel = 1 if TrueMotion else 0

    # Motion blur settings
    if ShutterAngleOut * FPSDivisor == ShutterAngleSrc:
        ShutterBlur = 0 # If motion blur output is same as clip

    # Miscellaneous
    if InputType < 2:
        ProgSADMask = 0
    if isGRAY:
        ChromaMotion = False
        ChromaNoise = False
        RepChroma = False

    # Get maximum temporal radius needed
    maxTR = SLRad if SLMode in [2, 4] else 0
    maxTR = max(maxTR, MatchTR2, TR1, TR2, NoiseTR)
    maxTR = max(maxTR, ForceTR)
    if ProgSADMask > 0 or StabilizeNoise or ShutterBlur > 0:
        maxTR = max(maxTR, 1)

    # Show settings
    if ShowSettings:
        text = "TR0={} | TR1={} | TR2={} | Rep0={} | Rep1={} | Rep2={} | RepChroma={} | EdiMode='{}' | NNSize={} | NNeurons={} | EdiQual={} | EdiMaxD={} | " + \
               "ChromaEdi='{}' | Sharpness={} | SMode={} | SLMode={} | SLRad={} | SOvs={} | SVThin={} | Sbb={} | SrchClipPP={} | SubPel={} | " + \
               "SubPelInterp={} | BlockSize={} | Overlap={} | Search={} | SearchParam={} | PelSearch={} | ChromaMotion={} | TrueMotion={} | Lambda={} | " + \
               "LSAD={} | PNew={} | PLevel={} | GlobalMotion={} | DCT={} | ThSAD1={} | ThSAD2={} | ThSCD1={} | ThSCD2={} | SourceMatch={} | " + \
               "MatchPreset='{}' | MatchEdi='{}' | MatchPreset2='{}' | MatchEdi2='{}' | MatchTR2={} | MatchEnhance={} | Lossless={} | NoiseProcess={} | " + \
               "Denoiser='{}' | FftThreads={} | DenoiseMC={} | NoiseTR={} | Sigma={} | ChromaNoise={} | ShowNoise={} | GrainRestore={} | NoiseRestore={} | " + \
               "NoiseDeint='{}' | StabilizeNoise={} | InputType={} | ProgSADMask={} | FPSDivisor={} | ShutterBlur={} | ShutterAngleSrc={} | " + \
               "ShutterAngleOut={} | SBlurLimit={} | Border={} | Precise={} | RefineMotion={} | Preset='{}' | ForceTR={}"
        text = text.format(TR0, TR1, TR2, Rep0, Rep1, Rep2, RepChroma, EdiMode, NNSize, NNeurons, EdiQual, EdiMaxD, ChromaEdi, Sharpness, SMode,
                           SLMode, SLRad, SOvs, SVThin, Sbb, SrchClipPP, SubPel, SubPelInterp, BlockSize, Overlap, Search, SearchParam, PelSearch,
                           ChromaMotion, TrueMotion, Lambda, LSAD, PNew, PLevel, GlobalMotion, DCT, ThSAD1, ThSAD2, ThSCD1, ThSCD2, SourceMatch,
                           MatchPreset, MatchEdi, MatchPreset2, MatchEdi2, MatchTR2, MatchEnhance, Lossless, NoiseProcess, Denoiser, FftThreads,
                           DenoiseMC, NoiseTR, Sigma, ChromaNoise, ShowNoise, GrainRestore, NoiseRestore, NoiseDeint, StabilizeNoise, InputType,
                           ProgSADMask, FPSDivisor, ShutterBlur, ShutterAngleSrc, ShutterAngleOut, SBlurLimit, Border, Precise, RefineMotion, Preset, ForceTR)
        return core.text.Text(clip, text)
    
    #---------------------------------------
    # Pre-Processing

    w = clip.width
    h = clip.height

    # Reverse "field" dominance for progressive repair mode 3 (only difference from mode 2)
    if InputType >= 3:
        TFF = not TFF

    # Pad vertically during processing (to prevent artifacts at top & bottom edges)
    if Border:
        h += 8
        clip = core.resize.Point(clip, w, h, src_top=-4, src_height=h)

    #---------------------------------------
    # Motion Analysis

    # Bob the input as a starting point for motion search clip
    if InputType <= 0:
        bobbed = haf.Bob(clip, 0, 0.5, TFF)
    elif InputType == 1:
        bobbed = clip
    else:
        bobbed = core.std.Convolution(clip, matrix=[1, 2, 1], mode='v')

    CMplanes = [0, 1, 2] if ChromaMotion else [0]

    # The bobbed clip will shimmer due to being derived from alternating fields. Temporally smooth over the neighboring frames using a binomial kernel.
    # Binomial kernels give equal weight to even and odd frames and hence average away the shimmer. The two kernels used are [1 2 1] and [1 4 6 4 1] for radius 1 and 2.
    # These kernels are approximately Gaussian kernels, which work well as a prefilter before motion analysis (hence the original name for this script)
    # Create linear weightings of neighbors first                                                     -2    -1    0     1     2
    if TR0 > 0: ts1 = haf.AverageFrames(bobbed, weights=[1]*3, scenechange=0.11, planes=CMplanes) # 0.00  0.33  0.33  0.33  0.00
    if TR0 > 1: ts2 = haf.AverageFrames(bobbed, weights=[1]*5, scenechange=0.11, planes=CMplanes) # 0.20  0.20  0.20  0.20  0.20

    # Combine linear weightings to give binomial weightings - TR0 = 0: (1), TR0 = 1: (1:2:1), TR0 = 2: (1:4:6:4:1)
    if TR0 <= 0:
        binomial0 = bobbed
    elif TR0 == 1:
        binomial0 = core.std.Merge(ts1, bobbed, [0.25] if ChromaMotion or isGRAY else [0.25, 0])
    else:
        binomial0 = core.std.Merge(core.std.Merge(ts1, ts2, [0.357] if ChromaMotion or isGRAY else [0.357, 0]), bobbed, [0.125] if ChromaMotion or isGRAY else [0.125, 0])

    # Remove areas of difference between temporal blurred motion search clip and bob that are not due to bob-shimmer - removes general motion blur
    repair0 = binomial0 if Rep0 <= 0 else QTGMC_KeepOnlyBobShimmerFixes(binomial0, bobbed, Rep0, RepChroma)

    # Blur image and soften edges to assist in motion matching of edge blocks. Blocks are matched by SAD (sum of absolute differences between blocks),
    # but even a slight change in an edge from frame to frame will give a high SAD due to the higher contrast of edges
    if SrchClipPP == 1:
        spatialBlur = core.resize.Bilinear(repair0, m4(w/2), m4(h/2)).std.Convolution(matrix=mat, planes=CMplanes).resize.Bilinear(w, h)
    elif SrchClipPP > 1:
        spatialBlur = core.tcanny.TCanny(repair0, sigma=2, mode=-1, planes=CMplanes)
        spatialBlur = core.std.Merge(spatialBlur, repair0, [0.1] if ChromaMotion or isGRAY else [0.1, 0])
    if SrchClipPP <= 0:
        srchClip = repair0
    elif SrchClipPP < 3:
        srchClip = spatialBlur
    else:
        expr1 = 'x {a} + y < x {a} + x {a} - y > x {a} - y ? ?'.format(a=3*i)
        tweaked = core.std.Expr([repair0, bobbed], [expr1] if ChromaMotion or isGRAY else [expr1, ''])
        expr2 = 'x {a} + y < x {b} + x {a} - y > x {b} - x y + 2 / ? ?'.format(a=7*i, b=2*i)
        srchClip = core.std.Expr([spatialBlur, tweaked], [expr2] if ChromaMotion or isGRAY else [expr2, ''])

    # Calculate forward and backward motion vectors from motion search clip
    analyse_args = dict(blksize=BlockSize, overlap=Overlap, search=Search, searchparam=SearchParam, pelsearch=PelSearch, truemotion=TrueMotion, _lambda=Lambda, lsad=LSAD, pnew=PNew, plevel=PLevel, _global=GlobalMotion, dct=DCT, chroma=ChromaMotion)
    srchSuper = S(DitherLumaRebuild(srchClip, s0=1, chroma=ChromaMotion), pel=SubPel, sharp=1, rfilter=4, hpad=hpad, vpad=vpad, chroma=ChromaMotion) if maxTR > 0 else None
    bVec1 = A(srchSuper, isb=True,  delta=1, **analyse_args) if maxTR > 0 else None
    fVec1 = A(srchSuper, isb=False, delta=1, **analyse_args) if maxTR > 0 else None
    bVec2 = A(srchSuper, isb=True,  delta=2, **analyse_args) if maxTR > 1 else None
    fVec2 = A(srchSuper, isb=False, delta=2, **analyse_args) if maxTR > 1 else None
    bVec3 = A(srchSuper, isb=True,  delta=3, **analyse_args) if maxTR > 2 else None
    fVec3 = A(srchSuper, isb=False, delta=3, **analyse_args) if maxTR > 2 else None

    if RefineMotion and maxTR > 0:
        recalculate_args = dict(blksize=BlockSize/2, overlap=Overlap/2, search=Search, searchparam=SearchParam, truemotion=TrueMotion, _lambda=Lambda/4, pnew=PNew, dct=DCT, chroma=ChromaMotion)
        bVec1 = R(srchSuper, bVec1, **recalculate_args)
        fVec1 = R(srchSuper, fVec1, **recalculate_args)
        bVec2 = R(srchSuper, bVec2, **recalculate_args) if maxTR > 1 else bVec2
        fVec2 = R(srchSuper, fVec2, **recalculate_args) if maxTR > 1 else fVec2
        bVec3 = R(srchSuper, bVec3, **recalculate_args) if maxTR > 2 else bVec3
        fVec3 = R(srchSuper, fVec3, **recalculate_args) if maxTR > 2 else fVec3

    #---------------------------------------
    # Noise Processing

    # Expand fields to full frame size before extracting noise (allows use of motion vectors which are frame-sized)
    if NoiseProcess > 0:
        fullClip = clip if InputType > 0 else haf.Bob(clip, 0, 1, TFF)
    if NoiseTR > 0:
        fullSuper = S(fullClip, pel=SubPel, sharp=SubPelInterp, levels=1, rfilter=1, hpad=hpad, vpad=vpad, chroma=ChromaNoise)

    CNplanes = [0, 1, 2] if ChromaNoise else [0]

    # Create a motion compensated temporal window around current frame and use to guide denoisers
    if NoiseProcess > 0:
        if not DenoiseMC or NoiseTR <= 0:
            noiseWindow = fullClip
        elif NoiseTR == 1:
            noiseWindow = core.std.Interleave([C(fullClip, fullSuper, fVec1, thscd1=ThSCD1, thscd2=ThSCD2), fullClip,
                                               C(fullClip, fullSuper, bVec1, thscd1=ThSCD1, thscd2=ThSCD2)])
        else:
            noiseWindow = core.std.Interleave([C(fullClip, fullSuper, fVec2, thscd1=ThSCD1, thscd2=ThSCD2),
                                               C(fullClip, fullSuper, fVec1, thscd1=ThSCD1, thscd2=ThSCD2), fullClip,
                                               C(fullClip, fullSuper, bVec1, thscd1=ThSCD1, thscd2=ThSCD2),
                                               C(fullClip, fullSuper, bVec2, thscd1=ThSCD1, thscd2=ThSCD2)])
        if Denoiser == 'dfttest':
            dnWindow = core.dfttest.DFTTest(noiseWindow, sigma=Sigma * 4, tbsize=noiseTD, planes=CNplanes)
        elif Denoiser == 'knlmeanscl':
            dnWindow = haf.KNLMeansCL(noiseWindow, d=NoiseTR, h=Sigma) if ChromaNoise else noiseWindow.knlm.KNLMeansCL(d=NoiseTR, h=Sigma)
        else:
            dnWindow = core.fft3dfilter.FFT3DFilter(noiseWindow, sigma=Sigma * i, planes=CNplanes, bt=noiseTD, ncpu=FftThreads)

        # Rework denoised clip to match source format - various code paths here: discard the motion compensation window, discard doubled lines (from PointResize)
        # Also reweave to get interlaced noise if source was interlaced (could keep the full frame of noise, but it will be poor quality from PointResize)
        if not DenoiseMC:
            denoised = dnWindow if InputType > 0 else haf.Weave(core.std.SeparateFields(dnWindow, TFF).std.SelectEvery(4, [0, 3]), TFF)
        elif InputType > 0:
            denoised = dnWindow if NoiseTR <= 0 else dnWindow[NoiseTR::noiseTD]
        else:
            denoised = haf.Weave(core.std.SeparateFields(dnWindow, TFF).std.SelectEvery(noiseTD * 4, [NoiseTR * 2, NoiseTR * 6 + 3]), TFF)

    # Get actual noise from difference. Then "deinterlace" where we have weaved noise - create the missing lines of noise in various ways
    if NoiseProcess > 0 and totalRestore > 0:
        noise = core.std.MakeDiff(clip, denoised, planes=CNplanes)
        if InputType > 0:
            deintNoise = noise
        elif NoiseDeint == 'bob':
            deintNoise = haf.Bob(noise, 0, 0.5, TFF)
        elif NoiseDeint == 'generate':
            deintNoise = QTGMC_Generate2ndFieldNoise(noise, denoised, ChromaNoise, TFF)
        else:
            deintNoise = core.std.SeparateFields(noise, TFF).std.DoubleWeave(TFF)

        # Motion-compensated stabilization of generated noise
        if StabilizeNoise:
            noiseSuper = S(deintNoise, pel=SubPel, sharp=SubPelInterp, levels=1, rfilter=1, hpad=hpad, vpad=vpad, chroma=ChromaNoise)
            mcNoise = C(deintNoise, noiseSuper, bVec1, thscd1=ThSCD1, thscd2=ThSCD2)
            expr3 = 'x abs y abs > x y ? 0.6 * x y + 0.2 * +' if isFLOAT else 'x {} - abs y {} - abs > x y ? 0.6 * x y + 0.2 * +'.format(mid, mid)
            finalNoise = core.std.Expr([deintNoise, mcNoise], [expr3] if ChromaNoise or isGRAY else [expr3, ''])
        else:
            finalNoise = deintNoise
        
        if ShowNoise > 0:
            expr4 = 'x {} *'.format(ShowNoise) if isFLOAT else 'x {} - {} * {} +'.format(mid, ShowNoise, mid)
            return core.std.Expr([finalNoise], [expr4] if ChromaNoise or isGRAY else [expr4, repr(mid)])

    # If NoiseProcess == 1 denoise input clip. If NoiseProcess == 2 leave noise in the clip and let the temporal blurs "denoise" it for a stronger effect
    innerClip = denoised if NoiseProcess == 1 else clip

    #---------------------------------------
    # Interpolation

    # Support badly deinterlaced progressive content - drop half the fields and reweave to get 1/2fps interlaced stream appropriate for QTGMC processing
    ediInput = haf.Weave(core.std.SeparateFields(innerClip, TFF).std.SelectEvery(4, [0, 3]), TFF) if InputType > 1 else innerClip

    # Create interpolated image as starting point for output
    if EdiExt is not None:
        edi1 = core.resize.Point(EdiExt, w, h, src_top=(EdiExt.height - h) / 2, src_height=h)
    else:
        edi1 = QTGMC_Interpolate(ediInput, InputType, EdiMode, NNSize, NNeurons, EdiQual, EdiMaxD, pscrn, int16_prescreener, int16_predictor, exp, alpha, beta, gamma, nrad, vcheck, bobbed, ChromaEdi, TFF, opencl, device)

    # InputType == 2 or 3: use motion mask to blend luma between original clip & reweaved clip based on ProgSADMask setting. Use chroma from original clip in any case
    if InputType < 2:
        edi = edi1
    elif ProgSADMask <= 0:
        edi = edi1 if isGRAY else core.std.ShufflePlanes([edi1, innerClip], [0, 1, 2], clip.format.color_family)
    else:
        inputTypeBlend = M(srchClip.fmtc.bitdepth(bits=8), bVec1, kind=1, ml=ProgSADMask).fmtc.bitdepth(bits=bd) if 8 < bd <= 16 else M(srchClip, bVec1, kind=1, ml=ProgSADMask)
        edi = core.std.MaskedMerge(innerClip, edi1, inputTypeBlend, [0])

    # Get the min/max value for each pixel over neighboring motion-compensated frames - used for temporal sharpness limiting
    if TR1 > 0 or SLMode in [2, 4]:
        ediSuper = S(edi, pel=SubPel, sharp=SubPelInterp, levels=1, rfilter=1, hpad=hpad, vpad=vpad)
    if SLMode in [2, 4]:
        bComp1 = C(edi, ediSuper, bVec1, thscd1=ThSCD1, thscd2=ThSCD2)
        fComp1 = C(edi, ediSuper, fVec1, thscd1=ThSCD1, thscd2=ThSCD2)
        tMax = core.std.Expr([core.std.Expr([edi, fComp1], ['x y max']), bComp1], ['x y max'])
        tMin = core.std.Expr([core.std.Expr([edi, fComp1], ['x y min']), bComp1], ['x y min'])
        if SLRad > 1:
            bComp3 = C(edi, ediSuper, bVec3, thscd1=ThSCD1, thscd2=ThSCD2)
            fComp3 = C(edi, ediSuper, fVec3, thscd1=ThSCD1, thscd2=ThSCD2)
            tMax = core.std.Expr([core.std.Expr([tMax, fComp3], ['x y max']), bComp3], ['x y max'])
            tMin = core.std.Expr([core.std.Expr([tMin, fComp3], ['x y min']), bComp3], ['x y min'])

    #---------------------------------------
    # Create basic output

    # Use motion vectors to blur interpolated image (edi) with motion-compensated previous and next frames. As above, this is done to remove shimmer
    # from alternate frames so the same binomial kernels are used. However, by using motion-compensated smoothing this time we avoid motion blur. 
    # The use of MDegrain1 (motion compensated) rather than TemporalSmooth makes the weightings *look* different, but they evaluate to the same values
    # Create linear weightings of neighbors first                                                         -2    -1    0     1     2
    if TR1 > 0: degrain1 = D1(edi, ediSuper, bVec1, fVec1, thsad=ThSAD1, thscd1=ThSCD1, thscd2=ThSCD2) # 0.00  0.33  0.33  0.33  0.00
    if TR1 > 1: degrain2 = D1(edi, ediSuper, bVec2, fVec2, thsad=ThSAD1, thscd1=ThSCD1, thscd2=ThSCD2) # 0.33  0.00  0.33  0.00  0.33

    # Combine linear weightings to give binomial weightings - TR1 == 0: (1), TR1 == 1: (1:2:1), TR1 == 2: (1:4:6:4:1)
    if TR1 <= 0:
        binomial1 = edi
    elif TR1 == 1:
        binomial1 = core.std.Merge(degrain1, edi, [0.25])
    else:
        binomial1 = core.std.Merge(core.std.Merge(degrain1, degrain2, [0.2]), edi, [0.0625])

    # Remove areas of difference between smoothed image and interpolated image that are not bob-shimmer fixes: repairs residual motion blur from temporal smooth
    repair1 = binomial1 if Rep1 <= 0 else QTGMC_KeepOnlyBobShimmerFixes(binomial1, edi, Rep1, RepChroma)

    # Apply source match - use difference between output and source to succesively refine output [extracted to function to clarify main code path]
    if SourceMatch <= 0:
        match = repair1
    else:
        match = QTGMC_ApplySourceMatch(repair1, InputType, ediInput, bVec1, fVec1, bVec2, fVec2, SubPel, SubPelInterp, hpad, vpad, ThSAD1, ThSCD1, ThSCD2, SourceMatch, MatchTR1,
                                       MatchEdi, MatchNNSize, MatchNNeurons, MatchEdiQual, MatchEdiMaxD, MatchTR2, MatchEdi2, MatchNNSize2, MatchNNeurons2, MatchEdiQual2,
                                       MatchEdiMaxD2, MatchEnhance, pscrn, int16_prescreener, int16_predictor, exp, alpha, beta, gamma, nrad, vcheck, TFF, opencl, device)

    # Lossless == 2 - after preparing an interpolated, de-shimmered clip, restore the original source fields into it and clean up any artifacts
    # This mode will not give a true lossless result because the resharpening and final temporal smooth are still to come, but it will add further detail
    # However, it can introduce minor combing. This setting is best used together with source-match (it's effectively the final source-match stage)
    lossed1 = QTGMC_MakeLossless(match, innerClip, InputType, TFF) if Lossless >= 2 else match

    #---------------------------------------
    # Resharpen / retouch output

    # Resharpen to counteract temporal blurs. Little sharpening needed for source-match mode since it has already recovered sharpness from source
    if SMode <= 0:
        resharp = lossed1
    elif SMode == 1:
        resharp = core.std.Expr([lossed1, MinBlur(lossed1)], ['x dup y - {} * +'.format(sharpAdj)])
    else:
        vresharp = core.std.Merge(core.std.Maximum(lossed1, coordinates=[0, 1, 0, 0, 0, 0, 1, 0]), core.std.Minimum(lossed1, coordinates=[0, 1, 0, 0, 0, 0, 1, 0]))
        if Precise:
            vresharp = core.std.Expr([vresharp, lossed1], ['x y < x {} + x y > x {} - x ? ?'.format(i, i)]) # Precise mode: reduce tiny overshoot
        resharp = core.std.Expr([lossed1, MinBlur(vresharp)], ['x dup y - {} * +'.format(sharpAdj)])

    # Slightly thin down 1-pixel high horizontal edges that have been widened into neigboring field lines by the interpolator
    SVThinSc = SVThin * 6
    if SVThin > 0:
        expr5 = 'y x - {} *'.format(SVThinSc) if isFLOAT else 'y x - {} * {} +'.format(SVThinSc, mid)
        vertMedD = core.std.Expr([lossed1, V(lossed1, [1] if isGRAY else [1, 0])], [expr5] if isGRAY else [expr5, ''])
        vertMedD = core.std.Convolution(vertMedD, matrix=[1, 2, 1], planes=[0], mode='h')
        expr6 = 'y abs x abs > y 0 ?' if isFLOAT else 'y {m} - abs x {m} - abs > y {m} ?'.format(m=mid)
        neighborD = core.std.Expr([vertMedD, core.std.Convolution(vertMedD, matrix=mat, planes=[0])], [expr6] if isGRAY else [expr6, ''])
        thin = core.std.MergeDiff(resharp, neighborD, [0])
    else:
        thin = resharp

    # Back blend the blurred difference between sharpened & unsharpened clip, before (1st) sharpness limiting (Sbb == 1 or 3). A small fidelity improvement
    if Sbb not in [1, 3]:
        backBlend1 = thin
    else:
        backBlend1 = core.std.MakeDiff(thin, core.std.MakeDiff(thin, lossed1, [0]).tcanny.TCanny(sigma=1.4, mode=-1, planes=[0]), [0])

    # Limit over-sharpening by clamping to neighboring (spatial or temporal) min/max values in original
    # Occurs here (before final temporal smooth) if SLMode == 1 or 2. This location will restrict sharpness more, but any artifacts introduced will be smoothed
    if SLMode == 1:
        if SLRad <= 1:
            sharpLimit1 = RE(backBlend1, edi, 1)
        else:
            sharpLimit1 = RE(backBlend1, RE(backBlend1, edi, 12), 1)
    elif SLMode == 2:
        sharpLimit1 = haf.Clamp(backBlend1, tMax, tMin, SOvs*i, SOvs*i)
    else:
        sharpLimit1 = backBlend1

    # Back blend the blurred difference between sharpened & unsharpened clip, after (1st) sharpness limiting (Sbb == 2 or 3). A small fidelity improvement
    if Sbb < 2:
        backBlend2 = sharpLimit1
    else:
        backBlend2 = core.std.MakeDiff(sharpLimit1, core.std.MakeDiff(sharpLimit1, lossed1, [0]).tcanny.TCanny(sigma=1.4, mode=-1, planes=[0]), [0])

    # Add back any extracted noise, prior to final temporal smooth - this will restore detail that was removed as "noise" without restoring the noise itself
    # Average luma of FFT3DFilter extracted noise is 128.5, so deal with that too
    if GrainRestore <= 0:
        addNoise1 = backBlend2
    else:
        expr7 = 'x {} *'.format(GrainRestore) if isFLOAT else 'x {} - {} * {} +'.format(noiseCentre, GrainRestore, mid)
        addNoise1 = core.std.MergeDiff(backBlend2, core.std.Expr([finalNoise], [expr7] if ChromaNoise or isGRAY else [expr7, '']), planes=CNplanes)

    # Final light linear temporal smooth for denoising
    if TR2 > 0:
        stableSuper = S(addNoise1, pel=SubPel, sharp=SubPelInterp, levels=1, rfilter=1, hpad=hpad, vpad=vpad)
    if TR2 <= 0:
        stable = addNoise1
    elif TR2 == 1:
        stable = D1(addNoise1, stableSuper, bVec1, fVec1, thsad=ThSAD2, thscd1=ThSCD1, thscd2=ThSCD2)
    elif TR2 == 2:
        stable = D2(addNoise1, stableSuper, bVec1, fVec1, bVec2, fVec2, thsad=ThSAD2, thscd1=ThSCD1, thscd2=ThSCD2)
    else:
        stable = D3(addNoise1, stableSuper, bVec1, fVec1, bVec2, fVec2, bVec3, fVec3, thsad=ThSAD2, thscd1=ThSCD1, thscd2=ThSCD2)

    # Remove areas of difference between final output & basic interpolated image that are not bob-shimmer fixes: repairs motion blur caused by temporal smooth
    repair2 = stable if Rep2 <= 0 else QTGMC_KeepOnlyBobShimmerFixes(stable, edi, Rep2, RepChroma)

    # Limit over-sharpening by clamping to neighboring (spatial or temporal) min/max values in original
    # Occurs here (after final temporal smooth) if SLMode == 3 or 4. Allows more sharpening here, but more prone to introducing minor artifacts
    if SLMode == 3:
        if SLRad <= 1:
            sharpLimit2 = RE(repair2, edi, 1)
        else:
            sharpLimit2 = RE(repair2, RE(repair2, edi, 12), 1)
    elif SLMode >= 4:
        sharpLimit2 = haf.Clamp(repair2, tMax, tMin, SOvs*i, SOvs*i)
    else:
        sharpLimit2 = repair2

    # Lossless == 1 - inject source fields into result and clean up inevitable artifacts. Provided NoiseRestore == 0.0 or 1.0, this mode will make the script
    # result properly lossless, but this will retain source artifacts and cause some combing (where the smoothed deinterlace doesn't quite match the source)
    lossed2 = QTGMC_MakeLossless(sharpLimit2, innerClip, InputType, TFF) if Lossless == 1 else sharpLimit2

    # Add back any extracted noise, after final temporal smooth. This will appear as noise/grain in the output
    # Average luma of FFT3DFilter extracted noise is 128.5, so deal with that too
    if NoiseRestore <= 0:
        addNoise2 = lossed2
    else:
        expr8 = 'x {} *'.format(NoiseRestore) if isFLOAT else 'x {} - {} * {} +'.format(noiseCentre, NoiseRestore, mid)
        addNoise2 = core.std.MergeDiff(lossed2, core.std.Expr([finalNoise], [expr8] if ChromaNoise or isGRAY else [expr8, '']), planes=CNplanes)

    #---------------------------------------
    # Post-Processing

    # Shutter motion blur - get level of blur depending on output framerate and blur already in source
    blurLevel = (ShutterAngleOut * FPSDivisor - ShutterAngleSrc) / 3.6
    if blurLevel < 0:
        raise ValueError('QTGMC: Cannot reduce motion blur already in source: increase ShutterAngleOut or FPSDivisor')
    elif blurLevel > 200:
        raise ValueError('QTGMC: Exceeded maximum motion blur level: decrease ShutterAngleOut or FPSDivisor')

    # Shutter motion blur - get finer resolution motion vectors to reduce blur "bleeding" into static areas
    rBlockDivide = [1, 1, 2, 4][ShutterBlur]
    rBlockSize = max(BlockSize // rBlockDivide, 4)
    rOverlap = max(Overlap // rBlockDivide, 2)
    rBlockDivide = BlockSize // rBlockSize
    rLambda = Lambda // (rBlockDivide ** 2)
    if ShutterBlur > 1:
        recalculate_args = dict(blksize=rBlockSize, overlap=rOverlap, search=Search, searchparam=SearchParam, truemotion=TrueMotion, _lambda=rLambda, pnew=PNew, dct=DCT, chroma=ChromaMotion)
        sbBVec1 = R(srchSuper, bVec1, **recalculate_args)
        sbFVec1 = R(srchSuper, fVec1, **recalculate_args)
    elif ShutterBlur > 0:
        sbBVec1 = bVec1
        sbFVec1 = fVec1

    # Shutter motion blur - use MFlowBlur to blur along motion vectors
    if ShutterBlur > 0:
        sblurSuper = S(addNoise2, pel=SubPel, sharp=SubPelInterp, levels=1, rfilter=1, hpad=hpad, vpad=vpad)
        sblur = FB(addNoise2, sblurSuper, sbBVec1, sbFVec1, blur=blurLevel, thscd1=ThSCD1, thscd2=ThSCD2)

    # Shutter motion blur - use motion mask to reduce blurring in areas of low motion - also helps reduce blur "bleeding" into static areas, then select blur type
    if ShutterBlur <= 0:
        sblurred = addNoise2
    elif SBlurLimit <= 0:
        sblurred = sblur
    else:
        sbMotionMask = M(srchClip.fmtc.bitdepth(bits=8), bVec1, kind=0, ml=SBlurLimit).fmtc.bitdepth(bits=bd) if 8 < bd <= 16 else M(srchClip, bVec1, kind=0, ml=SBlurLimit)
        sblurred = core.std.MaskedMerge(addNoise2, sblur, sbMotionMask)

    # Reduce frame rate
    decimated = sblurred[::FPSDivisor] if FPSDivisor > 1 else sblurred

    # Crop off temporary vertical padding
    # Show output of choice
    return decimated.std.Crop(top=4, bottom=4) if Border else decimated

#---------------------------------------

# Helper function: Interpolate input clip using method given in EdiMode. Use Fallback or Bob as result if mode not in list. If ChromaEdi is set then interpolate
# chroma separately with that method (only really useful for EEDIx). The function is used as main algorithm starting point and for first two source-match stages.
def QTGMC_Interpolate(clip, InputType, EdiMode, NNSize, NNeurons, EdiQual, EdiMaxD, pscrn, int16_prescreener, int16_predictor, exp, alpha, beta, gamma, nrad, vcheck, Fallback, ChromaEdi, TFF, opencl, device):
    if opencl:
        NNEDI3 = core.nnedi3cl.NNEDI3CL
        EEDI3 = core.eedi3m.EEDI3CL
        nnedi3_args = dict(nsize=NNSize, nns=NNeurons, qual=EdiQual, pscrn=pscrn, device=device)
        eedi3_args = dict(alpha=alpha, beta=beta, gamma=gamma, nrad=nrad, mdis=EdiMaxD, vcheck=vcheck, device=device)
    else:
        NNEDI3 = core.znedi3.nnedi3 if hasattr(core, 'znedi3') and clip.format.sample_type != vs.FLOAT else core.nnedi3.nnedi3
        EEDI3 = core.eedi3m.EEDI3 if hasattr(core, 'eedi3m') else core.eedi3.eedi3
        nnedi3_args = dict(nsize=NNSize, nns=NNeurons, qual=EdiQual, pscrn=pscrn, int16_prescreener=int16_prescreener, int16_predictor=int16_predictor, exp=exp)
        eedi3_args = dict(alpha=alpha, beta=beta, gamma=gamma, nrad=nrad, mdis=EdiMaxD, vcheck=vcheck)

    isGRAY = clip.format.color_family == vs.GRAY
    ChromaEdi = '' if isGRAY else ChromaEdi
    planes = [0] if ChromaEdi or isGRAY else [0, 1, 2]
    field = 3 if TFF else 2

    if InputType == 1:
        return clip
    elif EdiMode == 'nnedi3':
        interp = NNEDI3(clip, field=field, planes=planes, **nnedi3_args)
    elif EdiMode == 'eedi3+nnedi3':
        interp =  EEDI3(clip, field=field, planes=planes, **eedi3_args, sclip=NNEDI3(clip, field=field, planes=planes, **nnedi3_args))
    elif EdiMode == 'eedi3':
        interp =  EEDI3(clip, field=field, planes=planes, **eedi3_args)
    else:
        interp = Fallback if isinstance(Fallback, vs.VideoNode) else haf.Bob(clip, 0, 0.5, TFF)

    if ChromaEdi == 'nnedi3':
        interpuv = NNEDI3(clip, field=field, planes=[1, 2], **nnedi3_args)
    elif ChromaEdi == 'eedi3':
        interpuv =  EEDI3(clip, field=field, planes=[1, 2], **eedi3_args)
    elif ChromaEdi == 'bob':
        interpuv = haf.Bob(clip, 0, 0.5, TFF)
    else:
        return interp

    return core.std.ShufflePlanes([interp, interpuv], [0, 1, 2], clip.format.color_family)


# Helper function: Compare processed clip with reference clip: only allow thin, horizontal areas of difference, i.e. bob shimmer fixes
# Rough algorithm: Get difference, deflate vertically by a couple of pixels or so, then inflate again. Thin regions will be removed by this process. 
# Restore remaining areas of difference back to as they were in reference clip.
def QTGMC_KeepOnlyBobShimmerFixes(clip, Ref, Rep=1, Chroma=True):
    bd = clip.format.bits_per_sample
    isFLOAT = clip.format.sample_type == vs.FLOAT
    isGRAY = clip.format.color_family == vs.GRAY
    mid = 0 if isFLOAT else 1 << (bd - 1)
    i = 0.00392 if isFLOAT else 1 << (bd - 8)
    planes = [0, 1, 2] if Chroma and not isGRAY else [0]
    diff = core.std.MakeDiff(Ref, clip)

    # ed is the erosion distance - how much to deflate then reflate to remove thin areas of interest: 0 = minimum to 7 = maximum
    # od is over-dilation level  - extra inflation to ensure areas to restore back are fully caught:  0 = none to 3 = one full pixel
    # If Rep < 10, then ed = Rep and od = 0, otherwise ed = 10s digit and od = 1s digit (nasty method, but kept for compatibility with original TGMC)
    ed = Rep if Rep < 10 else Rep // 10
    od = 0 if Rep < 10 else Rep % 10

    # Areas of positive difference                                                              # ed = 0 1 2 3 4 5 6 7
    choke1 = core.std.Minimum(diff, planes, coordinates=[0, 1, 0, 0, 0, 0, 1, 0])               #      x x x x x x x x    1 pixel  \
    if ed > 2: choke1 = core.std.Minimum(choke1, planes, coordinates=[0, 1, 0, 0, 0, 0, 1, 0])  #      . . . x x x x x    1 pixel   | Deflate to remove thin areas
    if ed > 5: choke1 = core.std.Minimum(choke1, planes, coordinates=[0, 1, 0, 0, 0, 0, 1, 0])  #      . . . . . . x x    1 pixel  /
    if ed % 3 != 0: choke1 = core.std.Deflate(choke1, planes)                                   #      . x x . x x . x    A bit more deflate & some horizonal effect
    if ed in [2, 5]: choke1 = core.std.Median(choke1, planes)                                   #      . . x . . x . .    Local median
    choke1 = core.std.Maximum(choke1, planes, coordinates=[0, 1, 0, 0, 0, 0, 1, 0])             #      x x x x x x x x    1 pixel  \
    if ed > 1: choke1 = core.std.Maximum(choke1, planes, coordinates=[0, 1, 0, 0, 0, 0, 1, 0])  #      . . x x x x x x    1 pixel   | Reflate again
    if ed > 4: choke1 = core.std.Maximum(choke1, planes, coordinates=[0, 1, 0, 0, 0, 0, 1, 0])  #      . . . . . x x x    1 pixel  /

    # Over-dilation - extra reflation up to about 1 pixel
    if od == 1:
        choke1 = core.std.Inflate(choke1, planes)
    elif od == 2:
        choke1 = core.std.Inflate(choke1, planes).std.Inflate(planes)
    elif od >= 3:
        choke1 = core.std.Maximum(choke1, planes)

    # Areas of negative difference (similar to above)
    choke2 = core.std.Maximum(diff, planes, coordinates=[0, 1, 0, 0, 0, 0, 1, 0])
    if ed > 2: choke2 = core.std.Maximum(choke2, planes, coordinates=[0, 1, 0, 0, 0, 0, 1, 0])
    if ed > 5: choke2 = core.std.Maximum(choke2, planes, coordinates=[0, 1, 0, 0, 0, 0, 1, 0])
    if ed % 3 != 0: choke2 = core.std.Inflate(choke2, planes)
    if ed in [2, 5]: choke2 = core.std.Median(choke2, planes)
    choke2 = core.std.Minimum(choke2, planes, coordinates=[0, 1, 0, 0, 0, 0, 1, 0])
    if ed > 1: choke2 = core.std.Minimum(choke2, planes, coordinates=[0, 1, 0, 0, 0, 0, 1, 0])
    if ed > 4: choke2 = core.std.Minimum(choke2, planes, coordinates=[0, 1, 0, 0, 0, 0, 1, 0])

    if od == 1:
        choke2 = core.std.Deflate(choke2, planes)
    elif od == 2:
        choke2 = core.std.Deflate(choke2, planes).std.Deflate(planes)
    elif od >= 3:
        choke2 = core.std.Minimum(choke2, planes)

    # Combine above areas to find those areas of difference to restore
    expr1 = 'x 0.00392 < x y 0 < 0 y ? ?'  if isFLOAT else 'x {} < x y {} < {} y ? ?'.format(129*i, mid, mid)
    expr2 = 'x -0.00392 > x y 0 > 0 y ? ?' if isFLOAT else 'x {} > x y {} > {} y ? ?'.format(127*i, mid, mid)
    restore = core.std.Expr([core.std.Expr([diff, choke1], [expr1] if Chroma or isGRAY else [expr1, '']), choke2], [expr2] if Chroma or isGRAY else [expr2, ''])

    return core.std.MergeDiff(clip, restore, planes)


# Given noise extracted from an interlaced source (i.e. the noise is interlaced), generate "progressive" noise with a new "field" of noise injected.
# The new noise is centered on a weighted local average and uses the difference between local min & max as an estimate of local variance.
def QTGMC_Generate2ndFieldNoise(clip, InterleavedClip, ChromaNoise, TFF):
    bd = clip.format.bits_per_sample
    isFLOAT = clip.format.sample_type == vs.FLOAT
    isGRAY = clip.format.color_family == vs.GRAY
    mid = 0 if isFLOAT else 1 << (bd - 1)
    GRAY = [0.5] if isFLOAT and isGRAY else [0.5, 0, 0] if isFLOAT else [mid] * clip.format.num_planes
    planes = [0, 1, 2] if ChromaNoise and not isGRAY else [0]

    origNoise = core.std.SeparateFields(clip, TFF)
    noiseMax = core.std.Maximum(origNoise, planes).std.Maximum(planes, coordinates=[0, 0, 0, 1, 1, 0, 0, 0])
    noiseMin = core.std.Minimum(origNoise, planes).std.Minimum(planes, coordinates=[0, 0, 0, 1, 1, 0, 0, 0])
    random = core.std.SeparateFields(InterleavedClip, TFF).std.BlankClip(color=GRAY).grain.Add(var=1800, uvar=1800 if ChromaNoise else 0)
    expr = 'x y *' if isFLOAT else 'x {} - y * {} / {} +'.format(mid, 1 << bd, mid)
    varRandom = core.std.Expr([core.std.MakeDiff(noiseMax, noiseMin, planes), random], [expr] if ChromaNoise or isGRAY else [expr, ''])
    newNoise = core.std.MergeDiff(noiseMin, varRandom, planes)

    return haf.Weave(core.std.Interleave([origNoise, newNoise]), TFF)


# Insert the source lines into the result to create a true lossless output. However, the other lines in the result have had considerable processing 
# and won't exactly match source lines. There will be some slight residual combing. Use vertical medians to clean a little of this away.
def QTGMC_MakeLossless(clip, Source, InputType, TFF):
    isFLOAT = clip.format.sample_type == vs.FLOAT
    mid = 0 if isFLOAT else 1 << (clip.format.bits_per_sample - 1)
    V = core.rgsf.VerticalCleaner if isFLOAT else core.rgvs.VerticalCleaner
    RE = core.rgsf.Repair if isFLOAT else core.rgvs.Repair
    RG = core.rgsf.RemoveGrain if isFLOAT else core.rgvs.RemoveGrain

    if InputType == 1:
        raise ValueError('QTGMC: Lossless modes are incompatible with InputType = 1')

    # Weave the source fields and the "new" fields that have generated in the clip
    srcFields = core.std.SeparateFields(Source, TFF) if InputType <= 0 else core.std.SeparateFields(Source, TFF).std.SelectEvery(4, [0, 3])
    newFields = core.std.SeparateFields(clip, TFF).std.SelectEvery(4, [1, 2])
    processed = haf.Weave(core.std.Interleave([srcFields, newFields]).std.SelectEvery(4, [0, 1, 3, 2]), TFF)

    # Clean some of the artifacts caused by the above - creating a second version of the "new" fields
    vertMedian = V(processed, 1)
    vertMedDiff = core.std.MakeDiff(processed, vertMedian)
    vmNewDiff1 = core.std.SeparateFields(vertMedDiff, TFF).std.SelectEvery(4, [1, 2])
    expr = 'x y * 0 < 0 x abs y abs < x y ? ?' if isFLOAT else 'x {m} - y {m} - * 0 < {m} x {m} - abs y {m} - abs < x y ? ?'.format(m=mid)
    vmNewDiff2 = core.std.Expr([V(vmNewDiff1, 1), vmNewDiff1], [expr])
    vmNewDiff3 = RE(vmNewDiff2, RG(vmNewDiff2, 2), 1)

    # Reweave final result
    return haf.Weave(core.std.Interleave([srcFields, core.std.MakeDiff(newFields, vmNewDiff3)]).std.SelectEvery(4, [0, 1, 3, 2]), TFF)


# Source-match, a three stage process that takes the difference between deinterlaced clip and the original interlaced source, 
# to shift the clip more towards the source without introducing shimmer. All other arguments defined in main script.
def QTGMC_ApplySourceMatch(Deinterlace, InputType, Source, bVec1, fVec1, bVec2, fVec2, SubPel, SubPelInterp, hpad, vpad, ThSAD1, ThSCD1, ThSCD2, SourceMatch,
                           MatchTR1, MatchEdi, MatchNNSize, MatchNNeurons, MatchEdiQual, MatchEdiMaxD, MatchTR2, MatchEdi2, MatchNNSize2, MatchNNeurons2, MatchEdiQual2, MatchEdiMaxD2,
                           MatchEnhance, pscrn, int16_prescreener, int16_predictor, exp, alpha, beta, gamma, nrad, vcheck, TFF, opencl, device):
    # Basic source-match. Find difference between source clip & equivalent fields in interpolated/smoothed clip (called the "error" in formula below).
    # Ideally there should be no difference, we want the fields in the output to be as close as possible to the source whilst remaining shimmer-free.
    # So adjust the *source* in such a way that smoothing it will give a result closer to the unadjusted source. 
    # Then rerun the interpolation (edi) and binomial smooth with this new source. Result will still be shimmer-free and closer to the original source.
    # Formula used for correction is P0' = P0 + (P0-P1)/(k+S(1-k)), where P0 is original image, P1 is the 1st attempt at interpolation/smoothing,
    # P0' is the revised image to use as new source for interpolation/smoothing, k is the weighting given to the current frame in the smooth,
    # and S is a factor indicating "temporal similarity" of the error from frame to frame, i.e. S = average over all pixels of [neighbor frame error / current frame error]
    # Decreasing S will make the result sharper, sensible range is about -0.25 to 1.0. Empirically, S = 0.5 is effective.
    errorTemporalSimilarity = 0.4 # S in formula described above
    errorAdjust1 = [1.0, 2 / (1 + errorTemporalSimilarity), 8 / (3 + 5 * errorTemporalSimilarity)][MatchTR1]
    isFLOAT = (Deinterlace.format.sample_type == vs.FLOAT)
    S = core.mvsf.Super if isFLOAT else core.mv.Super
    D1 = core.mvsf.Degrain1 if isFLOAT else core.mv.Degrain1
    match1Clip = Deinterlace if SourceMatch < 1 or InputType == 1 else haf.Weave(core.std.SeparateFields(Deinterlace, TFF).std.SelectEvery(4, [0, 3]), TFF)
    match1Update = Source if SourceMatch < 1 or MatchTR1 <= 0 else core.std.Expr([Source, match1Clip], ['x {} * y {} * -'.format(errorAdjust1+1, errorAdjust1)])
    if SourceMatch > 0:
        match1Edi = QTGMC_Interpolate(match1Update, InputType, MatchEdi, MatchNNSize, MatchNNeurons, MatchEdiQual, MatchEdiMaxD, pscrn, int16_prescreener, int16_predictor, exp, alpha, beta, gamma, nrad, vcheck, None, '', TFF, opencl, device)
        if MatchTR1 > 0:
            match1Super = S(match1Edi, pel=SubPel, sharp=SubPelInterp, levels=1, rfilter=1, hpad=hpad, vpad=vpad)
            match1Degrain1 = D1(match1Edi, match1Super, bVec1, fVec1, thsad=ThSAD1, thscd1=ThSCD1, thscd2=ThSCD2)
        if MatchTR1 > 1:
            match1Degrain2 = D1(match1Edi, match1Super, bVec2, fVec2, thsad=ThSAD1, thscd1=ThSCD1, thscd2=ThSCD2)
    if SourceMatch < 1:
        match1 = Deinterlace
    elif MatchTR1 <= 0:
        match1 = match1Edi
    elif MatchTR1 == 1:
        match1 = core.std.Merge(match1Degrain1, match1Edi, [0.25])
    else:
        match1 = core.std.Merge(core.std.Merge(match1Degrain1, match1Degrain2, [0.2]), match1Edi, [0.0625])
    if SourceMatch < 2:
        return match1

    # Enhance effect of source-match stages 2 & 3 by sharpening clip prior to refinement (source-match tends to underestimate so this will leave result sharper)
    match1Shp = core.std.Expr([match1, MinBlur(match1)], ['x dup y - {} * +'.format(MatchEnhance)]) if SourceMatch > 1 and MatchEnhance > 0 else match1

    # Source-match refinement. Find difference between source clip & equivalent fields in (updated) interpolated/smoothed clip.
    # Interpolate & binomially smooth this difference then add it back to output. Helps restore differences that the basic match missed.
    # However, as this pass works on a difference rather than the source image it can be prone to occasional artifacts (difference images are not ideal for interpolation).
    # In fact a lower quality interpolation such as a simple bob often performs nearly as well as advanced, slower methods (e.g. NNEDI3)
    match2Clip = match1Shp if SourceMatch < 2 or InputType == 1 else haf.Weave(core.std.SeparateFields(match1Shp, TFF).std.SelectEvery(4, [0, 3]), TFF)
    if SourceMatch > 1:
        match2Diff = core.std.MakeDiff(Source, match2Clip)
        match2Edi = QTGMC_Interpolate(match2Diff, InputType, MatchEdi2, MatchNNSize2, MatchNNeurons2, MatchEdiQual2, MatchEdiMaxD2, pscrn, int16_prescreener, int16_predictor, exp, alpha, beta, gamma, nrad, vcheck, None, '', TFF, opencl, device)
        if MatchTR2 > 0:
            match2Super = S(match2Edi, pel=SubPel, sharp=SubPelInterp, levels=1, rfilter=1, hpad=hpad, vpad=vpad)
            match2Degrain1 = D1(match2Edi, match2Super, bVec1, fVec1, thsad=ThSAD1, thscd1=ThSCD1, thscd2=ThSCD2)
        if MatchTR2 > 1:
            match2Degrain2 = D1(match2Edi, match2Super, bVec2, fVec2, thsad=ThSAD1, thscd1=ThSCD1, thscd2=ThSCD2)
    if SourceMatch < 2:
        match2 = match1
    elif MatchTR2 <= 0:
        match2 = match2Edi
    elif MatchTR2 == 1:
        match2 = core.std.Merge(match2Degrain1, match2Edi, [0.25])
    else:
        match2 = core.std.Merge(core.std.Merge(match2Degrain1, match2Degrain2, [0.2]), match2Edi, [0.0625])

    # Source-match second refinement - correct error introduced in the refined difference by temporal smoothing. Similar to error correction from basic step
    errorAdjust2 = [1.0, 2 / (1 + errorTemporalSimilarity), 8 / (3 + 5 * errorTemporalSimilarity)][MatchTR2]
    match3Update = match2Edi if SourceMatch < 3 or MatchTR2 <= 0 else core.std.Expr([match2Edi, match2], ['x {} * y {} * -'.format(errorAdjust2+1, errorAdjust2)])
    if SourceMatch > 2:
        if MatchTR2 > 0:
            match3Super = S(match3Update, pel=SubPel, sharp=SubPelInterp, levels=1, rfilter=1, hpad=hpad, vpad=vpad)
            match3Degrain1 = D1(match3Update, match3Super, bVec1, fVec1, thsad=ThSAD1, thscd1=ThSCD1, thscd2=ThSCD2)
        if MatchTR2 > 1:
            match3Degrain2 = D1(match3Update, match3Super, bVec2, fVec2, thsad=ThSAD1, thscd1=ThSCD1, thscd2=ThSCD2)
    if SourceMatch < 3:
        match3 = match2
    elif MatchTR2 <= 0:
        match3 = match3Update
    elif MatchTR2 == 1:
        match3 = core.std.Merge(match3Degrain1, match3Update, [0.25])
    else:
        match3 = core.std.Merge(core.std.Merge(match3Degrain1, match3Degrain2, [0.2]), match3Update, [0.0625])

    # Apply difference calculated in source-match refinement
    return core.std.MergeDiff(match1Shp, match3)


def SMDegrain(clip, tr=2, thSAD=314, thSADC=None, RefineMotion=False, contrasharp=None, CClip=None, interlaced=False, TFF=None,
              pel=None, subpixel=2, prefilter=-1, mfilter=None, blksize=None, overlap=None, search=5, truemotion=None, DCT=0,
              MVglobal=None, limit=255, limitc=None, thSCD1=400, thSCD2=128, chroma=True, searchparam=2, Str=1, Amp=0.0625):
    """
    Simple MDegrain Mod - SMDegrain()
    Mod by Dogway - Original idea by Caroliano
    Special Thanks: Didée, cretindesalpes, Sagekilla, Gavino and MVTools people

    General purpose simple degrain function. Pure temporal denoiser.
    Basically a wrapper(function)/frontend of MVTools2 + MDegrain with some added common related options.
    Goal is accessibility and quality but not targeted to any specific kind of source.
    The reason behind is to keep it simple so you will only need MVTools2.
    Check documentation for deep explanation on settings and defaults.
    VideoHelp thread: (https://forum.videohelp.com/threads/369142)
    """
    # tr > 3 uses mvsf hence requires float input, also requires mvmulti module.
    # Modified from havsfunc: https://github.com/HomeOfVapourSynthEvolution/havsfunc/blob/r31/havsfunc.py#L3186

    bd = clip.format.bits_per_sample
    isFLOAT = clip.format.sample_type == vs.FLOAT
    isGRAY = clip.format.color_family == vs.GRAY
    chroma = False if isGRAY else chroma
    peak = 1.0 if isFLOAT else (1 << bd) - 1
    i = 0.00392 if isFLOAT else 1 << (bd - 8)
    A = core.mvsf.Analyse if isFLOAT else core.mv.Analyse
    S = core.mvsf.Super if isFLOAT else core.mv.Super
    R = core.mvsf.Recalculate if isFLOAT else core.mv.Recalculate
    D1 = core.mvsf.Degrain1 if isFLOAT else core.mv.Degrain1
    D2 = core.mvsf.Degrain2 if isFLOAT else core.mv.Degrain2
    D3 = core.mvsf.Degrain3 if isFLOAT else core.mv.Degrain3
    w = clip.width
    h = clip.height
    preclip = isinstance(prefilter, vs.VideoNode)
    ifC = isinstance(contrasharp, bool)
    if1 = isinstance(CClip, vs.VideoNode)
    planes = [0, 1, 2] if chroma else [0]
    plane = 4 if chroma else 0
    limit = scale(limit, peak)
    limitc = limit if limitc is None else scale(limitc, peak)

    # Defaults & Conditionals
    if thSADC is None:
        thSADC = thSAD // 2

    if contrasharp is None:
        contrasharp = if1

    if pel is None:
        pel = 1 if w > 960 else 2

    if pel < 2:
        subpixel = min(subpixel, 2)
    
    ppp = pel > 1 and subpixel > 2

    if blksize is None:
        blksize = 16 if w > 960 else 8

    if overlap is None:
        overlap = blksize // 2

    if truemotion is None:
        truemotion = w <= 960

    # Error Report
    if not isinstance(clip, vs.VideoNode) or clip.format.color_family not in [vs.GRAY, vs.YUV]:
        raise TypeError("SMDegrain: This is not a GRAY or YUV clip!")
    if not (ifC or isinstance(contrasharp, int)):
        raise TypeError("SMDegrain: contrasharp only accepts bool and integer inputs")
    if if1 and (CClip.format.id != clip.format.id):
        raise TypeError("SMDegrain: CClip must be the same format as clip")
    if if1 and (CClip.width != clip.width or CClip.height != clip.height):
        raise TypeError("SMDegrain: CClip must be the same size as clip")
    if interlaced and h & 3:
        raise ValueError('SMDegrain: Interlaced source requires mod 4 height sizes')
    if interlaced and not isinstance(TFF, bool):
        raise TypeError("SMDegrain: TFF must be set if source is interlaced. Setting TFF to True means top field first or False means bottom field first")
    if not (isinstance(prefilter, int) or preclip):
        raise TypeError("SMDegrain: prefilter only accepts integer and clip inputs")
    if preclip and prefilter.format.id != clip.format.id:
        raise TypeError("SMDegrain: prefilter must be the same format as clip")
    if preclip and (prefilter.width != clip.width or prefilter.height != clip.height):
        raise TypeError("SMDegrain: prefilter must be the same size as clip")
    if mfilter is not None and (not isinstance(mfilter, vs.VideoNode) or mfilter.format.id != clip.format.id):
        raise TypeError("SMDegrain: mfilter must be the same format as clip")
    if mfilter is not None and (mfilter.width != clip.width or mfilter.height != clip.height):
        raise TypeError("SMDegrain: mfilter must be the same size as clip")
    if RefineMotion and blksize < 8:
        raise ValueError('SMDegrain: For RefineMotion you need a blksize of at least 8')
    if not isFLOAT and tr > 3:
        raise ValueError("SMDegrain: tr > 3 requires input of float sample type")

    # RefineMotion Variables
    if RefineMotion:
        halfblksize = blksize // 2          # MRecalculate works with half block size
        halfoverlap = max(overlap // 2, 2)  # Halve the overlap to suit the halved block size
        halfthSAD = thSAD // 2              # MRecalculate uses a more strict thSAD, which defaults to 157 (half of function's default of 314)

    # clip preparation for Interlacing
    inpu = clip if not interlaced else core.std.SeparateFields(clip, TFF)

    # Prefilter & Motion Filter
    if mfilter is None:
        mfilter = inpu

    if preclip:
        pref = prefilter
    elif 0 <= prefilter < 3:
        pref = MinBlur(inpu, prefilter, planes)
    elif prefilter == 3:
        expr = 'x {a} < {p} x {b} > 0 {p} x {a} - {p} {b} {a} - / * - ? ?'.format(a=16*i, b=75*i, p=peak)
        mask = core.std.Expr([core.std.ShufflePlanes(inpu, [0], vs.GRAY)], [expr])
        pref = core.std.MaskedMerge(core.dfttest.DFTTest(inpu, tbsize=1, sstring='0.0:4.0 0.2:9.0 1.0:15.0', planes=planes), inpu, mask)
    elif prefilter == 4:
        pref = haf.KNLMeansCL(inpu, d=1, a=1, h=7) if chroma else inpu.knlm.KNLMeansCL(d=1, a=1, h=7)
    elif prefilter == 5:
        pref = SpotLess(inpu, chroma=chroma)
    else:
        pref = inpu

    # Default Auto-Prefilter - Luma expansion TV → PC (up to 16% more values for motion estimation)
    pref = DitherLumaRebuild(pref, s0=Str, c=Amp, chroma=chroma)

    # Subpixel 3
    # Motion vectors search
    super_args = dict(hpad=blksize, vpad=blksize, pel=pel)
    analyse_args = dict(blksize=blksize, search=search, chroma=chroma, truemotion=truemotion, _global=MVglobal, overlap=overlap, dct=DCT, searchparam=searchparam)

    if RefineMotion:
        recalculate_args = dict(thsad=halfthSAD, blksize=halfblksize, search=search, chroma=chroma, truemotion=truemotion, overlap=halfoverlap, dct=DCT, searchparam=searchparam)

    if ppp:
        cshift = 0.25 if pel == 2 else 0.375
        pclip  = nnrs.nnedi3_resample(pref, w * pel, h * pel, cshift, cshift, nns=4)
        pclip2 = nnrs.nnedi3_resample(inpu, w * pel, h * pel, cshift, cshift, nns=4)
        super_search = S(pref, chroma=chroma, rfilter=4, pelclip=pclip, **super_args)
        super_render = S(inpu, chroma=chroma, rfilter=1, pelclip=pclip2, levels=1, **super_args)
    else:
        super_search = S(pref, chroma=chroma, rfilter=4, sharp=1, **super_args)
        super_render = S(inpu, chroma=chroma, rfilter=1, sharp=subpixel, levels=1, **super_args)

    if tr < 4:
        if interlaced:
            if tr > 2:
                bv6 = A(super_search, isb=True,  delta=6, **analyse_args)
                fv6 = A(super_search, isb=False, delta=6, **analyse_args)
                if RefineMotion:
                    bv6 = R(super_search, bv6, **recalculate_args)
                    fv6 = R(super_search, fv6, **recalculate_args)
            if tr > 1:
                bv4 = A(super_search, isb=True,  delta=4, **analyse_args)
                fv4 = A(super_search, isb=False, delta=4, **analyse_args)
                if RefineMotion:
                    bv4 = R(super_search, bv4, **recalculate_args)
                    fv4 = R(super_search, fv4, **recalculate_args)
        else:
            if tr > 2:
                bv3 = A(super_search, isb=True,  delta=3, **analyse_args)
                fv3 = A(super_search, isb=False, delta=3, **analyse_args)
                if RefineMotion:
                    bv3 = R(super_search, bv3, **recalculate_args)
                    fv3 = R(super_search, fv3, **recalculate_args)
            bv1 = A(super_search, isb=True,  delta=1, **analyse_args)
            fv1 = A(super_search, isb=False, delta=1, **analyse_args)
            if RefineMotion:
                bv1 = R(super_search, bv1, **recalculate_args)
                fv1 = R(super_search, fv1, **recalculate_args)
        if interlaced or tr > 1:
            bv2 = A(super_search, isb=True,  delta=2, **analyse_args)
            fv2 = A(super_search, isb=False, delta=2, **analyse_args)
            if RefineMotion:
                bv2 = R(super_search, bv2, **recalculate_args)
                fv2 = R(super_search, fv2, **recalculate_args)
    else:
        vec = Analyze(super_search, tr=tr, d=interlaced, **analyse_args)
        if RefineMotion:
            vec = mvmulti.Recalculate(super_search, vec, tr=tr, **recalculate_args)

    # Finally, MDegrain
    if isFLOAT:
        degrain_args = dict(thsad=[thSAD, thSADC, thSADC], plane=plane, limit=[limit, limitc, limitc], thscd1=thSCD1, thscd2=thSCD2)
    else:
        degrain_args = dict(thsad=thSAD, thsadc=thSADC, plane=plane, limit=limit, limitc=limitc, thscd1=thSCD1, thscd2=thSCD2)

    if tr < 4:
        if interlaced:
            if tr == 3:
                output = D3(mfilter, super_render, bv2, fv2, bv4, fv4, bv6, fv6, **degrain_args)
            elif tr == 2:
                output = D2(mfilter, super_render, bv2, fv2, bv4, fv4, **degrain_args)
            else:
                output = D1(mfilter, super_render, bv2, fv2, **degrain_args)
        else:
            if tr == 3:
                output = D3(mfilter, super_render, bv1, fv1, bv2, fv2, bv3, fv3, **degrain_args)
            elif tr == 2:
                output = D2(mfilter, super_render, bv1, fv1, bv2, fv2, **degrain_args)
            else:
                output = D1(mfilter, super_render, bv1, fv1, **degrain_args)
    else:
        output = mvmulti.DegrainN(mfilter, super_render, vec, tr=tr, **degrain_args)

    # Contrasharp (only sharpens luma)
    if contrasharp:
        if if1:
            if interlaced:
                CClip = core.std.SeparateFields(CClip, TFF)
        else:
            CClip = inpu
            
    # Output
    if contrasharp:
        if interlaced:
            if ifC:
                return haf.Weave(ContraSharpening(output, CClip, planes=planes), TFF)
            else:
                return haf.Weave(LSFmod(output, contrasharp, source=CClip, kernel=-1, ssx=1, Lmode=0, soothe=False), TFF)
        elif ifC:
            return ContraSharpening(output, CClip, planes=planes)
        else:
            return LSFmod(output, contrasharp, source=CClip, kernel=-1, ssx=1, Lmode=0, soothe=False)
    elif interlaced:
        return haf.Weave(output, TFF)
    else:
        return output


def TemporalDegrain2(clip, degrainTR=2, degrainPlane=4, meAlg=5, meAlgPar=None, meSubpel=None, meBlksz=None, meTM=False,
    limitSigma=None, limitBlksz=None, fftThreads=None, postFFT=0, postTR=1, postSigma=1, knlDevId=0, ppSAD1=10, ppSAD2=5, 
    ppSCD1=4, thSCD2=100, DCT=0, SubPelInterp=2, SrchClipPP=3, GlobalMotion=True, ChromaMotion=True, rec=False, extraSharp=False):
    """
    Temporal Degrain Updated by ErazorTT                               
                                                                          
    Based on function by Sagekilla, idea + original script created by Didee
    Works as a simple temporal degraining function that'll remove             
    MOST or even ALL grain and noise from video sources,                      
    including dancing grain, like the grain found on 300.                     
    Also note, the parameters don't need to be tweaked much.                  
                                                                           
    Required plugins:                                                         
    FFT3DFilter: https://github.com/myrsloik/VapourSynth-FFT3DFilter   
    MVtools(sf): https://github.com/dubhater/vapoursynth-mvtools (https://github.com/IFeelBloated/vapoursynth-mvtools-sf)                   
                                                                           
    Optional plugins:                                                         
    dfttest: https://github.com/HomeOfVapourSynthEvolution/VapourSynth-DFTTest            
    KNLMeansCL: https://github.com/Khanattila/KNLMeansCL
    
    recommendation: 
    1. start with default settings
    2a.if there is too much denoising for your taste use degrainTR=1
    2b.if more denoising is needed try postFFT=1 with postSigma=1, then tune postSigma (obvious blocking and banding of the sky are indications of a value which is at least a factor 2 too high)
    3. do not increase degrainTR above 1/8 of the fps (at 24fps up to 3)
    4. if there are any issues with banding switch to postFFT=3

    use only the following knobs (all other settings should already be were they need to be):
    - degrainTR, temporal radius of degrain, usefull range: min=1, default=2, max=fps/8. Higher values do clean the video more, but also increase probability of wrongly identified motion meAlg which leads to washed out regions
    - postFFT, if you want to remove absolutely all remaining noise suggestion is to use 3 (dfttest) for its quality, 1/2 (ff3dfilter) is much faster but can introduce banding, 4 is KNLMeansCL.
    - postSigma, increase it to remove all the remaining noise you want removed, but do not increase too much since unnecessary high values have severe negative impact on either banding and/or sharpness
    - degrainPlane, if you just want to denoise the chroma use 3 (helps with compressability with the clip being almost identical to the original)
    """

    if not isinstance(clip, vs.VideoNode) or clip.format.color_family not in [vs.GRAY, vs.YUV]:
        raise TypeError("TemporalDegrain2: This is not a GRAY or YUV clip!")
    
    w = clip.width
    h = clip.height
    WH = max(w, h)
    bd = clip.format.bits_per_sample
    isFLOAT = clip.format.sample_type == vs.FLOAT
    isGRAY = clip.format.color_family == vs.GRAY
    i = 0.00392 if isFLOAT else 1 << (bd - 8)
    mid = 0.5 if isFLOAT else 1 << (bd - 1)
    S = core.mvsf.Super if isFLOAT else core.mv.Super
    A = core.mvsf.Analyse if isFLOAT else core.mv.Analyse
    C = core.mvsf.Compensate if isFLOAT else core.mv.Compensate
    R = core.mvsf.Recalculate if isFLOAT else core.mv.Recalculate
    D1 = core.mvsf.Degrain1 if isFLOAT else core.mv.Degrain1
    D2 = core.mvsf.Degrain2 if isFLOAT else core.mv.Degrain2
    D3 = core.mvsf.Degrain3 if isFLOAT else core.mv.Degrain3
    RG = core.rgsf.RemoveGrain if isFLOAT else core.rgvs.RemoveGrain
    rad = 3 if extraSharp else None
    mat = [1, 2, 1, 2, 4, 2, 1, 2, 1]
    ChromaNoise = (degrainPlane > 0)
    hpad = meBlksz
    vpad = meBlksz
    
    if meSubpel is None:
        if WH < 960:
            meSubpel = 4
        elif WH < 2080:
            meSubpel = 2
        else:
            meSubpel = 1
    
    if meBlksz is None:
        if WH < 1280:
            meBlksz = 8
        elif WH < 2080:
            meBlksz = 16
        else:
            meBlksz = 32
    
    if limitSigma is None:
        if WH < 960:
            limitSigma = 8
        elif WH < 1280:
            limitSigma = 12
        elif WH < 2080:
            limitSigma = 16
        else:
            limitSigma = 32
    
    if limitBlksz is None:
        if WH < 960:
            limitBlksz = 16
        elif WH < 1280:
            limitBlksz = 24
        elif WH < 2080:
            limitBlksz = 32
        else:
            limitBlksz = 64
    
    if postFFT <= 0:
        postTR = 0
    
    if isGRAY:
        ChromaMotion = False
        ChromaNoise = False
        degrainPlane = 0
    
    if degrainPlane == 0:
        fPlane = [0]
    elif degrainPlane == 1:
        fPlane = [1]
    elif degrainPlane == 2:
        fPlane = [2]
    elif degrainPlane == 3:
        fPlane = [1, 2]
    else:
        fPlane = [0, 1, 2]

    if postFFT == 3:
        postTR = min(postTR, 7)

    if postFFT in [1, 2]:
        postTR = min(postTR, 2)

    postTD  = postTR * 2 + 1
    maxTR = max(degrainTR, postTR)
    Overlap = meBlksz / 2
    Lambda = (1000 if meTM else 100) * (meBlksz ** 2) // 64
    LSAD = 1200 if meTM else 400
    PNew = 50 if meTM else 25
    PLevel = 1 if meTM else 0
    thSAD1 = int(ppSAD1 * 64)
    thSAD2 = int(ppSAD2 * 64)
    thSCD1 = int(ppSCD1 * 64)
    CMplanes = [0, 1, 2] if ChromaMotion else [0]
    
    if maxTR > 3 and not isFLOAT:
        raise ValueError("TemporalDegrain2: maxTR > 3 requires input of float sample type")
    
    if SrchClipPP == 1:
        spatialBlur = core.resize.Bilinear(clip, m4(w/2), m4(h/2)).std.Convolution(matrix=mat, planes=CMplanes).resize.Bilinear(w, h)
    elif SrchClipPP > 1:
        spatialBlur = core.tcanny.TCanny(clip, sigma=2, mode=-1, planes=CMplanes)
        spatialBlur = core.std.Merge(spatialBlur, clip, [0.1] if ChromaMotion or isGRAY else [0.1, 0])
    else:
        spatialBlur = clip
    if SrchClipPP < 3:
        srchClip = spatialBlur
    else:
        expr = 'x {a} + y < x {b} + x {a} - y > x {b} - x y + 2 / ? ?'.format(a=7*i, b=2*i)
        srchClip = core.std.Expr([spatialBlur, clip], [expr] if ChromaMotion or isGRAY else [expr, ''])

    analyse_args = dict(blksize=meBlksz, overlap=Overlap, search=meAlg, searchparam=meAlgPar, pelsearch=meSubpel, truemotion=meTM, _lambda=Lambda, lsad=LSAD, pnew=PNew, plevel=PLevel, _global=GlobalMotion, dct=DCT, chroma=ChromaMotion)
    recalculate_args = dict(blksize=Overlap, overlap=Overlap/2, search=meAlg, searchparam=meAlgPar, truemotion=meTM, _lambda=Lambda/4, pnew=PNew, dct=DCT, chroma=ChromaMotion)
    srchSuper = S(DitherLumaRebuild(srchClip, s0=1, chroma=ChromaMotion), pel=meSubpel, sharp=1, rfilter=4, hpad=hpad, vpad=vpad, chroma=ChromaMotion)
    
    if (maxTR > 0) and (degrainTR < 4 or postTR < 4):
        bVec1 = A(srchSuper, isb=True,  delta=1, **analyse_args)
        fVec1 = A(srchSuper, isb=False, delta=1, **analyse_args)
        if rec:
            bVec1 = R(srchSuper, bVec1, **recalculate_args)
            fVec1 = R(srchSuper, fVec1, **recalculate_args)
        if maxTR > 1:
            bVec2 = A(srchSuper, isb=True,  delta=2, **analyse_args)
            fVec2 = A(srchSuper, isb=False, delta=2, **analyse_args)
            if rec:
                bVec2 = R(srchSuper, bVec2, **recalculate_args)
                fVec2 = R(srchSuper, fVec2, **recalculate_args)
        if maxTR > 2:
            bVec3 = A(srchSuper, isb=True,  delta=3, **analyse_args)
            fVec3 = A(srchSuper, isb=False, delta=3, **analyse_args)
            if rec:
                bVec3 = R(srchSuper, bVec3, **recalculate_args)
                fVec3 = R(srchSuper, fVec3, **recalculate_args)

    if degrainTR > 3:
        vmulti1 = mvmulti.Analyze(srchSuper, tr=degrainTR, **analyse_args)
        if rec:
            vmulti1 = mvmulti.Recalculate(srchSuper, vmulti1, tr=tr, **recalculate_args)

    if postTR > 3:
        vmulti2 = mvmulti.Analyze(srchSuper, tr=postTR, **analyse_args)
        if rec:
            vmulti2 = mvmulti.Recalculate(srchSuper, vmulti2, tr=tr, **recalculate_args)
    #---------------------------------------
    # Degrain
    # "spat" is a prefiltered clip which is used to limit the effect of the 1st MV-denoise stage.
    if degrainTR > 0:
        limitSigma *= i
        s2 = limitSigma * 0.625
        s3 = limitSigma * 0.375
        s4 = limitSigma * 0.250
        spat = core.fft3dfilter.FFT3DFilter(clip, planes=fPlane, sigma=limitSigma, sigma2=s2, sigma3=s3, sigma4=s4, bt=3, bw=limitBlksz, bh=limitBlksz, ncpu=fftThreads)
        spatD  = core.std.MakeDiff(clip, spat)
  
    # First MV-denoising stage. Usually here's some temporal-medianfiltering going on.
    # For simplicity, we just use MDegrain.
    if degrainTR > 0:
        supero = S(clip, pel=meSubpel, sharp=SubPelInterp, levels=1, rfilter=1, hpad=hpad, vpad=vpad, chroma=ChromaNoise)

        if degrainTR < 2:
            NR1 = D1(clip, supero, bVec1, fVec1, plane=degrainPlane, thsad=thSAD1, thscd1=thSCD1, thscd2=thSCD2)
        elif degrainTR < 3:
            NR1 = D2(clip, supero, bVec1, fVec1, bVec2, fVec2, plane=degrainPlane, thsad=thSAD1, thscd1=thSCD1, thscd2=thSCD2)
        elif degrainTR < 4:
            NR1 = D3(clip, supero, bVec1, fVec1, bVec2, fVec2, bVec3, fVec3, plane=degrainPlane, thsad=thSAD1, thscd1=thSCD1, thscd2=thSCD2)
        else:
            NR1 = mvmulti.DegrainN(clip, supero, vmulti1, tr=degrainTR, plane=degrainPlane, thsad=thSAD1, thscd1=thSCD1, thscd2=thSCD2)

    # Limit NR1 to not do more than what "spat" would do.
    if degrainTR > 0:
        NR1D = core.std.MakeDiff(clip, NR1)
        expr = 'x abs y abs < x y ?' if isFLOAT else 'x {} - abs y {} - abs < x y ?'.format(mid, mid)
        DD   = core.std.Expr([spatD, NR1D], [expr])
        NR1x = core.std.MakeDiff(clip, DD, [0])
  
    # Second MV-denoising stage. We use MDegrain.
    if degrainTR > 0:
        NR1x_super = S(NR1x, pel=meSubpel, sharp=SubPelInterp, levels=1, rfilter=1, hpad=hpad, vpad=vpad, chroma=ChromaNoise)

        if degrainTR < 2:
            NR2 = D1(NR1x, NR1x_super, bVec1, fVec1, plane=degrainPlane, thsad=thSAD2, thscd1=thSCD1, thscd2=thSCD2)
        elif degrainTR < 3:
            NR2 = D2(NR1x, NR1x_super, bVec1, fVec1, bVec2, fVec2, plane=degrainPlane, thsad=thSAD2, thscd1=thSCD1, thscd2=thSCD2)
        elif degrainTR < 4:
            NR2 = D3(NR1x, NR1x_super, bVec1, fVec1, bVec2, fVec2, bVec3, fVec3, plane=degrainPlane, thsad=thSAD2, thscd1=thSCD1, thscd2=thSCD2)
        else:
            NR2 = mvmulti.DegrainN(NR1x, NR1x_super, vmulti1, tr=degrainTR, plane=degrainPlane, thsad=thSAD2, thscd1=thSCD1, thscd2=thSCD2)
    else:
        NR2 = clip
    
    NR2 = RG(NR2, mode=1) # Filter to remove last bits of dancing pixels, YMMV.

    #---------------------------------------
    # post FFT
    if postTR > 0:
        fullSuper = S(NR2, pel=meSubpel, sharp=SubPelInterp, levels=1, rfilter=1, hpad=hpad, vpad=vpad, chroma=ChromaNoise)

    if postTR > 0:
        if postTR == 1:
            noiseWindow = core.std.Interleave([C(NR2, fullSuper, fVec1, thscd1=thSCD1, thscd2=thSCD2), NR2,
                                               C(NR2, fullSuper, bVec1, thscd1=thSCD1, thscd2=thSCD2)])
        elif postTR == 2:
            noiseWindow = core.std.Interleave([C(NR2, fullSuper, fVec2, thscd1=thSCD1, thscd2=thSCD2),
                                               C(NR2, fullSuper, fVec1, thscd1=thSCD1, thscd2=thSCD2), NR2,
                                               C(NR2, fullSuper, bVec1, thscd1=thSCD1, thscd2=thSCD2),
                                               C(NR2, fullSuper, bVec2, thscd1=thSCD1, thscd2=thSCD2)])
        elif postTR == 3:
            noiseWindow = core.std.Interleave([C(NR2, fullSuper, fVec3, thscd1=thSCD1, thscd2=thSCD2),
                                               C(NR2, fullSuper, fVec2, thscd1=thSCD1, thscd2=thSCD2),
                                               C(NR2, fullSuper, fVec1, thscd1=thSCD1, thscd2=thSCD2), NR2,
                                               C(NR2, fullSuper, bVec1, thscd1=thSCD1, thscd2=thSCD2),
                                               C(NR2, fullSuper, bVec2, thscd1=thSCD1, thscd2=thSCD2),
                                               C(NR2, fullSuper, bVec3, thscd1=thSCD1, thscd2=thSCD2)])
        else:
            noiseWindow = mvmulti.Compensate(NR2, fullSuper, vmulti2, thscd1=thSCD1, thscd2=thSCD2, tr=postTR)
    else:
        noiseWindow = NR2
    
    if postFFT == 3:
        dnWindow = core.dfttest.DFTTest(noiseWindow, sigma=postSigma*4, tbsize=postTD, planes=fPlane)
    elif postFFT == 4:
        dnWindow = haf.KNLMeansCL(noiseWindow, d=postTR, a=2, h=postSigma/2, device_id=knlDevId) if ChromaNoise else noiseWindow.knlm.KNLMeansCL(dnWindow, d=postTR, a=2, h=postSigma/2, device_id=knlDevId)
    elif postFFT > 0:
        dnWindow = core.fft3dfilter.FFT3DFilter(noiseWindow, sigma=postSigma*i, planes=fPlane, bt=postTD, ncpu=fftThreads)
    else:
        dnWindow = noiseWindow
    
    if postTR > 0:
        dnWindow = dnWindow[postTR::postTD]
    
    return ContraSharpening(dnWindow, clip, rad)


def mClean(clip, thSAD=400, chroma=True, sharp=10, rn=14, deband=0, depth=0, strength=20, outbits=None, icalc=False, rgmode=18):
    """
    From: https://forum.doom9.org/showthread.php?t=174804 by burfadel
    mClean spatio/temporal denoiser

    +++ Description +++
    Typical spatial filters work by removing large variations in the image on a small scale, reducing noise but also making the image less
    sharp or temporally stable. mClean removes noise whilst retaining as much detail as possible, as well as provide optional image enhancement.

    mClean works primarily in the temporal domain, although there is some spatial limiting.
    Chroma is processed a little differently to luma for optimal results.
    Chroma processing can be disabled with chroma = False.

    +++ Artifacts +++
    Spatial picture artifacts may remain as removing them is a fine balance between removing the unwanted artifact whilst not removing detail.
    Additional dering/dehalo/deblock filters may be required, but should ONLY be uses if required due the detail loss/artifact removal balance.

    +++ Sharpening +++
    Applies a modified unsharp mask to edges and major detected detail. Range of normal sharpening is 0-20. There are 4 additional settings,
    21-24 that provide 'overboost' sharpening. Overboost sharpening is only suitable typically for high definition, high quality sources.
    Actual sharpening calculation is scaled based on resolution.

    +++ ReNoise +++
    ReNoise adds back some of the removed luma noise. Re-adding original noise would be counterproductive, therefore ReNoise modifies this noise
    both spatially and temporally. The result of this modification is the noise becomes much nicer and it's impact on compressibility is greatly
    reduced. It is not applied on areas where the sharpening occurs as that would be counterproductive. Settings range from 0 to 20.
    The strength of renoise is affected by the the amount of original noise removed and how this noise varies between frames.
    It's main purpose is to reduce the 'flatness' that occurs with any form of effective denoising.

    +++ Deband +++
    This will perceptibly improve the quality of the image by reducing banding effect and adding a small amount of temporally stabilised grain
    to both luma and chroma. The settings are not adjustable as the default settings are suitable for most cases without having a large effect
    on compressibility. 0 = disabled, 1 = deband only, 2 = deband and veed

    +++ Depth +++
    This applies a modified warp sharpening on the image that may be useful for certain things, and can improve the perception of image depth.
    Settings range up from 0 to 5. This function will distort the image, for animation a setting of 1 or 2 can be beneficial to improve lines.

    +++ Strength +++
    The strength of the denoising effect can be adjusted using this parameter. It ranges from 20 percent denoising effect with strength 0, up to the
    100 percent of the denoising with strength 20. This function works by blending a scaled percentage of the original image with the processed image.

    +++ Outbits +++
    Specifies the bits per component (bpc) for the output for processing by additional filters. It will also be the bpc that mClean will process.
    If you output at a higher bpc keep in mind that there may be limitations to what subsequent filters and the encoder may support.
    """
    # New parameter icalc, set to True to enable pure integer processing for faster speed. (Ignored if input is of float sample type)
    
    defH = max(clip.height, clip.width // 4 * 3) # Resolution calculation for auto blksize settings
    sharp = min(max(sharp, 0), 24) # Sharp multiplier
    rn = min(max(rn, 0), 20) # Luma ReNoise strength
    deband = min(max(deband, 0), 5)  # Apply deband/veed
    depth = min(max(depth, 0), 5) # Depth enhancement
    strength = min(max(strength, 0), 20) # Strength of denoising
    bd = clip.format.bits_per_sample
    isFLOAT = clip.format.sample_type == vs.FLOAT
    icalc = False if isFLOAT else icalc
    S = core.mv.Super if icalc else core.mvsf.Super
    A = core.mv.Analyse if icalc else core.mvsf.Analyse
    R = core.mv.Recalculate if icalc else core.mvsf.Recalculate

    if not isinstance(clip, vs.VideoNode) or clip.format.color_family != vs.YUV:
        raise TypeError("mClean: This is not a YUV clip!")

    if outbits is None: # Output bits, default input depth
        outbits = bd

    if deband or depth:
        outbits = min(outbits, 16)

    RE = core.rgsf.Repair if outbits == 32 else core.rgvs.Repair
    RG = core.rgsf.RemoveGrain if outbits == 32 else core.rgvs.RemoveGrain
    sc = 8 if defH > 2880 else 4 if defH > 1440 else 2 if defH > 720 else 1
    i = 0.00392 if outbits == 32 else 1 << (outbits - 8)
    peak = 1.0 if outbits == 32 else (1 << outbits) - 1
    bs = 16 if defH / sc > 360 else 8
    ov = 6 if bs > 12 else 2
    pel = 1 if defH > 720 else 2
    truemotion = False if defH > 720 else True
    lampa = 777 * (bs ** 2) // 64
    depth2 = -depth*3
    depth = depth*2

    if sharp > 20:
        sharp += 30
    elif defH <= 2500:
        sharp = 15 + defH * sharp * 0.0007
    else:
        sharp = 50

    # Denoise preparation
    c = core.vcmod.Median(clip, plane=[0, 1, 1]) if chroma else clip

    # Temporal luma noise filter
    if not (isFLOAT or icalc):
        c = c.fmtc.bitdepth(flt=1)
    cy = core.std.ShufflePlanes(c, [0], vs.GRAY)

    super1 = S(c if chroma else cy, hpad=bs, vpad=bs, pel=pel, rfilter=4, sharp=1)
    super2 = S(c if chroma else cy, hpad=bs, vpad=bs, pel=pel, rfilter=1, levels=1)
    analyse_args = dict(blksize=bs, overlap=ov, search=5, truemotion=truemotion)
    recalculate_args = dict(blksize=bs, overlap=ov, search=5, truemotion=truemotion, thsad=180, _lambda=lampa)

    # Analysis
    bvec4 = R(super1, A(super1, isb=True,  delta=4, **analyse_args), **recalculate_args) if not icalc else None
    bvec3 = R(super1, A(super1, isb=True,  delta=3, **analyse_args), **recalculate_args)
    bvec2 = R(super1, A(super1, isb=True,  delta=2, badsad=1100, lsad=1120, **analyse_args), **recalculate_args)
    bvec1 = R(super1, A(super1, isb=True,  delta=1, badsad=1500, lsad=980, badrange=27, **analyse_args), **recalculate_args)
    fvec1 = R(super1, A(super1, isb=False, delta=1, badsad=1500, lsad=980, badrange=27, **analyse_args), **recalculate_args)
    fvec2 = R(super1, A(super1, isb=False, delta=2, badsad=1100, lsad=1120, **analyse_args), **recalculate_args)
    fvec3 = R(super1, A(super1, isb=False, delta=3, **analyse_args), **recalculate_args)
    fvec4 = R(super1, A(super1, isb=False, delta=4, **analyse_args), **recalculate_args) if not icalc else None

    # Applying cleaning
    if not icalc:
        clean = core.mvsf.Degrain4(c if chroma else cy, super2, bvec1, fvec1, bvec2, fvec2, bvec3, fvec3, bvec4, fvec4, thsad=thSAD)
    else:
        clean = core.mv.Degrain3(c if chroma else cy, super2, bvec1, fvec1, bvec2, fvec2, bvec3, fvec3, thsad=thSAD)

    if c.format.bits_per_sample != outbits:
        c = c.fmtc.bitdepth(bits=outbits)
        cy = cy.fmtc.bitdepth(bits=outbits)
        clean = clean.fmtc.bitdepth(bits=outbits)

    uv = core.std.MergeDiff(clean, core.tmedian.TemporalMedian(core.std.MakeDiff(c, clean, [1, 2]), 1, [1, 2]), [1, 2]) if chroma else c
    clean = core.std.ShufflePlanes(clean, [0], vs.GRAY) if clean.format.num_planes != 1 else clean

    # Post clean, pre-process deband
    filt = core.std.ShufflePlanes([clean, uv], [0, 1, 2], vs.YUV)

    if deband:
        filt = filt.f3kdb.Deband(range=16, preset="high" if chroma else "luma", grainy=defH/15, grainc=defH/16 if chroma else 0, output_depth=outbits)
        clean = core.std.ShufflePlanes(filt, [0], vs.GRAY)
        filt = core.vcmod.Veed(filt) if deband == 2 else filt

    # Spatial luma denoising
    clean2 = RG(clean, rgmode)

    # Unsharp filter for spatial detail enhancement
    if sharp:
        if sharp <= 50:
            clsharp = core.std.MakeDiff(clean, muf.Blur(clean2, amountH=0.08+0.03*sharp))
        else:
            clsharp = core.std.MakeDiff(clean, clean2.tcanny.TCanny(sigma=(sharp-46)/4, mode=-1))
        clsharp = core.std.MergeDiff(clean2, RE(clsharp.tmedian.TemporalMedian(), clsharp, 12))

    # If selected, combining ReNoise
    noise_diff = core.std.MakeDiff(clean2, cy)
    if rn:
        expr = "x {a} < 0 x {b} > {p} 0 x {c} - {p} {a} {d} - / * - ? ?".format(a=32*i, b=45*i, c=35*i, d=65*i, p=peak)
        clean1 = core.std.Merge(clean2, core.std.MergeDiff(clean2, Tweak(noise_diff.tmedian.TemporalMedian(), cont=1.008+0.00016*rn)), 0.3+rn*0.035)
        clean2 = core.std.MaskedMerge(clean2, clean1, core.std.Expr([core.std.Expr([clean, clean.std.Invert()], 'x y min')], [expr]))

    # Combining spatial detail enhancement with spatial noise reduction using prepared mask
    noise_diff = noise_diff.std.Binarize().std.Invert()
    clean2 = core.std.MaskedMerge(clean2, clsharp if sharp else clean, core.std.Expr([noise_diff, clean.std.Sobel()], 'x y max'))

    # Combining result of luma and chroma cleaning
    output = core.std.ShufflePlanes([clean2, filt], [0, 1, 2], vs.YUV)
    output = core.std.Merge(c, output, 0.2+0.04*strength) if strength < 20 else output
    return core.std.MergeDiff(output, core.std.MakeDiff(output.warp.AWarpSharp2(128, 3, 1, depth2, 1), output.warp.AWarpSharp2(128, 2, 1, depth, 1))) if depth else output


def STPressoHD(clip, limit=4, bias=20, tlimit=4, tbias=40, conv=[1]*25, thSAD=400, tr=2, back=2, rec=False, chroma=True, analyse_args=None, recalculate_args=None):
    """
    Original STPresso by Didée: https://forum.doom9.org/showthread.php?t=163819
    Dampen the grain just a little, to keep the original look
    Spatial part uses a 5×5 BoxFilter / Temporal part uses MDegrain

    Args:
        limit  (int)   - The spatial part won't change a pixel more than this.
        bias   (int)   - The percentage of the spatial filter that will apply.
        tlimit (int)   - The temporal part won't change a pixel more than this.
        tbias  (int)   - The percentage of the temporal filter that will apply.
        conv   (int[]) - Convolution kernel used in the spatial part.
        thSAD  (int)   - Soft threshold of block sum absolute differences.
                         Low value can result in staggered denoising,
                         High value can result in ghosting and artifacts.
        tr     (int)   - Strength of temporal denoising (0-24). 0 means no temporal processing.
        back   (int)   - After all changes have been calculated, reduce all pixel changes by this value (shift "back" towards original value)
        rec    (bool)  - Recalculate the motion vectors to obtain more precision.
        chroma (bool)  - Whether to process chroma.
    """

    if not isinstance(clip, vs.VideoNode) or clip.format.color_family not in [vs.GRAY, vs.YUV]:
        raise TypeError('STPressoHD: This is not a GRAY or YUV clip!')

    isFLOAT = clip.format.sample_type == vs.FLOAT
    isGRAY = clip.format.color_family == vs.GRAY
    S = core.mvsf.Super if isFLOAT else core.mv.Super
    A = core.mvsf.Analyse if isFLOAT else core.mv.Analyse
    R = core.mvsf.Recalculate if isFLOAT else core.mv.Recalculate
    D1 = core.mvsf.Degrain1 if isFLOAT else core.mv.Degrain1
    D2 = core.mvsf.Degrain2 if isFLOAT else core.mv.Degrain2
    D3 = core.mvsf.Degrain3 if isFLOAT else core.mv.Degrain3
    bia = min(max(bias/100, 0), 1)
    tbia = min(max(tbias/100, 0), 1)
    bs = 32 if clip.width > 2400 else 16 if clip.width > 960 else 8
    pel = 1 if clip.width > 960 else 2
    truemotion = False if clip.width > 960 else True
    chroma = False if isGRAY else chroma
    plane = 4 if chroma else 0
    planes = [0, 1, 2] if chroma else 0

    if analyse_args is None:
        analyse_args = dict(blksize=bs, overlap=bs//2, search=5, chroma=chroma, truemotion=truemotion)
        
    if recalculate_args is None:
        recalculate_args = dict(blksize=bs//2, overlap=bs//4, search=5, chroma=chroma, truemotion=truemotion)

    if tr > 3 and not isFLOAT:
        raise TypeError("STPressoHD: DegrainN is only available in float")
    
    if tr > 0:
        super_b = S(DitherLumaRebuild(clip, 1), hpad=bs, vpad=bs, pel=pel, sharp=1, rfilter=4)
        super_rend = S(clip, hpad=bs, vpad=bs, pel=pel, levels=1, rfilter=1)

        if tr < 4:
            bv1 = A(super_b, isb=True,  delta=1, **analyse_args)
            fv1 = A(super_b, isb=False, delta=1, **analyse_args)
            if tr > 1:
                bv2 = A(super_b, isb=True,  delta=2, **analyse_args)
                fv2 = A(super_b, isb=False, delta=2, **analyse_args)
            if tr > 2:
                bv3 = A(super_b, isb=True,  delta=3, **analyse_args)
                fv3 = A(super_b, isb=False, delta=3, **analyse_args)
        else:
            vec = mvmulti.Analyze(super_b, tr=tr, **analyse_args)
        
        if rec:
            if tr < 4:
                bv1 = R(super_b, bv1, **recalculate_args)
                fv1 = R(super_b, fv1, **recalculate_args)
                if tr > 1:
                    bv2 = R(super_b, bv2, **recalculate_args)
                    fv2 = R(super_b, fv2, **recalculate_args)
                if tr > 2:
                    bv3 = R(super_b, bv3, **recalculate_args)
                    fv3 = R(super_b, fv3, **recalculate_args)    
            else:
                vec = mvmulti.Recalculate(super_b, vec, tr=tr, **recalculate_args)

    bd = clip.format.bits_per_sample
    peak = 1.0 if isFLOAT else (1 << bd) - 1
    i = 0.00392 if isFLOAT else 1 << (bd - 8)
    lim = scale(limit, peak)
    tlim = scale(tlimit, peak)
    back = scale(back, peak)
    LIM = lim/bia - i if limit > 0 else scale(1/bia, peak)
    TLIM = tlim/tbia - i if tlimit > 0 else scale(1/tbia, peak)

    sexpr = 'x y - abs {i} < x x {L1} + y < x {L2} + x {L1} - y > x {L2} - x {j} * y {B} * + ? ? ?'.format(i=i, L1=LIM, L2=lim, j=1-bia, B=bia) if limit > 0 else 'x y - abs {} < x x x y > {} -{} ? - ?'.format(LIM, i, i)
    texpr = 'x y - abs {i} < x x {T1} + y < x {T2} + x {T1} - y > x {T2} - x {j} * y {T} * + ? ? ?'.format(i=i, T1=TLIM, T2=tlim, j=1-tbia, T=tbia) if tlimit > 0 else 'x y - abs {} < x x x y > {} -{} ? - ?'.format(TLIM, i, i)

    if isinstance(conv, list):
        spat = core.std.Convolution(clip, matrix=conv, planes=planes)
    elif callable(conv):
        spat = conv(clip)
    else:
        raise TypeError("STPressoHD: conv must be a list or a function!")
    
    down = core.std.Expr([clip, spat], [sexpr if i in planes else '' for i in range(clip.format.num_planes)])

    if tr > 0:
        if tr == 1:
            temp = D1(spat, super_rend, bv1, fv1, thSAD, plane=plane)
        elif tr == 2:
            temp = D2(spat, super_rend, bv1, fv1, bv2, fv2, thSAD, plane=plane)
        elif tr == 3:
            temp = D3(spat, super_rend, bv1, fv1, bv2, fv2, bv3, fv3, thSAD, plane=plane)
        else:
            temp = mvmulti.DegrainN(spat, super_rend, vec, tr=tr, thsad=thSAD, plane=plane)
        
        diff = core.std.Expr([down, temp, spat], ['x y + z -' if i in planes else '' for i in range(clip.format.num_planes)])
    
        down = core.std.Expr([down, diff], [texpr if i in planes else '' for i in range(clip.format.num_planes)])
    
    if back > 0:
        bexpr = 'x {BK} + y < x {BK} + x {BK} - y > x {BK} - y ? ?'.format(BK=back)
        return core.std.Expr([down, clip], [bexpr if i in planes else '' for i in range(clip.format.num_planes)])
    else:
        return down


def MLDegrain(clip, scale1=1.5, scale2=2, thSAD=400, tr=3, rec=False, chroma=True, soft=[0]*3):
    """
    Multi-Level MDegrain
    Multi level in the sense of using multiple scalings.
    The observation was that when downscaling the source to a smallish resolution, then a vanilla MDegrain can produce a very stable result. 
    Hence, it's an obvious approach to first make a small-but-stable-denoised clip, and then work the way upwards to the original resolution.
    From: https://forum.doom9.org/showthread.php?p=1512413 by Didée

    Args:
        scale1 (float) - Scaling factor between original and medium scale
        scale2 (float) - Scaling factor between medium and small scale
        tr     (int)   - Strength of the denoising (1-24).
        thSAD  (int)   - Soft threshold of block sum absolute differences.
                         Low value can result in staggered denoising,
                         High value can result in ghosting and artifacts.
        rec    (bool)  - Recalculate the motion vectors to obtain more precision.
        chroma (bool)  - Whether to process chroma.
        soft (float[]) - [small, medium, original] ranges from 0 to 1, 0 means disabled, 1 means 100% strength.
                         Do slight sharpening where motionmatch is good, do slight blurring where motionmatch is bad.
    """

    isFLOAT = clip.format.sample_type == vs.FLOAT

    if not isinstance(clip, vs.VideoNode) or clip.format.color_family not in [vs.GRAY, vs.YUV]:
        raise TypeError('MLDegrain: This is not a GRAY or YUV clip!')
    
    if tr > 3 and not isFLOAT:
        raise TypeError("MLDegrain: DegrainN is only available in float")

    w = clip.width
    h = clip.height
    w1 = m4(w / scale1)
    h1 = m4(h / scale1)
    w2 = m4(w1 / scale2)
    h2 = m4(h1 / scale2)
    sm1 = clip.resize.Bicubic(w1, h1) # medium scale
    sm2 = sm1.resize.Bicubic(w2, h2)  # small scale
    D12 = core.std.MakeDiff(sm2.resize.Bicubic(w1, h1), sm1) # residual of (small)<>(medium)
    D10 = core.std.MakeDiff(sm1.resize.Bicubic(w, h), clip)  # residual of (medium)<>(original)
    lev2 = MLD_helper(sm2, sm2, tr, thSAD, rec, chroma, soft[0]) # Filter on smalle scale
    up1 = lev2.resize.Bicubic(w1, h1)
    up2 = up1.resize.Bicubic(w, h)
    M1 = MLD_helper(D12, up1, tr, thSAD, rec, chroma, soft[1])   # Filter on medium scale
    lev1 = core.std.MakeDiff(up1, M1)
    up3 = lev1.resize.Bicubic(w, h)
    M2 = MLD_helper(D10, up2, tr, thSAD, rec, chroma, soft[2])   # Filter on original scale

    return core.std.MakeDiff(up3, M2)


def MLD_helper(clip, srch, tr, thSAD, rec, chroma, soft):
    """ Helper function used in Multi-Level MDegrain"""

    if not isinstance(clip, vs.VideoNode) or clip.format.color_family not in [vs.GRAY, vs.YUV]:
        raise TypeError('MLD_helper: This is not a GRAY or YUV clip!')
    
    isFLOAT = clip.format.sample_type == vs.FLOAT
    isGRAY = clip.format.color_family == vs.GRAY
    S = core.mvsf.Super if isFLOAT else core.mv.Super
    A = core.mvsf.Analyse if isFLOAT else core.mv.Analyse
    R = core.mvsf.Recalculate if isFLOAT else core.mv.Recalculate
    D1 = core.mvsf.Degrain1 if isFLOAT else core.mv.Degrain1
    D2 = core.mvsf.Degrain2 if isFLOAT else core.mv.Degrain2
    D3 = core.mvsf.Degrain3 if isFLOAT else core.mv.Degrain3
    bs = 32 if clip.width > 2400 else 16 if clip.width > 960 else 8
    pel = 1 if clip.width > 960 else 2
    truemotion = False if clip.width > 960 else True
    chroma = False if isGRAY else chroma
    planes = [0, 1, 2] if chroma else [0]
    plane = 4 if chroma else 0

    analyse_args = dict(blksize=bs, overlap=bs//2, search=5, chroma=chroma, truemotion=truemotion)
    recalculate_args = dict(blksize=bs//2, overlap=bs//4, search=5, chroma=chroma, truemotion=truemotion)
    sup1 = S(DitherLumaRebuild(srch, 1), hpad=bs, vpad=bs, pel=pel, sharp=1, rfilter=4)

    if soft > 0:
        if clip.width > 1280:
            RG = core.std.Convolution(clip, matrix=[1, 1, 1, 1, 1, 1, 1, 1, 1], planes=planes)
        elif clip.width > 640:
            RG = core.std.Convolution(clip, matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1], planes=planes)
        else:
            RG = MinBlur(clip, 1, planes)
        RG = core.std.Merge(clip, RG, [soft] if chroma or isGRAY else [soft, 0]) if soft < 1 else RG
        sup2 = S(core.std.Expr([clip, RG], ['x dup y - +'] if chroma or isGRAY else ['x dup y - +', '']), hpad=bs, vpad=bs, pel=pel, levels=1, rfilter=1)
    else:
        RG = clip
        sup2 = S(clip, hpad=bs, vpad=bs, pel=pel, levels=1, rfilter=1)

    if tr < 4:
        bv1 = A(sup1, isb=True,  delta=1, **analyse_args)
        fv1 = A(sup1, isb=False, delta=1, **analyse_args)
        if tr > 1:
            bv2 = A(sup1, isb=True,  delta=2, **analyse_args)
            fv2 = A(sup1, isb=False, delta=2, **analyse_args)
        if tr > 2:
            bv3 = A(sup1, isb=True,  delta=3, **analyse_args)
            fv3 = A(sup1, isb=False, delta=3, **analyse_args)
    else:
        vec = mvmulti.Analyze(sup1, tr=tr, **analyse_args)
        
    if rec:
        if tr < 4:
            bv1 = R(sup1, bv1, **recalculate_args)
            fv1 = R(sup1, fv1, **recalculate_args)
            if tr > 1:
                bv2 = R(sup1, bv2, **recalculate_args)
                fv2 = R(sup1, fv2, **recalculate_args)
            if tr > 2:
                bv3 = R(sup1, bv3, **recalculate_args)
                fv3 = R(sup1, fv3, **recalculate_args)    
        else:
            vec = mvmulti.Recalculate(sup1, vec, tr=tr, **recalculate_args)
    
    if tr < 4:
        if tr == 1:
            return D1(RG, sup2, bv1, fv1, thsad=thSAD, plane=plane)
        elif tr == 2:
            return D2(RG, sup2, bv1, fv1, bv2, fv2, thsad=thSAD, plane=plane)
        else:
            return D3(RG, sup2, bv1, fv1, bv2, fv2, bv3, fv3, thsad=thSAD, plane=plane)
    else:
        return mvmulti.DegrainN(RG, sup2, vec, tr=tr, thsad=thSAD, plane=plane)


def NonlinUSM(clip, z=6.0, power=1.6, sstr=1, rad=9.0, ldmp=0.01, hdmp=0.01):
    """
    From: https://forum.doom9.org/showthread.php?p=1555234 by Didée.
    Non-linear Unsharp Masking, uses a wide-range Gaussian instead of a small-range kernel.

    Args:
        z     (float) - zero point of sharpening.
        power (float) - exponent for non-linear sharpening.
        sstr  (float) - strength of sharpening.
        rad   (float) - radius for gauss.
        ldmp  (float) - "low damp", damping for very small differences.
        hdmp  (float) - "high damp", this damping term has a larger effect than ldmp
                         when the sharp-difference is larger than 1, vice versa.
    
    Examples:
        NonlinUSM(power=4)                              # enhance: for low bitrate sources
        NonlinUSM(z=3, power=4, sstr=1, rad=6)          # enhance less
        NonlinUSM(z=3, sstr=0.5, rad=9, power=1)        # enhance less
        NonlinUSM(z=6, sstr=1.5, rad=0.6).Sharpen(0.3)  # sharpen: for hi-q sources
        NonlinUSM(z=3, sstr=2.5, rad=0.6)               # sharpen: less noise
        NonlinUSM(z=6, power=1, sstr=1, rad=6)          # unsharp
        NonlinUSM(z=6, power=1, rad=2,  sstr=0.7)       # "smoothen" for noisy sources
        NonlinUSM(z=6, power=1, rad=18, sstr=0.5)       # smear: soft glow
        NonlinUSM(z=6, power=4, sstr=1, rad=36)         # local contrast
        NonlinUSM(z=6, power=1, sstr=1, rad=36)         # local contrast
        NonlinUSM(z=16, power=4, sstr=18, rad=6)        # B+W psychedelic
        NonlinUSM(z=16, power=2, sstr=2, rad=36)        # solarized
        NonlinUSM(z=16, power=4, sstr=3, rad=6)         # sepia/artistic
    """

    ldmp = max(ldmp, 0)
    hdmp = max(hdmp, 0)
    bd = clip.format.bits_per_sample
    isFLOAT = clip.format.sample_type == vs.FLOAT
    i = 0.00392 if isFLOAT else 1 << (bd - 8)
    xy = 'x y - {} /'.format(i) if bd != 8 else 'x y -'
    color = clip.format.color_family
    
    if not isinstance(clip, vs.VideoNode):
        raise TypeError("NonlinUSM: This is not a clip!")
    if color == vs.COMPAT:
        raise TypeError("NonlinUSM: COMPAT color family is not supported!")

    tmp = core.std.ShufflePlanes(clip, [0], vs.GRAY) if color in [vs.YUV, vs.YCOCG] else clip
    gauss = core.resize.Bicubic(tmp, m4(clip.width/rad), m4(clip.height/rad)).resize.Bicubic(clip.width, clip.height, filter_param_a=1, filter_param_b=0)
    expr = 'x y = x dup {} dup dup dup abs {} / {} pow swap3 dup * dup {} + / swap2 abs {} + / * * {} * + ?'
    tmp = core.std.Expr([tmp, gauss], [expr.format(xy, z, 1/power, ldmp, hdmp, sstr*z*i)])

    return core.std.ShufflePlanes([tmp, clip], [0, 1, 2], color) if color in [vs.YUV, vs.YCOCG] else tmp


def DetailSharpen(clip, z=4, sstr=1.5, power=4, ldmp=1, mode=1, med=False):
    """
    From: https://forum.doom9.org/showthread.php?t=163598
    Didée: Wanna some sharpening that causes no haloing, without any edge masking?
    
    Args:
        z     (float) - zero point.
        sstr  (float) - strength of non-linear sharpening.
        power (float) - exponent of non-linear sharpening.
        ldmp  (float) - "low damp", to not over-enhance very small differences.
        mode   (int)  - 0: gaussian kernel 1: box kernel
        med   (bool)  - When True, median is used to achieve stronger sharpening.
        
    Examples:
        DetailSharpen() # Original DetailSharpen by Didée.
        DetailSharpen(power=1.5, mode=0, med=True) # Mini-SeeSaw...just without See, and without Saw.
    """
    
    ldmp = max(ldmp, 0)
    bd = clip.format.bits_per_sample
    isFLOAT = clip.format.sample_type == vs.FLOAT
    i = 0.00392 if isFLOAT else 1 << (bd - 8)
    xy = 'x y - {} /'.format(i) if bd != 8 else 'x y -'
    color = clip.format.color_family
    
    if not isinstance(clip, vs.VideoNode):
        raise TypeError("DetailSharpen: This is not a clip!")
    if color == vs.COMPAT:
        raise TypeError("DetailSharpen: COMPAT color family is not supported!")

    tmp = core.std.ShufflePlanes(clip, [0], vs.GRAY) if color in [vs.YUV, vs.YCOCG] else clip

    if mode == 1:
        blur = core.std.Convolution(tmp, matrix=[1, 1, 1, 1, 1, 1, 1, 1, 1])
    else:
        blur = core.std.Convolution(tmp, matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1])
    if med:
        blur = blur.std.Median()

    expr = 'x y = x dup {} dup dup abs {} / {} pow swap2 abs {} + / * {} * + ?'
    tmp = core.std.Expr([tmp, blur], [expr.format(xy, z, 1/power, ldmp, sstr*z*i)])

    return core.std.ShufflePlanes([tmp, clip], [0, 1, 2], color) if color in [vs.YUV, vs.YCOCG] else tmp
    

def Hysteria(clip, sstr=1.0, usemask=True, lthr=7, hthr=20, lcap=192, maxchg=255, minchg=0, showmask=False):
    """
    Hysteria, a line darkening script by Scintilla

    Args:
        sstr    (float) - This is a multiplicative factor for the amounts by which the pixels are darkened.
                          Ordinarily, each pixel is darkened by the difference between its luma value and the average
                          luma value of its brighter spatial neighbours. So if you want more darkening, increase this value.
    
        usemask (bool)  - Whether or not to apply the mask. If False, the entire image will have its edges darkened 
                          instead of just the edges detected in the mask. Could be useful on some sources
                          (specifically, it will often make dark lines look thicker), but you will probably 
                          want to stick to a lower value of strength if you choose to go that route.
    
        lthr    (float) - This is the threshold used for the noisy mask. Increase this value if your mask is picking up 
                          too much noise around the edges, or decrease it if the mask is not being grown thick enough.
    
        hthr    (float) - This is the threshold used for the clean mask. Increase this value if your mask is picking up 
                          too many weaker edges, or decrease it if the mask is not picking up enough.
    
        lcap    (float) - Luma cap, an idea swiped from FLD/VMToon. Any pixels brighter than this value will not be darkened at all,
                          no matter what the rest of the parameters are. This is useful if you have lighter edges that you do not want darkened.
                          0 will result in no darkening at all, while 255 will turn off the cap.
    
        maxchg  (float) - No pixel will be darkened by more than this amount, no matter how high you set the strength parameter.
                          This can be useful if you want to darken your weaker lines more without going overboard on the stronger ones.
                          0 will result in no darkening at all, while 255 will turn off the limiting.
    
        minchg  (float) - Another idea swiped from FLD/VMToon (though in those functions it was called "threshold"). 
                          Any pixels that would have been darkened by less than this amount will instead not be darkened at all.
                          This can be useful if you have noise that is getting darkened slightly. 
                          0 will turn off the thresholding, while 255 will result in no darkening at all.
    
        showmask (bool) - When True, the function will display the current edge mask plus the chroma from the original image.
                          Use this to find the optimal values of lowthresh and highthresh.
    """
    
    color = clip.format.color_family
    isFLOAT = clip.format.sample_type == vs.FLOAT
    bd = clip.format.bits_per_sample
    peak = 1.0 if isFLOAT else (1 << bd) - 1
    lcap = scale(lcap, peak)
    maxchg = scale(maxchg, peak)
    minchg = scale(minchg, peak)
    mat = [1, 2, 1, 2, 4, 2, 1, 2, 1]

    if not isinstance(clip, vs.VideoNode) or color not in [vs.YUV, vs.YCOCG]:
        raise ValueError("Hysteria: This is not a YUV or YCOCG clip!")

    tmp = core.std.ShufflePlanes(clip, [0], vs.GRAY)

    noisymask = EdgeDetect(tmp, mode='cartoon', lthr=lthr, hthr=lthr)
    cleanmask = EdgeDetect(tmp, mode='cartoon', lthr=hthr, hthr=hthr)
    themask = core.misc.Hysteresis(cleanmask, noisymask).std.Inflate().std.Convolution(matrix=mat).std.Convolution(matrix=mat).std.Deflate()

    if showmask:
        return core.std.ShufflePlanes([themask.std.Levels(min_in=0, max_in=peak, min_out=scale(80, peak), max_out=peak), clip], [0, 1, 2], color)

    diffs = core.std.Expr([core.std.Inflate(tmp), tmp], ['x y - {} *'.format(sstr)])

    darkened = core.std.Expr([tmp, diffs], ['x dup {l} > 0 y {max} > {max} y {min} < 0 y ? ? ? -'.format(l=lcap, max=maxchg, min=minchg)])

    darkened = core.std.ShufflePlanes([darkened, clip], [0, 1, 2], color)

    return core.std.MaskedMerge(clip, darkened, themask, [0]) if usemask else darkened

    
def SuperToon(clip, power=.69, mode=0, nthr=4, ncap=32, lthr=0, hthr=255, lcap=255, cont=188, show=False):
    """
    From: https://forum.doom9.org/showthread.php?t=163987 by Haiden
    Haiden: SuperToon is my attempt to optimize/speed up the previous versions of mfToon, vmToon, etc.

    Note: Unfilter related part not implemented, mode 3 is the original mode 4.

    Args:
        power (float) - 0.2-1.5 is usually the desired range. Positive values darken lines, negative values brighten them.
                        When setting the nthr, ncap, lcap, and pretty much any other setting, you can set this to an extreme value,
                        this way you can easily see (as if the edge mask was overlayed on the original) what SuperToon will be darkening.
                        (Whether it would be the lines you want it to darken or the noise you didn't want it to detect)
                                   
        mode   (int)  - Ranges from 0-3, tells SuperToon to try different methods. No mode is always better than the other,
                        its just that one mode can fare better for one source and not another.
                        So the user has the option of which mode is best for their source.
                        If you choose a number outside of the range it will default to 0.
                        mode = 0: Just applies a very simple edge detection mask with no effective noise detection.
                        mode = 1: Creates an edge mask and filters out noise by finding the average of all neighboring pixels
                        and determines if the current pixel is an "outlier" and not noise, meaning it is an edge to darken.
                        mode = 2: Silmilar to mode 1 but does things differently, it runs a sieve on all the neighboring pixels
                        and then resets the edge mask to be the median of the results from that sieve.
                        Sometimes its the only one that can seem to pull it off that the others can't.
                        mode = 3: Works to ignore non-edge lines like shadow lines so it doesn't darken lines that weren't meant to be darkened.
                        If you're getting something like black outlines around stars or snow, this mode is very good at preventing that.

        lthr  (float) - 0-255, raising this value makes darkened lines thicker and helps with edge detection,
                        but it'll usually give very very thick lines if you set this too high.

        hthr  (float) - 0-255, lowering of this value has a stronger impact on line detection than the low threshold.

        lcap  (float) - Brightness cap which tells the function to only darken pixels under a certain luma value.
                        This is mostly intended to give more control to modes 0-2, mode 3 shouldn't ever need it. Setting it to 255 basiaclly disables this.

        cont  (float) - Pixels are darkened by an amount relative to their spatial luma change. Pixels with very low
                        luma change (like noise) is darkened very very slightly. Pixels with large luma change (like a black line)
                        are greatly darkened. While moderate change (like a shadowline) is partially darkened.
                        This value controls the contrast of these changes. Lowering this values makes it so that both low luma
                        and high luma changes are subtracted by near the same amount. If you lower this value "power" will have
                        to be raised to show a visible difference when used with ncap, which is very effective at removing noise.

        nthr  (float) - Threshold to try to detect noise around the lines to be sharpened, however setting it too
                        high and it'll begin to confuse the lines you want to sharpen as noise. This parameter works
                        differently depending on the mode so in certain modes it works better in certain ranges.
                        It is used to clip lower values from the generated mask (as noise).
                        For modes 1 and 2 it is used to clean the noise from the lines.
                        For mode 3 it is used to overlay mode 0's mask onto mode 3's mask, lower value add more weight to mode 0's mask.

        ncap  (float) - 0-255, increasing it is the best (as far as filter speed) at preventing noise from being sharpened,
                        however setting it too high and you won't be darkening any lines. This setting helps determine how
                        "sensitive" it should be when looking at the edge mask. If it looks like the entire picture is darker,
                        then it is likely this value is too low. 18 is a good starting point for most sources I've tried.
    """
    
    if not isinstance(clip, vs.VideoNode) or clip.format.color_family != vs.YUV:
        raise ValueError("SuperToon: This is not a YUV clip!")

    bd = min(clip.format.bits_per_sample, 16)
    isFLOAT = clip.format.sample_type == vs.FLOAT
    mid  = 1 << (bd - 1)
    peak = (1 << bd) - 1
    R = core.rgvs.RemoveGrain
    lthr = scale(lthr, peak)
    hthr = scale(hthr, peak)
    minin = scale(128 + nthr, peak)
    cont = scale(cont, peak)
    ncap = scale(ncap, peak)
    lcap = scale(lcap, peak)
    sthr = scale(nthr, peak)

    tmp = clip.fmtc.bitdepth(bits=16) if isFLOAT else clip
    mask = core.std.MakeDiff(tmp.std.Maximum(threshold=hthr), tmp.std.Minimum(threshold=lthr))

    if mode == 1:
        mask = mask.avs.mt_lutf(mask, mode='avg', expr='y x - abs x / {} 255 / > y {} ?'.format(nthr, mid), U=1, V=1)
    elif mode == 2:
        mask = mask.avs.mt_luts(mask, pixels=core.avs.mt_square(4), mode='med', expr='y x - abs {} < x y > & {} y x + {} - ?'.format(sthr, mid, mid), U=1, V=1)
    elif mode == 3:
        diff = core.std.MakeDiff(core.std.Expr([clip.ctmf.CTMF(radius=2, planes=[0]), clip], ['x y max', '']), clip, [0])
        mask = core.std.Expr([diff, mask], ['x y {} - {} / +'.format(mid, nthr), ''])

    mask = R(R(core.std.ShufflePlanes(mask, [0], vs.GRAY), [1]).std.Levels(min_in=minin, max_in=peak, min_out=0, max_out=cont), [1])
    sharp = core.std.Expr([core.std.ShufflePlanes(clip, [0], vs.GRAY), mask], ['y {} < x {} > or x dup y {} - {} * - ?'.format(ncap, lcap, ncap, power)])
    sharp = sharp.fmtc.bitdepth(flt=1) if isFLOAT else sharp

    return mask if show else core.std.ShufflePlanes([sharp, clip], [0, 1, 2], vs.YUV)


def EdgeDetect(clip, mode="kirsch", lthr=0, hthr=255, multi=1):
    """
    Generates edge mask based on convolution kernel.
    The result of the convolution is then thresholded with lthr and hthr.

    Args:
        mode (string) - Chooses a predefined kernel used for the mask computing.
                        Vaild choices are "sobel", "prewitt", "scharr", "kirsch",
                        "robinson", "roberts", "cartoon", "min/max", "laplace", 
                        "frei-chen", "kayyali", "LoG", "FDOG" and "TEdge".
        lthr  (float) - 0-255, low threshold. Anything below lthr will be set to 0.
        hthr  (float) - 0-255, high threshold. Anything above hthr will be set to range_max.
        multi (float) - Multiply all pixels by this before thresholding. This can be used
                        to increase or decrease the intensity of edges in the output.
    """

    if not isinstance(clip, vs.VideoNode):
        raise TypeError("EdgeDetect: This is not a clip!")
    if clip.format.color_family == vs.COMPAT:
        raise TypeError("EdgeDetect: COMPAT color family is not supported!")

    if clip.format.color_family in [vs.YUV, vs.YCOCG]:
        clip = core.std.ShufflePlanes(clip, 0, vs.GRAY)

    bd = clip.format.bits_per_sample
    isFLOAT = clip.format.sample_type == vs.FLOAT
    peak = (1 << bd) - 1 if not isFLOAT else 1.0
    hthr = min(max(scale(hthr, peak), 0), peak)
    lthr = min(max(scale(lthr, peak), 0), hthr)

    if mode == "sobel":
        mask = core.std.Sobel(clip, scale=multi)
    elif mode == "prewitt":
        mask = core.std.Prewitt(clip, scale=multi)
    elif mode == "scharr":
        gx = core.std.Convolution(clip, [-3, 0, 3, -10, 0, 10, -3, 0, 3], divisor=3, saturate=False)
        gy = core.std.Convolution(clip, [-3, -10, -3, 0, 0, 0, 3, 10, 3], divisor=3, saturate=False)
        mask = core.std.Expr([gx, gy], 'x dup * y dup * + sqrt')
    elif mode == "kirsch":
        N  = core.std.Convolution(clip, [5, 5, 5, -3, 0, -3, -3, -3, -3], divisor=3, saturate=False)
        NW = core.std.Convolution(clip, [5, 5, -3, 5, 0, -3, -3, -3, -3], divisor=3, saturate=False)
        W  = core.std.Convolution(clip, [5, -3, -3, 5, 0, -3, 5, -3, -3], divisor=3, saturate=False)
        SW = core.std.Convolution(clip, [-3, -3, -3, 5, 0, -3, 5, 5, -3], divisor=3, saturate=False)
        S  = core.std.Convolution(clip, [-3, -3, -3, -3, 0, -3, 5, 5, 5], divisor=3, saturate=False)
        SE = core.std.Convolution(clip, [-3, -3, -3, -3, 0, 5, -3, 5, 5], divisor=3, saturate=False)
        E  = core.std.Convolution(clip, [-3, -3, 5, -3, 0, 5, -3, -3, 5], divisor=3, saturate=False)
        NE = core.std.Convolution(clip, [-3, 5, 5, -3, 0, 5, -3, -3, -3], divisor=3, saturate=False)
        mask = core.std.Expr([N, NW, W, SW, S, SE, E, NE], ['x y max z max a max b max c max d max e max'])
    elif mode == "robinson":
        g1 = core.std.Convolution(clip, [-1, 0, 1, -2, 0, 2, -1, 0, 1], saturate=False)
        g2 = core.std.Convolution(clip, [0, 1, 2, -1, 0, 1, -2, -1, 0], saturate=False)
        g3 = core.std.Convolution(clip, [1, 2, 1, 0, 0, 0, -1, -2, -1], saturate=False)
        g4 = core.std.Convolution(clip, [2, 1, 0, 1, 0, -1, 0, -1, -2], saturate=False)
        mask = core.std.Expr([g1, g2, g3, g4], ['x y max z max a max'])
    elif mode == "roberts":
        mask = core.std.Convolution(clip, matrix=[0, 0, 0, 0, 2, -1, 0, -1, 0], saturate=False)
    elif mode == "cartoon":
        mask = core.std.Convolution(clip, [0, -2, 1, 0, 1, 0, 0, 0, 0], saturate=True)
    elif mode == "min/max":
        mask = core.std.Expr([core.std.Maximum(clip), core.std.Minimum(clip)], 'x y -')
    elif mode == "laplace":
        mask = core.std.Convolution(clip, [-1, -1, -1, -1, 8, -1, -1, -1, -1], saturate=False)
    elif mode == "frei-chen":
        gx = core.std.Convolution(clip, [-7, 0, 7, -10, 0, 10, -7, 0, 7], divisor=7, saturate=False)
        gy = core.std.Convolution(clip, [-7, -10, -7, 0, 0, 0, 7, 10, 7], divisor=7, saturate=False)
        mask = core.std.Expr([gx, gy], 'x dup * y dup * + sqrt')
    elif mode == "kayyali":
        mask = core.std.Convolution(clip, [-6, 0, 6, 0, 0, 0, 6, 0, -6], saturate=False)
    elif mode == "LoG":
        mask = core.std.Convolution(clip, [0, 0, -1, 0, 0, 0, -1, -2, -1, 0, -1, -2, 16, -2, -1, 0, -1, -2, -1, 0, 0, 0, -1, 0, 0], saturate=False)
    elif mode == "FDOG":
        gx = core.std.Convolution(clip, [1, 1, 0, -1, -1, 2, 2, 0, -2, -2, 3, 3, 0, -3, -3, 2, 2, 0, -2, -2, 1, 1, 0, -1, -1], divisor=2, saturate=False)
        gy = core.std.Convolution(clip, [-1, -2, -3, -2, -1, -1, -2, -3, -2, -1, 0, 0, 0, 0, 0, 1, 2, 3, 2, 1, 1, 2, 3, 2, 1], divisor=2, saturate=False)
        mask = core.std.Expr([gx, gy], 'x dup * y dup * + sqrt')
    elif mode == "TEdge":
        gx = core.std.Convolution(clip, [4, -25, 0, 25, -4], divisor=2, saturate=False, mode='h')
        gy = core.std.Convolution(clip, [-4, 25, 0, -25, 4], divisor=2, saturate=False, mode='v')
        mask = core.std.Expr([gx, gy], 'x dup * y dup * + sqrt')
    else:
        raise ValueError("EdgeDetect: Unsupported mode!")

    if multi != 1 and isinstance(multi, (int, float)) and mode not in ["sobel", "prewitt"]:
        mask = core.std.Expr([mask], ['x {} *'.format(multi)])
    
    if lthr > 0 or hthr < peak:
        mask = core.std.Expr([mask], ['x {hthr} > {peak} x {lthr} <= 0 x ? ?'.format(lthr=lthr, hthr=hthr, peak=peak)])
        
    return mask


def JohnFPS(clip, num=None, den=None, pre=None, pel=None, sharp=2, blksize=16, overlap=8, blend=False, ml=200, analyse_args=None, recalculate_args=None):
    """
    From: https://forum.doom9.org/showthread.php?p=1847109.
    Motion Protected FPS converter script by johnmeyer.
    Slightly modified interface by Manolito, and a smidgen more by ssS.

    Args:
        num     (int) - Output framerate numerator.
        den     (int) - Output framerate denominator.
        pre    (clip) - pre-filtered clip used in motion vectors calculation.
        pel     (int) - accuracy of the motion estimation.
        sharp   (int) - subpixel interpolation method for pel > 1. (sharp = 3 → nnedi3)
        blksize (int) - blksize used in motion vectors calculation.
        overlap (int) - overlap used in motion vectors calculation.
        blend  (bool) - Whether to blend frames at scene change.
        ml      (int) - mask scale parameter. Greater values correspond to more weak occlusion mask.
    """

    isFLOAT = clip.format.sample_type == vs.FLOAT
    RG = core.rgsf.RemoveGrain if isFLOAT else core.rgvs.RemoveGrain
    A = core.mvsf.Analyse if isFLOAT else core.mv.Analyse
    S = core.mvsf.Super if isFLOAT else core.mv.Super
    R = core.mvsf.Recalculate if isFLOAT else core.mv.Recalculate
    F = core.mvsf.FlowFPS if isFLOAT else core.mv.FlowFPS
    enum = clip.fps.numerator
    eden = clip.fps.denominator
    w = clip.width
    h = clip.height

    if not isinstance(clip, vs.VideoNode) or clip.format.color_family not in [vs.GRAY, vs.YUV]:
        raise TypeError("JohnFPS: This is not a GRAY or YUV clip!")

    if isinstance(num, float) or isinstance(den, float):
        raise ValueError("JohnFPS: Please use exact fraction instead of float.")

    if num is None and den is None:
        enum *= 2
    elif num is not None and den is None:
        enum = num
        eden = 1
    elif num is None and den is not None:
        raise ValueError("JohnFPS: denominator must be used with numerator.")
    else:
        enum = num
        eden = den

    if analyse_args is None:
        analyse_args = dict(blksize=blksize, overlap=overlap, search=5, searchparam=3, dct=5)

    if recalculate_args is None:
        recalculate_args = dict(blksize=blksize//2, overlap=overlap//2, search=5, dct=5, thsad=100)

    if pre is None:
        pre = RG(clip, [22])

    if pel is None:
        pel = 1 if w > 960 else 2

    if pel < 2:
        sharp = min(sharp, 2)
    
    ppp = pel > 1 and sharp > 2
    pre = DitherLumaRebuild(pre, 1)

    if ppp:
        cshift = 0.25 if pel == 2 else 0.375
        pclip  = nnrs.nnedi3_resample(pre,  w * pel, h * pel, cshift, cshift, nns=4)
        pclip2 = nnrs.nnedi3_resample(clip, w * pel, h * pel, cshift, cshift, nns=4)
        supero = S(clip, hpad=blksize, vpad=blksize, pel=pel, pelclip=pclip2, rfilter=1, levels=1)
        superb = S(pre,  hpad=blksize, vpad=blksize, pel=pel, pelclip=pclip,  rfilter=4)
    else:
        supero = S(clip, hpad=blksize, vpad=blksize, pel=pel, rfilter=1, sharp=sharp, levels=1)
        superb = S(pre,  hpad=blksize, vpad=blksize, pel=pel, rfilter=4, sharp=1)

    bv = A(superb, isb=True,  **analyse_args)
    fv = A(superb, isb=False, **analyse_args)
    bv = R(superb, bv, **recalculate_args)
    fv = R(superb, fv, **recalculate_args)
    
    return F(clip, supero, bv, fv, num=enum, den=eden, blend=blend, ml=ml)

def SpotLess(clip, chroma=True, rec=False, analyse_args=None, recalculate_args=None):
    """
    From: https://forum.doom9.org/showthread.php?p=1402690 by Didée

    ChaosKing:
        In my experience this filter works very good as a prefilter for SMDegrain(). 
        Filtering only luma seems to help to avoid ghost artifacts.

    Args:
        chroma (bool) - Whether to process chroma.
        rec    (bool) - Recalculate the motion vectors to obtain more precision.
    """
    # modified from lostfunc: https://github.com/theChaosCoder/lostfunc/blob/v1/lostfunc.py#L10

    if not isinstance(clip, vs.VideoNode) or clip.format.color_family not in [vs.GRAY, vs.YUV]:
        raise TypeError("SpotLess: This is not a GRAY or YUV clip!")

    isFLOAT = clip.format.sample_type == vs.FLOAT
    isGRAY = clip.format.color_family == vs.GRAY
    chroma = False if isGRAY else chroma
    planes = [0, 1, 2] if chroma else [0]
    A = core.mvsf.Analyse if isFLOAT else core.mv.Analyse
    C = core.mvsf.Compensate if isFLOAT else core.mv.Compensate
    S = core.mvsf.Super if isFLOAT else core.mv.Super
    R = core.mvsf.Recalculate if isFLOAT else core.mv.Recalculate
    bs = 32 if clip.width > 2400 else 16 if clip.width > 960 else 8
    pel = 1 if clip.width > 960 else 2
    sup = S(clip, pel=pel, sharp=1, rfilter=4)

    if analyse_args is None:
        analyse_args = dict(blksize=bs, overlap=bs//2, search=5)

    if recalculate_args is None:
        recalculate_args = dict(blksize=bs//2, overlap=bs//4, search=5)

    bv1 = A(sup, isb=True,  delta=1, **analyse_args)
    fv1 = A(sup, isb=False, delta=1, **analyse_args)

    if rec:
        bv1 = R(sup, bv1, **recalculate_args)
        fv1 = R(sup, fv1, **recalculate_args)

    bc1 = C(clip, sup, bv1)
    fc1 = C(clip, sup, fv1)
    fcb = core.std.Interleave([fc1, clip, bc1])

    return fcb.tmedian.TemporalMedian(1, planes)[1::3]


def HQDeringmod(clip, p=None, ringmask=None, mrad=1, msmooth=1, incedge=False, mthr=57, minp=1, nrmode=None, sharp=1, drrep=13,
                thr=12., elast=2., darkthr=None, sbsize=None, sosize=None, sigma=128, sigma2=None, planes=[0], show=False):
    """
    HQDering modded by mawen1250

    Requirements: Miscellaneous Filters, RGVS/RGSF, CTMF, DFTTEST

    Applies deringing by using a smart smoother near edges (where ringing occurs) only

    Parameters:
    mrad    (int)   - Expanding of edge mask, higher value means more aggressive processing.
    msmooth (int)   - Inflate of edge mask, smooth boundaries of mask.
    incedge (bool)  - Whether to include edge in ring mask, by default ring mask only include area near edges.
    mthr    (int)   - Threshold of edge mask, lower value means more aggressive processing. Or define your own mask clip "ringmask".
                      But for strong ringing, lower value will treat some ringing as edge, which protects this ringing from being processed.
    minp    (int)   - Inpanding of edge mask, higher value means more aggressive processing.
    nrmode  (int)   - Kernel of dering - 0: dfttest 1: MinBlur(radius=1), 2: MinBlur(radius=2), 3: MinBlur(radius=3). Or define your own smoothed clip "p".
    sharp   (int)   - Whether to use contra-sharpening to resharp deringed clip, 1-3 represents radius, 0 means no sharpening.
    drrep   (int)   - Use repair for details retention, recommended values are 24/13/12/1.
    thr     (float) - Threshold to limit filtering diff.
    elast   (float) - Elasticity of the soft threshold.
                      Larger "thr" will result in more pixels being taken from processed clip
                      Larger "thr" will result in less pixels being taken from clip clip
                      Larger "elast" will result in more pixels being blended from processed&clip clip, for smoother merging
    darkthr (float) - Threshold for darker area near edges, set it lower if you think deringing destroys too much lines, etc.
                      When "darkthr" is not equal to "thr", "thr" limits darkening while "darkthr" limits brightening
    sigma   (float) - dfttest: sigma for medium frequecies
    sigma2  (float) - dfttest: sigma for low&high frequecies
    sbsize  (int)   - dfttest: length of the sides of the spatial window
    sosize  (int)   - dfttest: spatial overlap amount
    planes  (int[]) - Whether to process the corresponding plane. The other planes will be passed through unchanged.
    show    (bool)  - Whether to output mask clip instead of filtered clip.
    """
    # Modified from havsfunc: https://github.com/HomeOfVapourSynthEvolution/havsfunc/blob/r31/havsfunc.py#L703

    bd = min(clip.format.bits_per_sample, 16)
    isFLOAT = clip.format.sample_type == vs.FLOAT
    if4 = clip.width > 960
    peak = (1 << bd) - 1
    mthr = scale(mthr, peak)
    R = core.rgsf.Repair if isFLOAT else core.rgvs.Repair

    if not isinstance(clip, vs.VideoNode):
        raise TypeError("HQDeringmod: This is not a clip!")
    if clip.format.color_family in [vs.RGB, vs.COMPAT]:
        raise TypeError("HQDeringmod: RGB and COMPAT color family are not supported!")
    if p is not None and (not isinstance(p, vs.VideoNode) or p.format.id != clip.format.id):
        raise TypeError("HQDeringmod: p must be the same format as clip!")
    if p is not None and (p.width != clip.width or p.height != clip.height):
        raise TypeError("HQDeringmod: p must be the same size as clip!")
    if ringmask is not None and not isinstance(ringmask, vs.VideoNode):
        raise TypeError("HQDeringmod: ringmask must be a clip!")
    if ringmask is not None and (ringmask.width != clip.width or ringmask.height != clip.height):
        raise TypeError("HQDeringmod: ringmask must be the same size as clip!")
    if nrmode is None:
        nrmode = 2 if if4 else 1
    if sbsize is None:
        sbsize = 8 if if4 else 6
    if sosize is None:
        sosize = 6 if if4 else 4
    if darkthr is None:
        darkthr = thr / 4
    if sigma2 is None:
        sigma2 = sigma / 8
    if clip.format.num_planes == 1:
        planes = [0]
    if isinstance(planes, int):
        planes = [planes]

    # Kernel: Smoothing
    if p is None:
        if nrmode == 0:
            ss = "0.0:{s2} 0.05:{s} 0.5:{s} 0.75:{s2} 1.0:0.0".format(s=sigma, s2=sigma2)
            p = core.dfttest.DFTTest(clip, sbsize=sbsize, sosize=sosize, tbsize=1, sstring=ss)
        else:
            p = MinBlur(clip, nrmode, planes)

    # Post-Process: Contra-Sharpening
    sclp = p if sharp <= 0 else ContraSharpening(p, clip, sharp, drrep, planes)

    # Post-Process: Repairing
    repclp = sclp if drrep <= 0 else R(clip, sclp, [drrep if i in planes else 0 for i in range(clip.format.num_planes)])

    # Post-Process: Limiting
    limitclp = mvf.LimitFilter(repclp, clip, thr=thr, elast=elast, brighten_thr=darkthr, planes=planes)

    # Post-Process: Ringing Mask Generating
    if ringmask is None:
        luma = core.std.ShufflePlanes(clip, [0], vs.GRAY)
        if isFLOAT:
            luma = luma.fmtc.bitdepth(bits=16)
        sobelm = core.std.Sobel(luma).std.Expr(['x {} < 0 x ?'.format(mthr)])
        fmask = core.misc.Hysteresis(core.std.Median(sobelm), sobelm)
        omask = haf.mt_expand_multi(fmask, sw=mrad, sh=mrad) if mrad > 0 else fmask
        if msmooth > 0:
            omask = haf.mt_inflate_multi(omask, radius=msmooth)
        if incedge:
            ringmask = omask
        else:
            if minp > 3:
                imask = core.std.Minimum(fmask).std.Minimum()
            elif minp > 2:
                imask = core.std.Inflate(fmask).std.Minimum().std.Minimum()
            elif minp > 1:
                imask = core.std.Minimum(fmask)
            elif minp > 0:
                imask = core.std.Inflate(fmask).std.Minimum()
            else:
                imask = fmask
            expr = 'x {} y - * {} /'.format(peak, peak)
            ringmask = core.std.Expr([omask, imask], [expr])
        if isFLOAT:
            ringmask = ringmask.fmtc.bitdepth(flt=1)

    # Mask Merging & Output
    return ringmask if show else core.std.MaskedMerge(clip, limitclp, ringmask, planes)


def MaskedDHA(clip, rx=2, ry=2, darkstr=1, brightstr=1, lowsens=50, highsens=50, maskpull=48, maskpush=192, ss=1, showmask=False):
    """
    From: https://forum.doom9.org/showthread.php?t=148498 by 'Orum.
    A combination of the best of DeHalo_alpha and BlindDeHalo3, plus a few minor tweaks to the masking.

    Args:
        rx, ry (float)
        As usual, the radii for halo removal.
        Note: this function is rather sensitive to the radius settings.
        Set it as low as possible! If radius is set too high, it will start missing small spots.
        
        darkkstr, brightstr (float)
        The strength factors for processing dark and bright halos. Default 1.0 both for symmetrical processing.
        On Anime, darkstr 0.4~0.8 sometimes might be better. In General, the function seems to preserve dark lines rather good.

        lowsens, highsens (float)
        Sensitivity settings, not that easy to describe them exactly...
        In a sense, they define a window between how weak an achieved effect has to be to get fully accepted,
        and how strong an achieved effect has to be to get fully discarded.

        ss (float) Supersampling factor, to avoid creation of aliasing.

        maskpull, maskpush (float) The two new parameters are to adjust the masking.
    """
    
    if not isinstance(clip, vs.VideoNode):
        raise TypeError("MaskedDHA: This is not a clip!")
    if clip.format.color_family in [vs.RGB, vs.COMPAT]:
        raise TypeError("MaskedDHA: RGB and COMPAT color family are not supported!")
    
    w = clip.width
    h = clip.height
    ss = max(ss, 1)
    rx = max(rx, 1)
    ry = max(ry, 1)
    color = clip.format.color_family
    darkstr = max(min(darkstr, 1), 0)
    lowsens = max(min(lowsens, 100), 0)
    highsens = max(min(highsens, 100), 0)
    maskpull = max(min(maskpull, 254), 0)
    maskpush = max(min(maskpush, 255), maskpull + 1)
    isFLOAT = clip.format.sample_type == vs.FLOAT
    R = core.rgsf.Repair if isFLOAT else core.rgvs.Repair
    bd = clip.format.bits_per_sample
    peak = 1.0 if isFLOAT else (1 << bd) - 1
    lowsens = scale(lowsens, peak)
    maskpull = scale(maskpull, peak)
    maskpush = scale(maskpush, peak)
    r = 1.0 if isFLOAT else 1 << bd
    tmp = core.std.ShufflePlanes(clip, [0], vs.GRAY) if color in [vs.YUV, vs.YCOCG] else clip
    sm = core.resize.Bicubic(tmp, m4(w/rx), m4(h/ry))
    lg = sm.resize.Bicubic(w, h, filter_param_a=1, filter_param_b=0)
    chl = core.std.Expr([tmp.std.Maximum(), tmp.std.Minimum()], 'x y -')
    lhl = core.std.Expr([ lg.std.Maximum(),  lg.std.Minimum()], 'x y -')
    mask_i = core.std.Expr([lhl, chl], 'y x - y 0.000001 + / {} * {} - y {} + {} / {} + *'.format(peak, lowsens, r, r * 2, highsens/100))
    mask_f = core.std.Expr([sm.std.Maximum(), sm.std.Minimum()], 'x y - 4 *').std.Convolution(matrix=[1]*9).resize.Bicubic(w, h, filter_param_a=1, filter_param_b=0)
    mask_f = core.std.Expr(mask_f, '{} {} {} - {} {} - - / x {} - *'.format(peak, peak, maskpull, peak, maskpush, maskpull))
    mmg = core.std.MaskedMerge(lg, tmp, mask_i)

    if showmask:
        return mask_f

    if ss == 1:
        ssc = R(tmp, mmg, [1])
    else:
        ssc = tmp.resize.Spline36(m4(w * ss), m4(h * ss))
        ssc = core.std.Expr([ssc, mmg.std.Maximum().resize.Bicubic(m4(w * ss), m4(h * ss))], 'x y min')
        ssc = core.std.Expr([ssc, mmg.std.Minimum().resize.Bicubic(m4(w * ss), m4(h * ss))], 'x y max')
        ssc = core.resize.Spline36(ssc, w, h)

    umfc = core.std.Expr([tmp, ssc], 'x y < x dup y - {} * - x dup y - {} * - ?'.format(darkstr, brightstr))
    mfc = core.std.MaskedMerge(tmp, umfc, mask_f)

    return core.std.ShufflePlanes([mfc, clip], [0, 1, 2], color) if color in [vs.YUV, vs.YCOCG] else mfc


def daamod(c, nsize=None, nns=None, qual=None, pscrn=None, exp=None, opencl=False, device=None, rep=9):
    """Anti-aliasing with contra-sharpening by Didée, modded by GMJCZP"""

    if not isinstance(c, vs.VideoNode):
        raise TypeError("daamod: This is not a clip")
    if c.format.color_family == vs.COMPAT:
        raise TypeError("daamod: COMPAT color family is not supported!")

    isFLOAT = c.format.sample_type == vs.FLOAT
    R = core.rgsf.Repair if isFLOAT else core.rgvs.Repair
    V = core.rgsf.VerticalCleaner if isFLOAT else core.rgvs.VerticalCleaner

    if opencl:
        NNEDI3 = core.nnedi3cl.NNEDI3CL
        nnedi3_args = dict(nsize=nsize, nns=nns, qual=qual, pscrn=pscrn, device=device)
    else:
        NNEDI3 = core.znedi3.nnedi3 if hasattr(core, 'znedi3') and not isFLOAT else core.nnedi3.nnedi3
        nnedi3_args = dict(nsize=nsize, nns=nns, qual=qual, pscrn=pscrn, exp=exp)

    nn = NNEDI3(c, field=3, **nnedi3_args)
    dbl = core.std.Merge(nn[::2], nn[1::2])
    dblD = core.std.MakeDiff(c, dbl)
    shrpD = dbl.std.MakeDiff(dbl.std.Convolution(matrix=[1]*9 if c.width > 1000 else [1, 2, 1, 2, 4, 2, 1, 2, 1]))
    shrpD = V(shrpD, mode=2)
    DD = R(shrpD, dblD, [rep])
    return core.std.MergeDiff(dbl, DD)


def LUSM(clip, blur=1, thr=3.0, elast=4.0, brighthr=None):
    """
    Limited USM - apply a simple USM and then use LimitFilter from mvsfunc to limit the sharpening effect.
    
    Args:
        blur (int, clip) - 0 to 2, weakest to strongest, or supply your own blurred clip.
        thr      (float) - threshold to limit filtering diff.
        elast    (float) - elasticity of the soft threshold.
        brighthr (float) - threshold for filtering diff that brighten the image.
        Larger thr will result in more pixels being taken from processed clip.
        Larger thr will result in less pixels being taken from input clip.
        Larger elast will result in more pixels being blended from processed&input clip, for smoother merging.
    """
        
    if not isinstance(clip, vs.VideoNode):
        raise TypeError("LUSM: This is not a clip!")
    if clip.format.color_family == vs.COMPAT:
        raise TypeError("LUSM: COMPAT color family is not supported!")
    if blur == 0:
        c2 = MinBlur(clip)
    elif blur == 1:
        c2 = core.std.Convolution(clip, matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1])
    elif blur == 2:
        c2 = core.std.Convolution(clip, matrix=[1, 1, 1, 1, 1, 1, 1, 1, 1])
    elif isinstance(blur, vs.VideoNode):
        c2 = blur
    else:
        raise TypeError("LUSM: blur must be 0-2 or a clip!")
    
    sharp = core.std.Expr([clip, c2], ['x dup y - +', ''] if clip.format.color_family in [vs.YUV, vs.YCOCG] else ['x dup y - +'])
    
    return mvf.LimitFilter(sharp, clip, thr=thr, elast=elast, brighten_thr=brighthr)


### Utility functions below
def Overlay(c1, c2, mask=None, opacity=1.0, mode='blend'):
    # Simplified Overlay(), does not perform any checking or fitting.
    # Users need to take care of inputs themselves.

    bd = c1.format.bits_per_sample
    peak = (1 << bd) - 1 if c1.format.sample_type != vs.FLOAT else 1.0
    mode = mode.lower()

    if mask is None:
        if isinstance(c2, list):
            clip2 = c2[0]
            mask = c2[1]
        else:
            clip2 = c2
            mask = core.std.BlankClip(c2)
    else:
        clip2 = c2
        mask = mask

    clip1 = c1

    merg = core.std.MaskedMerge(clip2, clip1, mask)

    if mode != 'blend':
        if mode == 'add':
            merg = core.std.Expr([clip1, merg], ['x y + {} min'.format(peak)])
            merg = core.std.MaskedMerge(merg, clip1, mask)
        elif mode == 'substract':
            merg = core.std.Expr([clip1, merg], ['x y - 0 max'])
            merg = core.std.MaskedMerge(merg, clip1, mask)
        elif mode == 'difference':
            merg = core.std.Expr([clip1, merg], ['x y - abs'])
            merg = core.std.MaskedMerge(merg, clip1, mask)
        elif mode == 'multiply':
            merg = core.std.Expr([clip1, merg], expr=['x y * {} /'.format(peak)])
            merg = core.std.MaskedMerge(merg, clip1, mask)
        elif mode == 'divide':
            merg = core.std.Expr([clip1, merg], expr=['{} x * y {} + /'.format(peak, 1e-6)])
            merg = core.std.MaskedMerge(merg, clip1, mask)
        elif mode == 'lighten':
            merg = core.std.Expr([clip1, merg], expr=['x y max'])
            merg = core.std.MaskedMerge(merg, clip1, mask)
        elif mode == 'darken':
            merg = core.std.Expr([clip1, merg], expr=['x y min'])
            merg = core.std.MaskedMerge(merg, clip1, mask)

    if opacity != 1:
        merg = core.std.Merge(clip1, merg, [opacity])

    return merg


def ContraSharpening(clip, src, radius=None, rep=13, planes=[0, 1, 2]):
    # contra-sharpening: sharpen the denoised clip, but don't add more to any pixel than what was removed previously.
    # Author: Didée at the VERY GRAINY thread (https://forum.doom9.org/showthread.php?p=1076491)

    # Args:
    #     radius (int)   - Spatial radius for contra-sharpening(1-3).
    #     rep (int)      - Mode of repair to limit the difference.
    #     planes (int[]) - Whether to process the corresponding plane. The other planes will be passed through unchanged.

    if not (isinstance(clip, vs.VideoNode) and isinstance(src, vs.VideoNode)):
        raise TypeError("ContraSharpening: This is not a clip")

    if clip.format.color_family == vs.COMPAT:
        raise TypeError("ContraSharpening: COMPAT color family is not supported!")

    if clip.format.id != src.format.id:
        raise TypeError("ContraSharpening: Both clips must have the same format")

    if clip.width != src.width or clip.height != src.height:
        raise TypeError("ContraSharpening: Both clips must have the same size")

    if radius is None:
        radius = 2 if clip.width > 960 else 1

    if clip.format.num_planes == 1:
        planes = [0]

    if isinstance(planes, int):
        planes = [planes]
    
    mat1 = [1, 2, 1, 2, 4, 2, 1, 2, 1]
    mat2 = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    bd = clip.format.bits_per_sample
    isFLOAT = clip.format.sample_type == vs.FLOAT
    mid = 0 if isFLOAT else 1 << (bd - 1) 
    num = clip.format.num_planes
    R = core.rgsf.Repair if isFLOAT else core.rgvs.Repair

    s = MinBlur(clip, radius, planes) # damp down remaining spots of the denoised clip

    if radius <= 1:
        RG11 = core.std.Convolution(s, matrix=mat1, planes=planes)
    elif radius == 2:
        RG11 = core.std.Convolution(s, matrix=mat1, planes=planes).std.Convolution(matrix=mat2, planes=planes)
    else:
        RG11 = core.std.Convolution(s, matrix=mat1, planes=planes).std.Convolution(matrix=mat2, planes=planes).std.Convolution(matrix=mat2, planes=planes)
    
    ssD = core.std.MakeDiff(s, RG11, planes) # the difference of a simple kernel blur
    
    allD = core.std.MakeDiff(src, clip, planes) # the difference achieved by the denoising
  
    ssDD = R(ssD, allD, [rep if i in planes else 0 for i in range(num)]) # limit the difference to the max of what the denoising removed locally
   
    expr = 'x abs y abs < x y ?' if isFLOAT else 'x {} - abs y {} - abs < x y ?'.format(mid, mid) # abs(diff) after limiting may not be bigger than before

    ssDD = core.std.Expr([ssDD, ssD], [expr if i in planes else '' for i in range(num)])
    
    return core.std.MergeDiff(clip, ssDD, planes) # apply the limited difference (sharpening is just inverse blurring)


def MinBlur(clip, rad=1, planes=[0, 1, 2]):
    # MinBlur by Didée (https://avisynth.nl/index.php/MinBlur)
    # Nifty Gauss/Median combination

    if not isinstance(clip, vs.VideoNode):
        raise TypeError("MinBlur: This is not a clip")

    if clip.format.color_family == vs.COMPAT:
        raise TypeError("MinBlur: COMPAT color family is not supported!")

    if clip.format.num_planes == 1:
        planes = [0]

    if isinstance(planes, int):
        planes = [planes]

    mat1 = [1, 2, 1, 2, 4, 2, 1, 2, 1]
    mat2 = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    bd = clip.format.bits_per_sample
    isFLOAT = clip.format.sample_type == vs.FLOAT

    if rad <= 0:
        RG11 = sbr(clip, 1, planes)
        RG4 = core.std.Median(clip, planes)
    elif rad == 1:
        RG11 = core.std.Convolution(clip, matrix=mat1, planes=planes)
        RG4 = core.std.Median(clip, planes)
    elif rad == 2:
        RG11 = core.std.Convolution(clip, matrix=mat1, planes=planes).std.Convolution(matrix=mat2, planes=planes)
        RG4 = core.ctmf.CTMF(clip.fmtc.bitdepth(bits=16), radius=2, planes=planes).fmtc.bitdepth(flt=1) if isFLOAT else core.ctmf.CTMF(clip, radius=2, planes=planes)
    else:
        RG11 = core.std.Convolution(clip, matrix=mat1, planes=planes).std.Convolution(matrix=mat2, planes=planes).std.Convolution(matrix=mat2, planes=planes)
        RG4 = core.ctmf.CTMF(clip.fmtc.bitdepth(bits=12), radius=3, planes=planes).fmtc.bitdepth(bits=bd) if bd > 12 else core.ctmf.CTMF(clip, radius=3, planes=planes)

    expr = 'x y - x z - * 0 < x dup y - abs x z - abs < y z ? ?'
    return core.std.Expr([clip, RG11, RG4], [expr if i in planes else '' for i in range(clip.format.num_planes)])


def sbr(clip, r=1, planes=[0, 1, 2]):
    # Make a highpass on a blur's difference (well, kind of that)
    
    if not isinstance(clip, vs.VideoNode):
        raise TypeError("sbr: This is not a clip")

    if clip.format.color_family == vs.COMPAT:
        raise TypeError("sbr: COMPAT color family is not supported!")

    if clip.format.num_planes == 1:
        planes = [0]

    if isinstance(planes, int):
        planes = [planes]

    mat1 = [1, 2, 1, 2, 4, 2, 1, 2, 1]
    mat2 = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    bd = clip.format.bits_per_sample
    isFLOAT = clip.format.sample_type == vs.FLOAT
    mid = 0 if isFLOAT else 1 << (bd - 1)

    if r <= 1:
        RG11 = core.std.Convolution(clip, matrix=mat1, planes=planes)
    elif r == 2:
        RG11 = core.std.Convolution(clip, matrix=mat1, planes=planes).std.Convolution(matrix=mat2, planes=planes)
    else:
        RG11 = core.std.Convolution(clip, matrix=mat1, planes=planes).std.Convolution(matrix=mat2, planes=planes).std.Convolution(matrix=mat2, planes=planes)

    RG11D = core.std.MakeDiff(clip, RG11, planes)

    if r <= 1:
        RG11DS = core.std.Convolution(RG11D, matrix=mat1, planes=planes)
    elif r == 2:
        RG11DS = core.std.Convolution(RG11D, matrix=mat1, planes=planes).std.Convolution(matrix=mat2, planes=planes)
    else:
        RG11DS = core.std.Convolution(RG11D, matrix=mat1, planes=planes).std.Convolution(matrix=mat2, planes=planes).std.Convolution(matrix=mat2, planes=planes)
    
    expr = 'x y - x * 0 < 0 x y - abs x abs < x y - x ? ?' if isFLOAT else 'x y - x {m} - * 0 < {m} x y - abs x {m} - abs < x y - {m} + x ? ?'.format(m=mid)

    RG11DD = core.std.Expr([RG11D, RG11DS], [expr if i in planes else '' for i in range(clip.format.num_planes)])

    return core.std.MakeDiff(clip, RG11DD, planes)


def DitherLumaRebuild(src, s0=2., c=0.0625, chroma=True):
    # Converts luma (and chroma) to PC levels, and optionally allows tweaking for pumping up the darks. (for the clip to be fed to motion search only)
    # By courtesy of cretindesalpes. (https://forum.doom9.org/showthread.php?p=1548318)

    if not isinstance(src, vs.VideoNode):
        raise TypeError("DitherLumaRebuild: This is not a clip!")
    
    if src.format.color_family == vs.COMPAT:
        raise TypeError("DitherLumaRebuild: COMPAT color family is not supported!")

    bd = src.format.bits_per_sample
    isFLOAT = src.format.sample_type == vs.FLOAT
    i = 0.00390625 if isFLOAT else 1 << (bd - 8)

    x = 'x {} /'.format(i) if bd != 8 else 'x'
    expr = 'x 128 * 112 /' if isFLOAT else '{} 128 - 128 * 112 / 128 + {} *'.format(x, i)
    k = (s0 - 1) * c
    t = '{} 16 - 219 / 0 max 1 min'.format(x)
    c1 = 1 + c
    c2 = c1 * c
    e = '{} {} {} {} {} + / - * {} 1 {} - * + {} *'.format(k, c1, c2, t, c, t, k, 256*i)
    
    return core.std.Expr([src], [e] if src.format.num_planes == 1 else [e, expr if chroma else ''])


def Tweak(clip, hue=None, sat=None, bright=None, cont=None, coring=True):

    bd = clip.format.bits_per_sample
    isFLOAT = clip.format.sample_type == vs.FLOAT
    isGRAY = clip.format.color_family == vs.GRAY
    mid = 0 if isFLOAT else 1 << (bd - 1)

    if clip.format.color_family in [vs.RGB, vs.COMPAT]:
        raise TypeError("Tweak: RGB and COMPAT color family are not supported!")
        
    if not (hue is None and sat is None or isGRAY):
        hue = 0.0 if hue is None else hue
        sat = 1.0 if sat is None else sat
        hue = hue * math.pi / 180
        sinh = math.sin(hue)
        cosh = math.cos(hue)
        cmin = -0.5 if isFLOAT else 16 << (bd - 8) if coring else 0
        cmax = 0.5 if isFLOAT else 240 << (bd - 8) if coring else (1 << bd) - 1
        expr_u = "x {} * y {} * + -0.5 max 0.5 min".format(cosh * sat, sinh * sat) if isFLOAT else "x {} - {} * y {} - {} * + {} + {} max {} min".format(mid, cosh * sat, mid, sinh * sat, mid, cmin, cmax)
        expr_v = "y {} * x {} * - -0.5 max 0.5 min".format(cosh * sat, sinh * sat) if isFLOAT else "y {} - {} * x {} - {} * - {} + {} max {} min".format(mid, cosh * sat, mid, sinh * sat, mid, cmin, cmax)
        src_u = core.std.ShufflePlanes(clip, [1], vs.GRAY)
        src_v = core.std.ShufflePlanes(clip, [2], vs.GRAY)
        dst_u = core.std.Expr([src_u, src_v], expr_u)
        dst_v = core.std.Expr([src_u, src_v], expr_v)
        clip = core.std.ShufflePlanes([clip, dst_u, dst_v], [0, 0, 0], clip.format.color_family)

    if not (bright is None and cont is None):
        bright = 0.0 if bright is None else bright
        cont = 1.0 if cont is None else cont

        if isFLOAT:
            expr = "x {} * {} + 0.0 max 1.0 min".format(cont, bright)
            clip =  core.std.Expr([clip], [expr] if isGRAY else [expr, ''])
        else:
            luma_lut = []
            luma_min = 16  << (bd - 8) if coring else 0
            luma_max = 235 << (bd - 8) if coring else (1 << bd) - 1

            for i in range(1 << bd):
                val = int((i - luma_min) * cont + bright + luma_min + 0.5)
                luma_lut.append(min(max(val, luma_min), luma_max))

            clip = core.std.Lut(clip, [0], luma_lut)

    return clip

# Modified from mvmulti: https://github.com/IFeelBloated/vapoursynth-mvtools-sf/blob/r9/src/mvmulti.py
def Analyze(super, blksize=None, blksizev=None, levels=None, search=None, searchparam=None, pelsearch=None, _lambda=None, chroma=None, tr=None, truemotion=None, lsad=None, plevel=None, _global=None, pnew=None, pzero=None, pglobal=None, overlap=None, overlapv=None, divide=None, badsad=None, badrange=None, meander=None, trymany=None, fields=None, TFF=None, search_coarse=None, dct=None, d=False):
    def getvecs(isb, delta):
        return core.mvsf.Analyze(super, isb=isb, blksize=blksize, blksizev=blksizev, levels=levels, search=search, searchparam=searchparam, pelsearch=pelsearch, _lambda=_lambda, chroma=chroma, delta=delta, truemotion=truemotion, lsad=lsad, plevel=plevel, _global=_global, pnew=pnew, pzero=pzero, pglobal=pglobal, overlap=overlap, overlapv=overlapv, divide=divide, badsad=badsad, badrange=badrange, meander=meander, trymany=trymany, fields=fields, tff=TFF, search_coarse=search_coarse, dct=dct)

    bv = [getvecs(True,  i) for i in range(tr, 0, -1)] if not d else [getvecs(True,  i*2) for i in range(tr, 0, -1)]
    fv = [getvecs(False, i) for i in range(1, tr + 1)] if not d else [getvecs(False, i*2) for i in range(1, tr + 1)]

    return core.std.Interleave(bv + fv)

### Helper functions below
def spline(x, coordinates):
    def get_matrix(px, py, l):
        matrix = []
        matrix.append([(i == 0) * 1.0 for i in range(l + 1)])
        for i in range(1, l - 1):
            p = [0 for t in range(l + 1)]
            p[i - 1] = px[i] - px[i - 1]
            p[i] = 2 * (px[i + 1] - px[i - 1])
            p[i + 1] = px[i + 1] - px[i]
            p[l] = 6 * (((py[i + 1] - py[i]) / p[i + 1]) - (py[i] - py[i - 1]) / p[i - 1])
            matrix.append(p)
        matrix.append([(i == l - 1) * 1.0 for i in range(l + 1)])
        return matrix
    def equation(matrix, dim):
        for i in range(dim):
            num = matrix[i][i]
            for j in range(dim + 1):
                matrix[i][j] /= num
            for j in range(dim):
                if i != j:
                    a = matrix[j][i]
                    for k in range(i, dim + 1):
                        matrix[j][k] -= a * matrix[i][k]
    if not isinstance(coordinates, dict):
        raise TypeError("coordinates must be a dict")
    length = len(coordinates)
    if length < 3:
        raise ValueError("coordinates require at least three pairs")
    px = [key for key in coordinates.keys()]
    py = [val for val in coordinates.values()]
    matrix = get_matrix(px, py, length)
    equation(matrix, length)
    for i in range(length + 1):
        if x >= px[i] and x <= px[i + 1]:
            break
    j = i + 1
    h = px[j] - px[i]
    s = matrix[j][length] * (x - px[i]) ** 3
    s -= matrix[i][length] * (x - px[j]) ** 3
    s /= 6 * h
    s += (py[j] / h - h * matrix[j][length] / 6) * (x - px[i])
    s -= (py[i] / h - h * matrix[i][length] / 6) * (x - px[j])
    
    return s

# Full-range scale function that scale a value from [0, 255] to [0, peak]
def scale(value, peak):
    return value * peak / 255

# mod-4 and at least 16 function
def m4(x):
    return 16 if x < 16 else int(x / 4 + 0.5) * 4
