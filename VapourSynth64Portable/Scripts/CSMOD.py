"""
    # CSMOD v0.2.5 ported from Contra-Sharpen mod 3.7 and Contra-Sharpen mod16 1.6
    2016.01.27 by Evalyn, Thanks [author]mawen1250 and [sis]Holy !
    
    Requirements: VapourSynth R28, havsfunc r20 or newer, MSmoosh, 
    fluxsmooth, MVtools, nnedi3, TemporalSoften
    
    Test on YV12/YV24(8~16bit INT) passed, and Greyscale now is supported.
    Other YUV colorfamily havn't been fully tested yet.
    *****************************************************
    Be aware of chroma placement issue when "ssout=True".
    That means you should use "ssout=True" at your own risk if the input clip isn't "MPEG2" style of chroma placement.
    Anyother condition of chroma placement will result in a 'chroma shifted' output when "ssout=True".
    But you can define your own method of supersample in case of this. See examples below.
    *****************************************************
    
    Changelog:
        1. change "ss_hq[bool]" to int(0~2).
            #0 using non-ringing Spline64Resize when (ss_w*ss_h < 3.0625) otherwise using nnedi3 with pscrn=2.
            #1 using nnedi3 in super sampling, but "pscrn" always sets to "1" (much slower but better quality)
            #2 using nnedi3 in super sampling, but "pscrn" always sets to "2" (faster, in other words, depth will be dithered to 8bit when necessary)
        2. add "showmask[bool]" to output internal mask.
        3. add greyscale support.
        4. remove "ssoutc[bool]", output chroma will always be fixed when "ssout=True"
        5. internal greyscale process when "chroma=False" (10% Faster).
        6. custom "kernel", "filter_ss", "filter_nr" are available but a little different.
        7. remove "deband" preset and related params. You can use custom "filter_ss" instead.
        8. change "secure[bool]" to float, controls Threshold(on an 8-bit scale) to avoid banding & oil painting (or face wax) effect of sharpening, set 0 to disable it.
"""
##############################################################################
#    Example of custom definition.

#    import vapoursynth as vs
#    import nnedi3_resample as nnrs
#    import CSMOD as cs
#
#    core = vs.get_core()
#
#    def SSMethod(clip):
#        ss_w = 2    # scale ratio in horizon
#        ss_h = 2    # scale ratio in vertical
#        width = clip.width        # source width
#        height = clip.height    # source height
#
#        clip = nnrs.nnedi3_resample(clip, width * ss_w, height * ss_h, nnrs=3)
#        
#        return clip
#    # using nnedi3_resample as internal supersample method.
#    # (No chroma placement issue when "ssout=True")
#
#    src = core.lsmas.LWLibavSource('xxx.mp4')
#    cs = cs.CSMOD(src, ssmethod=SSMethod, ss_w=2, ss_h=2, ssout=True)
#    # you should keep scale ratio set correctly
#    
#    cs.set_output()
#    
##############################################################################
"""
    Other params such as "kernel" "filter_ss" "filter_nr" can be customized in the same way.
    But "Smode" should be defined as str with Reverse Polish notation if you want to customize yourself.
"""

import vapoursynth as vs
import havsfunc as haf
import math

def CSMOD(filtered, source=None, pclip=None, chroma=None, preset=None, edgemode=None, edgemask=None,
          edgethr=None, tcannysigma=None, showmask=None, mergesrc=None, ss_w=None, ss_h=None, ss_hq=None,
          nr=None, ssmethod=None, filter_ss=None, ssrep=None, ssout=None, preblur=None, prec=None, preR=None,
          usepasf=None, sspre=None, Smethod=None, kernel=None, secure=None, filter_nr=None, Smode=None, strength=None,
          divisor=None, index=None, Szrp=None, Spwr=None, SdmpLo=None, SdmpHi=None, Slimit=None, Tlimit=None,
          limitsrc=None, Sovershoot=None, Sundershoot=None, Tovershoot=None, Tundershoot=None, Soft=None,
          Soothe=None, limit=None, Repmode=None, RepmodeU=None, thr=None, thrc=None, chromamv=None, blksize=None,
          overlap=None, thSAD=300, thSCD1=300, thSCD2=100, truemotion=False, MVglobal=False, pel=None,
          pelsearch=None, search=None, searchparam=None, MVsharp=2, DCT=0):
    
    #get vs core
    core = vs.get_core()
    # format check
    inputFormatid = filtered.format.id  
    sColorFamily = filtered.format.color_family
    sSType = filtered.format.sample_type
    if sColorFamily == vs.YUV or sColorFamily == vs.GRAY:
        if sSType != vs.INTEGER:
            raise TypeError('CSMOD: \"filtered\" must be INTEGER format !')
    else:
        raise TypeError('CSMOD: Only YUV colorfmaily is supported !')
    # constant value
    sw = filtered.width
    sh = filtered.height
    #Greyscale?
    if sColorFamily == vs.GRAY:
        GRAYS = True
        chroma = False
        sGRAY = True
    else:
        GRAYS = False
        sGRAY = False
        
    #Showmask ?
    if showmask is None:
        showmask = False
    
    # define CALL , custom yourself
    def CALL(func, clip):
        return func(clip)
    
    #generate default param
    if source is not None and isinstance(source, vs.VideoNode):
        defsrc = True
        if (filtered.width != source.width or filtered.height != source.height):
            raise TypeError('CSMOD: resolution of \"source\" and \"filtered\" must match !')
    else:
        if source is not None:
            raise TypeError('CSMOD: \"source\" is not a clip !')
        else:
            defsrc = False
    
    if pclip is not None and isinstance(pclip, vs.VideoNode):
        defpclp = True
    else:
        if pclip is not None:
            raise TypeError('CSMOD: \"pdclip\" is not a clip !')
        else:
            defpclp = False
    
    # custom filter ?
    if type(filter_ss) == type(CALL):
        deffss = True
    elif filter_ss is not None:
        raise TypeError('CSMOD: \"filter_ss\" is not a function !')
    else:
        deffss = False
    
    if type(filter_nr) == type(CALL):
        deffnr = True
    elif filter_nr is not None:
        raise TypeError('CSMOD: \"filter_nr\" is not a function !')
    else:
        deffnr = False
    
    # Whether to limit sharpening to source clip, only takes effect when (Defined(source) || Defined(filter_ss) || usepasf) == true.    
    if limit is None:
        limit = True
    elif not isinstance(limit, bool):
        raise TypeError('CSMOD: \"limit\" must be bool !')
    if not (defsrc or deffss or usepasf or deffnr):
        limit = False
    
    # process chroma or not
    if chroma is None or not isinstance(chroma, bool):
        chroma = False
    
    # HD ?
    if sw > 1024 or sh > 576:
        HD = True
    else:
        HD = False
        
    if HD:
        bs = 16
        bs2 = 32
    else:
        bs = 8
        bs2 = 16
    
    # presets
    if preset is None:
        if defsrc:
            preset = "faster"
        else:
            preset = "medium"
    
    preset = preset.lower()
    if preset == "very fast":
        pnum = 0
    elif preset == "faster":
        pnum = 1
    elif preset == "fast":
        pnum = 2
    elif preset == "medium":
        pnum = 3
    elif preset == "slow":
        pnum = 4
    elif preset == "slower":
        pnum = 5
    elif preset == "very slow":
        pnum = 6
    elif preset == "noise":
        pnum = 7
    elif preset == "grain":
        pnum = 8
    elif preset == "detail":
        pnum = 9
    else:
        pnum = 10
        raise TypeError('CSMOD: \"preset\" is invalid !')
    
            
                          # preset = veryF faster  fast  medium  slow  slower veryS  noise  grain  detail 
    if edgemode    is None : edgemode    = [0,     0,     0,     0,     0,     0,     0,     2,     2,     2][pnum]
    # 0 = Sharpening all, 1 = Sharpening only edge, 2 = Sharpening only non-edge.
    # By default, edgemode=2 is tuned for enhancing noise/small detail.
    # It will automatically set [ss_w=1, ss_h=1, preblur=0, kernel=5, Slimit=False, Tlimit=True(when limit=False)].
    
    if edgemask    is None : edgemask    = [1,     1,     3,     6,     6,     6,     5,     6,     6,     6][pnum]
    #  1: Same as edgemaskHQ=False in LSFmod(min/max), 2: Same as edgemaskHQ=True in LSFmod,
    #  3: Same as sharpening mask in MCTD(prewitt),    4: MSharpen mask,
    #  5: tcanny mask(less sensitive to noise),        6: prewitt mask with mt_hysteresis(less sensitive to noise),
    # -1: Same as mtype=1 in TAA(sobel),              -2: Same as mtype=2 in TAA(roberts),
    # -3: Same as mtype=3 in TAA(prewitt),            -4: Same as mtype=4 in TAA(TEdgeMask),
    # -5: Same as mtype=5 in TAA(tcanny),             -6: Same as mtype=6 in TAA(MSharpen),
    # -7: My own method of tcanny usage of AA mask.
    # 1~6 are masks tweaked for Sharpening, -1~-7 are masks tweaked for AA.
    # Otherwise define a custom edge mask clip, only luma is taken to merge all planes.
    
    if edgethr     is None : edgethr     = [32,      32,      32,   32,    32,    32,    32,    24,    32,    48][pnum]
    # Tweak edge mask threshold EXCEPT edgemask mode -2/-1.
    
    if ss_w        is None : ss_w         = [1,     1,      1.25,    1.25,    1.5,   1.5,   1.5,       1,      1,     1][pnum]
    # Super sampling multiplier of width.
    if ss_h        is None : ss_h         = [1,     1,      1.25,    1.25,    1.5,   1.5,   1.5,       1,      1,     1][pnum]
    # Super sampling multiplier of height.
    
    if nr            is None : nr             = [False, False, False, False, True, True, True, False, False, False][pnum]
    # True using non-ringing resize in super sampling.
    
    if preblur     is None : preblur     = [0,     -1,    -1,    -1,   -6,    -6,    -6,     0,     1,    -6][pnum]
    # Pre-filtering, 0 = disable, -1 = Gaussian Blur radius=1(RG11), -2 = Gaussian Blur radius=2(RG11+RG20),
    # -3 = Gaussian Blur radius=3(RG11+RG20+RG20), -4 = Median Blur radius=1(RG4),
    # -5 = Average Blur radius=1(RG20) -6 = (RG4+RG11), -7 = (RG19+RG4),
    # 1 = MinBlur, 2 = MinBlur(Uses SBR by default) mixed with MinBlur+FluxSmoothT,
    # 3 = MinBlur(Uses SBR by default)+FluxSmoothT.
    # "preblur" is ignored when pclip is defined.

    if kernel       is None : kernel      = [1,      1,     6,     6,    3,     3,     3,     8,     7,     3][pnum]
    # 1: Gaussian Blur radius=1(RG11),      2: Average Blur(RG20),
    # 3: Gaussian Blur radius=2(RG11+RG20), 4: Gaussian Blur radius=3(RG11+RG20+RG20),
    # 5: Median Blur(RG4),                  6: Median Blur + Gaussian Blur(RG4+RG11)
    # 7: for grain enhance(RG19+RG4),       8: for noise enhance(MinBlur radius=1)
    # Otherwise define a custom kernel in string such as kernel="RemoveGrain(20, 11)".

    if secure      is None : secure      = [0.25,  0.25,  0.125, 0.125, 0,     0,        0,       0,      0,     0][pnum]
    # Threshold(on an 8-bit scale) to avoid banding & oil painting (or face wax) effect of sharpening, set 0 to disable it.(from LSFmod)
    
    if Smode       is None : Smode         = [0,     3,     3,     3,     3,     3,     3,     3,     3,     3][pnum]
    # Sharpen Mode - 0: None, 1: Linear, 2: Non-linear 1, 3:Non-linear 2(from LSFmod Smode=5), Otherwise define a custom Smode in string.
    
    if Soft           is None : Soft         = [0,     0,    -2,    -2,    -2,    -2,    -2,    0,     -2,    -2][pnum]
    # Soft the sharpening effect (-1 = old autocalculate, -2 = new autocalculate, 0 = disable, (0, 100] = enable).
    # Disabled when (limit==True && thr==0 && thrc==0).(from LSFmod)
    
    if Slimit        is None : Slimit      = [False, False, not limit, not limit, not limit, not limit, not limit, False, False, not limit][pnum]
    # Spatial limit with overshoot and undershoot. Disabled when (limit==True && thr==0 && thrc==0).
    
    if Tlimit       is None : Tlimit      = [False, False, False, False, False, not limit, not limit, not limit, not limit, False][pnum]
    # Use MC Temporal limit at the end of sharpening.(from MCTD)
    
    if chromamv    is None : chromamv    = [False, False, False, chroma, chroma, True, True, chroma, chroma, chroma][pnum]
    if pel           is None : pel         = [1,     1,     1,     1, 1 if HD else 2, 1 if HD else 2, 1 if HD else 2, 1,  1, 1][pnum]
    if pelsearch   is None : pelsearch   = [1,     2,     2,     2,     2,     2,     2,     2,     2,     2][pnum]
    if search       is None : search      = [2,     2,     4,     4,     4,     5,     3,     4,     4,     4][pnum]
    if searchparam is None : searchparam = [1,     2,     2,     2,     2,     2,     2,     2,     2,     2][pnum]
    if blksize       is None : blksize     = [bs2,  bs2,  bs2,   bs2,    bs,    bs,    bs,    bs,    bs,    bs][pnum]
    
    ol = round(blksize / 2)
    ol2 = round(blksize / 4)
    
    if overlap       is None : overlap     = [ol2,  ol2,  ol2,   ol2,    ol,    ol,    ol,    ol,    ol,    ol][pnum]
    
    # custom supersample method ?    
    if type(ssmethod) == type(CALL):
        defssmthd = True
    elif ssmethod is not None:
        raise TypeError('CSMOD: \"ssmethod\" is not a function !')
    else:
        defssmthd = False
    
    
    if limitsrc is None:
        limitsrc = False
    elif not isinstance(limitsrc, bool):
        raise TypeError('CSMOD: \"limitsrc\" must be bool !')
        
    # edgemode
    if not isinstance(edgemode, int):
        raise TypeError('CSMOD: \"edgemode\" must be int !')
    elif edgemode < 0 or edgemode > 2:
        raise TypeError('CSMOD: \"edgemode\" must be int(0~2) !')    
    # 0 = Sharpening all, 1 = Sharpening only edge, 2 = Sharpening only non-edge.
    # By default, edgemode=2 is tuned for enhancing noise/small detail.
    # It will automatically set [edgemask=-2, ss_w=1, ss_h=1, preblur=0, kernel=5, Slimit=false, Tlimit=true(when limit=false)].
    
    # edgemask
    if not (isinstance(edgemask, int) or isinstance(edgemask, vs.VideoNode)):
        raise TypeError('CSMOD: \"edgemask\" must be int or clip !')    
    #  1: Same as edgemaskHQ=False in LSFmod(min/max), 2: Same as edgemaskHQ=True in LSFmod,
    #  3: Same as sharpening mask in MCTD(prewitt),    4: MSharpen mask,
    #  5: tcanny mask(less sensitive to noise),        6: prewitt mask with mt_hysteresis(less sensitive to noise),
    # -1: Same as mtype=1 in TAA(sobel),              -2: Same as mtype=2 in TAA(roberts),
    # -3: Same as mtype=3 in TAA(prewitt),            -4: Same as mtype=4 in TAA(TEdgeMask),
    # -5: Same as mtype=5 in TAA(tcanny),             -6: Same as mtype=6 in TAA(MSharpen),
    # -7: My own method of tcanny usage of AA mask.
    # 1~6 are masks tweaked for Sharpening, -1~-7 are masks tweaked for AA.
    # Otherwise define a custom edge mask clip, only luma is taken to merge all planes.
    
    if not (isinstance(edgethr, float) or isinstance(edgethr, int)):
        raise TypeError('CSMOD: \"edgethr\" must be float !')
    # Tweak edge mask threshold EXCEPT edgemask mode -2/-1.
    
    if tcannysigma is None:
        tcannysigma = 1.2
    elif not (isinstance(tcannysigma, float) or isinstance(tcannysigma, int)):
        raise TypeError('CSMOD: \"tcannysigma\" must be float !')
    #Tweak tcanny's sigma in edgemask mode -7/-5/5.
        
    if mergesrc is None:
        mergesrc = False
    elif not isinstance(mergesrc, bool):
        raise TypeError('CSMOD: \"mergesrc\" must be bool !')
    #Whether to merge clip "source" instead of clip "filtered" at the end of processing.
        
    if not (isinstance(ss_w, float) or isinstance(ss_w, int)):
        raise TypeError('CSMOD: \"ss_w\" must be float !')
    elif ss_w < 1.00:
        ss_w = 1.00
    #Super sampling multiplier of width.

    if not (isinstance(ss_h, float) or isinstance(ss_h, int)):
        raise TypeError('CSMOD: \"ss_w\" must be float !')
    elif ss_h < 1.00:
        ss_h = 1.00
    #Super sampling multiplier of height.
    
    if ss_hq is None:
        ss_hq = 0
    elif not isinstance(ss_hq, int):
        raise TypeError('CSMOD: \"ss_hq\" must be int(0~2) !')
    #0 using non-ringing Spline64Resize when (ss_w*ss_h < 3.0625) otherwise using nnedi3 with pscrn=2.
    #1 using nnedi3 in super sampling, but "pscrn" always sets to "1" (much slower but better quality)
    #2 using nnedi3 in super sampling, but "pscrn" always sets to "2" (faster, in other words, depth will be dithered to 8bit when necessary)
    
    if ssout is None:
        ssout = False
    elif not isinstance(ssout, bool):
        raise TypeError('CSMOD: \"ssout\" must be bool !')
    if not (ss_w > 1.0 or ss_h > 1.0):
        ssout = False
    #Whether to output in super sampling resolution.
    
    # chroma ?
    if not chroma and not sGRAY:
        U = core.std.ShufflePlanes(filtered, 1, vs.GRAY)
        V = core.std.ShufflePlanes(filtered, 2, vs.GRAY)
            # TO DO NEXT
        filtered = core.std.ShufflePlanes(filtered, 0, vs.GRAY)
        source = core.std.ShufflePlanes(source, 0, vs.GRAY) if defsrc else source
        pclip = core.std.ShufflePlanes(pclip, 0, vs.GRAY) if defpclp else pclip
        GRAYS = True
        sGRAY = False
    
    if ssrep is None:
        ssrep = False
    elif not isinstance(ssrep, bool):
        raise TypeError('CSMOD: \"ssrep\" must be bool !')
    if ssout:
        ssrep = True
    #When limiting sharpening to source clip, whether to Repair in super sampling resolution.

    if not isinstance(preblur, int):
        raise TypeError('CSMOD: \"preblur\" must be int !')
    # Pre-filtering, 0 = disable, -1 = Gaussian Blur radius=1(RG11), -2 = Gaussian Blur radius=2(RG11+RG20),
    # -3 = Gaussian Blur radius=3(RG11+RG20+RG20), -4 = Median Blur radius=1(RG4),
    # -5 = Average Blur radius=1(RG20) -6 = (RG4+RG11), -7 = (RG19+RG4),
    # 1 = MinBlur, 2 = MinBlur(Uses SBR by default) mixed with MinBlur+FluxSmoothT,
    # 3 = MinBlur(Uses SBR by default)+FluxSmoothT.
    # "preblur" is ignored when pclip is defined.

    
    if prec is None:
        prec = True
    #Whether to process chroma plane in preblur.
    
    if preR is None:
        if preblur >= 2:
            preR = 0
        elif ss_w*ss_h > 6.25:
            preR = 3
        elif ss_w*ss_h > 2.25:
            preR = 2
        else:
            preR = 1
    # MinBlur setting, 1-3 sets radius of MinBlur(Gaussian|Median), 0 uses SBR instead of normal Gaussian Blur in MinBlur.
    
    if sspre is None:
        sspre = deffss
    # When true apply pre-filter in super sampling clip, when false apply pre-filter in original resolution.
    # By default it is false unless filter_ss is defined.

    
    if usepasf is None:
        usepasf = False
    # Whether to use pre-filtered clip as input(filtered) clip, which will reduce noise/halo/ring from input clip.
    
    if Smethod is None:
        if ss_w*ss_h > 1:
            Smethod = 3
        else:
            Smethod = 1
    #Sharpen Method - 1: 3x3 kernel, 2: Min/Max, 3: Min/Max + 3x3 kernel.
    
    if not (isinstance(kernel, int) or type(kernel) == type(CALL)):
        raise TypeError('CSMOD: \"kernel\" must be int or function !')
    elif isinstance(kernel, int) and (kernel < 1 or kernel > 8):
        raise ValueError('CSMOD: \"kernel\" is out of range [int(1~8)] !')
    # 1: Gaussian Blur radius=1(RG11),      2: Average Blur(RG20),
    # 3: Gaussian Blur radius=2(RG11+RG20), 4: Gaussian Blur radius=3(RG11+RG20+RG20),
    # 5: Median Blur(RG4),                  6: Median Blur + Gaussian Blur(RG4+RG11)
    # 7: for grain enhance(RG19+RG4),       8: for noise enhance(MinBlur radius=1)
    # Otherwise define a custom kernel in string such as kernel="removegrain(20, 11)".
    
    if not (isinstance(secure, int) or isinstance(secure, float)):
        raise TypeError('CSMOD: \"secure\" must be float !')
    elif secure < 0 or secure > 255:
        raise ValueError('CSMOD: \"secure\" must be non-negative float[0~255] !')
    # Threshold(on an 8-bit scale) to avoid banding & oil painting (or face wax) effect of sharpening, set 0 to disable it.(from LSFmod)
    
    if Smode is None:
        Smode = 3
    elif not (isinstance(Smode, int) or isinstance(Smode, str)):
        raise TypeError('CSMOD: \"Smode\" must be int or string !')
    elif isinstance(Smode, int) and (Smode < 0 or Smode > 3):
        raise ValueError('CSMOD: \"Smode\" is out of range [int(0~3)] !')
    # Sharpen Mode - 0: None, 1: Linear, 2: Non-linear 1, 3:Non-linear 2(from LSFmod Smode=5), Otherwise define a custom Smode in string.
    
    if divisor is None:
        divisor = 1.5
    
    if index is None:
        index = 0.8
        
    if Szrp is None:
        Szrp = 16
        
    if Spwr is None:
        Spwr = 4
        
    if SdmpLo is None:
        SdmpLo = 4
        
    if SdmpHi is None:
        SdmpHi = 48
    
    
    if thr is None:
        thr = 0
        
    if thrc is None:
        if chroma:
            thrc = thr
        else:
            thrc = 0
    # Allow pixels sharpen more than the source by [thr/thrc](luma/chroma). Set to 0 to disable this function.

    
    if Repmode is None:
        Repmode = 1
    
    if RepmodeU is None:
        RepmodeU = Repmode
        
    if not isinstance(Slimit, bool):
        raise TypeError('CSMOD: \"Slimit\" must be bool !')
    # Spatial limit with overshoot and undershoot. Disabled when (limit==true && thr==0 && thrc==0).
    
    if not isinstance(Tlimit, bool):
        raise TypeError('CSMOD: \"Tlimit\" must be bool !')
    # Use MC Temporal limit at the end of sharpening.(from MCTD)
    
    if strength is None:
        strength = 100
            
    if Sovershoot is None:
        Sovershoot = round(strength / 160)
    Sovershoot = max(Sovershoot, 0)
        
    if Sundershoot is None:
        Sundershoot = Sovershoot
    Sundershoot = max(Sundershoot, 0)
        
    if Tovershoot is None:
        Tovershoot = round(strength / 48)
    Tovershoot = max(Tovershoot, 0)

    if Tundershoot is None:
        Tundershoot = Tovershoot
    Tundershoot = max(Tundershoot, 0)
    
    if not isinstance(Soft, int):
        raise TypeError('CSMOD: \"Soft\" must be int !')
    elif Soft < -2 or Soft > 100:
        raise ValueError('CSMOD: \"Soft\" is out of range [int(-2~100)] !')
    # enhanced Soft    
    if limit and thr == 0 and thrc == 0:
        Soft = 0
    elif Soft <= -2:
        Soft = (1.0 + (2.0 / (ss_w+ss_h))) * math.sqrt(strength)
    elif Soft == -1:
        Soft = math.sqrt((((ss_w + ss_h) / 2.0 - 1.0) * 100.0)) * 10
    else:
        Soft = Soft
    # Soft the sharpening effect (-1 = old autocalculate, -2 = new autocalculate, 0 = disable, (0, 100] = enable).
    # Disabled when (limit==true && thr==0 && thrc==0).(from LSFmod)
    
    if Soothe is None:
        if limit:
            Soothe = -1
        else:
            Soothe = 24
    elif not isinstance(Soothe, int):
        raise TypeError('CSMOD: \"Soothe\" must be int !')
    elif Soothe < -1 or Soothe > 100:
        raise ValueError('CSMOD: \"Soothe\" is out of range [int(-1~100)] !')
    if Tlimit:
        Soothe = -1
    # Soothe temporal stabilization, 0-100 sets minimum percent of the original sharpening to keep, -1 disables Soothe. Disabled when (chroma==true && Tlimit=true).
    
    if not isinstance(chromamv, bool):
        raise TypeError('CSMOD: \"chromamv\" must be bool !')
        
    if not isinstance(nr, bool):
        raise TypeError('CSMOD: \"nr\" must be bool !')
    
    
    # Avisynth Function: Spline    
    def Spline(x, x1, y1, x2, y2, x3, y3, cubic=True):
        n = 3
        xa = [0.0, x1, x2, x3]
        ya = [0.0, y1, y2, y3]
        y2a = [0.0, 0.0, 0.0, 0.0]
        y = 0
        
        def spline(x, y, n, y2):
            u = [0.0, 0.0, 0.0]
            y2[1] = u[1] = float(0)
            
            idx = 2
            sig = (x[idx] - x[idx-1]) / (x[idx+1] - x[idx-1])
            p = sig * y2[idx-1] + float(2)
            y2[idx] = (sig - float(1)) / p
            u[idx] = (y[idx+1] - y[idx]) / (x[idx+1] - x[idx]) - (y[idx] - y[idx-1]) / (x[idx] - x[idx-1])
            u[idx] = (float(6) * u[idx] / (x[idx+1] - x[idx-1]) - sig * u[idx-1]) / p
            idx = idx + 1
            
            un = float(0)
            qn = float(0)
            
            y2[n] = (un - qn * u[n-1]) / (qn * y2[n-1] + float(1))
            
            for i in range(2):
                idx = 2
                y2[idx] = y2[idx] * y2[idx+1] + u[idx]
                idx = idx - 1
            return None
            
        def    splint(xa, ya, y2a, n, x, y, cubic):
            klo = 1
            khi = n
            while (khi - klo > 1):
                k = (khi + klo) >> 1
                if xa[k] > x:
                    khi = k
                else:
                    klo = k
                    
            h = xa[khi] - xa[klo]
                
            a = (xa[khi] - x) / h
            b = (x - xa[klo]) / h
            
            if cubic:
                y = a * ya[klo] + b * ya[khi] + (( a * a * a - a) * y2a[klo] + (b * b * b - b) * y2a[khi]) * (h * h) / float(6)
            else:
                y = a * ya[klo] + b * ya[khi]
            return y
        
        spline(xa, ya, n, y2a)
        
        return splint(xa, ya, y2a, n, x, y, cubic)
        
    # Internal Function
    def Depth(input, depth=None):
        sbitPS = input.format.bits_per_sample
        if sbitPS == depth:
            return input
        else:
            return core.fmtc.bitdepth(input, bits=depth, flt=0, dmode=3)
        
    
    # Internal Function: SuperSample
    def CSmod_nrSpline64Resize(input, target_width=None, target_height=None, chroma=None, nr=nr, ss_wf=ss_w, ss_hf=ss_h):
        w = input.width
        h = input.height
        sbitPS = input.format.bits_per_sample
        wss = round(w * ss_wf / 8) * 8
        hss = round(h * ss_hf / 8) * 8
        if target_width is None:
            target_width = wss
        if target_height is None:
            target_height = hss
        if chroma is None:
            chroma = True
            
        res_mul = float(target_width * target_height) / float(w * h)
        res_mul = min(max(res_mul, 1), 2.25)
        if not (isinstance(nr, float) or isinstance(nr, int)):
            if nr:
                nr_weight = Spline(res_mul, 1, 0, 2.25, 1, 3.5, 0, True)
            else:
                nr_weight = 0
        else:
            nr_weight = nr
        nr_weight = min(max(nr_weight, 0), 1)
        
        inputp = input
        
        resize = core.fmtc.resample(inputp, w=target_width, h=target_height, kernel="spline64")
        resize = Depth(resize, sbitPS)
        nrres = core.fmtc.resample(inputp, w=target_width, h=target_height, kernel="gaussian", a1=100)
        nrres = Depth(nrres, sbitPS)
        
        if nr_weight == 0:
            resize = resize
        elif nr_weight == 1:
            resize = core.rgvs.Repair(resize, nrres, [1] if chroma or GRAYS else [1,0])
        else:
            if chroma or GRAYS:
                resize = core.std.Merge(resize, core.rgvs.Repair(resize, nrres, [1]), nr_weight)
            else:
                resize = core.std.Merge(resize, core.rgvs.Repair(resize, nrres, [1,0]), [nr_weight,0])
                
        return resize
    
    def CSmod_nnedi3_SuperSample(input, ss_wf=ss_w, ss_hf=ss_h):
        # get cycle times
        cyclew = math.ceil(math.log(ss_wf, 2))
        cycleh = math.ceil(math.log(ss_hf, 2))
        # constant values
        sbitPS = input.format.bits_per_sample
        wss = round(input.width * ss_wf / 8) * 8
        hss = round(input.height * ss_hf / 8) * 8
        # nothing to say..
        def SS(src):
            
            if ss_hq != 1:
                src = Depth(src, 8)
                ss = core.nnedi3.nnedi3(src, qual=2, nsize=0, nns=3, pscrn=2 if ss_hq==0 else ss_hq, field=1, dh=True)
            else:
                ss = core.nnedi3.nnedi3(src, qual=2, field=1, dh=True, pscrn=1)
            fix = core.fmtc.resample(ss, sy=[-0.5, -0.5*(1<<ss.format.subsampling_h)], kernel="spline64")
            
            return fix
        #SuperSampleing in vertical
        fh = input
        while cycleh > 0:
            clp = SS(fh)
            fh = clp
            cycleh = cycleh - 1
        #SuperSampleing in horizon
        fw = core.std.Transpose(fh)
        while cyclew > 0:
            clp = SS(fw)
            fw = clp
            cyclew = cyclew - 1
        
        final = core.std.Transpose(fw)
        # Resize to target size
        if final.width == wss and final.height == hss:
            output = final
        else:
            output = CSmod_nrSpline64Resize(final, target_width=wss, target_height=hss)
        return Depth(output,sbitPS)
    
    # ssout ? TO DO UV
    if not chroma and not sGRAY:
        if ssout:
            U = CSmod_nrSpline64Resize(U, int(U.width*ss_w), int(U.height*ss_h))
            V = CSmod_nrSpline64Resize(V, int(V.width*ss_w), int(V.height*ss_h))
    
    # Auto Select SuperSampleing Method
    def SSMethod(input, target_width=None, target_height=None, chroma=chroma, nr=nr, ss_wf=ss_w, ss_hf=ss_h):
        if not defssmthd:
            if ss_hq != 0 :
                return CSmod_nnedi3_SuperSample(input, ss_wf=ss_wf, ss_hf=ss_hf)
            else:
                if ss_w * ss_h > 3.0625:
                    return CSmod_nnedi3_SuperSample(input, ss_wf=ss_wf, ss_hf=ss_hf)
                else:
                    return CSmod_nrSpline64Resize(input, target_width=target_width, target_height=target_height, chroma=chroma, nr=nr, ss_wf=ss_wf, ss_hf=ss_hf)
        else:
            return CALL(ssmethod, input)
    # NEXT super sampling, filtering in ss clip
    if not defsrc:
        source = filtered
    
    filtered_os = filtered
    
    if ss_w > 1.0 or ss_h > 1.0:
        filtered_ss = SSMethod(filtered)
        if defsrc:
            source_ss = SSMethod(source)
        else:
            source_ss = filtered_ss
    else:
        filtered_ss = filtered
        source_ss = source
    
    # custom filter_ss
    if deffss:
        filtered = CALL(filter_ss, filtered_ss)
    else:
        filtered = filtered_ss
        
    if ss_w > 1.0 or ss_h > 1.0:
        if deffss:
            filtered_ds = CSmod_nrSpline64Resize(filtered, sw, sh, chroma, 0)
        else:
            filtered_ds = filtered_os
    else:
        filtered_ds = filtered
        
    # pre-filtering before sharpening
    
    if sspre:
        fforpre = filtered
    else:
        fforpre = filtered_ds
    
    if defpclp:
        pre = pclip
    elif preblur <= -7:
        pre = core.rgvs.RemoveGrain(fforpre, [19] if (chroma and prec) or GRAYS else [19, 0])
        pre = coer.rgvs.RemoveGrain(pre, [4] if (chroma and prec) or GRAYS else [4, 0])
    elif preblur == -6:
        pre = core.rgvs.RemoveGrain(fforpre, [4] if (chroma and prec) or GRAYS else [4, 0])
        pre = core.rgvs.RemoveGrain(pre, [11] if (chroma and prec) or GRAYS else [11, 0])
    elif preblur == -5:
        pre = core.rgvs.RemoveGrain(fforpre, [20] if (chroma and prec) or GRAYS else [20, 0])
    elif preblur == -4:
        pre = core.rgvs.RemoveGrain(fforpre, [4] if (chroma and prec) or GRAYS else [4, 0])
    elif preblur == -3:
        pre = core.rgvs.RemoveGrain(fforpre, [11] if (chroma and prec) or GRAYS else [11, 0])
        pre = core.rgvs.RemoveGrain(pre, [20] if (chroma and prec) or GRAYS else [20, 0])
        pre = core.rgvs.RemoveGrain(pre, [20] if (chroma and prec) or GRAYS else [20, 0])
    elif preblur == -2:
        pre = core.rgvs.RemoveGrain(fforpre, [11] if (chroma and prec) or GRAYS else [11, 0])
        pre = core.rgvs.RemoveGrain(pre, [20] if (chroma and prec) or GRAYS else [20, 0])
    elif preblur == -1:
        pre = core.rgvs.RemoveGrain(fforpre, [11] if (chroma and prec) or GRAYS else [11, 0])
    elif preblur == 1:
        spatial = haf.MinBlur(fforpre, preR, [0, 1, 2] if (chroma and prec) or GRAYS else [0])
        pre = spatial
    elif preblur == 2:
        spatial = haf.MinBlur(fforpre, preR, [0, 1, 2] if (chroma and prec) or GRAYS else [0])
        temporal = core.flux.SmoothT(spatial, 7, [0] if not (chroma and prec) or GRAYS else [0, 1, 2])
        temporal = core.rgvs.Repair(temporal, spatial, [1] if prec or GRAYS else [1, 0])
        mixed = core.std.Merge(temporal, spatial, 0.251)
        pre = mixed
    elif preblur >= 3:
        spatial = haf.MinBlur(fforpre, preR, [0, 1, 2] if (chroma and prec) or GRAYS else [0])
        temporal = core.flux.SmoothT(spatial, 7, [0, 1, 2] if (chroma and prec) or GRAYS else [0])
        temporal = core.rgvs.Repair(temporal, spatial, [1] if prec or GRAYS else [1, 0])
        pre = temporal
    else:
        pre = filtered
    # You can define your own pre-filter clip with pclip.
    if (pre.width == sw and pre.height == sh):
        pre_ds = pre
    else:
        pre_ds = CSmod_nrSpline64Resize(pre, sw, sh, chroma)
    if (ss_w > 1.0 or ss_h > 1.0):
        wss = round(sw * ss_w / 8) * 8
        hss = round(sh * ss_h / 8) * 8
        if (pre.width == wss and pre.height == hss):
            pre = pre
        else:
            pre = SSMethod(pre)
    else:
        pre = pre_ds
    
    # whether to use pre-filtered clip as main clip
    if usepasf:
        filtered = pre
        filtered_ds = pre_ds
        
    # generate edge mask
    if ssout:
        if (Tlimit and not chroma and chromamv) and not GRAYS:
            prefinal = core.std.Merge(pre, filtered_ss, [0, 1])
        else:
            prefinal = pre
    else:
        if (Tlimit and not chroma and chromamv) and not GRAYS:
            prefinal = core.std.Merge(pre_ds, filtered_os, [0, 1])
        else:
            prefinal = pre_ds
            
    if ssout:
        srcfinal = source_ss
    else:
        srcfinal = source
    
    if isinstance(edgemask, vs.VideoNode):
        edgemask = edgemask
    else:
        sbitPS = prefinal.format.bits_per_sample
        prefinal8 = Depth(prefinal, 8)
        srcfinal8 = Depth(srcfinal, 8)
        # -1: Same as mtype=1 in TAA(sobel)
        if edgemask == -1:
            edgemask = core.std.Convolution(prefinal8, [0, -1, 0, -1, 0, 1, 0, 1, 0], planes=0)
            mt = "x 7 < 0 255 ?"
            edgemask = core.std.Expr(edgemask, [mt] if GRAYS else [mt, ""])
            edgemask = core.std.Inflate(edgemask, planes=0)
        # -2: Same as mtype=2 in TAA(roberts)
        elif edgemask == -2:
            edgemask = core.std.Convolution(prefinal8, [0, 0, 0, 0, 2, -1, 0, -1, 0], planes=0)
            mt = "x 4 > 255 x ?"
            edgemask = core.std.Expr(edgemask, [mt] if GRAYS else [mt, ""])
            edgemask = core.std.Inflate(edgemask, planes=0)
        # -3: Same as mtype=3 in TAA(prewitt)
        elif edgemask == -3:
            edgemask1 = core.std.Convolution(prefinal8, [1, 1, 0, 1, 0, -1, 0, -1, -1], divisor=1, saturate=False, planes=0)
            edgemask2 = core.std.Convolution(prefinal8, [1, 1, 1, 0, 0, 0, -1, -1, -1], divisor=1, saturate=False, planes=0)
            edgemask3 = core.std.Convolution(prefinal8, [1, 0, -1, 1, 0, -1, 1, 0, -1], divisor=1, saturate=False, planes=0)
            edgemask4 = core.std.Convolution(prefinal8, [0, -1, -1, 1, 0, -1, 1, 1, 0], divisor=1, saturate=False, planes=0)
            mt = "x y max z max a max 1.8 pow"
            edgemask = core.std.Expr([edgemask1, edgemask2, edgemask3, edgemask4], [mt] if GRAYS else [mt, ""])
            edgemask = core.rgvs.RemoveGrain(edgemask, [4] if GRAYS else [4, 0])
            edgemask = core.std.Inflate(edgemask, planes=0)
            edgemask = core.rgvs.RemoveGrain(edgemask, [20] if GRAYS else [20, 0])
        # -4: Same as mtype=4 in TAA(TEdgeMask)
        elif edgemask == -4:
            edgemask = core.generic.TEdge(prefinal8, planes=0)
            mt = "x " + str(edgethr * 3.0) + " <= x 2 / x 16 * ?"
            edgemask = core.std.Expr(edgemask, [mt] if GRAYS else [mt, ""])
            edgemask = core.std.Deflate(edgemask, planes=0)
            edgemask = core.rgvs.RemoveGrain(edgemask, [20 if HD else 11] if GRAYS else [20 if HD else 11,0])
        # -5: Same as mtype=5 in TAA(tcanny)
        elif edgemask == -5:
            edgemask = core.tcanny.TCanny(srcfinal8, sigma=tcannysigma, mode=1, planes=0)
            mt = "x " + str(edgethr) + " <= x 2 / x 2 * ?"
            edgemask = core.std.Expr(edgemask, [mt] if GRAYS else [mt, ""])
            edgemask = core.rgvs.RemoveGrain(edgemask, [20 if HD else 11] if GRAYS else [20 if HD else 11,0])
            edgemask = core.std.Inflate(edgemask, planes=0)
        # -6: Same as mtype=6 in TAA(MSharpen)
        elif edgemask == -6:
            edgemask = core.msmoosh.MSharpen(prefinal8, threshold=edgethr//5, strength=0, mask=True, planes=0)
            edgemask = core.rgvs.RemoveGrain(edgemask, [20 if HD else 11] if GRAYS else [20 if HD else 11,0])
        # -7: My own method of tcanny usage of AA mask
        elif edgemask <= -7:
            edgemask = core.tcanny.TCanny(srcfinal8, sigma=tcannysigma, mode=1, planes=0)
            mt = "x " + str(edgethr) + " <= 0 x " + str(edgethr) + " - 64 * ?"
            edgemask = core.std.Expr(edgemask, [mt] if GRAYS else [mt, ""])
            edgemask = core.rgvs.RemoveGrain(edgemask, [20 if HD else 11] if GRAYS else [20 if HD else 11, 0])
            edgemask = core.std.Inflate(edgemask, planes=0)
        # 1: Same as edgemaskHQ=False in LSFmod(min/max)
        elif edgemask == 1:
            edgemask1 = core.std.Maximum(prefinal8, planes=0)
            edgemask2 = core.std.Minimum(prefinal8, planes=0)
            mt = "x y - " + str(edgethr) + " / 0.86 pow 255 *"
            edgemask = core.std.Expr([edgemask1,edgemask2], [mt] if GRAYS else [mt, ""])
            edgemask = core.std.Inflate(edgemask, planes=0)
            edgemask = core.std.Inflate(edgemask, planes=0)
            edgemask = core.rgvs.RemoveGrain(edgemask, [11] if GRAYS else [11, 0])
        # 2: Same as edgemaskHQ=True in LSFmod,
        elif edgemask == 2:
            edgemask1 = core.std.Convolution(prefinal8, [8, 16, 8, 0, 0, 0, -8, -16, -8], divisor=4, saturate=False, planes=0)
            edgemask2 = core.std.Convolution(prefinal8, [8, 0, -8, 16, 0, -16, 8, 0, -8], divisor=4, saturate=False, planes=0)
            mt = "x y max " + str(edgethr * 4) + " / 0.86 pow 255 *"
            edgemask = core.std.Expr([edgemask1, edgemask2], [mt] if GRAYS else [mt, ""])
            edgemask = core.std.Inflate(edgemask, planes=0)
            edgemask = core.std.Inflate(edgemask, planes=0)
            edgemask = core.rgvs.RemoveGrain(edgemask, [11] if GRAYS else [11, 0])
        # 3: Same as sharpening mask in MCTD(prewitt)
        elif edgemask == 3:
            edgemask1 = core.std.Convolution(prefinal8, [1, 1, 0, 1, 0, -1, 0, -1, -1], divisor=1, saturate=False, planes=0)
            edgemask2 = core.std.Convolution(prefinal8, [1, 1, 1, 0, 0, 0, -1, -1, -1], divisor=1, saturate=False, planes=0)
            edgemask3 = core.std.Convolution(prefinal8, [1, 0, -1, 1, 0, -1, 1, 0, -1], divisor=1, saturate=False, planes=0)
            edgemask4 = core.std.Convolution(prefinal8, [0, -1, -1, 1, 0, -1, 1, 1, 0], divisor=1, saturate=False, planes=0)
            mt = "x y max z max a max"
            edgemask = core.std.Expr([edgemask1, edgemask2, edgemask3, edgemask4], [mt] if GRAYS else [mt, ""])
            mt = "x " + str(round(edgethr * 0.25)) + " < 0 x ? 1.8 pow"
            edgemask = core.std.Expr(edgemask, [mt] if GRAYS else [mt, ""])
            edgemask = core.rgvs.RemoveGrain(edgemask, [4] if GRAYS else [4, 0])
            edgemask = core.std.Inflate(edgemask, planes=0)
            edgemask = core.rgvs.RemoveGrain(edgemask, [20] if GRAYS else [20, 0])
        # 4: MSharpen mask,
        elif edgemask == 4:
            edgemask = core.msmoosh.MSharpen(prefinal8, threshold=edgethr//10, strength=0, mask=True, planes=0)
            edgemask = core.rgvs.RemoveGrain(edgemask, [20 if HD else 11] if GRAYS else [20 if HD else 11, 0])
        # 5: tcanny mask(less sensitive to noise)
        elif edgemask == 5:
            edgemask = core.tcanny.TCanny(srcfinal8, sigma=tcannysigma, mode=1, planes=0)
            mt = "x " + str(edgethr * 0.5) + " <= 0 x " + str(edgethr * 0.5) + " - 2.4 pow ?"
            edgemask = core.std.Expr(edgemask, [mt] if GRAYS else [mt, ""])
            edgemask = core.rgvs.RemoveGrain(edgemask, [20 if HD else 11] if GRAYS else [20 if HD else 11, 0])
            edgemask = core.std.Inflate(edgemask, planes=0)
        # 6: prewitt mask with mt_hysteresis(less sensitive to noise)
        else:
            edgemask1 = core.std.Convolution(prefinal8, [1, 1, 0, 1, 0, -1, 0, -1, -1], divisor=1, saturate=False, planes=0)
            edgemask2 = core.std.Convolution(prefinal8, [1, 1, 1, 0, 0, 0, -1, -1, -1], divisor=1, saturate=False, planes=0)
            edgemask3 = core.std.Convolution(prefinal8, [1, 0, -1, 1, 0, -1, 1, 0, -1], divisor=1, saturate=False, planes=0)
            edgemask4 = core.std.Convolution(prefinal8, [0, -1, -1, 1, 0, -1, 1, 1, 0], divisor=1, saturate=False, planes=0)
            mt = "x y max z max a max"
            prewittm = core.std.Expr([edgemask1, edgemask2, edgemask3, edgemask4], [mt] if GRAYS else [mt, ""])
            mt = "x " + str(round(edgethr * 0.5)) + " < 0 x ?"
            prewittm = core.std.Expr(prewittm, [mt] if GRAYS else [mt, ""])
            prewitt = core.rgvs.RemoveGrain(prewittm, [4] if GRAYS else [4, 0])
            edgemask = core.misc.Hysteresis(prewitt, prewittm, [0])
            edgemask = core.rgvs.RemoveGrain(edgemask, [20 if HD else 11] if GRAYS else [20 if HD else 11, 0])
        
        # 1~6 are masks tweaked for Sharpening, -1~-7 are masks tweaked for AA.
        # Otherwise define a custom edge mask clip, only luma is taken to merge all planes.    
        
        edgemask = Depth(edgemask, sbitPS)
    
    def average(clipa, clipb, planes=[0, 1, 2]):
        if 1 or 2 in planes:
            return (core.std.Expr(clips=[clipa, clipb], expr=["x y + 2 /"]))        
        else:
            if not GRAYS:
                return (core.std.Expr(clips=[clipa, clipb], expr=["x y + 2 /", ""]))        
            else:
                return (core.std.Expr(clips=[clipa, clipb], expr=["x y + 2 /"]))
    # unsharp    
    dark_limit = haf.mt_inpand_multi(pre, planes=[0, 1, 2] if chroma or GRAYS else [0])
    bright_limit = haf.mt_expand_multi(pre, planes=[0, 1, 2] if chroma or GRAYS else [0])
    minmaxavg = average(dark_limit,bright_limit, planes=[0, 1, 2] if chroma or GRAYS else [0])
    if Smethod <= 1:
        method = pre
    else:
        method = minmaxavg
    
    if Smethod == 2:
        method = minmaxavg
    else:
        if type(kernel) != type(CALL):
            if kernel <= 1:
                method = core.rgvs.RemoveGrain(method, [11] if chroma or GRAYS else [11, 0])
            elif kernel == 2:
                method = core.rgvs.RemoveGrain(method, [20] if chroma or GRAYS else [20, 0])
            elif kernel == 3:
                method = core.rgvs.RemoveGrain(method, [11] if chroma or GRAYS else [11, 0])
                method = core.rgvs.RemoveGrain(method, [20] if chroma or GRAYS else [20, 0])
            elif kernel == 4:
                method = core.rgvs.RemoveGrain(method, [11] if chroma or GRAYS else [11, 0])
                method = core.rgvs.RemoveGrain(method, [20] if chroma or GRAYS else [20, 0])
                method = core.rgvs.RemoveGrain(method, [20] if chroma or GRAYS else [20, 0])
            elif kernel == 5:
                method = core.rgvs.RemoveGrain(method, [4] if chroma or GRAYS else [4, 0])
            elif kernel == 6:
                method = core.rgvs.RemoveGrain(method, [4] if chroma or GRAYS else [4, 0])
                method = core.rgvs.RemoveGrain(method, [11] if chroma or GRAYS else [11, 0])
            elif kernel == 7:
                method = core.rgvs.RemoveGrain(method, [19] if chroma or GRAYS else [19, 0])
                method = core.rgvs.RemoveGrain(method, [4] if chroma or GRAYS else [4, 0])
            else:
                method == haf.MinBlur(method, 1, [0, 1, 2] if chroma or GRAYS else [0])
        else:
            method = CALL(kernel, method)
    
    # a simplified version of LimitFilter from mvsfunc
    if secure > 0:
        bits = method.format.bits_per_sample
        shift = bits - 8
        neutral = 128 << shift
        peak = (1 << bits) - 1
        multiple = peak / 255
        
        elast = 10.0
        thr = secure * multiple
        thr_1 = thr
        thr_2 = thr * elast
        thr_slope = 1 / (thr_2 - thr_1)
        
        expr = "x y - abs {thr_1} <= x x y - abs {thr_2} >= y y x y - {thr_2} x y - abs - * {thr_slope} * + ? ?".format(thr_1=thr_1, thr_2=thr_2, thr_slope=thr_slope)
        
        method = core.std.Expr([pre, method], [expr] if chroma or GRAYS else [expr, ""])
    
    # making difference clip for sharpening
    sharpdiff = core.std.MakeDiff(pre, method,[0, 1, 2] if chroma else [0])
    
    # filtering in nr clip
    if deffnr:
        filtered = CALL(filter_nr, method)
        filtered = core.std.MergeDiff(filtered, sharpdiff, [0, 1, 2] if chroma else [0])
    
    if (ss_w > 1.0 or ss_h > 1.0):
        if deffnr:
            filtered_ds = CSmod_nrSpline64Resize(filtered, sw, sh, chroma=chroma, nr=0)
        else:
            filtered_ds = filtered_ds
    else:
        filtered_ds = filtered
        
    # sharpening diff generate mode
    if isinstance(Smode, str):
        sharpdiff = core.std.Expr(sharpdiff, [Smode] if chroma or GRAYS else [Smode, ""])
    elif Smode <= 0:
        sharpdiff = sharpdiff
    elif Smode == 1:
        bits = sharpdiff.format.bits_per_sample
        shift = bits - 8
        neutral = 128 << shift
        peak = (1 << bits) - 1
        multiple = peak / 255
        def get_lut1(x):
            if x == neutral:
                return x
            else:
                y = ((x - neutral) * (strength / 50.0) + neutral)
                return min(max(round(y), 0), peak)
        sharpdiff = core.std.Lut(sharpdiff, function=get_lut1, planes=[0, 1, 2] if chroma else [0])
    elif Smode == 2:
        bits = sharpdiff.format.bits_per_sample
        shift = bits - 8
        neutral = 128 << shift
        peak = (1 << bits) - 1
        multiple = peak / 255
        def get_lut2(x):
            if x == neutral:
                return x
            else:
                y = (((x - neutral) / divisor) ** abs(index) * (strength / 20.0) * (1 if x > neutral else -1) + neutral)
                return min(max(round(y.real), 0), peak)
        sharpdiff = core.std.Lut(sharpdiff, function=get_lut2, planes=[0, 1, 2] if chroma else [0])
    else:
        bits = sharpdiff.format.bits_per_sample
        shift = bits - 8
        neutral = 128 << shift
        peak = (1 << bits) - 1
        multiple = peak / 255
        def get_lut3(x):
            if x == neutral:
                return x
            else:
                tmp1 = (x - neutral) / multiple
                tmp2 = tmp1 ** 2
                tmp3 = Szrp ** 2
                return min(max(round(x + (abs(tmp1) / Szrp) ** (1 / Spwr) * Szrp * (strength / 100 * multiple) * (1 if x > neutral else -1) * (tmp2 * (tmp3 + SdmpLo) / ((tmp2 + SdmpLo) * tmp3)) * ((1 + (0 if SdmpHi == 0 else (Szrp / SdmpHi) ** 4)) / (1 + (0 if SdmpHi == 0 else (abs(tmp1) / SdmpHi) ** 4)))), 0), peak)
        
        #"x 128 == x x 128 - abs " + str(Szrp) + " / " + str(miSpwr) + " ^ " + str(Szrp) + " * " + str(strength / 100.0) + " * x 128 > 1 -1 ? * x 128 - 2 ^ " + str(Szrp) + " 2 ^ " + str(SdmpLo) + " + * x 128 - 2 ^ " + str(SdmpLo) + " + " + str(Szrp) + " 2 ^ * / * 1 " + str(SdmpHi) + " 0 == 0 " + str(Szrp) + " " + str(SdmpHi) + " / 4 ^ ? + 1 " + str(SdmpHi) + " 0 == 0 x 128 - abs " + str(SdmpHi) + " / 4 ^ ? + / * 128 + ?"
        sharpdiff = core.std.Lut(sharpdiff, function=get_lut3, planes=[0, 1, 2] if chroma else [0])
    
    def clamp(clip, Bclip, Dclip, overshoot=0, undershoot=0, planes=[0, 1, 2]):
        if 1 or 2 in planes:
            mt = 'x y {overshoot} + > y {overshoot} + x ? z {undershoot} - < z {undershoot} - x y {overshoot} + > y {overshoot} + x ? ?'.format(overshoot=overshoot, undershoot=undershoot)
            output = core.std.Expr([clip, Bclip, Dclip], [mt])
            return output
        else:
            mt = 'x y {overshoot} + > y {overshoot} + x ? z {undershoot} - < z {undershoot} - x y {overshoot} + > y {overshoot} + x ? ?'.format(overshoot=overshoot, undershoot=undershoot)
            if not GRAYS:
                output = core.std.Expr([output, Dclip, Dclip], [mt, ""])
            else:
                output = core.std.Expr([clip, Bclip, Dclip], [mt])
            return output
    
    # spatial limit
    if ssout:
        fltfinal = filtered
    else:
        fltfinal = filtered_ds
    if limitsrc:
        limitclp = srcfinal
    else:
        limitclp = fltfinal
    if (ss_w > 1.00 or ss_h > 1.00):
        if limitsrc:
            limitclpss = source_ss
        else:
            limitclpss = filtered
    else:
        limitclpss = limitclp
    
    if Slimit:
        sclp = core.std.MergeDiff(pre, sharpdiff, [0, 1, 2] if chroma else [0])
        sclp = clamp(sclp, bright_limit, dark_limit, Sovershoot, Sundershoot, [0, 1, 2] if chroma else [0])
    else:
        sclp = None
    
    # Soft
    if Slimit:
        sharpdiff = core.std.MakeDiff(sclp, pre, [0, 1, 2] if chroma else [0])
    if Soft != 0:
        bits = sharpdiff.format.bits_per_sample
        shift = bits - 8
        neutral = 128 << shift
        peak = (1 << bits) - 1
        multiple = peak / 255
        const = 100 * multiple
        val = (100-Soft) * multiple
        SoftMP = multiple * Soft
        sharpdiff2 = core.rgvs.RemoveGrain(sharpdiff, [19] if chroma or GRAYS else [19, 0])
        mt = 'x {neutral} - abs y {neutral} - abs > y {SoftMP} * x {val} * + {const} / x ?'.format(neutral=neutral, val=val, const=const, SoftMP=SoftMP)
        sharpdiff = core.std.Expr([sharpdiff, sharpdiff2], [mt] if chroma or GRAYS else [mt, ""])

    
    # Soothe
    if Soothe >= 0 and Soothe <= 100:
        bits = sharpdiff.format.bits_per_sample
        shift = bits - 8
        neutral = 128 << shift
        peak = (1 << bits) - 1
        multiple = peak / 255
        const = 100 * multiple
        sharpdiff2 = core.focus.TemporalSoften(sharpdiff, 1, 255, 255 if chroma else 0, 32, 2)
        mt = 'x {neutral} - y {neutral} - * 0 < x {neutral} - {const} / {Soothe} * {neutral} + x {neutral} - abs y {neutral} - abs > x {Soothe} * y {const} {Soothe} - * + {const} / x ? ?'.format(neutral=neutral, Soothe=Soothe, const=const)
        sharpdiff = core.std.Expr([sharpdiff, sharpdiff2], [mt] if chroma or GRAYS else [mt, ""])
        
    # the difference achieved by filtering
    if limit:
        if ssrep:
            allD = core.std.MakeDiff(source_ss, filtered, [0, 1, 2] if chroma else [0])
        else:
            allD = core.std.MakeDiff(source, filtered_ds, [0, 1, 2] if chroma else [0])
    else:
        allD = None
        
    # limiting sharpening to source clip
    if not ssrep and (ss_w > 1.0 or ss_h > 1.0):
        sharpdiff = CSmod_nrSpline64Resize(sharpdiff, sw,sh, chroma, 0)
    if limit:
        if not GRAYS:
            ssDD = core.rgvs.Repair(sharpdiff, allD, [Repmode, RepmodeU] if chroma else [Repmode, 0])
        else:
            ssDD = core.rgvs.Repair(sharpdiff, allD, [Repmode])
    else:
        ssDD = sharpdiff
    bits = sharpdiff.format.bits_per_sample
    shift = bits - 8
    neutral = 128 << shift
    peak = (1 << bits) - 1
    multiple = peak / 255
    if limit and (thr > 0 or thrc > 0):
        thrMP = thr * multiple
        thrcMP = thrc * multiple
        yexpr = 'x {neutral} - abs y {neutral} - abs {thrMP} + <= x y {neutral} < y {thrMP} - y {thrMP} + ? ?'.format(neutral=neutral, thrMP=thrMP)
        uvexpr = 'x {neutral} - abs y {neutral} - abs {thrcMP} + <= x y {neutral} < y {thrcMP} - y {thrcMP} + ? ?'.format(neutral=neutral, thrcMP=thrcMP)
        if not GRAYS:
            ssDD = core.std.Expr([sharpdiff, ssDD], [yexpr, uvexpr] if chroma else [yexpr, ""])
        else:
            ssDD = core.std.Expr([sharpdiff, ssDD], [yexpr])
    if not limit and (thr > 0 or thrc > 0):
        thrMP = thr * multiple
        thrcMP = thrc * multiple
        valm = neutral - thrMP
        vala = neutral + thrMP
        valcm = neutral - thrcMP
        valca = neutral + thrcMP
        yexpr = 'x {neutral} - abs {thrMP} <= x x {neutral} < {valm} {vala} ? ?'.format(neutral=neutral, thrMP=thrMP, valm=valm, vala=vala)
        uvexpr = 'x {neutral} - abs {thrcMP} <= x x {neutral} < {valcm} {valca} ? ?'.format(neutral=neutral, thrcMP=thrcMP, valcm=valcm, valca=valca)
        if not GRAYS:
            ssDD = core.std.Expr(sharpdiff, [yexpr, uvexpr] if chroma else [yexpr, ""])
        else:
            ssDD = core.std.Expr(sharpdiff, [yexpr])
    if limit and (thr == 0 and thrc == 0):
        mt = 'x {neutral} - abs y {neutral} - abs <= x y ?'.format(neutral=neutral)
        ssDD = core.std.Expr([ssDD, sharpdiff], [mt] if chroma or GRAYS else [mt, ""])
    if ssrep and (ss_w > 1.0 or ss_h > 1.0) and not ssout:
        ssDD = CSmod_nrSpline64Resize(ssDD, sw,sh, chroma,0)
        
    # add difference clip to clip "filtered" of ss/original resolution
    if ssout:
        sclp = core.std.MergeDiff(filtered, ssDD, [0, 1, 2] if chroma else [0])
    else:
        sclp = core.std.MergeDiff(filtered_ds,ssDD, [0, 1, 2] if chroma else [0])
        
    # temporal limit
    sMVS = core.mv.Super(prefinal, hpad=0, vpad=0, pel=pel, levels=0, sharp=MVsharp, chroma=chroma)
    if usepasf or (preblur == 0 and not defpclp):
        rMVS = sMVS
    else:
        rMVS = core.mv.Super(limitclp, hpad=0, vpad=0, pel=pel, levels=1, sharp=MVsharp, chroma=chroma)
    
    f1v = core.mv.Analyse(super=sMVS, blksize=blksize, search=search, searchparam=searchparam,
                          pelsearch=pelsearch, isb=False, chroma=chromamv, truemotion=truemotion,
                          _global=MVglobal, overlap=overlap, dct=DCT)

    b1v = core.mv.Analyse(super=sMVS, blksize=blksize, search=search, searchparam=searchparam,
                          pelsearch=pelsearch, isb=True, chroma=chromamv, truemotion=truemotion,
                          _global=MVglobal, overlap=overlap, dct=DCT)
    
    f1c = core.mv.Compensate(limitclp, rMVS, f1v, thsad=thSAD, thscd1=thSCD1, thscd2=thSCD2)
    b1c = core.mv.Compensate(limitclp, rMVS, b1v, thsad=thSAD, thscd1=thSCD1, thscd2=thSCD2)
    Tmax = core.std.Expr([limitclp, f1c, b1c], ["x y max z max"] if chroma or GRAYS else ["x y max z max", ""])
    Tmin = core.std.Expr([limitclp, f1c, b1c], ["x y min z min"] if chroma or GRAYS else ["x y min z min", ""])
    if Tlimit:
        sclp = clamp(sclp, Tmax, Tmin, Tovershoot, Tundershoot, [0, 1, 2] if chroma else [0])

    # merge with edge mask and output correct chroma
    if mergesrc:
        merged_ss = source_ss
        merged_os = source
    else:
        merged_ss = filtered_ss
        merged_os = filtered_os
        
    if edgemode <= 0:
        if chroma:
            end = sclp
        else:
            if ssout:
                end = core.std.ShufflePlanes([sclp, U, V], [0, 0, 0], vs.YUV) if not chroma and not sGRAY else sclp
            else:
                end = core.std.ShufflePlanes([sclp, U, V], [0, 0, 0], vs.YUV) if not chroma and not sGRAY else sclp
    elif edgemode == 1:
        if ssout:
            end = core.std.MaskedMerge(merged_ss, sclp, edgemask, planes=[0, 1, 2] if chroma else [0], first_plane=True)
            end = core.std.ShufflePlanes([end, U, V], [0, 0, 0], vs.YUV) if not chroma and not sGRAY else end
        else:
            end = core.std.MaskedMerge(merged_os, sclp, edgemask, planes=[0, 1, 2] if chroma else [0], first_plane=True)
            end = core.std.ShufflePlanes([end, U, V], [0, 0, 0], vs.YUV) if not chroma and not sGRAY else end
    else:
        if ssout:
            end = core.std.MaskedMerge(sclp, merged_ss, edgemask, planes=[0, 1, 2] if chroma else [0], first_plane=True)
            end = core.std.ShufflePlanes([end, U, V], [0, 0, 0], vs.YUV) if not chroma and not sGRAY else end
        else:
            end = core.std.MaskedMerge(sclp, merged_os, edgemask, planes=[0, 1, 2] if chroma else [0], first_plane=True)
            end = core.std.ShufflePlanes([end, U, V], [0, 0, 0], vs.YUV) if not chroma and not sGRAY else end
    
    return end if not showmask else edgemask
