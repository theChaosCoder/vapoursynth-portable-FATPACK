import vapoursynth as vs
import math
import Dither

def EDIResample (src, w=None, h=None, sx=0, sy=0, sw=None, sh=None, kernel_u="spline64", kernel_d="spline36", taps=4, a1=None, a2=None, a3=None, css=None, fulls=False, fulld=None, cplace="mpeg2", nsize=0, nns=4, qual=2, etype=0, pscrn=1, noring=False, ratiothr=1.125, thr=1.0, elast=1.5, curve="linear", gcor=1.0, sigmoid=True, sigmoidthr=0.5, cont=6.5):
    core   = vs.get_core ()
    scss   = GetCSS (src)
    css    = scss if css is None else css
    fulld  = fulls if fulld is None else fulld

    Gray   = scss == "GRAY" or css == "GRAY"
    scss   = "444" if Gray else scss
    ocss   = "444" if Gray else css

    InterK = Dither.Resize16nr if noring else core.fmtc.resample

    Y      = 3
    U      = 1 if Gray else 3
    V      = 1 if Gray else 3
    Ut     = U == 3
    Vt     = V == 3

    cw     = src.width
    ch     = src.height
    cwc    = cw if scss == "444" else cw // 2
    chc    = ch // 2 if scss == "420" else ch

    ow     = cw if w is None else w
    oh     = ch if h is None else h
    owc    = ow if ocss == "444" else ow // 2
    ohc    = oh // 2 if ocss == "420" else oh

    sw     = cw if sw is None else sw
    sh     = ch if sh is None else sh

    prel   = int (sx / 2) * 2
    pret   = int (sy / 2) * 2
    prer   = int ((-cw + sx + sw if sw > 0 else sw) / 2) * 2
    preb   = int ((-ch + sy + sh if sh > 0 else sh) / 2) * 2
    prew   = cw - prel + prer
    preh   = ch - pret + preb

    if scss == "444":
       cwmod2 = int (cw / 2) * 2 == cw
       pwmod2 = int (prew / 2) * 2 == prew
       wpre   = prew < cw
       prel   = prel if wpre else 0
       prer   = (prer if pwmod2 else prer + 1) if wpre else (0 if cwmod2 else 1)
       prew   = cw - prel + prer
       wpre   = prew < cw or cwmod2 == False
    else:
       cwmod4 = int (cw / 4) * 4 == cw
       pwmod4 = int (prew / 4) * 4 == prew
       wpre   = prew < cw
       prel   = prel if wpre else 0
       prer   = (prer if pwmod4 else prer + 2) if wpre else (0 if cwmod4 else 2)
       prew   = cw - prel + prer
       wpre   = prew < cw or cwmod4 == False

    if scss == "420":
       chmod4 = int (ch / 4) * 4 == ch
       phmod4 = int (preh / 4) * 4 == preh
       hpre   = preh < ch
       pret   = pret if hpre else 0
       preb   = (preb if phmod4 else preb + 2) if hpre else (0 if chmod4 else 2)
       preh   = ch - pret + preb
       hpre   = preh < ch or chmod4 == False
    else:
       chmod2 = int (ch / 2) * 2 == ch
       phmod2 = int (preh / 2) * 2 == preh
       hpre   = preh < ch
       pret   = pret if hpre else 0
       preb   = (preb if phmod2 else preb + 1) if hpre else (0 if chmod2 else 1)
       preh   = ch - pret + preb
       hpre   = preh < ch or chmod2 == False

    sw     = cw - sx + sw if sw <= 0 else sw
    sh     = ch - sy + sh if sh <= 0 else sh
    sx     = sx - prel if wpre else sx
    sy     = sy - pret if hpre else sy
    
    sxc    = sx if scss == "444" else float (sx / 2)
    syc    = float (sy / 2) if scss == "420" else sy
    swc    = sw if scss == "444" else float (sw / 2)
    shc    = float (sh / 2) if scss == "420" else sh

    yhratio = float (ow / sw)
    yvratio = float (oh / sh)
    chratio = float (owc / swc)
    cvratio = float (ohc / shc)

    enable   = yhratio != 1 or yvratio != 1 or chratio != 1 or cvratio != 1 or sw != int (sw) or sh != int (sh) or swc != int (swc) or shc != int (shc) or sx != int (sx) or sy != int (sy) or sxc != int (sxc) or syc != int (syc)

    yhct     = math.ceil ((math.log (yhratio / ratiothr)) / math.log (2)) if yhratio > ratiothr else 0
    yhrf     = int (math.pow (2, yhct))
    yrhratio = yhratio / yhrf
    yvct     = math.ceil ((math.log (yvratio / ratiothr)) / math.log (2)) if yvratio > ratiothr else 0
    yvrf     = int (math.pow (2, yvct))
    yrvratio = yvratio / yvrf

    chct     = math.ceil ((math.log (chratio / ratiothr)) / math.log (2)) if chratio > ratiothr else 0
    chrf     = int (math.pow (2, chct))
    crhratio = chratio / chrf
    cvct     = math.ceil ((math.log (cvratio / ratiothr)) / math.log (2)) if cvratio > ratiothr else 0
    cvrf     = int (math.pow (2, cvct))
    crvratio = cvratio / cvrf

    noediy = yhct <= 0 and yvct <= 0
    noedic = chct <= 0 and cvct <= 0
    noedi  = noediy or noedic

    Yedit  = noediy == False
    Uedit  = Ut and noedic == False
    Vedit  = Vt and noedic == False

    edit   = Yedit or Uedit or Vedit
    mixed  = False if edit == False or enable == False else True

    yhchift = 0.5 if yhrf >= 2 else 0
    yvchift = 0.5 if yvrf >= 2 else 0
    yhfix   = -yhchift
    yvfix   = -yvchift

    chshift = ((0.5 if chrf >= 2 else 0) if scss == "444" else ((0.5 if chrf >= 2 else 0) if cplace == "mpeg1" else (0.5 - float (chrf / 4) if chrf >= 2 else -0.25))) if ocss == "444" else (((0.5 if chrf >= 2 else 0) if cplace == "mpeg1" else (0.75 if chrf >= 2 else 0.25)) if scss == "444" else ((0.5 if chrf >= 2 else 0) if cplace == "mpeg1" else (0.75 - float (chrf / 4) if chrf >= 2 else 0)))
    cvshift = 0.5 if cvrf >= 2 else 0
    chfix   = -chshift
    cvfix   = -cvshift

    cphfixe = 0 if ocss == "444" else (0 if cplace == "mpeg1" else 0.25 - float (0.25 / crhratio))
    cphfix  = (0 if scss == "444" else (0 if cplace == "mpeg1" else 0.25)) if ocss=="444" else ((0 if cplace == "mpeg1" else -0.5) if scss == "444" else (0 if cplace == "mpeg1" else 0.25 - float (0.25 / chratio)))

    Luma    = core.std.ShufflePlanes (src, planes=0, colorfamily=vs.GRAY) if GetCSS (src) != "GRAY" else src
    LumaL   = Dither.gamma_to_linear (Luma, fulls=fulls, fulld=fulls, curve=curve, gcor=gcor, sigmoid=sigmoid, thr=sigmoidthr, cont=cont) if curve != "linear" else Luma
    ChromaU = core.std.ShufflePlanes (src, planes=1, colorfamily=vs.GRAY) if Gray == False else 0
    ChromaV = core.std.ShufflePlanes (src, planes=2, colorfamily=vs.GRAY) if Gray == False else 0
    Linear  = core.std.ShufflePlanes ([LumaL, ChromaU, ChromaV], planes=[0, 0, 0], colorfamily=vs.YUV) if Gray == False else LumaL
    input   = core.fmtc.resample (Linear, (prew if wpre else cw), (preh if hpre else ch), (prel if wpre else 0), (pret if hpre else 0), (prew if wpre else cw), (preh if hpre else ch), kernel="point", fulls=fulls, fulld=fulls) if wpre or hpre else Linear

    if enable == False and edit == False:
       0
    elif yhct == chct and yvct == cvct and scss == "420":
       edgeedi  = EDInter (input, yvct, yhct, 1, 1, True, True, True, nsize, nns, qual, etype, pscrn)
       edgeediU = core.std.ShufflePlanes (edgeedi, planes=1, colorfamily=vs.GRAY)
       edgeediV = core.std.ShufflePlanes (edgeedi, planes=2, colorfamily=vs.GRAY)
       edgeediY = core.std.ShufflePlanes (edgeedi, planes=0, colorfamily=vs.GRAY)
    else:
       edgeediY = core.std.ShufflePlanes (input, planes=0, colorfamily=vs.GRAY) if GetCSS (src) != "GRAY" else input
       edgeediY = edgeediY if Yedit == False else EDInter (edgeediY, yvct, yhct, 1, 1, True, False, False, nsize, nns, qual, etype, pscrn)
       edgeediU = core.std.ShufflePlanes (input, planes=1, colorfamily=vs.GRAY) if Gray == False else 0
       edgeediU = 0 if Uedit == False else EDInter (edgeediU, cvct, chct, 1, 1, Ut, False, False, nsize, nns, qual, etype, pscrn)
       edgeediV = core.std.ShufflePlanes (input, planes=2, colorfamily=vs.GRAY) if Gray == False else 0
       edgeediV = 0 if Vedit == False else EDInter (edgeediV, cvct, chct, 1, 1, Vt, False, False, nsize, nns, qual, etype, pscrn)

    yrh = yrhratio > ratiothr
    yrv = yrvratio > ratiothr
    crh = crhratio > ratiothr
    crv = crvratio > ratiothr

    if enable == False and edit == False:
       0
    else:
       edgeY   = edgeediY if Yedit == False else core.fmtc.resample (edgeediY, ow, oh, (sx * yhrf + yhfix), (sy * yvrf + yvfix), (sw * yhrf), (sh * yvrf), kernelh=(kernel_u if yrh else kernel_d), kernelv=(kernel_u if yrv else kernel_d), taps=taps, a1=a1, a2=a2, a3=a3, fulls=fulls, fulld=fulls)
       edgeU   = 0 if Uedit == False else core.fmtc.resample (edgeediU, owc, ohc, (sxc * chrf + chfix + cphfixe), (syc * cvrf + cvfix), (swc * chrf), (shc * cvrf), kernelh=(kernel_u if crh else kernel_d), kernelv=(kernel_u if crv else kernel_d), taps=taps, a1=a1, a2=a2, a3=a3, fulls=fulls, fulld=fulls)	
       edgeV   = 0 if Vedit == False else core.fmtc.resample (edgeediV, owc, ohc, (sxc * chrf + chfix + cphfixe), (syc * cvrf + cvfix), (swc * chrf), (shc * cvrf), kernelh=(kernel_u if crh else kernel_d), kernelv=(kernel_u if crv else kernel_d), taps=taps, a1=a1, a2=a2, a3=a3, fulls=fulls, fulld=fulls)
       edge    = edgeY if Gray or Uedit == False or Vedit == False else core.std.ShufflePlanes ([edgeY, edgeU, edgeV], planes=[0, 0, 0], colorfamily=vs.YUV)

    yh = yhratio > ratiothr
    yv = yvratio > ratiothr
    ch = chratio > ratiothr
    cv = cvratio > ratiothr

    if enable == False and (mixed == False or (Yedit and Uedit and Vedit)):
       0
    elif yhratio == chratio and yvratio == cvratio and (mixed == False or (Yedit and Uedit and Vedit)):
       flat = InterK (input, ow, oh, sx, sy, sw, sh, cplace=cplace, kernelh=(kernel_u if yh else kernel_d), kernelv=(kernel_u if yv else kernel_d), taps=taps, a1=a1, a2=a2, a3=a3, fulls=fulls, fulld=fulls)
    else:
       flatY = core.std.ShufflePlanes (input, planes=0, colorfamily=vs.GRAY) if GetCSS (src) != "GRAY" else input
       flatY = InterK (flatY, ow, oh, sx, sy, sw, sh, kernelh=(kernel_u if yh else kernel_d), kernelv=(kernel_u if yv else kernel_d), taps=taps, a1=a1, a2=a2, a3=a3, fulls=fulls, fulld=fulls) if mixed or Yedit == False else flatY
       flatU = core.std.ShufflePlanes (input, planes=1, colorfamily=vs.GRAY) if (mixed or Uedit == False) and Ut else 0
       flatU = InterK (flatU, owc, ohc, (sxc + cphfix), syc, swc, shc, kernelh=(kernel_u if ch else kernel_d), kernelv=(kernel_u if cv else kernel_d), taps=taps, a1=a1, a2=a2, a3=a3, fulls=fulls, fulld=fulls) if (mixed or Uedit == False) and Ut else 0
       flatV = core.std.ShufflePlanes (input, planes=2, colorfamily=vs.GRAY) if (mixed or Vedit == False) and Vt else 0
       flatV = InterK (flatV, owc, ohc, (sxc + cphfix), syc, swc, shc, kernelh=(kernel_u if ch else kernel_d), kernelv=(kernel_u if cv else kernel_d), taps=taps, a1=a1, a2=a2, a3=a3, fulls=fulls, fulld=fulls) if (mixed or Vedit == False) and Vt else 0	
       flat  = flatY if Gray else core.std.ShufflePlanes ([flatY, flatU, flatV], planes=[0, 0, 0], colorfamily=vs.YUV)

    merge  = (Dither.limit_dif16 (flat, edge, thr=thr, elast=elast) if mixed else edge) if edit else flat
    mergeY = core.std.ShufflePlanes (merge, planes=0, colorfamily=vs.GRAY)
    mergeY = Dither.linear_to_gamma (mergeY, fulls=fulls, fulld=fulls, curve=curve, gcor=gcor, sigmoid=sigmoid, thr=sigmoidthr, cont=cont) if curve != "linear" else mergeY
    mergeU = core.std.ShufflePlanes (merge, planes=1, colorfamily=vs.GRAY) if Gray == False else 0
    mergeV = core.std.ShufflePlanes (merge, planes=2, colorfamily=vs.GRAY) if Gray == False else 0
    mergeF = core.std.ShufflePlanes ([mergeY, mergeU, mergeV], planes=[0, 0, 0], colorfamily=vs.YUV) if css != "GRAY" else mergeY
    Range  = core.fmtc.bitdepth (mergeF, fulls=fulls, fulld=fulld) if fulls != fulld else mergeF

    return Range

def EDInter (src, vct=1, hct=1, vfield=1, hfield=1, Y=True, U=False, V=False, nsize=0, nns=4, qual=2, etype=0, pscrn=1, honly=False):
    core = vs.get_core ()
    clip = src

    if hct >= 1:
       clip   = clip if honly else core.std.Transpose (clip).std.FlipHorizontal ()
       clip   = core.nnedi3.nnedi3 (clip, hfield, True, Y, U, V, nsize, nns, qual, etype, pscrn)
       hct    = hct - 1
       honly  = hct >= 1
       hfield = 0
       clip   = clip if honly else core.std.FlipHorizontal (clip).std.Transpose ()
    else:
       0

    if vct >= 1 and honly == False:
       clip   = core.nnedi3.nnedi3(clip, vfield, True, Y, U, V, nsize, nns, qual, etype, pscrn)
       vct    = vct - 1
       vfield = 0
    else: 
       0

    clip = (clip if vct <= 0 and hct <= 0 else EDInter (clip, vct, hct, vfield, hfield, Y, U, V, nsize, nns, qual, etype, pscrn, honly)) if Y or U or V else src
    return clip

def GetCSS (src):
    if src.format.id == vs.YUV420P16:
       css = "420"
    elif src.format.id == vs.YUV422P16:
       css = "422"
    elif src.format.id == vs.YUV444P16:
       css = "444"
    elif src.format.id == vs.GRAY16:
       css = "GRAY"
    else:
       css = 0
    return css
