import vapoursynth as vs

def get_msb (src):
    core = vs.get_core ()
    clip = core.fmtc.nativetostack16 (src)
    return core.std.CropRel (clip, 0, 0, 0, (clip.height // 2))

def get_lsb (src):
    core = vs.get_core ()
    clip = core.fmtc.nativetostack16 (src)
    return core.std.CropRel (clip, 0, 0, (clip.height // 2), 0)

def add16 (src1, src2, dif=True):
    core = vs.get_core ()
    if dif:
       clip = core.std.MergeDiff (src1, src2)
    else:
       clip = core.std.Expr ([src1, src2], ["x y +"])
    return clip

def sub16 (src1, src2, dif=True):
    core = vs.get_core ()
    if dif:
       clip = core.std.MakeDiff (src1, src2)
    else:
       clip = core.std.Expr ([src1, src2], ["x y -"])
    return clip

def max_dif16 (src1, src2, ref):
    core = vs.get_core ()
    clip = core.std.Expr ([src1, src2, ref], ["x z - abs y z - abs > x y ?"])
    return clip

def min_dif16 (src1, src2, ref):
    core = vs.get_core ()
    clip = core.std.Expr ([src1, src2, ref], ["x z - abs y z - abs > y x ?"])
    return clip

def limit_dif16 (flt, src, ref=None, thr=0.25, elast=3.0):
    core  = vs.get_core ()
    thr   = thr * 256
    alpha = 1 / (thr * (elast - 1))
    beta  = elast * thr
    ref   = src if ref is None else ref
    clip  = core.std.Expr ([flt, src, ref], ["x z - abs {thr} <= x x z - abs {beta} >= ? y y {alpha} x y - * {beta} x y - abs - * + ?".format (thr=thr, alpha=alpha, beta=beta)])
    return clip

def merge16_8 (src1, src2, mask):
    core   = vs.get_core ()
    mask16 = core.fmtc.bitdepth (mask, bits=16, fulls=True, fulld=True)
    clip   = core.std.MaskedMerge (src1, src2, mask16)
    return clip

def build_sigmoid_expr (string, inv=False, thr=0.5, cont=6.5):
    core = vs.get_core ()
    x1m0 = "1 {thr} 1 - {cont} * exp 1 + / 1 {cont} {thr} * exp 1 + / -".format (thr=thr, cont=cont)
    x0   = "1 {cont} {thr} * exp 1 + /".format (thr=thr, cont=cont)

    if inv:
       expr = "{thr} 1 " + string + " {x1m0} * {x0} + 0.000001 max / 1 - 0.000001 max log {cont} / -".format (x1m0=x1m0, x0=x0, thr=thr, cont=cont)
    else:
       expr = "1 1 {cont} {thr} " + string + " - * exp + / {x0} - {x1m0} /".format (x1m0=x1m0, x0=x0, thr=thr, cont=cont)
    return expr.format (thr=thr, cont=cont)

def sigmoid_direct (src, thr=0.5, cont=6.5):
    core = vs.get_core ()
    expr = build_sigmoid_expr ("x 65536 /", False, thr, cont)
    clip = core.std.Expr ([src], [expr + " 65536 *"])
    return clip

def sigmoid_inverse (src, thr=0.5, cont=6.5):
    core = vs.get_core ()
    expr = build_sigmoid_expr ("x 65536 /", True, thr, cont)
    clip = core.std.Expr ([src], [expr + " 65536 *"])
    return clip

def linear_and_gamma (src, l2g_flag=True, fulls=True, fulld=None, curve="srgb", gcor=1.0, sigmoid=False, thr=0.5, cont=6.5):
    core = vs.get_core ()
    if curve == "srgb":
       k0    = "0.04045"
       phi   = "12.92"
       alpha = "0.055"
       gamma = "2.4"
    elif curve == "709":
       k0    = "0.081"
       phi   = "4.5"
       alpha = "0.099"
       gamma = "2.22222"
    elif curve == "240":
       k0    = "0.0912"
       phi   = "4.0"
       alpha = "0.1115"
       gamma = "2.22222"
    elif curve == "2020":
       k0    = "0.08145"
       phi   = "4.5"
       alpha = "0.0993"
       gamma = "2.22222"
    else:
       k0    = "0.04045"
       phi   = "12.92"
       alpha = "0.055"
       gamma = "2.4"

    fulld = fulls if fulld is None else fulld

    if fulls == False:
       expr = "x 4096 - 56064 /"
    else:
       expr = "x 65536 /"

    g2l = "{expr} {k0} <= {expr} {phi} / {expr} {alpha} + 1 {alpha} + / log {gamma} * exp ?".format (expr=expr, k0=k0, phi=phi, alpha=alpha, gamma=gamma)
          
    if gcor != 1.0:
       g2l = "{g2l} 0 >= {g2l} log {gcor} * exp {g2l} ?".format (g2l=g2l, gcor=gcor)

    if sigmoid:
       g2l = build_sigmoid_expr (g2l , True , thr, cont)
       l2g = build_sigmoid_expr (expr , False , thr, cont)
    else:
       l2g = expr

    if gcor != 1.0:
       l2g = "{l2g} 0 >= {l2g} log {gcor} * exp {l2g} ?".format (l2g=l2g, gcor=gcor)

    l2g = "{l2g} {k0} {phi} / <= {l2g} {phi} * {l2g} log 1 {gamma} / * exp {alpha} 1 + * {alpha} - ?".format (l2g=l2g, k0=k0, phi=phi, alpha=alpha, gamma=gamma)

    if l2g_flag:
       expr = l2g
    else:
       expr = g2l

    if fulld == False:
       expr = expr + " 56064 * 4096 +"
    else:
       expr = expr + " 65536 *"

    clip  = core.std.Expr ([src], [expr])
    return clip

def gamma_to_linear (src, fulls=True, fulld=None, curve="srgb", gcor=1.0, sigmoid=False, thr=0.5, cont=6.5):
    core  = vs.get_core ()
    clip  = linear_and_gamma (src, False, fulls, fulld, curve, gcor, sigmoid, thr, cont)
    return clip

def linear_to_gamma (src, fulls=True, fulld=None, curve="srgb", gcor=1.0, sigmoid=False, thr=0.5, cont=6.5):
    core  = vs.get_core ()
    clip  = linear_and_gamma (src, True, fulls, fulld, curve, gcor, sigmoid, thr, cont)
    return clip

def sbr16 (src):
    core   = vs.get_core ()
    rg11   = core.rgvs.RemoveGrain(src, 11)
    rg11D  = core.std.MakeDiff (src, rg11)
    rg11DR = core.rgvs.RemoveGrain(rg11D, 11)
    rg11DD = core.std.Expr ([rg11D, rg11DR], ["x y - x 32768 - * 0 < 32768 x y - abs x 32768 - abs < x y - 32768 + x ? ?"])
    clip   = core.std.MakeDiff (src, rg11DD)
    return clip

def clamp16 (src, bright_limit, dark_limit, overshoot=0, undershoot=0):
    core = vs.get_core ()
    os   = overshoot * 256
    us   = undershoot * 256

    clip = core.std.Expr ([src, bright_limit, dark_limit], ["x y {os} + > y {os} + x ? z {us} - < z {us} - x ?".format (os=os, us=us)])
    return clip

def Resize16nr (src, w=None, h=None, sx=0, sy=0, sw=0, sh=0, kernel="spline36", kernelh=None, kernelv=None, fh=1, fv=1, taps=4, a1=None, a2=None, a3=None, kovrspl=1, cnorm=True, center=True, fulls=None, fulld=None, cplace="mpeg2", invks=False, invkstaps=4, noring=True):
    core    = vs.get_core ()
    w       = src.width if w is None else w
    h       = src.height if h is None else h
    kernelh = kernel if kernelh is None else kernelh
    kernelv = kernel if kernelv is None else kernelv
    sr_h    = float (w / src.width)
    sr_v    = float (h / src.height)
    sr_up   = max (sr_h, sr_v)
    sr_dw   = 1.0 / min (sr_h, sr_v)
    sr      = max (sr_up, sr_dw)

    thr   = 2.5
    nrb   = (sr > thr)
    nrf   = (sr < thr + 1.0 and noring)
    nrr   = min (sr - thr, 1.0) if nrb else 1.0
    nrv   = [round ((1.0 - nrr) * 65535), round ((1.0 - nrr) * 65535), round ((1.0 - nrr) * 65535)] if nrb else [0, 0, 0]
    nrm   = core.std.BlankClip (clip=src, width=w, height=h, color=nrv) if nrb and nrf else 0

    main  = core.fmtc.resample (src, w=w, h=h, sx=sx, sy=sy, sw=sw, sh=sh, kernel=kernel, kernelh=kernelh, kernelv=kernelv, fh=fh, fv=fv, taps=taps, a1=a1, a2=a2, a3=a3, kovrspl=kovrspl, cnorm=cnorm, center=center, fulls=fulls, fulld=fulld, cplace=cplace, invks=invks, invkstaps=invkstaps)
    nrng  = core.fmtc.resample (src, w=w, h=h, sx=sx, sy=sy, sw=sw, sh=sh, kernel="gauss", a1=100, center=center, fulls=fulls, fulld=fulld, cplace=cplace) if nrf else main

    clip  = core.rgvs.Repair (main, nrng, 1) if nrf else main
    clip  = core.std.MaskedMerge (main, clip, nrm) if nrf and nrb else clip
    return clip
