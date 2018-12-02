import vapoursynth as vs


def RainbowSmooth(clip, radius=3, lthresh=0, hthresh=220, mask="original"):
    core = vs.get_core()
    
    if isinstance(mask, str):
        if mask == "original":
            mask = core.std.Expr(clips=[clip.std.Maximum(planes=0), clip.std.Minimum(planes=0)], expr=["x y - 90 > 255 x y - 255 90 / * ?", "", ""])
        elif mask == "prewitt":
            mask = core.std.Prewitt(clip=clip, planes=0)
        elif mask == "sobel":
            mask = core.std.Sobel(clip=clip, planes=0)
        elif mask == "tcanny":
            mask = core.tcanny.TCanny(clip)
        elif mask == "fast_sobel":
            import kagefunc as kage

            mask = kage.fast_sobel(clip)
        elif mask == "kirsch":
            import kagefunc as kage

            mask = kage.kirsch(clip)
        elif mask == "retinex_edgemask":
            import kagefunc as kage

            mask = kage.retinex_edgemask(clip)

    lderain = clip

    if lthresh > 0:
        lderain = clip.smoothuv.SmoothUV(radius=radius, threshold=lthresh, interlaced=False)

    hderain = clip.smoothuv.SmoothUV(radius=radius, threshold=hthresh, interlaced=False)

    if hthresh > lthresh:
        return core.std.MaskedMerge(clipa=lderain, clipb=hderain, mask=mask, planes=[1, 2], first_plane=True)
    else:
        return lderain
