import vapoursynth as vs


def BlurForDFMDerainbow(clip, amount=0, planes=None):
    lower_limit = 0
    upper_limit = 1.5849625
    if amount < lower_limit or amount > upper_limit:
        raise ValueError("BlurForDFMDerainbow: amount must be between {} and {}.".format(lower_limit, upper_limit))

    center_weight = 1 / pow(2, amount)
    side_weight = (1 - center_weight) / 2

    corner = int(side_weight * side_weight * 1000 + 0.5)
    side = int(side_weight * center_weight * 1000 + 0.5)
    center = int(center_weight * center_weight * 1000 + 0.5)

    blur_matrix = [corner,   side, corner,
                     side, center,   side,
                   corner,   side, corner]

    return clip.std.Convolution(matrix=blur_matrix, planes=planes)


def Flux5FramesT(clip, temporal_threshold=7, planes=None):
    core = vs.get_core()

    median = core.tmedian.TemporalMedian(clip=clip, radius=2, planes=planes)

    if planes is None or 0 in planes:
        luma_threshold = temporal_threshold
    else:
        luma_threshold = 0

    if planes is None or (1 in planes and 2 in planes):
        chroma_threshold = temporal_threshold
    else:
        chroma_threshold = 0

    average = core.focus2.TemporalSoften2(clip=clip, radius=2, luma_threshold=luma_threshold, chroma_threshold=chroma_threshold, scenechange=24)

    median_diff = core.std.MakeDiff(clipa=clip, clipb=median, planes=planes)
    average_diff = core.std.MakeDiff(clipa=clip, clipb=average, planes=planes)

    expression = "x 128 - y 128 - * 0 < 128 x 128 - abs y 128 - abs < x y ? ?"

    expr = [
            expression if planes is None or 0 in planes else "",
            expression if planes is None or 1 in planes else "",
            expression if planes is None or 2 in planes else ""
    ]

    DD = core.std.Expr(clips=[median_diff, average_diff], expr=expr)

    return core.std.MakeDiff(clipa=clip, clipb=DD, planes=planes)


def DFMDerainbow(clip, maskthresh=10, mask=False, interlaced=False, radius=None):
    if radius is None:
        radius = 1

    if radius < 1 or radius > 2:
        raise ValueError("DFMDerainbow: radius must be 1 or 2.")

    core = vs.get_core()

    if interlaced:
        clip = clip.std.SeparateFields(tff=True)

    if radius == 1:
        first = clip.flux.SmoothT(temporal_threshold=17, planes=[1, 2])
    elif radius == 2:
        first = Flux5FramesT(clip=clip, temporal_threshold=17, planes=[1, 2])
    first = BlurForDFMDerainbow(clip=clip, amount=1.5, planes=[1, 2])

    themask = core.std.MakeDiff(clipa=clip, clipb=first, planes=[1, 2])
    themask = core.std.Levels(clip=themask, min_in=108, gamma=1, max_in=148, min_out=0, max_out=255, planes=[1, 2])
    themask = core.msmoosh.MSharpen(clip=themask, mask=True, threshold=maskthresh * 100 / 255, planes=[1, 2])
    themask = core.std.Invert(clip=themask, planes=[1, 2])
    themask = BlurForDFMDerainbow(clip=themask, amount=0.5, planes=[1, 2])
    themask = core.std.Levels(clip=themask, min_in=0, gamma=2, max_in=255, min_out=0, max_out=255, planes=[1, 2])
    themask = BlurForDFMDerainbow(clip=themask, amount=0.5, planes=[1, 2])

    if mask:
        return themask
    else:
        fixed = clip.flux.SmoothST(temporal_threshold=17, spatial_threshold=14, planes=[1, 2])
        if radius == 1:
            fixed = fixed.flux.SmoothT(temporal_threshold=17, planes=[1, 2])
        elif radius == 2:
            fixed = Flux5FramesT(clip=fixed, temporal_threshold=17, planes=[1, 2])
        fixed = fixed.minideen.MiniDeen(radius=4, threshold=14, planes=[1, 2])
        fixed = BlurForDFMDerainbow(clip=fixed, amount=1.0, planes=[1, 2])

        output = core.std.MaskedMerge(clipa=fixed, clipb=clip, mask=themask, planes=[1, 2])

        if interlaced:
            output = core.std.DoubleWeave(clip=output, tff=True)
            output = core.std.SelectEvery(clip=output, cycle=2, offsets=0)

        return output


def DFMDerainbowMC(clip, maskthresh=12, radius=1, motion_vectors=None):
    # Do inverse telecine first.

    if radius < 1 or radius > 2:
        raise ValueError("DFMDerainbowMC: radius must be 1 or 2.")

    if motion_vectors is not None:
        if not isinstance(motion_vectors, list):
            raise TypeError("DFMDerainbowMC: motion_vectors must be a list.")

        if len(motion_vectors) != radius * 2:
            raise ValueError("DFMDerainbowMC: motion_vectors must be a list of {} clips (radius * 2).".format(radius * 2))

        for i in range(len(motion_vectors)):
            if not isinstance(motion_vectors[i], vs.VideoNode):
                raise TypeError("DFMDerainbowMC: motion_vectors[{}] must be a clip, not {}.".format(i, type(motion_vectors[i])))

    core = vs.get_core()

    prefiltered = clip.fft3dfilter.FFT3DFilter(sigma=1.5, bw=16, bh=16, bt=3, ow=8, oh=8, planes=0)

    derbsuperfilt = core.mv.Super(clip=prefiltered) # all levels for Analyse
    derbsuper = core.mv.Super(clip=clip, levels=1) # one level is enough for Compensate

    if radius == 1:
        if motion_vectors is None:
            derbforward_vectors = core.mv.Analyse(super=derbsuperfilt, isb=False, chroma=False)
            derbbackward_vectors = core.mv.Analyse(super=derbsuperfilt, isb=True, chroma=False)
        else:
            derbforward_vectors = motion_vectors[0]
            derbbackward_vectors = motion_vectors[1]

        derbforward_compensation = core.mv.Compensate(clip=clip, super=derbsuper, vectors=derbforward_vectors)
        derbbackward_compensation = core.mv.Compensate(clip=clip, super=derbsuper, vectors=derbbackward_vectors)

        compensated = core.std.Interleave(clips=[derbforward_compensation, clip, derbbackward_compensation])
    elif radius == 2:
        if motion_vectors is None:
            derbforward_vectors2 = core.mv.Analyse(super=derbsuperfilt, isb=False, delta=2, overlap=4, chroma=True, search=5, searchparam=4)
            derbforward_vectors1 = core.mv.Analyse(super=derbsuperfilt, isb=False, delta=1, overlap=4, chroma=True, search=5, searchparam=4)
            derbbackward_vectors1 = core.mv.Analyse(super=derbsuperfilt, isb=True, delta=1, overlap=4, chroma=True, search=5, searchparam=4)
            derbbackward_vectors2 = core.mv.Analyse(super=derbsuperfilt, isb=True, delta=2, overlap=4, chroma=True, search=5, searchparam=4)
        else:
            derbforward_vectors2 = motion_vectors[0]
            derbforward_vectors1 = motion_vectors[1]
            derbbackward_vectors1 = motion_vectors[2]
            derbbackward_vectors2 = motion_vectors[3]

        derbforward_compensation2 = core.mv.Compensate(clip=clip, super=derbsuper, vectors=derbforward_vectors2, thscd1=600, thscd2=160)
        derbforward_compensation1 = core.mv.Compensate(clip=clip, super=derbsuper, vectors=derbforward_vectors1, thscd1=600, thscd2=160)
        derbbackward_compensation1 = core.mv.Compensate(clip=clip, super=derbsuper, vectors=derbbackward_vectors1, thscd1=600, thscd2=160)
        derbbackward_compensation2 = core.mv.Compensate(clip=clip, super=derbsuper, vectors=derbbackward_vectors2, thscd1=600, thscd2=160)

        compensated = core.std.Interleave(clips=[derbforward_compensation2, derbforward_compensation1, clip, derbbackward_compensation1, derbbackward_compensation2])


    derainbowed = DFMDerainbow(clip=compensated, maskthresh=maskthresh, radius=radius)

    derainbowed = core.std.SelectEvery(clip=derainbowed, cycle=radius * 2 + 1, offsets=radius)

    derbmask = clip.std.Prewitt(planes=0) #.std.Deflate(planes=0)
    
    derainbowed = core.std.MaskedMerge(clipa=clip, clipb=derainbowed, mask=derbmask, first_plane=True, planes=[1, 2])

    return derainbowed
