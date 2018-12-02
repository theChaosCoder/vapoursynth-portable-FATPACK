import vapoursynth as vs


# Based on ASTDR DeRainbow function v1.74 for Avisynth


# amount doesn't go below 0 because std.Convolution doesn't take coefficients greater than 1023.
def BlurForASTDR(input_clip, amount=0, planes=None):
    lower_limit = 0
    upper_limit = 1.5849625
    if amount < lower_limit or amount > upper_limit:
        raise ValueError("BlurForASTDR: amount must be between {} and {}.".format(lower_limit, upper_limit))

    center_weight = 1 / pow(2, amount)
    side_weight = (1 - center_weight) / 2

    corner = int(side_weight * side_weight * 1000 + 0.5)
    side = int(side_weight * center_weight * 1000 + 0.5)
    center = int(center_weight * center_weight * 1000 + 0.5)

    blur_matrix = [corner,   side, corner,
                     side, center,   side,
                   corner,   side, corner]

    return input_clip.std.Convolution(matrix=blur_matrix, planes=planes)


def ASTDR(input_clip, strength=None, tempsoftth=None, tempsoftrad=None, tempsoftsc=None, blstr=None, tht=None, fluxstv=None, dcn=None, edgem=None, exmc=None, edgemprefil=None, separated=None):
    core = vs.get_core()

    sisfield = separated
    fnomc = sisfield and not exmc


    if strength is None:
        strength = 5

    if tempsoftth is None:
        tempsoftth = 30

    if tempsoftrad is None:
        tempsoftrad = 3

    if tempsoftsc is None:
        tempsoftsc = 3
    
    if blstr is None:
        if sisfield:
            blstr = 0.3
        else:
            blstr = 0.5

    if tht is None:
        tht = 255

    if fluxstv is None:
        if sisfield:
            fluxstv = 60
        else:
            fluxstv = 75

    if dcn is None:
        dcn=15

    if edgem is None:
        edgem = False

    if exmc is None:
        exmc = False

    if separated is None:
        separated = False

    inrainev = input_clip
    if fnomc:
        inrainev = input_clip.std.SelectEvery(cycle=2, offsets=0)

    filtered_uv = inrainev
    if not exmc:
        filtered_uv = inrainev.decross.DeCross(thresholdy=15, noise=dcn, margin=1)

    flux_spatial_threshold = fluxstv
    if sisfield:
        flux_spatial_threshold = fluxstv // 2

    filtered_uv = filtered_uv.flux.SmoothST(temporal_threshold=fluxstv, spatial_threshold=flux_spatial_threshold, planes=[1, 2])

    if not sisfield:
        filtered_uv = filtered_uv.warp.AWarpSharp2(depth=4, chroma=0, cplace="mpeg2", planes=[1, 2])

    chrom_spac = strength * 3 / 5
    if sisfield:
        chrom_spac = strength * 2 / 5
    filtered_uv = filtered_uv.hqdn3d.Hqdn3d(lum_spac=0, lum_tmp=0, chrom_spac=chrom_spac, chrom_tmp=strength).focus2.TemporalSoften2(radius=tempsoftrad, luma_threshold=0, chroma_threshold=tempsoftth, scenechange=tempsoftsc, mode=2)
    filtered_uv = BlurForASTDR(input_clip=filtered_uv, amount=blstr, planes=[1, 2])

    if not sisfield:
        filtered_uv = filtered_uv.warp.AWarpSharp2(depth=4, chroma=0, cplace="mpeg2", planes=[1, 2])

    sigma = 1
    sigma3 = 4
    if sisfield:
        sigma = 0.7
        sigma3 = 3

    filtered_uv = filtered_uv.fft3dfilter.FFT3DFilter(sigma=sigma, sigma3=sigma3, planes=[1, 2], degrid=1)

    if fnomc:
        filtered_odd = input_clip.std.SelectEvery(cycle=2, offsets=1)
        filtered_odd = filtered_odd.decross.DeCross(thresholdy=15, noise=dcn, margin=1).flux.SmoothST(temporal_threshold=fluxstv, spatial_threshold=flux_spatial_threshold, planes=[1, 2])
        filtered_odd = filtered_odd.hqdn3d.Hqdn3d(lum_spac=0, lum_tmp=0, chrom_spac=chrom_spac, chrom_tmp=strength).focus2.TemporalSoften2(radius=tempsoftrad, luma_threshold=0, chroma_threshold=tempsoftth, scenechange=tempsoftsc, mode=2)
        filtered_odd = BlurForASTDR(input_clip=filtered_odd, amount=blstr, planes=[1, 2])
        filtered_odd = filtered_odd.fft3dfilter.FFT3DFilter(sigma=sigma, sigma3=sigma3, planes=[1, 2], degrid=1)
        filtered_uv = core.std.Interleave([filtered_uv, filtered_odd])


    last = filtered_uv

    if not exmc:
        import adjust

        momask = adjust.Tweak(clip=input_clip, sat=1.1).motionmask.MotionMask(th1=1, th2=1, tht=tht)
        momaskinv = momask.std.Maximum(planes=0).std.Inflate(planes=0).std.Invert(planes=0).std.Levels(min_in=0, gamma=2, max_in=255, min_out=0, max_out=255, planes=0)

        filtered = core.std.MaskedMerge(clipa=filtered_uv, clipb=input_clip, mask=momaskinv, first_plane=True, planes=[1, 2])
        last = core.std.MaskedMerge(clipa=input_clip, clipb=filtered, mask=momask.std.Maximum(planes=[1, 2]).std.Inflate(planes=[1, 2]), planes=[1, 2])

    if edgem:
        if edgemprefil is None:
            edgemprefil = input_clip

        edgemclip = edgemprefil.std.Sobel(planes=0).std.Binarize(threshold=5, planes=0).std.Maximum(planes=0).std.Inflate(planes=0)
        last = core.std.MaskedMerge(clipa=input_clip, clipb=last, mask=edgemclip, first_plane=True, planes=[1, 2])

    return last


# Partial port of the sbr function from SMDegrain Avisynth script.
# Only the parts needed by MinBlurForASTDRmc are included.

def sbrForASTDRmc(input_clip):
    core = vs.get_core()

    matrix11 = [1, 2, 1,
                2, 4, 2,
                1, 2, 1]

    rg11 = input_clip.std.Convolution(matrix=matrix11, planes=0)

    rg11D = core.std.MakeDiff(clipa=input_clip, clipb=rg11, planes=0)

    rg11DD = core.std.MakeDiff(clipa=rg11D, clipb=rg11D.std.Convolution(matrix=matrix11, planes=0), planes=0)
    
    rg11DD = core.std.Expr(clips=[rg11DD, rg11D], expr=["x 128 - y 128 - * 0 < 128 x 128 - abs y 128 - abs < x y ? ?", ""])

    return core.std.MakeDiff(clipa=input_clip, clipb=rg11DD, planes=0)


# Partial port of the MinBlur function from SMDegrain Avisynth script.
# Only the parts needed by ASTDRmc are included.

def MinBlurForASTDRmc(input_clip, r=1, blurrep=False, planes=None):
    core = vs.get_core()

    if r < 0 or r > 3:
        raise ValueError("MinBlurForASTDRmc: r must be between 0 and 3 (inclusive).")


    matrix11 = [1, 2, 1,
                2, 4, 2,
                1, 2, 1]

    matrix20 = [1, 1, 1,
                1, 1, 1,
                1, 1, 1]

    if r == 0:
        RG11D = sbrForASTDRmc(input_clip=input_clip)
    else:
        RG11D = input_clip.std.Convolution(matrix=matrix11, planes=planes)
        # Zero..two passes:
        for i in range(1, r):
            RG11D = RG11D.std.Convolution(matrix=matrix20, planes=planes)

    RG11D = core.std.MakeDiff(clipa=input_clip, clipb=RG11D, planes=planes)

    if r < 2:
        RG4D = input_clip.std.Median(planes=planes)
    else:
        RG4D = input_clip.ctmf.CTMF(radius=r, planes=planes)

    RG4D = core.std.MakeDiff(clipa=input_clip, clipb=RG4D, planes=planes)

    expr = "x 128 - y 128 - * 0 < 128 x 128 - abs y 128 - abs < x y ? ?"

    DD = core.std.Expr(clips=[RG11D, RG4D], expr=[expr if i in planes else '' for i in range(input_clip.format.num_planes)])

    last = core.std.MakeDiff(input_clip, DD, planes=planes)

    if blurrep:
        last = core.rgvs.Repair(last, input_clip.rgvs.RemoveGrain(mode=[17, 0]), mode=[9, 0])

    return last


def mc4ASTDRmc(input_clip, radius, prefil, thsad, chroma, motion_vectors=None):
    core = vs.get_core()

    if radius == 1:
        thsad = None

    masuper = prefil.mv.Super()
    mcsuper = input_clip.mv.Super(levels=1)

    f = []
    b = []

    for i in range(1, radius + 1):
        if motion_vectors is None:
            forward_vectors = core.mv.Analyse(super=masuper, delta=i, isb=False, chroma=chroma)
            backward_vectors = core.mv.Analyse(super=masuper, delta=i, isb=True, chroma=chroma)
        else:
            forward_vectors = motion_vectors[(radius - 1) - (i - 1)]
            backward_vectors = motion_vectors[radius + (i - 1)]

        f.append(core.mv.Compensate(clip=input_clip, super=mcsuper, vectors=forward_vectors, thsad=thsad))
        b.append(core.mv.Compensate(clip=input_clip, super=mcsuper, vectors=backward_vectors, thsad=thsad))


    f.reverse()
    f.append(input_clip)
    f.extend(b)

    return core.std.Interleave(clips=f)


def ASTDRmc(input_clip, strength=None, tempsoftth=None, tempsoftrad=None, tempsoftsc=None, blstr=None, tht=255, fluxstv=None, dcn=None, edgem=None, thsad=None, prefil=None, chroma=False, edgemprefil=None, separated=False, motion_vectors=None):
    core = vs.get_core()

    sisfield = separated

    if tempsoftrad is None:
        tempsoftrad = 3
        if sisfield:
            tempsoftrad = 5

    tempsoftrad = min(tempsoftrad, 5)

    if tempsoftth is None:
        tempsoftth = 30
        if sisfield:
            tempsoftth = 50

    if thsad is None:
        thsad = tht

    if edgem is None:
        edgem = sisfield

    exprefil = prefil is not None

    if motion_vectors is not None:
        if sisfield:
            raise ValueError("ASTDRmc: motion_vectors cannot be used when separated is True.")

        if not isinstance(motion_vectors, list):
            raise TypeError("ASTDRmc: motion_vectors must be a list.")

        if len(motion_vectors) != tempsoftrad * 2:
            raise ValueError("ASTDRmc: motion_vectors must be a list of {} clips (tempsoftrad * 2).".format(tempsoftrad * 2))

        for i in range(len(motion_vectors)):
            if not isinstance(motion_vectors[i], vs.VideoNode):
                raise TypeError("ASTDRmc: motion_vectors[{}] must be a clip, not {}.".format(i, type(motion_vectors[i])))


    if prefil is None:
        # XXX planes parameter?
        if chroma:
            if sisfield:
                prefil = BlurForASTDR(input_clip=MinBlurForASTDRmc(input_clip=input_clip, r=3, planes=[1, 2]), amount=1)
            else:
                prefil = MinBlurForASTDRmc(input_clip=input_clip, r=3, blurrep=True)
        else:
            prefil = BlurForASTDR(input_clip=input_clip, amount=1.5)


    if sisfield:
        if edgemprefil is not None:
            edgemprefil_even = edgemprefil_odd = edgemprefil
        elif not exprefil:
            edgemprefil_even = edgemprefil_odd = prefil
        else:
            edgemprefil_even = edgemprefil_odd = None

        if edgemprefil_even is not None:
            # SelectEven followed by duplicating every frame radius * 2 + 1 times
            edgemprefil_even = edgemprefil_even.std.SelectEvery(cycle=2, offsets=[0 for i in range(tempsoftrad * 2 + 1)])

        if edgemprefil_odd is not None:
            # SelectOdd followed by duplicating every frame radius * 2 + 1 times
            edgemprefil_even = edgemprefil_even.std.SelectEvery(cycle=2, offsets=[1 for i in range(tempsoftrad * 2 + 1)])

        ieven = mc4ASTDRmc(input_clip=input_clip.std.SelectEvery(cycle=2, offsets=0), radius=tempsoftrad, prefil=prefil.std.SelectEvery(cycle=2, offsets=0), thsad=thsad, chroma=chroma)
        asteven = ASTDR(input_clip=ieven, strength=strength, tempsoftth=tempsoftth, tempsoftrad=tempsoftrad, tempsoftsc=tempsoftsc, blstr=blstr, tht=tht, fluxstv=fluxstv, dcn=dcn, edgem=edgem, exmc=True, edgemprefil=edgemprefil_even)
        asteven = asteven.std.SelectEvery(cycle=tempsoftrad * 2 + 1, offsets=tempsoftrad)

        iodd = mc4ASTDRmc(input_clip=input_clip.std.SelectEvery(cycle=2, offsets=1), radius=tempsoftrad, prefil=prefil.std.SelectEvery(cycle=2, offsets=1), thsad=thsad, chroma=chroma)
        astodd = ASTDR(input_clip=iodd, strength=strength, tempsoftth=tempsoftth, tempsoftrad=tempsoftrad, tempsoftsc=tempsoftsc, blstr=blstr, tht=tht, fluxstv=fluxstv, dcn=dcn, edgem=edgem, exmc=True, edgemprefil=edgemprefil_odd)
        astodd = astodd.std.SelectEvery(cycle=tempsoftrad * 2 + 1, offsets=tempsoftrad)

        ASTDRclip = core.std.Interleave(clips=[asteven, astodd])
    else: # not sisfield
        if edgemprefil is None:
            edgemprefil = prefil

        # duplicate every frame radius * 2 + 1 times
        edgemprefil = core.std.Interleave(clips=[edgemprefil for i in range(tempsoftrad * 2 + 1)])

        mcclip = mc4ASTDRmc(input_clip=input_clip, radius=tempsoftrad, prefil=prefil, thsad=thsad, chroma=chroma, motion_vectors=motion_vectors)

        ASTDRclip = ASTDR(input_clip=mcclip, strength=strength, tempsoftth=tempsoftth, tempsoftrad=tempsoftrad, tempsoftsc=tempsoftsc, blstr=blstr, tht=tht, fluxstv=fluxstv, dcn=dcn, edgem=edgem, exmc=True, edgemprefil=edgemprefil)
        ASTDRclip = ASTDRclip.std.SelectEvery(cycle=tempsoftrad * 2 + 1, offsets=tempsoftrad)

        # XXX technically it should be TEdgeMask
        derbmask = input_clip.std.Prewitt(planes=0).std.Inflate(planes=0)

        ASTDRclip = core.std.MaskedMerge(clipa=input_clip, clipb=ASTDRclip, mask=derbmask, first_plane=True, planes=[1, 2])

    return ASTDRclip
