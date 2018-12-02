import vapoursynth as vs


def nnedi3_rpow2(clip, rfactor=2, width=None, height=None, correct_shift=True,
                 kernel="spline36", nsize=0, nns=3, qual=None, etype=None, pscrn=None,
                 opt=True, int16_prescreener=None, int16_predictor=None, exp=None):
    """nnedi3_rpow2 is for enlarging images by powers of 2.

    Args:
        rfactor (int): Image enlargement factor.
            Must be a power of 2 in the range [2 to 1024].
        correct_shift (bool): If False, the shift is not corrected.
            The correction is accomplished by using the subpixel
            cropping capability of fmtc's resizers.
        width (int): If correcting the image center shift by using the
            "correct_shift" parameter, width/height allow you to set a
            new output resolution.
        kernel (string): Sets the resizer used for correcting the image
            center shift that nnedi3_rpow2 introduces. This can be any of
            fmtc kernels, such as "cubic", "spline36", etc.
            spline36 is the default one.
        nnedi3_args (mixed): For help with nnedi3 args
            refert to nnedi3 documentation.
    """
    core = vs.get_core()

    # Setting up variables

    if width is None:
        width = clip.width*rfactor
    if height is None:
        height = clip.height*rfactor
    hshift = 0.0
    vshift = -0.5
    pkdnnedi = dict(dh=True, nsize=nsize, nns=nns, qual=qual, etype=etype,
                    pscrn=pscrn, opt=opt, int16_prescreener=int16_prescreener,
                    int16_predictor=int16_predictor, exp=exp)
    pkdchroma = dict(kernel=kernel, sy=-0.5, planes=[2, 3, 3])

    tmp = 1
    times = 0
    while tmp < rfactor:
        tmp *= 2
        times += 1

    # Checks

    if rfactor < 2 or rfactor > 1024:
        raise ValueError("nnedi3_rpow2: rfactor must be between 2 and 1024")

    if tmp != rfactor:
        raise ValueError("nnedi3_rpow2: rfactor must be a power of 2")

    if hasattr(core, 'nnedi3') is not True:
        raise RuntimeError("nnedi3_rpow2: nnedi3 plugin is required")

    if correct_shift or clip.format.subsampling_h:
        if hasattr(core, 'fmtc') is not True:
            raise RuntimeError("nnedi3_rpow2: fmtconv plugin is required")

    # Processing

    last = clip

    for i in range(times):
        field = 1 if i == 0 else 0
        last = core.nnedi3.nnedi3(last, field=field, **pkdnnedi)
        last = core.std.Transpose(last)
        if last.format.subsampling_w:
            # Apparently always using field=1 for the horizontal pass somehow
            # keeps luma/chroma alignment.
            field = 1
            hshift = hshift*2 - 0.5
        else:
            hshift = -0.5
        last = core.nnedi3.nnedi3(last, field=field, **pkdnnedi)
        last = core.std.Transpose(last)

    # Correct vertical shift of the chroma.

    if clip.format.subsampling_h:
        last = core.fmtc.resample(last, w=last.width, h=last.height, **pkdchroma)

    if correct_shift is True:
        last = core.fmtc.resample(last, w=width, h=height, kernel=kernel,
                                  sx=hshift, sy=vshift)

    if last.format.id != clip.format.id:
        last = core.fmtc.bitdepth(last, csp=clip.format.id)

    return last
