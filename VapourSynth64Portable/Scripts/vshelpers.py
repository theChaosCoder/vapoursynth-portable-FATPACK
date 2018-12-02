import vapoursynth as vs


def clamp(minimum, x, maximum):
    return int(max(minimum, min(round(x), maximum)))


def m4(x, m=4.0):
    return 16 if x < 16 else int(round(x / m) * m)


def build_mask(c, edgelvl=40, mode=0):
    core = vs.get_core()

    if mode < 0 or mode > 1:
        raise ValueError('mode should be an int between 1 and 3.')

    edgelvl = edgelvl << c.format.bits_per_sample-8

    if c.format.id != vs.GRAY:
        c = core.std.ShufflePlanes(clips=c, planes=[0], colorfamily=vs.GRAY)

    if mode == 0:
        m = core.std.Prewitt(c, edgelvl, edgelvl)
        m = core.std.Maximum(m)
    elif mode == 1:
        m = core.std.Convolution(clip=c, matrix=[1, 2, 1, 0, 0, 0, -1, -2, -1])
        m = core.std.Binarize(m)
        m = core.std.Maximum(m)

    return m


def mt_lut(c1, expr, planes=[0]):
    core = vs.get_core()
    max_ = 2 ** c1.format.bits_per_sample - 1
    lut_range = range(max_ + 1)
    lut = [clamp(0, expr(x), max_) for x in lut_range]
    return core.std.Lut(clip=c1, lut=lut, planes=planes)


def mt_lutxy(c1, c2, expr, planes=[0]):
    core = vs.get_core()
    max_ = 2 ** c1.format.bits_per_sample - 1
    lut_range = range(max_ + 1)
    lut = []
    for y in lut_range:
        for x in lut_range:
            lut.append(clamp(0, expr(x, y), max_))
    return core.std.Lut2(c1, c2, lut=lut, planes=planes)


def logic(c1, c2, mode, th1=0, th2=0, planes=[0]):
    core = vs.get_core()
    mode = mode.lower()
    if mode == 'and':
        expr = 'x y and'
    elif mode == 'andn':
        expr = 'x y not and'
    elif mode == 'or':
        expr = 'x y or'
    elif mode == 'xor':
        expr = 'x y xor'
    elif mode == 'min':
        expr = 'x {th1} + y {th2} + min'.format(th1=th1, th2=th2)
    elif mode == 'max':
        expr = 'x {th1} + y {th2} + max'.format(th1=th1, th2=th2)
    else:
        raise ValueError('"{}" is not a valid mode for logic (and, andn, or, xor, min, max)'.format(mode))
    expr = [expr if 0 in planes else '', expr if 1 in planes else '', expr if 2 in planes else '']
    return core.std.Expr([c1, c2], expr=expr)


def get_luma(c):
    core = vs.get_core()
    return core.std.ShufflePlanes(clips=c, planes=[0], colorfamily=vs.GRAY)


def merge_chroma(c1, c2):
    core = vs.get_core()
    return core.std.ShufflePlanes(clips=[c1, c2], planes=[0, 1, 2], colorfamily=c2.format.color_family)


def rsoften(clip, radius=5, kernel='bicubic', bits=None):
    core = vs.get_core()

    blur = core.fmtc.resample(clip, m4(clip.width//radius), m4(clip.height//radius), kernel=kernel)
    blur = core.fmtc.resample(clip, clip.width, clip.height, kernel=kernel)

    if blur.format.bits_per_sample != clip.format.bits_per_sample or bits is not None:
        blur = core.fmtc.bitdepth(blur, bits=clip.format.bits_per_sample)

    return blur


def fit(clipa, clipb):
    core = vs.get_core()

    bd = clipb.format.bits_per_sample
    max_ = 2 ** bd - 1
    mid = (max_ + 1) // 2

    if clipb.format.num_planes > 1:
        if clipb.format.color_family == vs.RGB:
            color = [max_, max_, max_]
        else:
            color = [max_, mid, mid]
    else:
        color = [max_]

    if clipa.width > clipb.width:
        clipb = core.std.AddBorders(clip=clipb, left=0, right=clipa.width - clipb.width, color=color)
    elif clipa.width < clipb.width:
        clipb = core.std.CropRel(clip=clipb, left=0, right=clipb.width - clipa.width)

    if clipa.height > clipb.height:
        clipb = core.std.AddBorders(clip=clipb, top=0, bottom=clipa.height - clipb.height, color=color)
    elif clipa.height < clipb.height:
        clipb = core.std.CropRel(clip=clipb, top=0, bottom=clipb.height - clipa.height)

    return clipb


def move(clips, x, y):
    core = vs.get_core()

    moved = None

    for clip in clips:
        if clip.format.num_planes == 1:
            color = [(2 ** clip.format.bits_per_sample) - 1]
        else:
            color = None

        if x != 0 or y != 0:
            if x >= 0:
                right = 0
                left = x
            else:
                right = abs(x)
                left = 0
            if y >= 0:
                top = 0
                bottom = y
            else:
                top = abs(y)
                bottom = 0

            clip = core.std.AddBorders(clip=clip, left=left, right=right, top=top, bottom=bottom, color=color)
            clip = core.std.CropRel(clip=clip, left=right, right=left, top=bottom, bottom=top)

        if clip is isinstance(list()):
            moved.append(clip)
        else:
            moved = clip

    return moved


def get_decoder():
    core = vs.get_core()

    try:
        core.lsmas
    except NameError:
        try:
            core.ffms2
        except NameError:
            raise NameError('No suitable source filter was found, please, install either ffms2 or lsmas.')
        else:
            vsource = core.ffms2.Source
    else:
        vsource = core.lsmas.LWLibavSource

    return vsource


def subtract(c1, c2, luma=126, planes=[0]):
    core = vs.get_core()

    expr = ('{luma} x + y -').format(luma=luma)
    expr = [(i in planes) * expr for i in range(c1.format.num_planes)]

    return core.std.Expr([c1, c2], expr)


def starmask(src, mode=1):
    core = vs.get_core()

    clip = get_luma(src)

    if mode == 1:
        clean = core.rgvs.RemoveGrain(clip, 17)
        diff = core.std.MakeDiff(clip, clean)
        final = core.std.ShufflePlanes(diff, 0, vs.GRAY).resize.Bicubic(format=vs.YUV420P8)
        final = core.std.Levels(final, 40, 168, 0.350, 0, 255)
        final = core.rgvs.RemoveGrain(final, [7, 0])
        final = core.std.Prewitt(clip=final, min=4, max=16)
    else:
        clean = core.rgvs.RemoveGrain(core.rgvs.Repair(core.avs.deen(src, 'a3d', 4, 12, 0), src, 15), 21)
        coord = [int(s) for s in core.avs.mt_circle(1).split(' ')]
        pmask = core.avs.mt_edge(src, 'roberts', 0, 2, 0, 2).std.Maximum(coordinates=coord).std.Invert()
        fmask = core.std.MaskedMerge(clean, src, pmask)
        subt = subtract(fmask, src)
        final = core.std.Deflate(core.avs.mt_edge(subt, 'roberts', 0, 0, 0, 0))

    return final
