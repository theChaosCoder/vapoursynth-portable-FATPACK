import vapoursynth as vs


def _clamp(minimum, x, maximum):
    return int(max(minimum, min(round(x), maximum)))


def _m4(x, m=4.0):
    return 16 if x < 16 else int(round(x / m) * m)


def psharpen(clip, strength=25, threshold=75, ss_x=1.0, ss_y=1.0,
             dest_x=None, dest_y=None):
    """From http://forum.doom9.org/showpost.php?p=683344&postcount=28

    Sharpeing function similar to LimitedSharpenFaster.

    Args:
        strength (int): Strength of the sharpening.
        threshold (int): Controls "how much" to be sharpened.
        ss_x (float): Supersampling factor (reduce aliasing on edges).
        ss_y (float): Supersampling factor (reduce aliasing on edges).
        dest_x (int): Output resolution after sharpening.
        dest_y (int): Output resolution after sharpening.
    """
    core = vs.get_core()

    ox = clip.width
    oy = clip.height

    bd = clip.format.bits_per_sample
    max_ = 2 ** bd - 1
    scl = (max_ + 1) // 256
    x = 'x {} /'.format(scl) if bd != 8 else 'x'
    y = 'y {} /'.format(scl) if bd != 8 else 'y'

    if dest_x is None:
        dest_x = ox
    if dest_y is None:
        dest_y = oy

    strength = _clamp(0, strength, 100)
    threshold = _clamp(0, threshold, 100)

    if ss_x < 1.0:
        ss_x = 1.0
    if ss_y < 1.0:
        ss_y = 1.0

    if ss_x != 1.0 or ss_y != 1.0:
        clip = core.resize.Lanczos(clip, width=_m4(ox*ss_x), height=_m4(oy*ss_y))

    orig = clip

    if orig.format.num_planes != 1:
        clip = core.std.ShufflePlanes(clips=clip, planes=[0],
                                      colorfamily=vs.GRAY)
    val = clip

    max_ = core.std.Maximum(clip)
    min_ = core.std.Minimum(clip)

    nmax = core.std.Expr([max_, min_], ['x y -'])
    nval = core.std.Expr([val, min_], ['x y -'])

    s = strength/100.0
    t = threshold/100.0
    x0 = t * (1.0 - s) / (1.0 - (1.0 - t) * (1.0 - s))

    expr = ('{x} {y} / 2 * 1 - abs {x0} < {s} 1 = {x} {y} 2 / = 0 {y} 2 / ? '
            '{x} {y} / 2 * 1 - abs 1 {s} - / ? {x} {y} / 2 * 1 - abs 1 {t} - '
            '* {t} + ? {x} {y} 2 / > 1 -1 ? * 1 + {y} * 2 / {scl} *').format(
                x=x, y=y, x0=x0, t=t, s=s, scl=scl)

    nval = core.std.Expr([nval, nmax], [expr])

    val = core.std.Expr([nval, min_], ['x y +'])

    if orig.format.num_planes != 1:
        clip = core.std.ShufflePlanes(clips=[val, orig], planes=[0, 1, 2],
                                      colorfamily=orig.format.color_family)

    if ss_x != 1.0 or ss_y != 1.0 or dest_x != ox or dest_y != oy:
        clip = core.resize.Lanczos(clip, width=dest_x, height=dest_y)

    return clip
