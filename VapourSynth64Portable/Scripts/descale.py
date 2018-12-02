from vapoursynth import core, GRAYS, RGBS, GRAY, YUV, RGB  # You need Vapoursynth R37 or newer
from functools import partial


# If yuv444 is True chroma will be upscaled instead of downscaled
# If gray is True the output will be grayscale
def Debilinear(src, width, height, yuv444=False, gray=False, chromaloc=None):
    return Descale(src, width, height, kernel='bilinear', b=None, c=None, taps=None, yuv444=yuv444, gray=gray, chromaloc=chromaloc)

def Debicubic(src, width, height, b=1/3, c=1/3, yuv444=False, gray=False, chromaloc=None):
    return Descale(src, width, height, kernel='bicubic', b=b, c=c, taps=None, yuv444=yuv444, gray=gray, chromaloc=chromaloc)

def Delanczos(src, width, height, taps=3, yuv444=False, gray=False, chromaloc=None):
    return Descale(src, width, height, kernel='lanczos', b=None, c=None, taps=taps, yuv444=yuv444, gray=gray, chromaloc=chromaloc)

def Despline16(src, width, height, yuv444=False, gray=False, chromaloc=None):
    return Descale(src, width, height, kernel='spline16', b=None, c=None, taps=None, yuv444=yuv444, gray=gray, chromaloc=chromaloc)

def Despline36(src, width, height, yuv444=False, gray=False, chromaloc=None):
    return Descale(src, width, height, kernel='spline36', b=None, c=None, taps=None, yuv444=yuv444, gray=gray, chromaloc=chromaloc)


def Descale(src, width, height, kernel='bilinear', b=1/3, c=1/3, taps=3, yuv444=False, gray=False, chromaloc=None):
    src_f = src.format
    src_cf = src_f.color_family
    src_st = src_f.sample_type
    src_bits = src_f.bits_per_sample
    src_sw = src_f.subsampling_w
    src_sh = src_f.subsampling_h

    descale_filter = get_filter(b, c, taps, kernel)

    if src_cf == RGB and not gray:
        rgb = descale_filter(to_rgbs(src), width, height)
        return rgb.resize.Point(format=src_f.id)

    y = descale_filter(to_grays(src), width, height)
    y_f = core.register_format(GRAY, src_st, src_bits, 0, 0)
    y = y.resize.Point(format=y_f.id)

    if src_cf == GRAY or gray:
        return y

    if not yuv444 and ((width % 2 and src_sw) or (height % 2 and src_sh)):
        raise ValueError('Descale: The output dimension and the subsampling are incompatible.')

    uv_f = core.register_format(src_cf, src_st, src_bits, 0 if yuv444 else src_sw, 0 if yuv444 else src_sh)
    uv = src.resize.Spline36(width, height, format=uv_f.id, chromaloc_s=chromaloc)

    return core.std.ShufflePlanes([y,uv], [0,1,2], YUV)


# Helpers

def to_grays(src):
    return src.resize.Point(format=GRAYS)


def to_rgbs(src):
    return src.resize.Point(format=RGBS)


def get_plane(src, plane):
    return core.std.ShufflePlanes(src, plane, GRAY)


def get_filter(b, c, taps, kernel):
    if kernel.lower() == 'bilinear':
        return core.descale.Debilinear
    elif kernel.lower() == 'bicubic':
        return partial(core.descale.Debicubic, b=b, c=c)
    elif kernel.lower() == 'lanczos':
        return partial(core.descale.Delanczos, taps=taps)
    elif kernel.lower() == 'spline16':
        return core.descale.Despline16
    elif kernel.lower() == 'spline36':
        return core.descale.Despline36
    else:
        raise ValueError('Descale: Invalid kernel specified.')
