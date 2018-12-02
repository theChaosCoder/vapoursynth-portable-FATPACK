import vapoursynth as vs
import mvsfunc as mvf
import fvsfunc as fvf
from functools import partial

core = vs.core


# TODO fixedge port


def inverse_scale(source: vs.VideoNode, width=None, height=720, kernel='bilinear', kerneluv='blackman', taps=4,
                  a1=1 / 3, a2=1 / 3, invks=True, mask_detail=False, masking_areas=None, mask_threshold=0.05,
                  show_mask=False, denoise=False, bm3d_sigma=1, knl_strength=0.4, use_gpu=True) -> vs.VideoNode:
    """
    source = input clip
    width, height, kernel, taps, a1, a2 are parameters for resizing.
    mask_detail, masking_areas, mask_threshold are parameters for masking; mask_detail = False to disable.
    masking_areas takes frame tuples to define areas which will be masked (e.g. opening and ending)
    masking_areas = [[1000, 2500], [30000, 32000]]. Start and end frame are inclusive.
    mask_threshold is the binarization threshold. Value must be normalized for floats (0-1) or an 8-bit integer.
    denoise, bm3d_sigma, knl_strength, use_gpu are parameters for denoising; denoise = False to disable
    use_gpu = True -> chroma will be denoised with KNLMeansCL (faster)
    """
    if source.format.bits_per_sample != 32:
        # If this returns an error, make sure you're using R39 or newer
        source = source.resize.Point(format=source.format.replace(bits_per_sample=32, sample_type=vs.FLOAT))
    luma = getY(source)
    if width is None:
        width = getw(height, ar=source.width / source.height)
    if mask_threshold > 1:
        mask_threshold /= 255
    planes = clip_to_plane_array(source)
    if denoise and use_gpu:
        # I'd really like to use the newer versions that support processing both chroma planes in a single call,
        # but I just can't get them to work on my system. I should probably report that as a bug at some point.
        planes[1], planes[2] = [core.knlm.KNLMeansCL(plane, a=2, h=knl_strength, d=3, device_type='gpu', device_id=0)
                                for plane in planes[1:]]
    planes = inverse_scale_clip_array(planes, width, height, kernel, kerneluv, taps, a1, a2, invks)

    if mask_detail:
        mask = generate_detail_mask(luma, planes[0], kernel, taps, a1, a2, mask_threshold)
        if show_mask:
            return mask
        if masking_areas is None:
            planes[0] = apply_mask(luma, planes[0], mask)
        else:
            planes[0] = apply_mask_to_area(luma, planes[0], mask, masking_areas)
    scaled = plane_array_to_clip(planes)
    if denoise:
        scaled = mvf.BM3D(scaled, radius1=1, sigma=[bm3d_sigma, 0] if use_gpu else bm3d_sigma)
    return scaled


# the following 6 functions are mostly called from inside inverse_scale
def inverse_scale_clip_array(planes, w, h, kernel, kerneluv, taps, a1, a2, invks=True):
    if hasattr(core, 'descale') and invks:
        planes[0] = get_descale_filter(kernel, b=a1, c=a2, taps=taps)(planes[0], w, h)
    elif kernel == 'bilinear' and hasattr(core, 'unresize') and invks:
        planes[0] = core.unresize.Unresize(planes[0], w, h)
    else:
        planes[0] = core.fmtc.resample(planes[0], w, h, kernel=kernel, invks=invks, invkstaps=taps, a1=a1, a2=a2)
    planes[1], planes[2] = [core.fmtc.resample(plane, w, h, kernel=kerneluv, sx=0.25) for plane in planes[1:]]
    return planes


def clip_to_plane_array(clip):
    return [core.std.ShufflePlanes(clip, x, colorfamily=vs.GRAY) for x in range(clip.format.num_planes)]


def plane_array_to_clip(planes, family=vs.YUV):
    return core.std.ShufflePlanes(clips=planes, planes=[0] * len(planes), colorfamily=family)


def generate_detail_mask(source, downscaled, kernel='bicubic', taps=4, a1=1 / 3, a2=1 / 3, threshold=0.05):
    upscaled = fvf.Resize(downscaled, source.width, source.height, kernel=kernel, taps=taps, a1=a1, a2=a2)
    mask = core.std.Expr([source, upscaled], 'x y - abs') \
        .resize.Bicubic(downscaled.width, downscaled.height).std.Binarize(threshold)
    mask = iterate(mask, core.std.Maximum, 2)
    return iterate(mask, core.std.Inflate, 2)


def apply_mask(source, scaled, mask):
    noalias = core.fmtc.resample(source, scaled.width, scaled.height, kernel='blackmanminlobe', taps=5)
    return core.std.MaskedMerge(scaled, noalias, mask)


def apply_mask_to_area(source, scaled, mask, area):
    if len(area) == 2 and isinstance(area[0], int):
        area = [[area[0], area[1]]]
    noalias = core.fmtc.resample(source, scaled.width, scaled.height, kernel='blackmanminlobe', taps=5)
    for a in area:  # TODO: use ReplaceFrames
        source_cut = core.std.Trim(noalias, a[0], a[1])
        scaled_cut = core.std.Trim(scaled, a[0], a[1])
        mask_cut = core.std.Trim(mask, a[0], a[1])
        masked = apply_mask(source_cut, scaled_cut, mask_cut)
        scaled = insert_clip(scaled, masked, a[0])
    return scaled


# less typing == more time to encode
split = clip_to_plane_array
join = plane_array_to_clip


def getY(c: vs.VideoNode) -> vs.VideoNode:
    return core.std.ShufflePlanes(c, 0, vs.GRAY)


# TODO: currently, this should fail for non mod4 subsampled input.
# Not really relevant, though, as 480p, 576p, 720p, and 1080p are all mod32
def generate_keyframes(clip: vs.VideoNode, out_path=None) -> None:
    """
    probably only useful for fansubbing
    generates qp-filename for keyframes to simplify timing
    """
    import os
    clip = core.resize.Bilinear(clip, 640, 360)  # speed up the analysis by resizing first
    clip = core.wwxd.WWXD(clip)
    out_txt = "# WWXD log file, using qpfile format\n\n"
    for i in range(clip.num_frames):
        if clip.get_frame(i).props.Scenechange == 1:
            out_txt += "%d I -1\n" % i
        if i % 1000 == 0:
            print(i)
    if out_path is None:
        out_path = os.path.expanduser("~") + "/Desktop/keyframes.txt"
    text_file = open(out_path, "w")
    text_file.write(out_txt)
    text_file.close()


def adaptive_grain(clip: vs.VideoNode, strength=0.25, static=True, luma_scaling=12, mask_bits=8,
                   show_mask=False) -> vs.VideoNode:
    """
    generates grain based on frame and pixel brightness.
    details can be found here: https://kageru.moe/article.php?p=adaptivegrain
    strength is the strength of the grain generated by AddGrain, static=True for static grain
    luma_scaling manipulates the grain alpha curve. Higher values will generate less grain (especially in brighter scenes)
    while lower values will generate more grain, even in brighter scenes
    Please note that 8 bit should be enough for the mask; 10, if you want to do everything in 10 bit.
    It is technically possible to set it to up to 16 (float does not work), but you won't gain anything.
    An 8 bit mask uses 1 MB of RAM, 10 bit need 4 MB, and 16 bit use 256 MB.
    Lookup times might also increase (they shouldn't, but you never know), as well as the initial generation time.
    """
    import numpy as np

    def fill_lut(y):
        """
        Using horner's method to compute this polynomial:
        (1 - (1.124 * x - 9.466 * x ** 2 + 36.624 * x ** 3 - 45.47 * x ** 4 + 18.188 * x ** 5)) ** ((y ** 2) * luma_scaling) * 255
        Using the normal polynomial is about 2.5x slower during the initial generation.
        I know it doesn't matter as it only saves a few ms (or seconds at most), but god damn, just let me have some fun here, will ya?
        Just truncating (rather than rounding) the array would also half the processing time,
        but that would decrease the precision and is also just unnecessary.
        """
        x = np.arange(0, 1, 1 / (1 << mask_bits))
        z = (1 - (x * (1.124 + x * (-9.466 + x * (36.624 + x * (-45.47 + x * 18.188)))))) ** ((y ** 2) * luma_scaling)
        if clip.format.sample_type == vs.INTEGER:
            z = z * ((1 << mask_bits) - 1)
            z = np.rint(z).astype(int)
        return z.tolist()

    def generate_mask(n, f, clip):
        frameluma = round(f.props.PlaneStatsAverage * 999)
        table = lut[int(frameluma)]
        return core.std.Lut(clip, lut=table)

    clip8 = fvf.Depth(clip, mask_bits)
    bits = clip.format.bits_per_sample

    lut = [None] * 1000
    for y in np.arange(0, 1, 0.001):
        lut[int(round(y * 1000))] = fill_lut(y)

    luma = core.std.ShufflePlanes(clip8, 0, vs.GRAY)
    luma = core.std.PlaneStats(luma)
    grained = core.grain.Add(clip, var=strength, constant=static)

    mask = core.std.FrameEval(luma, partial(generate_mask, clip=luma), prop_src=luma)
    mask = core.resize.Spline36(mask, clip.width, clip.height)

    if bits != mask_bits:
        mask = core.fmtc.bitdepth(mask, bits=bits, dmode=1)

    if show_mask:
        return mask

    return core.std.MaskedMerge(clip, grained, mask)


# TODO: implement blending zone in which both clips are merged to aviod abrupt and visible kernel changes.
def conditional_resize(src: vs.VideoNode, kernel='bilinear', w=1280, h=720, thr=0.00015, debug=False) -> vs.VideoNode:
    """
    Fix oversharpened upscales by comparing a regular downscale with a blurry bicubic kernel downscale.
    Similar to the avisynth function. thr is lower in vapoursynth because it's normalized (between 0 and 1)
    """

    def compare(n, down, os, diff_default, diff_os):
        error_default = diff_default.get_frame(n).props.PlaneStatsDiff
        error_os = diff_os.get_frame(n).props.PlaneStatsDiff
        if debug:
            debugstring = "error when scaling with {:s}: {:.5f}\nerror when scaling with bicubic (b=0, c=1): " \
                          "{:.5f}\nUsing debicubic OS: {:s}".format(kernel, error_default, error_os,
                                                                    str(error_default - thr > error_os))
            os = os.sub.Subtitle(debugstring)
            down = down.sub.Subtitle(debugstring)
        if error_default - thr > error_os:
            return os
        return down

    src_w, src_h = src.width, src.height

    if hasattr(core, 'descale'):
        down = get_descale_filter(kernel)(w, h)
        os = core.descale.Debicubic(w, h, b=0, c=1)
    else:
        down = src.fmtc.resample(w, h, kernel=kernel, invks=True)
        os = src.fmtc.resample(w, h, kernel='bicubic', a1=0, a2=1, invks=True)

    # we only need luma for the comparison
    up = core.std.ShufflePlanes([down], [0], vs.GRAY).fmtc.resample(src_w, src_h, kernel=kernel)
    os_up = core.std.ShufflePlanes([os], [0], vs.GRAY).fmtc.resample(src_w, src_h, kernel='bicubic', a1=0, a2=1)

    src_luma = core.std.ShufflePlanes([src], [0], vs.GRAY)
    diff_default = core.std.PlaneStats(up, src_luma)
    diff_os = core.std.PlaneStats(os_up, src_luma)

    return core.std.FrameEval(down,
                              partial(compare, down=down, os=os, diff_os=diff_os, diff_default=diff_default))


def squaremask(clip: vs.VideoNode, width: int, height: int, offset_x: int, offset_y: int) -> vs.VideoNode:
    """
    “There must be a better way!”
    Basically a small script that draws white rectangles on a black background.
    Python-only replacement for manual paint/photoshop/gimp masks, as long as these don't go beyond a simple rectangle.
    Can be merged with an edgemask to only mask certain edge areas.
    TL;DR: Unless you're scenefiltering, this is useless.
    """
    bits = clip.format.bits_per_sample
    src_w = clip.width
    src_h = clip.height
    mask_format = clip.format.replace(color_family=vs.GRAY, subsampling_w=0, subsampling_h=0)

    if mask_format.sample_type == vs.FLOAT:
        white = 1
    else:
        white = (1 << bits) - 1

    center = core.std.BlankClip(width=width, height=height, _format=mask_format, color=white,
                                length=clip.num_frames, fpsnum=clip.fps.numerator, fpsden=clip.fps.denominator)

    if offset_x:
        left = core.std.BlankClip(center, width=offset_x, height=height, color=0)
        center = core.std.StackHorizontal([left, center])

    if center.width < src_w:
        right = core.std.BlankClip(center, width=src_w - center.width, height=height, color=0)
        center = core.std.StackHorizontal([center, right])

    if offset_y:
        top = core.std.BlankClip(center, width=src_w, height=offset_y, color=0)
        center = core.std.StackVertical([top, center])

    if center.height < src_h:
        bottom = core.std.BlankClip(center, width=src_w, height=src_h - center.height, color=0)
        center = core.std.StackVertical([center, bottom])

    return center


def retinex_edgemask(src: vs.VideoNode, sigma=1) -> vs.VideoNode:
    """
    Use retinex to greatly improve the accuracy of the edge detection in dark scenes.
    sigma is the sigma of tcanny
    """
    luma = mvf.GetPlane(src, 0)
    ret = core.retinex.MSRCP(luma, sigma=[50, 200, 350], upper_thr=0.005)
    mask = core.std.Expr([kirsch(luma), ret.tcanny.TCanny(mode=1, sigma=sigma).std.Minimum(
        coordinates=[1, 0, 1, 0, 0, 1, 0, 1])], 'x y +')
    return mask


def kirsch(src: vs.VideoNode) -> vs.VideoNode:
    """
    Kirsch edge detection. This uses 8 directions, so it's slower but better than Sobel (4 directions).
    more information: https://ddl.kageru.moe/konOJ.pdf
    """
    w = [5] * 3 + [-3] * 5
    weights = [w[-i:] + w[:-i] for i in range(4)]
    c = [core.std.Convolution(src, (w[:4] + [0] + w[4:]), saturate=False) for w in weights]
    return core.std.Expr(c, 'x y max z max a max')


def fast_sobel(src: vs.VideoNode) -> vs.VideoNode:
    """
    Should behave similar to std.Sobel() but faster since it has no additional high-/lowpass, gain, or the sqrt.
    The internal filter is also a little brighter
    """
    sx = src.std.Convolution([-1, -2, -1, 0, 0, 0, 1, 2, 1], saturate=False)
    sy = src.std.Convolution([-1, 0, 1, -2, 0, 2, -1, 0, 1], saturate=False)
    return core.std.Expr([sx, sy], 'x y max')


def get_descale_filter(kernel: str, **kwargs):
    """
    Stolen from a declined pull request.
    Originally written by @stuxcrystal on Github.
    """
    FILTERS = {
        'bilinear': (lambda **kwargs: core.descale.Debilinear),
        'spline16': (lambda **kwargs: core.descale.Despline16),
        'spline36': (lambda **kwargs: core.descale.Despline36),
        'bicubic': (lambda b, c, **kwargs: partial(core.descale.Debicubic, b=b, c=c)),
        'lanczos': (lambda taps, **kwargs: partial(core.descale.Delanczos, taps=taps)),
    }
    return FILTERS[kernel](**kwargs)


def hardsubmask(clip: vs.VideoNode, ref: vs.VideoNode, mode='default', expandN=None, highpass=25) -> vs.VideoNode:
    """
    Uses multiple techniques to mask the hardsubs in video streams like Anime on Demand or Wakanim.
    Might (should) work for other hardsubs, too, as long as the subs are somewhat close to black/white.
    It's kinda experimental, but I wanted to try something like this.
    It works by finding the edge of the subtitle (where the black border and the white fill color touch),
    and it grows these areas into a regular brightness + difference mask via hysteresis.
    This should (in theory) reliably find all hardsubs in the image with barely any false positives (or none at all).
    Output depth and processing precision are the same as the input
    It is not necessary for 'clip' and 'ref' to have the same bit depth, as 'ref' will be dithered to match 'clip'
    Most of this code was written by Zastin (https://github.com/Z4ST1N)
    Clean code soon(tm)
    """

    clp_f = clip.format
    bits = clp_f.bits_per_sample
    st = clp_f.sample_type
    peak = 1 if st == vs.FLOAT else (1 << bits) - 1

    if expandN is None:
        expandN = clip.width // 200
    # if mode in ['default', None]:

    out_fmt = core.register_format(vs.GRAY, st, bits, 0, 0)
    YUV_fmt = core.register_format(clp_f.color_family, vs.INTEGER, 8, clp_f.subsampling_w, clp_f.subsampling_h)

    y_range = 219 << (bits - 8) if st == vs.INTEGER else 1
    uv_range = 224 << (bits - 8) if st == vs.INTEGER else 1
    offset = 16 << (bits - 8) if st == vs.INTEGER else 0

    uv_abs = ' abs ' if st == vs.FLOAT else ' {} - abs '.format((1 << bits) // 2)
    yexpr = 'x y - abs {thr} > 255 0 ?'.format(thr=y_range * 0.7)
    uvexpr = 'x {uv_abs} {thr} < y {uv_abs} {thr} < and 255 0 ?'.format(uv_abs=uv_abs, thr=uv_range * 0.1)

    difexpr = 'x {upper} > x {lower} < or x y - abs {mindiff} > and 255 0 ?'.format(upper=y_range * 0.8 + offset,
                                                                                    lower=y_range * 0.2 + offset,
                                                                                    mindiff=y_range * 0.1)

    # right shift by 4 pixels.
    # fmtc uses at least 16 bit internally, so it's slower for 8 bit,
    # but its behaviour when shifting/replicating edge pixels makes it faster otherwise
    if bits < 16:
        right = core.resize.Point(clip, src_left=4)
    else:
        right = core.fmtc.resample(clip, sx=4, flt=False)
    subedge = core.std.Expr([clip, right], [yexpr, uvexpr], YUV_fmt.id)
    c444 = split(subedge.resize.Bicubic(format=vs.YUV444P8, filter_param_a=0, filter_param_b=0.5))
    subedge = core.std.Expr(c444, 'x y z min min')

    clip, ref = getY(clip), getY(ref)
    ref = ref if clip.format == ref.format else fvf.Depth(ref, bits)

    clips = [clip.std.Convolution([1] * 9), ref.std.Convolution([1] * 9)]
    diff = core.std.Expr(clips, difexpr, vs.GRAY8).std.Maximum().std.Maximum()

    mask = core.misc.Hysteresis(subedge, diff)
    mask = iterate(mask, core.std.Maximum, expandN)
    mask = mask.std.Inflate().std.Inflate().std.Convolution([1] * 9)
    mask = fvf.Depth(mask, bits, range=1, range_in=1)

    """
    # needs some more testing
    elif mode == 'fast':
        highpass = highpass << (bits - 8) if st == vs.INTEGER else highpass / 255
        clip, ref = getY(clip), getY(ref)
        ref = ref if clip.format == ref.format else fvf.Depth(ref, bits)
        edge = clip.std.Sobel()
        diff = core.std.Expr([clip, ref], 'x y - abs {:d} < 0 {:d} ?'.format(highpass, peak))
        mask = core.misc.Hysteresis(edge, diff)
        mask = iterate(mask, core.std.Maximum, expandN)
        mask = mask.std.Convolution([1] * 9)
    else:
        raise ValueError('hardsubmask: Unknown mode')
    """
    return mask


# TODO: add as mode to hardsubmask
def hardsubmask_fades(clip, ref, expandN=8, highpass=5000):
    """
    Uses Sobel edge detection to find edges that are only present in the main clip.
    These should (theoretically) be the subtitles.
    The video is blurred beforehand to prevent compression artifacts from being recognized as subtitles.
    This may create more false positives than the other hardsubmask,
    but it is capable of finding subtitles of any color and subtitles during fadein/fadeout.
    Setting highpass to a lower value may catch very slight changes (e.g. the last frame of a low-contrast fade),
    but it will make the mask more susceptible to artifacts.
    """
    clip = core.fmtc.bitdepth(clip, bits=16).std.Convolution([1] * 9)
    ref = core.fmtc.bitdepth(ref, bits=16).std.Convolution([1] * 9)
    clipedge = getY(clip).std.Sobel()
    refedge = getY(ref).std.Sobel()
    mask = core.std.Expr([clipedge, refedge], 'x y - {} < 0 65535 ?'.format(highpass)).std.Median()
    mask = iterate(mask, core.std.Maximum, expandN)
    mask = iterate(mask, core.std.Inflate, 4)
    return mask


def crossfade(clipa, clipb, duration):
    """
    Crossfade clipa into clipb. Duration is the length of the blending zone.
    For example, crossfade(a, b, 100) will fade the last 100 frames of a into b.
    """

    def fade_image(n, clipa, clipb):
        return core.std.Merge(clipa, clipb, weight=n / clipa.num_frames)

    # lol, >error handling
    if clipa.format.id != clipb.format.id or clipa.height != clipb.height or clipa.width != clipb.width:
        raise ValueError('Crossfade: Both clips must have the same dimensions and format.')
    fade = core.std.FrameEval(clipa[-duration:], partial(fade_image, clipa=clipa[-duration:], clipb=clipb[:duration]))
    return clipa[:-duration] + fade + clipb[duration:]


def hybriddenoise(src, knl=0.5, sigma=2, radius1=1):
    """
    denoise luma with BM3D (CPU-based) and chroma with KNLMeansCL (GPU-based)
    sigma = luma denoise strength
    knl = chroma denoise strength. The algorithm is different, so this value is different from sigma
    BM3D's sigma default is 5, KNL's is 1.2, to give you an idea of the order of magnitude
    radius1 = temporal radius of luma denoising, 0 for purely spatial denoising
    """
    planes = clip_to_plane_array(src)
    planes[0] = mvf.BM3D(planes[0], radius1=radius1, sigma=sigma)
    planes[1], planes[2] = [core.knlm.KNLMeansCL(plane, a=2, h=knl, d=3, device_type='gpu', device_id=0)
                            for plane in planes[1:]]
    return core.std.ShufflePlanes(clips=planes, planes=[0, 0, 0], colorfamily=vs.YUV)


def insert_clip(ep, op, startframe):
    """
    convenience function to insert things like Non-credit OP/ED into episodes
    """
    if startframe == 0:
        return op + ep[op.num_frames:]
    pre = ep[:startframe]
    if startframe + op.num_frames == ep.num_frames - 1:
        return pre + op
    post = ep[startframe + op.num_frames:]
    return pre + op + post


# helpers

def get_subsampling(src):
    """
    returns string to be used with fmtc.resample
    """
    if src.format.subsampling_w == 1 and src.format.subsampling_h == 1:
        css = '420'
    elif src.format.subsampling_w == 1 and src.format.subsampling_h == 0:
        css = '422'
    elif src.format.subsampling_w == 0 and src.format.subsampling_h == 0:
        css = '444'
    elif src.format.subsampling_w == 2 and src.format.subsampling_h == 2:
        css = '410'
    elif src.format.subsampling_w == 2 and src.format.subsampling_h == 0:
        css = '411'
    elif src.format.subsampling_w == 0 and src.format.subsampling_h == 1:
        css = '440'
    else:
        raise ValueError('Unknown subsampling')
    return css


def iterate(base, filter, count):
    for _ in range(count):
        base = filter(base)
    return base


def is16bit(clip):
    """
    returns bool. Yes, I was lazy enough to write a function that saves ~20 characters
    """
    return clip.format.bits_per_sample == 16


def getw(h, ar=16 / 9, only_even=True):
    """
    returns width for image
    """
    w = h * ar
    w = int(round(w))
    if only_even:
        w = w // 2 * 2
    return w


def fit_subsampling(x, sub):
    """
    Makes a value (e.g. resolution or crop value) compatible with the specified subsampling.
    sub is given by the properties (clip.format.subsampling_w/_h)
    The number is then truncated to be a compatible resolution.
    """
    return (x >> bits) << bits
    # return x & (0xffffffff - 1 << sub -1);
