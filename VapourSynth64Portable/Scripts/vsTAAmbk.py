import vapoursynth as vs
import mvsfunc as mvf
import havsfunc as haf
import functools

MODULE_NAME = 'vsTAAmbk'


class Clip:
    def __init__(self, clip):
        self.core = vs.get_core()
        self.clip = clip
        if not isinstance(clip, vs.VideoNode):
            raise TypeError(MODULE_NAME + ': clip is invalid.')
        self.clip_width = clip.width
        self.clip_height = clip.height
        self.clip_bits = clip.format.bits_per_sample
        self.clip_color_family = clip.format.color_family
        self.clip_sample_type = clip.format.sample_type
        self.clip_id = clip.format.id
        self.clip_subsample_w = clip.format.subsampling_w
        self.clip_subsample_h = clip.format.subsampling_h
        self.clip_is_gray = True if clip.format.num_planes == 1 else False
        # Register format for GRAY10
        vs.GRAY10 = self.core.register_format(vs.GRAY, vs.INTEGER, 10, 0, 0).id


class AAParent(Clip):
    def __init__(self, clip, strength=0.0, down8=False):
        super(AAParent, self).__init__(clip)
        self.aa_clip = self.clip
        self.dfactor = 1 - max(min(strength, 0.5), 0)
        self.dw = round(self.clip_width * self.dfactor / 4) * 4
        self.dh = round(self.clip_height * self.dfactor / 4) * 4
        self.upw4 = round(self.dw * 0.375) * 4
        self.uph4 = round(self.dh * 0.375) * 4
        self.down8 = down8
        self.process_depth = self.clip_bits
        if down8 is True:
            self.down_8()
        if self.dfactor != 1:
            self.aa_clip = self.resize(self.aa_clip, self.dw, self.dh, shift=0)
        if self.clip_color_family is vs.GRAY:
            if self.clip_sample_type is not vs.INTEGER:
                raise TypeError(MODULE_NAME + ': clip must be integer format.')
        else:
            raise TypeError(MODULE_NAME + ': clip must be GRAY family.')

    def resize(self, clip, w, h, shift):
        try:
            resized = self.core.resize.Spline36(clip, w, h, src_top=shift)
        except vs.Error:
            resized = self.core.fmtc.resample(clip, w, h, sy=shift)
            if resized.format.bits_per_sample != self.process_depth:
                mvf.Depth(resized, self.process_depth)
        return resized

    def down_8(self):
        self.process_depth = 8
        self.aa_clip = mvf.Depth(self.aa_clip, 8)

    def output(self, aaed):
        if self.process_depth != self.clip_bits:
            return mvf.LimitFilter(self.clip, mvf.Depth(aaed, self.clip_bits), thr=1.0, elast=2.0)
        else:
            return aaed


class AANnedi3(AAParent):
    def __init__(self, clip, strength=0, down8=False, **args):
        super(AANnedi3, self).__init__(clip, strength, down8)
        self.nnedi3_args = {
            'nsize': args.get('nsize', 3),
            'nns': args.get('nns', 1),
            'qual': args.get('qual', 2),
        }
        self.opencl = args.get('opencl', False)
        if self.opencl is True:
            try:
                self.nnedi3 = self.core.nnedi3cl.NNEDI3CL
                self.nnedi3_args['device'] = args.get('opencl_device', 0)
            except AttributeError:
                try:
                    self.nnedi3 = self.core.znedi3.nnedi3
                except AttributeError:
                    self.nnedi3 = self.core.nnedi3.nnedi3
        else:
            try:
                self.nnedi3 = self.core.znedi3.nnedi3
            except AttributeError:
                self.nnedi3 = self.core.nnedi3.nnedi3

    def out(self):
        aaed = self.nnedi3(self.aa_clip, field=1, dh=True, **self.nnedi3_args)
        aaed = self.resize(aaed, self.clip_width, self.clip_height, -0.5)
        aaed = self.core.std.Transpose(aaed)
        aaed = self.nnedi3(aaed, field=1, dh=True, **self.nnedi3_args)
        aaed = self.resize(aaed, self.clip_height, self.clip_width, -0.5)
        aaed = self.core.std.Transpose(aaed)
        return self.output(aaed)


class AANnedi3SangNom(AANnedi3):
    def __init__(self, clip, strength=0, down8=False, **args):
        super(AANnedi3SangNom, self).__init__(clip, strength, down8, **args)
        self.aa = args.get('aa', 48)

    def out(self):
        aaed = self.nnedi3(self.aa_clip, field=1, dh=True, **self.nnedi3_args)
        aaed = self.resize(aaed, self.clip_width, self.uph4, shift=-0.5)
        aaed = self.core.std.Transpose(aaed)
        aaed = self.nnedi3(aaed, field=1, dh=True, **self.nnedi3_args)
        aaed = self.resize(aaed, self.uph4, self.upw4, shift=-0.5)
        aaed = self.core.sangnom.SangNom(aaed, aa=self.aa)
        aaed = self.core.std.Transpose(aaed)
        aaed = self.core.sangnom.SangNom(aaed, aa=self.aa)
        aaed = self.resize(aaed, self.clip_width, self.clip_height, shift=0)
        return self.output(aaed)


class AANnedi3UpscaleSangNom(AANnedi3SangNom):
    def __init__(self, clip, strength=0, down8=False, **args):
        super(AANnedi3UpscaleSangNom, self).__init__(clip, strength, down8, **args)
        self.nnedi3_args = {
            'nsize': args.get('nsize', 1),
            'nns': args.get('nns', 3),
            'qual': args.get('qual', 2),
        }
        if self.opencl is True:
            self.nnedi3_args['device'] = args.get('opencl_device', 0)


class AAEedi3(AAParent):
    def __init__(self, clip, strength=0, down8=False, **args):
        super(AAEedi3, self).__init__(clip, strength, down8)
        self.eedi3_args = {
            'alpha': args.get('alpha', 0.5),
            'beta': args.get('beta', 0.2),
            'gamma': args.get('gamma', 20),
            'nrad': args.get('nrad', 3),
            'mdis': args.get('mdis', 30),
        }

        self.opencl = args.get('opencl', False)
        if self.opencl is True:
            try:
                self.eedi3 = self.core.eedi3m.EEDI3CL
                self.eedi3_args['device'] = args.get('opencl_device', 0)
            except AttributeError:
                self.eedi3 = self.core.eedi3.eedi3
                if self.process_depth > 8:
                    self.down_8()
        else:
            try:
                self.eedi3 = self.core.eedi3m.EEDI3
            except AttributeError:
                self.eedi3 = self.core.eedi3.eedi3
                if self.process_depth > 8:
                    self.down_8()

    '''
    def build_eedi3_mask(self, clip):
        eedi3_mask = self.core.nnedi3.nnedi3(clip, field=1, show_mask=True)
        eedi3_mask = self.core.std.Expr([eedi3_mask, clip], "x 254 > x y - 0 = not and 255 0 ?")
        eedi3_mask_turn = self.core.std.Transpose(eedi3_mask)
        if self.dfactor != 1:
            eedi3_mask_turn = self.core.resize.Bicubic(eedi3_mask_turn, self.clip_height, self.dw)
        return eedi3_mask, eedi3_mask_turn
    '''

    def out(self):
        aaed = self.eedi3(self.aa_clip, field=1, dh=True, **self.eedi3_args)
        aaed = self.resize(aaed, self.dw, self.clip_height, shift=-0.5)
        aaed = self.core.std.Transpose(aaed)
        aaed = self.eedi3(aaed, field=1, dh=True, **self.eedi3_args)
        aaed = self.resize(aaed, self.clip_height, self.clip_width, shift=-0.5)
        aaed = self.core.std.Transpose(aaed)
        return self.output(aaed)


class AAEedi3SangNom(AAEedi3):
    def __init__(self, clip, strength=0, down8=False, **args):
        super(AAEedi3SangNom, self).__init__(clip, strength, down8, **args)
        self.aa = args.get('aa', 48)

    '''
    def build_eedi3_mask(self, clip):
        eedi3_mask = self.core.nnedi3.nnedi3(clip, field=1, show_mask=True)
        eedi3_mask = self.core.std.Expr([eedi3_mask, clip], "x 254 > x y - 0 = not and 255 0 ?")
        eedi3_mask_turn = self.core.std.Transpose(eedi3_mask)
        eedi3_mask_turn = self.core.resize.Bicubic(eedi3_mask_turn, self.uph4, self.dw)
        return eedi3_mask, eedi3_mask_turn
    '''

    def out(self):
        aaed = self.eedi3(self.aa_clip, field=1, dh=True, **self.eedi3_args)
        aaed = self.resize(aaed, self.dw, self.uph4, shift=-0.5)
        aaed = self.core.std.Transpose(aaed)
        aaed = self.eedi3(aaed, field=1, dh=True, **self.eedi3_args)
        aaed = self.resize(aaed, self.uph4, self.upw4, shift=-0.5)
        aaed = self.core.sangnom.SangNom(aaed, aa=self.aa)
        aaed = self.core.std.Transpose(aaed)
        aaed = self.core.sangnom.SangNom(aaed, aa=self.aa)
        aaed = self.resize(aaed, self.clip_width, self.clip_height, shift=0)
        return self.output(aaed)


class AAEedi2(AAParent):
    def __init__(self, clip, strength=0, down8=False, **args):
        super(AAEedi2, self).__init__(clip, strength, down8)
        self.mthresh = args.get('mthresh', 10)
        self.lthresh = args.get('lthresh', 20)
        self.vthresh = args.get('vthresh', 20)
        self.maxd = args.get('maxd', 24)
        self.nt = args.get('nt', 50)

    def out(self):
        aaed = self.core.eedi2.EEDI2(self.aa_clip, 1, self.mthresh, self.lthresh, self.vthresh, maxd=self.maxd,
                                     nt=self.nt)
        aaed = self.resize(aaed, self.dw, self.clip_height, shift=-0.5)
        aaed = self.core.std.Transpose(aaed)
        aaed = self.core.eedi2.EEDI2(aaed, 1, self.mthresh, self.lthresh, self.vthresh, maxd=self.maxd, nt=self.nt)
        aaed = self.resize(aaed, self.clip_height, self.clip_width, shift=-0.5)
        aaed = self.core.std.Transpose(aaed)
        return self.output(aaed)


class AAEedi2SangNom(AAEedi2):
    def __init__(self, clip, strength=0, down8=False, **args):
        super(AAEedi2SangNom, self).__init__(clip, strength, down8, **args)
        self.aa = args.get('aa', 48)

    def out(self):
        aaed = self.core.eedi2.EEDI2(self.aa_clip, 1, self.mthresh, self.lthresh, self.vthresh, maxd=self.maxd,
                                     nt=self.nt)
        aaed = self.resize(aaed, self.dw, self.uph4, shift=-0.5)
        aaed = self.core.std.Transpose(aaed)
        aaed = self.core.eedi2.EEDI2(aaed, 1, self.mthresh, self.lthresh, self.vthresh, maxd=self.maxd, nt=self.nt)
        aaed = self.resize(aaed, self.uph4, self.upw4, shift=-0.5)
        aaed = self.core.sangnom.SangNom(aaed, aa=self.aa)
        aaed = self.core.std.Transpose(aaed)
        aaed = self.core.sangnom.SangNom(aaed, aa=self.aa)
        aaed = self.resize(aaed, self.clip_width, self.clip_height, shift=0)
        return self.output(aaed)


class AASpline64NRSangNom(AAParent):
    def __init__(self, clip, strength=0, down8=False, **args):
        super(AASpline64NRSangNom, self).__init__(clip, strength, down8)
        self.aa = args.get('aa', 48)

    def out(self):
        aa_spline64 = self.core.fmtc.resample(self.aa_clip, self.upw4, self.uph4, kernel='spline64')
        aa_spline64 = mvf.Depth(aa_spline64, self.process_depth)
        aa_gaussian = self.core.fmtc.resample(self.aa_clip, self.upw4, self.uph4, kernel='gaussian', a1=100)
        aa_gaussian = mvf.Depth(aa_gaussian, self.process_depth)
        aaed = self.core.rgvs.Repair(aa_spline64, aa_gaussian, 1)
        aaed = self.core.sangnom.SangNom(aaed, aa=self.aa)
        aaed = self.core.std.Transpose(aaed)
        aaed = self.core.sangnom.SangNom(aaed, aa=self.aa)
        aaed = self.core.std.Transpose(aaed)
        aaed = self.resize(aaed, self.clip_width, self.clip_height, shift=0)
        return self.output(aaed)


class AASpline64SangNom(AAParent):
    def __init__(self, clip, strength=0, down8=False, **args):
        super(AASpline64SangNom, self).__init__(clip, strength, down8)
        self.aa = args.get('aa', 48)

    def out(self):
        aaed = self.core.fmtc.resample(self.aa_clip, self.clip_width, self.uph4, kernel="spline64")
        aaed = mvf.Depth(aaed, self.process_depth)
        aaed = self.core.sangnom.SangNom(aaed, aa=self.aa)
        aaed = self.core.std.Transpose(self.resize(aaed, self.clip_width, self.clip_height, 0))
        aaed = self.core.fmtc.resample(aaed, self.clip_height, self.upw4, kernel="spline64")
        aaed = mvf.Depth(aaed, self.process_depth)
        aaed = self.core.sangnom.SangNom(aaed, aa=self.aa)
        aaed = self.core.std.Transpose(self.resize(aaed, self.clip_height, self.clip_width, 0))
        return self.output(aaed)


class AAPointSangNom(AAParent):
    def __init__(self, clip, strength=0, down8=False, **args):
        super(AAPointSangNom, self).__init__(clip, 0, down8)
        self.aa = args.get('aa', 48)
        self.upw = self.clip_width * 2
        self.uph = self.clip_height * 2
        self.strength = strength  # Won't use this

    def out(self):
        aaed = self.core.resize.Point(self.aa_clip, self.upw, self.uph)
        aaed = self.core.sangnom.SangNom(aaed, aa=self.aa)
        aaed = self.core.std.Transpose(aaed)
        aaed = self.core.sangnom.SangNom(aaed, aa=self.aa)
        aaed = self.core.std.Transpose(aaed)
        aaed = self.resize(aaed, self.clip_width, self.clip_height, 0)
        return self.output(aaed)


def mask_sobel(mthr, opencl=False, opencl_device=-1, **kwargs):
    core = vs.get_core()
    if opencl is True:
        try:
            canny = functools.partial(core.tcanny.TCannyCL, device=opencl_device)
        except AttributeError:
            canny = core.tcanny.TCanny
    else:
        canny = core.tcanny.TCanny
    mask_kwargs = {
        'gmmax': kwargs.get('gmmax', max(round(-0.14 * mthr + 61.87), 80)),
        'sigma': kwargs.get('sigma', 1.0),
        't_h': kwargs.get('t_h', 8.0),
        't_l': kwargs.get('t_l', 1.0),
    }
    return lambda clip: canny(clip, mode=1, op=2, **mask_kwargs)


def mask_prewitt(mthr, **kwargs):
    core = vs.get_core()

    def wrapper(clip):
        eemask_1 = core.std.Convolution(clip, [1, 1, 0, 1, 0, -1, 0, -1, -1], divisor=1, saturate=False)
        eemask_2 = core.std.Convolution(clip, [1, 1, 1, 0, 0, 0, -1, -1, -1], divisor=1, saturate=False)
        eemask_3 = core.std.Convolution(clip, [1, 0, -1, 1, 0, -1, 1, 0, -1], divisor=1, saturate=False)
        eemask_4 = core.std.Convolution(clip, [0, -1, -1, 1, 0, -1, 1, 1, 0], divisor=1, saturate=False)
        eemask = core.std.Expr([eemask_1, eemask_2, eemask_3, eemask_4], 'x y max z max a max')
        eemask = core.std.Expr(eemask, 'x %d <= x 2 / x 1.4 pow ?' % mthr).rgvs.RemoveGrain(4).std.Inflate()
        return eemask

    return wrapper


def mask_canny_continuous(mthr, opencl=False, opencl_device=-1, **kwargs):
    core = vs.get_core()
    if opencl is True:
        try:
            canny = functools.partial(core.tcanny.TCannyCL, device=opencl_device)
        except AttributeError:
            canny = core.tcanny.TCanny
    else:
        canny = core.tcanny.TCanny
    mask_kwargs = {
        'sigma': kwargs.get('sigma', 1.0),
        't_h': kwargs.get('t_h', 8.0),
        't_l': kwargs.get('t_l', 1.0),
    }
    return lambda clip: (canny(clip, mode=1, **mask_kwargs)
                         .std.Expr('x %d <= x 2 / x 2 * ?' % mthr)
                         .rgvs.RemoveGrain(20 if clip.width > 1100 else 11))


def mask_canny_binarized(mthr, opencl=False, opencl_device=-1, **kwargs):
    core = vs.get_core()
    if opencl is True:
        try:
            canny = functools.partial(core.tcanny.TCannyCL, device=opencl_device)
        except AttributeError:
            canny = core.tcanny.TCanny
    else:
        canny = core.tcanny.TCanny
    mask_kwargs = {
        'sigma': kwargs.get('sigma', max(min(0.01772 * mthr + 0.4823, 5.0), 0.5)),
        't_h': kwargs.get('t_h', 8.0),
        't_l': kwargs.get('t_l', 1.0),
    }
    return lambda clip: canny(clip, mode=0, **mask_kwargs).std.Maximum()


def mask_tedge(mthr, **kwargs):
    """
    Mainly based on Avisynth's plugin TEMmod(type=2) (https://github.com/chikuzen/TEMmod)
    """
    core = vs.get_core()
    mthr /= 5

    def wrapper(clip):
        # The Maximum value of these convolution is 21930, thus we have to store the result in 16bit clip
        fake16 = core.std.Expr(clip, 'x', eval('vs.' + clip.format.name.upper()[:-1] + '16'))
        ix = core.std.Convolution(fake16, [12, -74, 0, 74, -12], saturate=False, mode='h')
        iy = core.std.Convolution(fake16, [-12, 74, 0, -74, 12], saturate=False, mode='v')
        mask = core.std.Expr([ix, iy], 'x x * y y * + 0.0001 * sqrt 255.0 158.1 / * 0.5 +',
                             eval('vs.' + fake16.format.name.upper()[:-2] + '8'))
        mask = core.std.Expr(mask, 'x %f <= x 2 / x 16 * ?' % mthr)
        mask = core.std.Deflate(mask).rgvs.RemoveGrain(20 if clip.width > 1100 else 11)
        return mask

    return wrapper


def mask_robert(mthr, **kwargs):
    core = vs.get_core()

    def wrapper(clip):
        m1 = core.std.Convolution(clip, [0, 0, 0, 0, -1, 0, 0, 0, 1], saturate=False)
        m2 = core.std.Convolution(clip, [0, 0, 0, 0, 0, -1, 0, 1, 0], saturate=False)
        mask = core.std.Expr([m1, m2], 'x y max').std.Expr('x %d < x 255 ?' % mthr).std.Inflate()
        return mask

    return wrapper


def mask_msharpen(mthr, **kwargs):
    core = vs.get_core()
    mthr /= 5
    return lambda clip: core.msmoosh.MSharpen(clip, threshold=mthr, strength=0, mask=True)


def mask_lthresh(clip, mthrs, lthreshes, mask_kernel, inexpand, **kwargs):
    core = vs.get_core()
    gray8 = mvf.Depth(clip, 8) if clip.format.bits_per_sample != 8 else clip
    gray8 = core.std.ShufflePlanes(gray8, 0, vs.GRAY) if clip.format.color_family != vs.GRAY else gray8
    mthrs = mthrs if isinstance(mthrs, (list, tuple)) else [mthrs]
    lthreshes = lthreshes if isinstance(lthreshes, (list, tuple)) else [lthreshes]
    inexpand = inexpand if isinstance(inexpand, (list, tuple)) and len(inexpand) >= 2 else [inexpand, 0]

    mask_kernels = [mask_kernel(mthr, **kwargs) for mthr in mthrs]
    masks = [kernel(gray8) for kernel in mask_kernels]
    mask = ((len(mthrs) - len(lthreshes) == 1) and functools.reduce(
        lambda x, y: core.std.Expr([x, y, gray8], 'z %d < x y ?' % lthreshes[masks.index(y) - 1]), masks)) or masks[0]
    mask = [mask] + [core.std.Maximum] * inexpand[0]
    mask = functools.reduce(lambda x, y: y(x), mask)
    mask = [mask] + [core.std.Minimum] * inexpand[1]
    mask = functools.reduce(lambda x, y: y(x), mask)

    bps = clip.format.bits_per_sample
    mask = (bps > 8 and core.std.Expr(mask, 'x %d *' % (((1 << clip.format.bits_per_sample) - 1) // 255),
                                      eval('vs.GRAY' + str(bps)))) or mask
    return lambda clip_a, clip_b, show=False: (show is False and core.std.MaskedMerge(clip_a, clip_b, mask)) or mask


def mask_fadetxt(clip, lthr=225, cthr=(2, 2), expand=2, fade_num=(5, 5), apply_range=None):
    core = vs.get_core()
    if clip.format.color_family != vs.YUV:
        raise TypeError(MODULE_NAME + ': fadetxt mask: only yuv clips are supported.')
    w = clip.width
    h = clip.height
    bps = clip.format.bits_per_sample
    ceil = (1 << bps) - 1
    neutral = 1 << (bps - 1)
    frame_count = clip.num_frames

    yuv = [core.std.ShufflePlanes(clip, i, vs.GRAY) for i in range(clip.format.num_planes)]
    try:
        yuv444 = [core.resize.Bicubic(plane, w, h, src_left=0.25) if yuv.index(plane) > 0 else plane for plane in yuv]
    except vs.Error:
        yuv444 = [mvf.Depth(core.fmtc.resample(plane, w, h, sx=0.25), 8)
                  if yuv.index(plane) > 0 else plane for plane in yuv]
    cthr_u = cthr if not isinstance(cthr, (list, tuple)) else cthr[0]
    cthr_v = cthr if not isinstance(cthr, (list, tuple)) else cthr[1]
    expr = 'x %d > y %d - abs %d < and z %d - abs %d < and %d 0 ?' % (lthr, neutral, cthr_u, neutral, cthr_v, ceil)
    mask = core.std.Expr(yuv444, expr)
    mask = [mask] + [core.std.Maximum] * expand
    mask = functools.reduce(lambda x, y: y(x), mask)

    if fade_num is not 0:
        def shift_backward(n, mask_clip, num):
            return mask_clip[frame_count - 1] if n + num > frame_count - 1 else mask_clip[n + num]

        def shift_forward(n, mask_clip, num):
            return mask_clip[0] if n - num < 0 else mask_clip[n - num]

        fade_in_num = fade_num if not isinstance(fade_num, (list, tuple)) else fade_num[0]
        fade_out_num = fade_num if not isinstance(fade_num, (list, tuple)) else fade_num[1]
        fade_in = core.std.FrameEval(mask, functools.partial(shift_backward, mask_clip=mask, num=fade_in_num))
        fade_out = core.std.FrameEval(mask, functools.partial(shift_forward, mask_clip=mask, num=fade_out_num))
        mask = core.std.Expr([mask, fade_in, fade_out], 'x y max z max')
        if apply_range is not None and isinstance(apply_range, (list, tuple)):
            try:
                blank = core.std.BlankClip(mask)
                if 0 in apply_range:
                    mask = mask[apply_range[0]:apply_range[1]] + blank[apply_range[1]:]
                elif frame_count in apply_range:
                    mask = blank[0:apply_range[0]] + mask[apply_range[0]:apply_range[1]]
                else:
                    mask = blank[0:apply_range[0]] + mask[apply_range[0]:apply_range[1]] + blank[apply_range[1]:]
            except vs.Error:
                raise ValueError(MODULE_NAME + ': incorrect apply range setting. Possibly end less than start')
            except IndexError:
                raise ValueError(MODULE_NAME + ': incorrect apply range setting. '
                                               'Apply range must be a tuple/list with 2 elements')
    return mask


def daa(clip, mode=-1, opencl=False, opencl_device=-1):
    core = vs.get_core()
    nnedi3_attr = ((opencl is True and getattr(core, 'nnedi3cl', getattr(core, 'znedi3', getattr(core, 'nnedi3'))))
                   or getattr(core, 'znedi3', getattr(core, 'nnedi3')))
    nnedi3 = (hasattr(nnedi3_attr, 'NNEDI3CL') and nnedi3_attr.NNEDI3CL) or nnedi3_attr.nnedi3
    nnedi3 = (nnedi3.name == 'NNEDI3CL' and functools.partial(nnedi3, device=opencl_device)) or nnedi3
    if mode == -1:
        nn = nnedi3(clip, field=3)
        nnt = nnedi3(core.std.Transpose(clip), field=3).std.Transpose()
        clph = core.std.Merge(core.std.SelectEvery(nn, cycle=2, offsets=0),
                              core.std.SelectEvery(nn, cycle=2, offsets=1))
        clpv = core.std.Merge(core.std.SelectEvery(nnt, cycle=2, offsets=0),
                              core.std.SelectEvery(nnt, cycle=2, offsets=1))
        clp = core.std.Merge(clph, clpv)
    elif mode == 1:
        nn = nnedi3(clip, field=3)
        clp = core.std.Merge(core.std.SelectEvery(nn, cycle=2, offsets=0),
                             core.std.SelectEvery(nn, cycle=2, offsets=1))
    elif mode == 2:
        nnt = nnedi3(core.std.Transpose(clip), field=3).std.Transpose()
        clp = core.std.Merge(core.std.SelectEvery(nnt, cycle=2, offsets=0),
                             core.std.SelectEvery(nnt, cycle=2, offsets=1))
    else:
        raise ValueError(MODULE_NAME + ': daa: at least one direction should be processed.')
    return clp


def temporal_stabilize(clip, src, delta=3, pel=1, retain=0.6):
    core = vs.get_core()
    clip_bits = clip.format.bits_per_sample
    src_bits = src.format.bits_per_sample
    if clip_bits != src_bits:
        raise ValueError(MODULE_NAME + ': temporal_stabilize: bits depth of clip and src mismatch.')
    if delta not in [1, 2, 3]:
        raise ValueError(MODULE_NAME + ': temporal_stabilize: delta (1~3) invalid.')

    diff = core.std.MakeDiff(src, clip)
    clip_super = core.mv.Super(clip, pel=pel)
    diff_super = core.mv.Super(diff, pel=pel, levels=1)

    backward_vectors = [core.mv.Analyse(clip_super, isb=True, delta=i + 1, overlap=8, blksize=16) for i in range(delta)]
    forward_vectors = [core.mv.Analyse(clip_super, isb=False, delta=i + 1, overlap=8, blksize=16) for i in range(delta)]
    vectors = [vector for vector_group in zip(backward_vectors, forward_vectors) for vector in vector_group]

    stabilize_func = {
        1: core.mv.Degrain1,
        2: core.mv.Degrain2,
        3: core.mv.Degrain3
    }
    diff_stabilized = stabilize_func[delta](diff, diff_super, *vectors)

    neutral = 1 << (clip_bits - 1)
    expr = 'x {neutral} - abs y {neutral} - abs < x y ?'.format(neutral=neutral)
    diff_stabilized_limited = core.std.Expr([diff, diff_stabilized], expr)
    diff_stabilized = core.std.Merge(diff_stabilized_limited, diff_stabilized, retain)
    clip_stabilized = core.std.MakeDiff(src, diff_stabilized)
    return clip_stabilized


def soothe(clip, src, keep=24):
    core = vs.get_core()
    clip_bits = clip.format.bits_per_sample
    src_bits = src.format.bits_per_sample
    if clip_bits != src_bits:
        raise ValueError(MODULE_NAME + ': temporal_stabilize: bits depth of clip and src mismatch.')

    neutral = 1 << (clip_bits - 1)
    ceil = (1 << clip_bits) - 1
    multiple = ceil // 255
    const = 100 * multiple
    kp = keep * multiple

    diff = core.std.MakeDiff(src, clip)
    try:
        diff_soften = core.misc.AverageFrame(diff, weights=[1, 1, 1], scenechange=32)
    except AttributeError:
        diff_soften = core.focus.TemporalSoften(diff, radius=1, luma_threshold=255,
                                                chroma_threshold=255, scenechange=32, mode=2)
    diff_soothed_expr = "x {neutral} - y {neutral} - * 0 < x {neutral} - {const} / {kp} * {neutral} + " \
                        "x {neutral} - abs y {neutral} - abs > " \
                        "x {kp} * y {const} {kp} - * + {const} / x ? ?".format(neutral=neutral, const=const, kp=kp)
    diff_soothed = core.std.Expr([diff, diff_soften], diff_soothed_expr)
    clip_soothed = core.std.MakeDiff(src, diff_soothed)
    return clip_soothed


def aa_cycle(clip, aa_class, cycle, *args, **kwargs):
    aaed = aa_class(clip, *args, **kwargs).out()
    return aaed if cycle <= 0 else aa_cycle(aaed, aa_class, cycle - 1, *args, **kwargs)


def TAAmbk(clip, aatype=1, aatypeu=None, aatypev=None, preaa=0, strength=0.0, cycle=0, mtype=None, mclip=None,
           mthr=None, mlthresh=None, mpand=(0, 0), txtmask=0, txtfade=0, thin=0, dark=0.0, sharp=0,
           aarepair=0, postaa=None, src=None, stabilize=0, down8=True, showmask=0, opencl=False, opencl_device=-1,
           **kwargs):
    core = vs.get_core()

    aatypeu = aatype if aatypeu is None else aatypeu
    aatypev = aatype if aatypev is None else aatypev
    if mtype is None:
        mtype = 0 if preaa == 0 and True not in (aatype, aatypeu, aatypev) else 1
    if postaa is None:
        postaa = True if abs(sharp) > 70 or (0.4 < abs(sharp) < 1) else False
    if src is None:
        src = clip
    else:
        if clip.format.id != src.format.id:
            raise ValueError(MODULE_NAME + ': clip format and src format mismatch.')
        elif clip.width != src.width or clip.height != src.height:
            raise ValueError(MODULE_NAME + ': clip resolution and src resolution mismatch.')

    preaa_clip = clip if preaa == 0 else daa(clip, preaa, opencl, opencl_device)
    edge_enhanced_clip = (thin != 0 and core.warp.AWarpSharp2(preaa_clip, depth=int(thin)) or preaa_clip)
    edge_enhanced_clip = (dark != 0 and haf.Toon(edge_enhanced_clip, str=float(dark)) or edge_enhanced_clip)

    aa_kernel = {
        0: lambda clip, *args, **kwargs: type('', (), {'out': lambda: clip}),
        1: AAEedi2,
        2: AAEedi3,
        3: AANnedi3,
        4: AANnedi3UpscaleSangNom,
        5: AASpline64NRSangNom,
        6: AASpline64SangNom,
        -1: AAEedi2SangNom,
        -2: AAEedi3SangNom,
        -3: AANnedi3SangNom,
        'Eedi2': AAEedi2,
        'Eedi3': AAEedi3,
        'Nnedi3': AANnedi3,
        'Nnedi3UpscaleSangNom': AANnedi3UpscaleSangNom,
        'Spline64NrSangNom': AASpline64NRSangNom,
        'Spline64SangNom': AASpline64SangNom,
        'Eedi2SangNom': AAEedi2SangNom,
        'Eedi3SangNom': AAEedi3SangNom,
        'Nnedi3SangNom': AANnedi3SangNom,
        'PointSangNom': AAPointSangNom,
        'Unknown': lambda clip, *args, **kwargs: type('', (), {
            'out': lambda: exec('raise ValueError(MODULE_NAME + ": unknown aatype, aatypeu or aatypev")')}),
        'Custom': kwargs.get('aakernel', lambda clip, *args, **kwargs: type('', (), {
            'out': lambda: exec('raise RuntimeError(MODULE_NAME + ": custom aatype: aakernel must be set.")')})),
    }

    if clip.format.color_family is vs.YUV:
        yuv = [core.std.ShufflePlanes(edge_enhanced_clip, i, vs.GRAY) for i in range(clip.format.num_planes)]
        aatypes = [aatype, aatypeu, aatypev]
        aa_classes = [aa_kernel.get(aatype, aa_kernel['Unknown']) for aatype in aatypes]
        aa_clips = [aa_cycle(plane, aa_class, cycle, strength if yuv.index(plane) == 0 else 0, down8, opencl=opencl,
                             opencl_device=opencl_device, **kwargs) for plane, aa_class in zip(yuv, aa_classes)]
        aaed_clip = core.std.ShufflePlanes(aa_clips, [0, 0, 0], vs.YUV)
    elif clip.format.color_family is vs.GRAY:
        gray = edge_enhanced_clip
        aa_class = aa_kernel.get(aatype, aa_kernel['Unknown'])
        aaed_clip = aa_cycle(gray, aa_class, cycle, strength, down8, **kwargs)
    else:
        raise ValueError(MODULE_NAME + ': Unsupported color family.')

    abs_sharp = abs(sharp)
    if sharp >= 1:
        sharped_clip = haf.LSFmod(aaed_clip, strength=int(abs_sharp), defaults='old', source=src)
    elif sharp > 0:
        per = int(40 * abs_sharp)
        matrix = [-1, -2, -1, -2, 52 - per, -2, -1, -2, -1]
        sharped_clip = core.std.Convolution(aaed_clip, matrix)
    elif sharp == 0:
        sharped_clip = aaed_clip
    elif sharp > -1:
        sharped_clip = haf.LSFmod(aaed_clip, strength=round(abs_sharp * 100), defaults='fast', source=src)
    elif sharp == -1:
        blured = core.rgvs.RemoveGrain(aaed_clip, mode=20 if aaed_clip.width > 1100 else 11)
        diff = core.std.MakeDiff(aaed_clip, blured)
        diff = core.rgvs.Repair(diff, core.std.MakeDiff(src, aaed_clip), mode=13)
        sharped_clip = core.std.MergeDiff(aaed_clip, diff)
    else:
        sharped_clip = aaed_clip

    postaa_clip = sharped_clip if postaa is False else soothe(sharped_clip, src, 24)
    repaired_clip = ((aarepair > 0 and core.rgvs.Repair(src, postaa_clip, aarepair)) or
                     (aarepair < 0 and core.rgvs.Repair(postaa_clip, src, -aarepair)) or postaa_clip)
    stabilized_clip = repaired_clip if stabilize == 0 else temporal_stabilize(repaired_clip, src, stabilize)

    if mclip is not None:
        try:
            masked_clip = core.std.MaskedMerge(src, stabilized_clip, mclip, first_plane=True)
            masker = type('', (), {'__call__': lambda *args, **kwargs: mclip})()
        except vs.Error:
            raise RuntimeError(
                MODULE_NAME + ': Something wrong with your mclip. Maybe format, resolution or bit_depth mismatch.')
    else:
        # Use lambda for lazy evaluation
        mask_kernel = {
            0: lambda: lambda a, b, *args, **kwargs: b,
            1: lambda: mask_lthresh(clip, mthr, mlthresh, mask_sobel, mpand, opencl=opencl,
                                    opencl_device=opencl_device, **kwargs),
            2: lambda: mask_lthresh(clip, mthr, mlthresh, mask_robert, mpand, **kwargs),
            3: lambda: mask_lthresh(clip, mthr, mlthresh, mask_prewitt, mpand, **kwargs),
            4: lambda: mask_lthresh(clip, mthr, mlthresh, mask_tedge, mpand, **kwargs),
            5: lambda: mask_lthresh(clip, mthr, mlthresh, mask_canny_continuous, mpand, opencl=opencl,
                                    opencl_device=opencl_device, **kwargs),
            6: lambda: mask_lthresh(clip, mthr, mlthresh, mask_msharpen, mpand, **kwargs),
            'Sobel': lambda: mask_lthresh(clip, mthr, mlthresh, mask_sobel, mpand, opencl=opencl,
                                          opencl_device=opencl_device, **kwargs),
            'Canny': lambda: mask_lthresh(clip, mthr, mlthresh, mask_canny_binarized, mpand, opencl=opencl,
                                          opencl_device=opencl_device, **kwargs),
            'Prewitt': lambda: mask_lthresh(clip, mthr, mlthresh, mask_prewitt, mpand, **kwargs),
            'Robert': lambda: mask_lthresh(clip, mthr, mlthresh, mask_robert, mpand, **kwargs),
            'TEdge': lambda: mask_lthresh(clip, mthr, mlthresh, mask_tedge, mpand, **kwargs),
            'Canny_Old': lambda: mask_lthresh(clip, mthr, mlthresh, mask_canny_continuous, mpand, opencl=opencl,
                                              opencl_device=opencl_device, **kwargs),
            'MSharpen': lambda: mask_lthresh(clip, mthr, mlthresh, mask_msharpen, mpand, **kwargs),
            'Unknown': lambda: exec('raise ValueError(MODULE_NAME + ": unknown mtype")')
        }
        mtype = 5 if mtype is None else mtype
        mthr = (24,) if mthr is None else mthr
        masker = mask_kernel.get(mtype, mask_kernel['Unknown'])()
        masked_clip = masker(src, stabilized_clip)

    if txtmask > 0 and clip.format.color_family is not vs.GRAY:
        text_mask = mask_fadetxt(clip, lthr=txtmask, fade_num=txtfade)
        txt_protected_clip = core.std.MaskedMerge(masked_clip, src, text_mask, first_plane=True)
    else:
        text_mask = src
        txt_protected_clip = masked_clip

    final_output = ((showmask == -1 and text_mask) or
                    (showmask == 1 and masker(None, src, show=True)) or
                    (showmask == 2 and core.std.StackVertical([core.std.ShufflePlanes([masker(None, src, show=True),
                                                               core.std.BlankClip(src)], [0, 1, 2], vs.YUV), src])) or
                    (showmask == 3 and core.std.Interleave([core.std.ShufflePlanes([masker(None, src, show=True),
                                                           core.std.BlankClip(src)], [0, 1, 2], vs.YUV), src])) or
                    txt_protected_clip)
    return final_output
