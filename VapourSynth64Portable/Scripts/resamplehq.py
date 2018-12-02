import vapoursynth as vs

__version__ = '2.0.0'


def resample_hq(clip, width=None, height=None, kernel='spline36', matrix='709', transfer='709',
                src_left=None, src_top=None, src_width=None, src_height=None, descale=False,
                filter_param_a=None, filter_param_b=None, range_in=None, precision=1):
    """Gamma correct resizing in linear light (RGB).

    Args:
        width (int): The target width.
        height (int): The target height.
        kernel (string): The kernel to use while resizing.
            Default is "spline36".
        matrix (string): The source matrix. Default is "709".
            Ignored if source colorspace is RGB.
        transfer (string): The transfer matrix. Default is "709".
        src_left (int): A sub‐pixel offset to crop the source from the left.
            Default 0.
        src_top (int): A sub‐pixel offset to crop the source from the top.
            Default 0.
        src_width (int): A sub‐pixel width to crop the source to. If negative,
            specifies offset from the right. Default is source width−src_left.
        src_height (int): A sub‐pixel height to crop the source to.
            If negative, specifies offset from the bottom.
            Default is source height − src_top.
        descale (bool): Activates the kernel inversion mode, allowing to “undo” a previous upsizing
            by compensating the loss in high frequencies, giving a sharper and more accurate output
            than classic kernels, closer to the original. Default is False.
        filter_param_a (float): For the bicubic filter, filter_param_a represent the “b” parameter ,
            for the lanczos filter, it represents the number of taps.
        filter_param_b (float): For the bicubic filter, it represent the “c” parameter.
        range_in (bool): Range of the input video, either "limited" or "full". Default is "limited".
        precision (bool): 0 uses half float precision , 1 uses single float precision. Default is 1.
    """
    core = vs.get_core()

    # Cheks

    if kernel == 'point' and descale is True:
        raise ValueError('Descale does not support point resizer.')

    if not isinstance(descale, bool):
        raise ValueError('"descale" must be True or False.')

    if precision < 0 or precision > 1:
        raise ValueError('"precision" must be either 0 (half) or 1 (single).')

    # Var stuff

    if descale is True:
        precision = 1

    kernel = kernel.lower().strip()

    if kernel == 'point':
        scaler = core.resize.Point
    elif kernel == 'linear' or kernel == 'bilinear':
        if descale is False:
            scaler = core.resize.Bilinear
        else:
            scaler = core.descale.Debilinear
    elif kernel == 'cubic' or kernel == 'bicubic':
        if descale is False:
            scaler = core.resize.Bicubic
        else:
            scaler = core.descale.Debicubic
    elif kernel == 'lanczos':
        if descale is False:
            scaler = core.resize.Lanczos
        else:
            scaler = core.descale.Delanczos
    elif kernel == 'spline16':
        if descale is False:
            scaler = core.resize.Spline16
        else:
            scaler = core.descale.Despline16
    elif kernel == 'spline36':
        if descale is False:
            scaler = core.resize.Spline36
        else:
            scaler = core.descale.Despline36

    scaler_opts = dict(width=width, height=height)

    if descale is True:
        scaler_opts.update(src_top=src_top, src_left=src_left)
        if kernel == 'cubic' or kernel == 'bicubic':
            scaler_opts.update(b=filter_param_a, c=filter_param_b)
        elif kernel == 'lanczos':
            scaler_opts.update(taps=filter_param_a)
    else:
        scaler_opts.update(src_left=src_left, src_top=src_top,
                           src_width=src_width, src_height=src_height,
                           filter_param_a=filter_param_a, filter_param_b=filter_param_b)

    if range_in is None:
        if clip.format.color_family == vs.RGB:
            range_in = 'full'
        else:
            range_in = 'limited'

    orig_format = clip.format.id

    if precision == 1:
        tmp_format = vs.RGBS
    else:
        tmp_format = vs.RGBH

    to_tmp_format_opts = dict(format=tmp_format, transfer_in_s=transfer, transfer_s='linear',
                              range_in_s=range_in, range_s='full')

    to_orig_format_opts = dict(format=orig_format, transfer_in_s='linear', transfer_s=transfer,
                               range_in_s='full', range_s=range_in)

    if clip.format.color_family != vs.RGB:
        to_tmp_format_opts.update(matrix_in_s=matrix)
        to_orig_format_opts.update(matrix_s=matrix)

    # Do stuff

    clip = core.resize.Bicubic(clip, **to_tmp_format_opts)

    clip = scaler(clip, **scaler_opts)

    clip = core.resize.Bicubic(clip, **to_orig_format_opts)

    return clip
