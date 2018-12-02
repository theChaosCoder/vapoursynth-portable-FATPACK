#####################################################
###                                               ###
###          logoNR port for VapourSynth          ###
###                                               ###
###   port by TundraWork - tundrawork@gmail.com   ###
###   original by 06_taro - astrataro@gmail.com   ###
###                                               ###
###             v0.1.1 - 11 August 2017           ###
###              v0.1 - 22 March 2012             ###
###                                               ###
#####################################################
###
### Post-denoise and artifacts clean filter of EraseLogo.
### Only process logo areas in logo frames, even if l/t/r/b are not set. Non-logo areas are left untouched.
###
###
### +-----------+
### |  WARNING  |
### +-----------+
###
### Added a required "core" parameter, read the usage info below for how to set it.
### The "GPU" parameter is removed since there is no replace for FFT3DGPU filter in VapourSynth right now.
### The "nr" parameter's format has changed, read the usage info below if you want to use it.
### This script has NOT properly tested, use at your own risk.
###
###
### +---------+
### |  USAGE  |
### +---------+
###
### logoNR(core, dlg, src, chroma, l, t, r, b, nr)
###
### Example script clip:
### ------------------
###    core = vs.get_core()
###    import logonr_vs as logonr
###    src = core.avisource.AVISource("input.avi")
###    dlg = core.delogo.EraseLogo(src, "logofile.lgd")
###    logonr.logoNR(core, dlg, src, True, 1280, 40, -40, -960)
###
### core [instance]
### ------------------
###    The VapourSynth core instance created by vs.get_core(), must be the instance which loaded the FFT3DFilter filter.
###
### dlg [clip]
### ------------------
###    Clip after delogo.
###
### src [clip]
### ------------------
###    Clip before delogo.
###
### chroma [bool, default: True]
### ------------------
###    Process chroma plane or not.
###
### l/t/r/b [int, default: 0]
### ------------------
###    left/top/right/bottom pixels to be cropped for logo area.
###    Have the same restriction as Crop, e.g., no odd value for YV12.
###    logoNR only filters the logo areas in logo frames, no matter l/t/r/b are set or not.
###    So if you have other heavy filters running in a pipeline and don't care much about the speed of logoNR,
###    it is safe to left these values unset.
###    Setting these values only makes logoNR run faster, with rarely noticeable difference in result,
###    unless you set wrong values and the logo is not covered in your cropped target area.
###
### nr [string]
### ------------------
###    Set your custom nr filter to process logo areas, e.g., "core.rgvs.RemoveGrain(last, 4)".
###    NOTE: The clip input parameter of your filter must be "last".
###
###
### +----------------+
### |  REQUIREMENTS  |
### +----------------+
###
### -> FFT3DFilter, or your custom nr filter's requirements
### -> A newer version of VapourSynth, of course
###
###
### +-----------+
### | CHANGELOG |
### +-----------+
###
### v0.1.1 - 11 August 2017
###      - Port to VapourSynth
###
### v0.1 - 22 Mar 2012
###      - First release
###

import vapoursynth as vs
import functools
import math

class ConfigError(Exception):
    def __init__(self, msg):
        Exception.__init__ (self, msg)
        self.msg = msg

def throw_config_error(config_name, config_value):
    raise ConfigError('[logoNR] The following parameter is set to an invalid value: '+config_name+' = '+config_value)
    quit()

# The GetPlane function is based on the one by mawen1250, thanks for his work!
def GetPlane(clip, plane=None):
    # Set core and function name
    core = vs.get_core()
    funcName = 'GetPlane'
    if not isinstance(clip, vs.VideoNode):
        raise TypeError(funcName + ': \"clip\" must be a clip!')
    # Get properties of input clip
    sFormat = clip.format
    sNumPlanes = sFormat.num_planes
    # Parameters
    if plane is None:
        plane = 0
    elif not isinstance(plane, int):
        raise TypeError(funcName + ': \"plane\" must be an int!')
    elif plane < 0 or plane > sNumPlanes:
        raise ValueError(funcName + ': valid range of \"plane\" is [0, {})!'.format(sNumPlanes))
    # Process
    return core.std.ShufflePlanes(clip, plane, vs.GRAY)

# The Overlay function is based on the one by HolyWu, thanks for his work!
def Overlay(clipa, clipb, x=0, y=0, mask=None):
    core = vs.get_core()
    if not (isinstance(clipa, vs.VideoNode) and isinstance(clipb, vs.VideoNode)):
        raise TypeError('Overlay: This is not a clip')
    if clipa.format.subsampling_w > 0 or clipa.format.subsampling_h > 0:
        clipa_src = clipa
        clipa = core.resize.Point(clipa, format=core.register_format(clipa.format.color_family, clipa.format.sample_type, clipa.format.bits_per_sample, 0, 0).id)
    else:
        clipa_src = None
    if clipb.format.id != clipa.format.id:
        clipb = core.resize.Point(clipb, format=clipa.format.id)
    if mask is None:
        mask = core.std.BlankClip(clipb, color=[(1 << clipb.format.bits_per_sample) - 1] * clipb.format.num_planes)
    elif not isinstance(mask, vs.VideoNode):
        raise TypeError("Overlay: 'mask' is not a clip")
    if mask.width != clipb.width or mask.height != clipb.height:
        raise TypeError("Overlay: 'mask' must be the same dimension as 'clipb'")
    mask = GetPlane(mask, 0)
    # Calculate padding sizes
    l, r = x, clipa.width - clipb.width - x
    t, b = y, clipa.height - clipb.height - y
    # Split into crop and padding values
    cl, pl = min(l, 0) * -1, max(l, 0)
    cr, pr = min(r, 0) * -1, max(r, 0)
    ct, pt = min(t, 0) * -1, max(t, 0)
    cb, pb = min(b, 0) * -1, max(b, 0)
    # Crop and padding
    clipb = core.std.CropRel(clipb, cl, cr, ct, cb)
    mask = core.std.CropRel(mask, cl, cr, ct, cb)
    clipb = core.std.AddBorders(clipb, pl, pr, pt, pb)
    mask = core.std.AddBorders(mask, pl, pr, pt, pb)
    # Return padded clip
    last = core.std.MaskedMerge(clipa, clipb, mask)
    if clipa_src is not None:
        last = core.resize.Point(last, format=clipa_src.format.id)
    return last

def logoNR(core, dlg, src, chroma=True, l=0, t=0, r=0, b=0, nr=None, GPU=False):
    b_crop = not (l == 0 and t == 0 and r == 0 and b == 0)
    src    = core.std.CropRel(src, l, r, t, b) if b_crop else src
    last   = core.std.CropRel(dlg, l, r, t, b) if b_crop else dlg
    if nr is not None:
        clp_nr = eval(nr)
    elif GPU:
        throw_config_error("GPU", GPU)
        # clp_nr = FFT3DGPU(last, sigma=4, plane=4 if chroma else 0) # Waiting for VapourSynth port of FFT3DGPU.
    else:
        clp_nr = core.fft3dfilter.FFT3DFilter(last, sigma=4, plane=4 if chroma else 0)
    # VapourSynth's Expr function doesn't provide a parameter to control which plane to process, so we simply process all planes.
    logoM = core.std.Expr(clips=[last, src], expr="x y - abs 16 *")\
                .std.Maximum(planes=[0,1,2] if chroma else [0], coordinates=[0, 1, 0, 1, 1, 0, 1, 0])\
                .std.Convolution(matrix=[1, 1, 1, 1, 0, 1, 1, 1, 1], planes=[0,1,2] if chroma else [0])\
                .std.Deflate(planes=[0,1,2] if chroma else [0])
    if chroma:
        core.std.MergeDiff(clp_nr, logoM)
    else:
        core.std.MergeDiff(clp_nr, logoM, planes=0)
    return Overlay(dlg, clp_nr, x=l, y=t) if b_crop else clp_nr
