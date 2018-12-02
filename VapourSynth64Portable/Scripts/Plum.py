import vapoursynth as vs
import mvmulti
import math

fmtc_args                    = dict(fulls=True, fulld=True)
msuper_args                  = dict(hpad=0, vpad=0, sharp=2, levels=0, chroma=False)
manalyze_args                = dict(search=3, truemotion=False, trymany=True, levels=0, badrange=-24, divide=0, dct=0, chroma=False)
mrecalculate_args            = dict(truemotion=False, search=3, smooth=1, divide=0, dct=0, chroma=False)
mdegrain_args                = dict(thsad=10000.0, plane=0, thscd1=16711680.0, thscd2=255.0)
mcompensate_args             = dict(thscd1=16711680.0, thscd2=255.0)
convolution_args             = dict(matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1])
deconvolution_args           = dict(line=0)
nnedi_args                   = dict(field=1, dh=True, nns=4, qual=2, etype=1, nsize=0)

class get_core:
      def __init__(self):
          self.MSuper        = vs.core.mvsf.Super
          self.MAnalyze      = mvmulti.Analyze
          self.MRecalculate  = mvmulti.Recalculate
          self.MDegrainN     = mvmulti.DegrainN
          self.MCompensate   = mvmulti.Compensate
          self.RGB2OPP       = vs.core.bm3d.RGB2OPP
          self.OPP2RGB       = vs.core.bm3d.OPP2RGB
          self.KNLMeansCL    = vs.core.knlm.KNLMeansCL
          self.NNEDI         = vs.core.nnedi3.nnedi3
          self.FQSharp       = vs.core.vcfreq.Sharp
          self.Resample      = vs.core.fmtc.resample
          self.MakeDiff      = vs.core.std.MakeDiff
          self.MergeDiff     = vs.core.std.MergeDiff
          self.AddBorders    = vs.core.std.AddBorders
          self.Crop          = vs.core.std.CropRel
          self.Expr          = vs.core.std.Expr
          self.Median        = vs.core.std.Median
          self.Merge         = vs.core.std.Merge
          self.Convolution   = vs.core.std.Convolution
          self.Transpose     = vs.core.std.Transpose
          self.ShufflePlanes = vs.core.std.ShufflePlanes
          self.SelectEvery   = vs.core.std.SelectEvery
          self.SetFieldBased = vs.core.std.SetFieldBased

      def CutOff(self, low, hi, p, margin):
          def inline(src, p):
              upsmp          = self.Resample(src, src.width*2, src.height*2, kernel="gauss", a1=100, **fmtc_args)
              clip           = self.Resample(upsmp, src.width, src.height, kernel="gauss", a1=p, **fmtc_args)
              return clip
          hif                = self.MakeDiff(hi, inline(hi, p+margin))
          clip               = self.MergeDiff(inline(low, p), hif)
          return clip

      def Pad(self, src, left, right, top, bottom):
          w                  = src.width
          h                  = src.height
          clip               = self.Resample(src, w+left+right, h+top+bottom, -left, -top, w+left+right, h+top+bottom, kernel="point", **fmtc_args)
          return clip

      def Deconvolution(self, src, radius, wn, fr, scale):
          src                = self.Pad(src, radius+fr, radius+fr, radius+fr, radius+fr)
          sharp              = self.FQSharp(src, x=radius, y=radius, wn=wn, fr=fr, scale=scale, **deconvolution_args)
          sharp              = self.CutOff(src, sharp, 1, 0)
          clip               = self.Crop(sharp, radius+fr, radius+fr, radius+fr, radius+fr)
          return clip

      def Shrink(self, src):
          blur               = self.Median(src)
          dif                = self.MakeDiff(blur, src)
          convD              = self.Convolution(dif, **convolution_args)
          DD                 = self.MakeDiff(dif, convD)
          convDD             = self.Convolution(DD, **convolution_args)
          DDD                = self.Expr([DD, convDD], ["x y - x * 0.0 < 0.0 x y - abs x abs < x y - x ? ?"])
          dif                = self.MakeDiff(dif, DDD)
          convD              = self.Convolution(dif, **convolution_args)
          dif                = self.Expr([dif, convD], ["y abs x abs > y 0.0 ?"])
          clip               = self.MergeDiff(src, dif)
          return clip

      def NLErrors(self, src, a, h, rclip):
          pad                = self.AddBorders(src, a, a, a, a)
          rclip              = self.AddBorders(rclip, a, a, a, a) if rclip is not None else None
          nlm                = self.KNLMeansCL(pad, d=0, a=a, s=0, h=h, rclip=rclip)
          clip               = self.Crop(nlm, a, a, a, a)
          return clip

      def TemporalExtremum(self, src, radius, mode):
          clip               = self.SelectEvery(src, radius * 2 + 1, 0)
          for i in range(1, radius * 2 + 1):
              clip           = self.Expr([clip, self.SelectEvery(src, radius * 2 + 1, i)], "x y " + mode)
          return clip

      def Clamp(self, src, bright_limit, dark_limit, overshoot, undershoot):
          clip               = self.Expr([src, bright_limit, dark_limit], "x y {os} + > y {os} + x ? z {us} - < z {us} - x ?".format(os=overshoot, us=undershoot))
          return clip

      def XYClosest(self, src1, src2, ref):
          clip               = self.Expr([src1, src2, ref], "x z - abs y z - abs > y x ?")
          return clip

class internal:
      def super(core, src, pel):
          src                = core.Pad(src, 128, 128, 128, 128)
          clip               = core.Transpose(core.NNEDI(core.Transpose(core.NNEDI(src, **nnedi_args)), **nnedi_args))
          if pel == 4:
             clip            = core.Transpose(core.NNEDI(core.Transpose(core.NNEDI(clip, **nnedi_args)), **nnedi_args))
          return clip

      def basic(core, src, strength, a, h, radius, wn, scale, cutoff):
          c1                 = 0.0980468750214585389567894354907
          c2                 = 0.0124360171036224062543798508968
          h                 += [c1 * h[1] * strength * (1.0 - math.exp(-1.0 / (c1 * strength)))]
          cutoff_array       = [cutoff]
          cutoff_array      += [int(max(1.0, c2 * math.pow(cutoff, 2.0) * math.log(1.0 + 1.0 / (c2 * cutoff))) + 0.5)]
          strength_floor     = math.floor(strength)
          strength_ceil      = math.ceil(strength)
          def inline(src):
              sharp          = core.Deconvolution(src, radius, wn, int(a / 2 + 0.5), scale)
              sharp          = core.Transpose(core.NNEDI(core.Transpose(core.NNEDI(sharp, **nnedi_args)), **nnedi_args))
              ref            = core.Transpose(core.NNEDI(core.Transpose(core.NNEDI(src, **nnedi_args)), **nnedi_args))
              sharp          = core.NLErrors(ref, a, h[0], sharp)
              dif            = core.MakeDiff(sharp, ref)
              dif            = core.Resample(dif, src.width, src.height, sx=-0.5, sy=-0.5, kernel="gauss", a1=100)
              sharp          = core.MergeDiff(src, dif)
              sharp          = core.CutOff(src, sharp, cutoff_array[0], 0)
              local_error    = core.NLErrors(src, radius, h[1], src)
              local_limit    = core.MergeDiff(src, core.MakeDiff(src, local_error))
              limited        = core.Expr([sharp, local_limit, src], ["x z - abs y z - abs > y x ?"])
              clip           = core.Shrink(limited)
              return clip
          sharp              = src
          for i in range(strength_floor):
              sharp          = inline(sharp)
          if strength_floor != strength_ceil:
             sharp_ceil      = inline(sharp)
             sharp           = core.Merge(sharp, sharp_ceil, strength - strength_floor)
          sharp_nr           = core.NLErrors(sharp, a, h[2], sharp)
          clip               = core.CutOff(sharp, sharp_nr, cutoff_array[1], 0)
          h.pop()
          return clip

      def final(core, src, super, radius, pel, sad, flexibility, strength, constants, cutoff, freq_margin):
          constant           = 0.0009948813682897925944723492342
          me_sad             = constant * math.pow(sad, 2.0) * math.log(1.0 + 1.0 / (constant * sad))
          expression         = "{x} {y} - abs {lstr} / 1 {pstr} / pow {sstr} * {x} {y} - {x} {y} - abs 0.001 + / * {x} {y} - 2 pow {x} {y} - 2 pow {ldmp} + / * 256 / y +".format(lstr=constants[0], pstr=constants[1], sstr=strength, ldmp=constants[2], x="x 256 *", y="y 256 *")
          for i in range(2):
              src[i]         = core.Pad(src[i], 128, 128, 128, 128)
          src               += [core.Merge(src[0], src[1], flexibility)]
          src[1]             = core.MakeDiff(src[1], src[0])
          super             += [core.Merge(super[0], super[1], flexibility) if super[0] is not None and super[1] is not None else None]
          super[1]           = core.MakeDiff(super[1], super[0]) if super[0] is not None and super[1] is not None else None
          blankdif           = core.Expr(src[0], "0.0")
          supersearch        = core.MSuper(src[0], pelclip=super[0], rfilter=4, pel=pel, **msuper_args)
          superdif           = core.MSuper(src[1], pelclip=super[1], rfilter=2, pel=pel, **msuper_args)
          superflex          = core.MSuper(src[2], pelclip=super[2], rfilter=2, pel=pel, **msuper_args)
          vmulti             = core.MAnalyze(supersearch, tr=radius, overlap=64, blksize=128, **manalyze_args)
          vmulti             = core.MRecalculate(supersearch, vmulti, tr=radius, overlap=32, blksize=64, thsad=me_sad, **mrecalculate_args)
          vmulti             = core.MRecalculate(supersearch, vmulti, tr=radius, overlap=16, blksize=32, thsad=me_sad, **mrecalculate_args)
          vmulti             = core.MRecalculate(supersearch, vmulti, tr=radius, overlap=8, blksize=16, thsad=me_sad, **mrecalculate_args)
          vmulti             = core.MRecalculate(supersearch, vmulti, tr=radius, overlap=4, blksize=8, thsad=me_sad, **mrecalculate_args)
          vmulti             = core.MRecalculate(supersearch, vmulti, tr=radius, overlap=2, blksize=4, thsad=me_sad, **mrecalculate_args)
          vmulti             = core.MRecalculate(supersearch, vmulti, tr=radius, overlap=1, blksize=2, thsad=me_sad, **mrecalculate_args)
          averaged_dif       = core.MDegrainN(blankdif, superdif, vmulti, tr=radius, **mdegrain_args)
          averaged_dif       = core.XYClosest(averaged_dif, src[1], blankdif)
          compensated        = core.MCompensate(src[0], superflex, vmulti, tr=radius, thsad=sad, **mcompensate_args)
          src[0]             = core.Crop(src[0], 128, 128, 128, 128)
          averaged_dif       = core.Crop(averaged_dif, 128, 128, 128, 128)
          compensated        = core.Crop(compensated, 128, 128, 128, 128)
          bright_limit       = core.TemporalExtremum(compensated, radius, "max")
          dark_limit         = core.TemporalExtremum(compensated, radius, "min")
          averaged           = core.MergeDiff(src[0], averaged_dif)
          clamped            = core.Clamp(averaged, bright_limit, dark_limit, 0.0, 0.0)
          amplified          = core.Expr([clamped, src[0]], expression)
          clip               = core.CutOff(src[0], amplified, cutoff, freq_margin)
          super.pop()
          return clip

def Super(src, pel=4):
    if not isinstance(src, vs.VideoNode):
       raise TypeError("Plum.Super: src has to be a video clip!")
    elif src.format.sample_type != vs.FLOAT or src.format.bits_per_sample < 32:
       raise TypeError("Plum.Super: the sample type of src has to be single precision!")
    if not isinstance(pel, int):
       raise TypeError("Plum.Super: pel has to be an integer!")
    elif pel != 2 and pel != 4:
       raise RuntimeError("Plum.Super: pel has to be 2 or 4!")
    core                     = get_core()
    src                      = core.SetFieldBased(src, 0)
    colorspace               = src.format.color_family
    if colorspace == vs.RGB:
       src                   = core.RGB2OPP(src, 1)
    if colorspace != vs.GRAY:
       src                   = core.ShufflePlanes(src, 0, vs.GRAY)
    clip                     = internal.super(core, src, pel)
    del core
    return clip

def Basic(src, strength=3.20, a=32, h=[6.4, 64.0], radius=1, wn=0.48, scale=0.28, cutoff=24):
    if not isinstance(src, vs.VideoNode):
       raise TypeError("Plum.Basic: src has to be a video clip!")
    elif src.format.sample_type != vs.FLOAT or src.format.bits_per_sample < 32:
       raise TypeError("Plum.Basic: the sample type of src has to be single precision!")
    if not isinstance(strength, float) and not isinstance(strength, int):
       raise TypeError("Plum.Basic: strength has to be a real number!")
    elif strength <= 0.0:
       raise RuntimeError("Plum.Basic: strength has to be greater than 0.0!")
    if not isinstance(a, int):
       raise TypeError("Plum.Basic: a has to be an integer!")
    if not isinstance(h, list):
       raise TypeError("Plum.Basic: h has to be an array!")
    elif len(h) != 2:
       raise RuntimeError("Plum.Basic: h has to contain 2 elements exactly!")
    for i in range(2):
        if not isinstance(h[i], float) and not isinstance(h[i], int):
           raise TypeError("Plum.Basic: elements in h must be real numbers!")
        elif h[i] <= 0:
           raise RuntimeError("Plum.Basic: elements in h must be greater than 0!")
    if not isinstance(radius, int):
       raise TypeError("Plum.Basic: radius has to be an integer!")
    elif radius < 1:
       raise RuntimeError("Plum.Basic: radius has to be greater than 0!")
    if not isinstance(wn, float) and not isinstance(wn, int):
       raise TypeError("Plum.Basic: wn has to be a real number!")
    if not isinstance(scale, float) and not isinstance(scale, int):
       raise TypeError("Plum.Basic: scale has to be a real number!")
    if not isinstance(cutoff, int):
       raise TypeError("Plum.Basic: cutoff has to be an integer!")
    elif cutoff < 1 or cutoff > 100:
       raise RuntimeError("Plum.Basic: cutoff must fall in (0, 100]!")
    core                     = get_core()
    src                      = core.SetFieldBased(src, 0)
    colorspace               = src.format.color_family
    if colorspace == vs.RGB:
       src                   = core.RGB2OPP(src, 1)
    if colorspace != vs.GRAY:
       src                   = core.ShufflePlanes(src, 0, vs.GRAY)
    clip                     = internal.basic(core, src, strength, a, h, radius, wn, scale, cutoff)
    del core
    return clip

def Final(src, super=[None, None], radius=6, pel=4, sad=400.0, flexibility=0.64, strength=3.20, constants=[1.49, 1.272, None], cutoff=12, freq_margin=20):
    if not isinstance(src, list):
       raise TypeError("Plum.Final: src has to be an array!")
    elif len(src) != 2:
       raise RuntimeError("Plum.Final: src has to contain 2 elements exactly!")
    elif not isinstance(src[0], vs.VideoNode) or not isinstance(src[1], vs.VideoNode):
       raise TypeError("Plum.Final: elements in src must be video clips!")
    elif src[0].format.sample_type != vs.FLOAT or src[0].format.bits_per_sample < 32:
       raise TypeError("Plum.Final: the sample type of src[0] has to be single precision!")
    elif src[1].format.id != vs.GRAYS:
       raise RuntimeError("Plum.Final: corrupted basic estimation!")
    if not isinstance(super, list):
       raise TypeError("Plum.Final: super has to be an array!")
    elif len(super) != 2:
       raise RuntimeError("Plum.Final: super has to contain 2 elements exactly!")
    for i in range(2):
        if not isinstance(super[i], vs.VideoNode) and super[i] is not None:
           raise TypeError("Plum.Final: elements in super must be video clips or None!")
        elif super[i] is not None:
           if super[i].format.id != vs.GRAYS:
              raise RuntimeError("Plum.Final: corrupted super clips!")
    if not isinstance(radius, int):
       raise TypeError("Plum.Final: radius has to be an integer!")
    elif radius < 1:
       raise RuntimeError("Plum.Final: radius has to be greater than 0!")
    if not isinstance(pel, int):
       raise TypeError("Plum.Final: pel has to be an integer!")
    elif pel != 1 and pel != 2 and pel != 4:
       raise RuntimeError("Plum.Final: pel has to be 1, 2 or 4!")
    if not isinstance(sad, float) and not isinstance(sad, int):
       raise TypeError("Plum.Final: sad has to be a real number!")
    elif sad <= 0:
       raise RuntimeError("Plum.Final: sad has to be greater than 0!")
    if not isinstance(strength, float) and not isinstance(strength, int):
       raise TypeError("Plum.Final: strength has to be a real number!")
    elif strength <= 0:
       raise RuntimeError("Plum.Final: strength has to be greater than 0!")
    if not isinstance(flexibility, float) and not isinstance(flexibility, int):
       raise TypeError("Plum.Final: flexibility has to be a real number!")
    elif flexibility < 0.0 or flexibility > 1.0:
       raise RuntimeError("Plum.Final: flexibility has to fall in [0.0, 1.0]!")
    if not isinstance(constants, list):
       raise TypeError("Plum.Final: constants has to be an array!")
    elif len(constants) != 3:
       raise RuntimeError("Plum.Final: constants has to contain 3 elements exactly!")
    for i in range(2):
        if not isinstance(constants[i], float) and not isinstance(constants[i], int):
           raise TypeError("Plum.Final: elements in constants must be real numbers!")
    if not isinstance(constants[2], float) and not isinstance(constants[2], int) and constants[2] is not None:
       raise TypeError("Plum.Final: constants[2] has to be a real number or None!")
    if not isinstance(cutoff, int):
       raise TypeError("Plum.Final: cutoff has to be an integer!")
    elif cutoff < 1 or cutoff > 100:
       raise RuntimeError("Plum.Final: cutoff must fall in (0, 100]!")
    if not isinstance(freq_margin, int):
       raise TypeError("Plum.Final: freq_margin has to be an integer!")
    elif freq_margin < 0 or freq_margin > 100-cutoff:
       raise RuntimeError("Plum.Final: freq_margin must fall in [0, 100-cutoff]!")
    constants[2]             = strength + 0.1 if constants[2] is None else constants[2]
    core                     = get_core()
    for i in range(2):
        src[i]               = core.SetFieldBased(src[i], 0)
        super[i]             = core.SetFieldBased(super[i], 0) if super[i] is not None else None
    colorspace               = src[0].format.color_family
    if colorspace == vs.RGB:
       src[0]                = core.RGB2OPP(src[0], 1)
    if colorspace != vs.GRAY:
       src_color             = src[0]
       src[0]                = core.ShufflePlanes(src[0], 0, vs.GRAY)
    clip                     = internal.final(core, src, super, radius, pel, sad, flexibility, strength, constants, cutoff, freq_margin)
    if colorspace != vs.GRAY:
       clip                  = core.ShufflePlanes([clip, src_color], [0, 1, 2], vs.YUV)
    if colorspace == vs.RGB:
       clip                  = core.OPP2RGB(clip, 1)
    del core
    return clip
