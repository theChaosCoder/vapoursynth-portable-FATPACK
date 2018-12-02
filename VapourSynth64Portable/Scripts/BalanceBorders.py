import vapoursynth as vs
import math

#BalanceBorders
#Original script: https://github.com/Neroldy/AviSynth_Filters/blob/master/BalanceBorders.avsi
#by PL — [2009.09.25] v0.2
#Ported by fdar0536 2017.08.17

#Dependency: VapourSynth R38 or later.

def BalanceBorders(c, cTop, cBottom, cLeft, cRight, thresh=128, blur=999):
    funcname = "BalanceBorders"

    if not isinstance(c, vs.VideoNode):
        raise TypeError(funcname+': This is not a clip.')
    
    if c.format.color_family != vs.YUV:
        raise TypeError(funcname+': Clip must be YUV family.')
    
    if c.format.sample_type != vs.INTEGER:
        raise TypeError(funcname+': Clip must be integer format.')
    
    if thresh < 0 or thresh > 128:
        raise ValueError(funcname+': \"thresh\" must between 0 and 128.')
    
    if blur < 0:
        raise ValueError(funcname+': \"blur\" must greater than 0.')
    
    del funcname
    core = vs.get_core()

    def BalanceTopBorder(c, cTop, thresh, blur):
        cWidth = c.width
        cHeight = c.height
        cTop = min(cTop, cHeight-1)
        BlurWidth = max(4, math.floor(cWidth / blur))

        C2 = core.resize.Point(c, cWidth<<1, cHeight<<1)

        C3 = core.std.CropRel(C2, 0, 0, cTop<<1, (cHeight-cTop-1)<<1) #(cHeight * 2 - cTop * 2) - 2
        C3 = core.resize.Point(C3, cWidth<<1, cTop<<1)
        C3 = core.resize.Bilinear(C3, BlurWidth<<1, cTop<<1)
        C3 = core.std.Convolution(C3, matrix=[0, 0, 0, 1, 1, 1, 0, 0, 0], planes=[0, 1, 2])
        ReferenceBlur = core.resize.Bilinear(C3, cWidth<<1, cTop<<1)
        
        Original = core.std.CropRel(C2, 0, 0, 0, (cHeight-cTop)<<1) #cHeight * 2 - 0 - cTop * 2

        C3 = core.resize.Bilinear(Original, BlurWidth<<1, cTop<<1)
        C3 = core.std.Convolution(C3, matrix=[0, 0, 0, 1, 1, 1, 0, 0, 0], planes=[0, 1, 2])
        OriginalBlur = core.resize.Bilinear(C3, cWidth<<1, cTop<<1)
        del C3

        Balanced = core.std.Expr(clips=[Original, OriginalBlur, ReferenceBlur], expr=["z y - x +", "z y - x +", "z y - x +"])
        del OriginalBlur
        del ReferenceBlur
        Difference = core.std.MakeDiff(Balanced, Original, planes=[0, 1, 2])
        del Balanced

        Tp = (128 + thresh) * ((1 << c.format.bits_per_sample) - 1) * 0.004 # 1 / 255 = 0.004
        Tm = (128 - thresh) * ((1 << c.format.bits_per_sample) - 1) * 0.004

        expr = 'x {0} > {0} x ?'.format(Tp)
        Difference = core.std.Expr(clips=Difference, expr=[expr, expr, expr])
        expr = 'x {0} < {0} x ?'.format(Tm)
        Difference = core.std.Expr(clips=Difference, expr=[expr, expr, expr])
        del expr
        del Tp
        del Tm

        res = core.std.MergeDiff(Original, Difference, planes=[0, 1, 2])
        del Difference

        res = core.std.StackVertical(clips=[res, core.std.CropRel(C2, 0, 0, cTop*2, 0)])
        #cHeight * 2 - cTop * 2 - (cHeight - cTop) * 2 = 0
        return core.resize.Point(res, cWidth, cHeight)

    res = BalanceTopBorder(c, cTop, thresh, blur).std.Transpose().std.FlipHorizontal() if cTop > 0 else core.std.Transpose(c).std.FlipHorizontal()
    res = BalanceTopBorder(res, cLeft, thresh, blur).std.Transpose().std.FlipHorizontal() if cLeft > 0 else core.std.Transpose(res).std.FlipHorizontal()
    res = BalanceTopBorder(res, cBottom, thresh, blur).std.Transpose().std.FlipHorizontal() if cBottom > 0 else core.std.Transpose(res).std.FlipHorizontal()
    res = BalanceTopBorder(res, cRight, thresh, blur).std.Transpose().std.FlipHorizontal() if cRight > 0 else core.std.Transpose(res).std.FlipHorizontal()
    
    return res