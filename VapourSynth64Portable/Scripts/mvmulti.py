from vapoursynth import core

def Analyze(super, blksize=8, blksizev=None, levels=0, search=4, searchparam=2, pelsearch=0, _lambda=None, chroma=True, tr=3, truemotion=True, lsad=None, plevel=None, _global=None, pnew=None, pzero=None, pglobal=0, overlap=0, overlapv=None, divide=0, badsad=10000.0, badrange=24, meander=True, trymany=False, fields=False, tff=None, search_coarse=3, dct=0):
    def getvecs(isb, delta):
        vectors  = core.mvsf.Analyze(super, isb=isb, blksize=blksize, blksizev=blksizev, levels=levels, search=search, searchparam=searchparam, pelsearch=pelsearch, _lambda=_lambda, chroma=chroma, delta=delta, truemotion=truemotion, lsad=lsad, plevel=plevel, _global=_global, pnew=pnew, pzero=pzero, pglobal=pglobal, overlap=overlap, overlapv=overlapv, divide=divide, badsad=badsad, badrange=badrange, meander=meander, trymany=trymany, fields=fields, tff=tff, search_coarse=search_coarse, dct=dct)
        return vectors
    bv           = [getvecs(True, i) for i in range(tr, 0, -1)]
    fv           = [getvecs(False, i) for i in range(1, tr+1)]
    vmulti       = bv + fv
    vmulti       = core.std.Interleave(vmulti)
    return vmulti

def Recalculate(super, vectors, thsad=200.0, smooth=1, blksize=8, blksizev=None, search=4, searchparam=2, _lambda=None, chroma=True, truemotion=True, pnew=None, overlap=0, overlapv=None, divide=0, meander=True, fields=False, tff=None, dct=0, tr=3):
    def refine(delta):
        analyzed = core.std.SelectEvery(vectors, 2*tr, delta)
        refined  = core.mvsf.Recalculate(super, analyzed, thsad=thsad, smooth=smooth, blksize=blksize, blksizev=blksizev, search=search, searchparam=searchparam, _lambda=_lambda, chroma=chroma, truemotion=truemotion, pnew=pnew, overlap=overlap, overlapv=overlapv, divide=divide, meander=meander, fields=fields, tff=tff, dct=dct)
        return refined
    vmulti       = [refine(i) for i in range(0, 2*tr)]
    vmulti       = core.std.Interleave(vmulti)
    return vmulti

def StoreVect(vectors, log):
    w            = vectors.get_frame(0).width
    with open(log, "w") as f:
         print(w, file=f)
    vectors      = core.std.CropAbs(vectors, width=w, height=1)
    return vectors

def RestoreVect(store, log):
    with open(log, "r") as f:
         w       = int(f.readline())
    vectors      = core.raws.Source(store, w, 1, src_fmt="Y8")
    blank        = core.std.BlankClip(vectors, width=1, length=1)
    vectors      = core.std.Splice([blank, vectors], mismatch=True)
    vectors      = core.std.Trim(vectors, 1)
    return vectors

def Compensate(clip, super, vectors, scbehavior=1, thsad=10000.0, fields=False, time=100.0, thscd1=400.0, thscd2=130.0, tff=None, tr=3, cclip=None):
    cclip        = clip if cclip is None else cclip
    def comp(delta):
        mv       = core.std.SelectEvery(vectors, 2*tr, delta)
        mc       = core.mvsf.Compensate(clip, super, mv, scbehavior=scbehavior, thsad=thsad, fields=fields, time=time, thscd1=thscd1, thscd2=thscd2, tff=tff)
        return mc
    bcomp        = [comp(i) for i in range(0, tr)]
    fcomp        = [comp(i) for i in range(tr, 2*tr)]
    compmulti    = bcomp + [cclip] + fcomp
    compmulti    = core.std.Interleave(compmulti)
    return compmulti

def Flow(clip, super, vectors, time=100.0, mode=0, fields=False, thscd1=400.0, thscd2=130.0, tff=None, tr=3, cclip=None):
    cclip        = clip if cclip is None else cclip
    def flow(delta):
        mv       = core.std.SelectEvery(vectors, 2*tr, delta)
        mc       = core.mvsf.Flow(clip, super, mv, time=time, mode=mode, fields=fields, thscd1=thscd1, thscd2=thscd2, tff=tff)
        return mc
    bflow        = [flow(i) for i in range(0, tr)]
    fflow        = [flow(i) for i in range(tr, 2*tr)]
    flowmulti    = bflow + [cclip] + fflow
    flowmulti    = core.std.Interleave(flowmulti)
    return flowmulti

def DegrainN(clip, super, mvmulti, tr=3, thsad=400.0, plane=4, limit=1.0, thscd1=400.0, thscd2=130.0):
    def bvn(n):
        bv       = core.std.SelectEvery(mvmulti, tr*2, tr-n)
        return bv
    def fvn(n):
        fv       = core.std.SelectEvery(mvmulti, tr*2, tr+n-1)
        return fv
    if tr == 1:
       dgn       = core.mvsf.Degrain1(clip, super, bvn(1), fvn(1), thsad=thsad, plane=plane, limit=limit, thscd1=thscd1, thscd2=thscd2)
    elif tr == 2:
       dgn       = core.mvsf.Degrain2(clip, super, bvn(1), fvn(1), bvn(2), fvn(2), thsad=thsad, plane=plane, limit=limit, thscd1=thscd1, thscd2=thscd2)
    elif tr == 3:
       dgn       = core.mvsf.Degrain3(clip, super, bvn(1), fvn(1), bvn(2), fvn(2), bvn(3), fvn(3), thsad=thsad, plane=plane, limit=limit, thscd1=thscd1, thscd2=thscd2)
    elif tr == 4:
       dgn       = core.mvsf.Degrain4(clip, super, bvn(1), fvn(1), bvn(2), fvn(2), bvn(3), fvn(3), bvn(4), fvn(4), thsad=thsad, plane=plane, limit=limit, thscd1=thscd1, thscd2=thscd2)
    elif tr == 5:
       dgn       = core.mvsf.Degrain5(clip, super, bvn(1), fvn(1), bvn(2), fvn(2), bvn(3), fvn(3), bvn(4), fvn(4), bvn(5), fvn(5), thsad=thsad, plane=plane, limit=limit, thscd1=thscd1, thscd2=thscd2)
    elif tr == 6:
       dgn       = core.mvsf.Degrain6(clip, super, bvn(1), fvn(1), bvn(2), fvn(2), bvn(3), fvn(3), bvn(4), fvn(4), bvn(5), fvn(5), bvn(6), fvn(6), thsad=thsad, plane=plane, limit=limit, thscd1=thscd1, thscd2=thscd2)
    elif tr == 7:
       dgn       = core.mvsf.Degrain7(clip, super, bvn(1), fvn(1), bvn(2), fvn(2), bvn(3), fvn(3), bvn(4), fvn(4), bvn(5), fvn(5), bvn(6), fvn(6), bvn(7), fvn(7), thsad=thsad, plane=plane, limit=limit, thscd1=thscd1, thscd2=thscd2)
    elif tr == 8:
       dgn       = core.mvsf.Degrain8(clip, super, bvn(1), fvn(1), bvn(2), fvn(2), bvn(3), fvn(3), bvn(4), fvn(4), bvn(5), fvn(5), bvn(6), fvn(6), bvn(7), fvn(7), bvn(8), fvn(8), thsad=thsad, plane=plane, limit=limit, thscd1=thscd1, thscd2=thscd2)
    elif tr == 9:
       dgn       = core.mvsf.Degrain9(clip, super, bvn(1), fvn(1), bvn(2), fvn(2), bvn(3), fvn(3), bvn(4), fvn(4), bvn(5), fvn(5), bvn(6), fvn(6), bvn(7), fvn(7), bvn(8), fvn(8), bvn(9), fvn(9), thsad=thsad, plane=plane, limit=limit, thscd1=thscd1, thscd2=thscd2)
    elif tr == 10:
       dgn       = core.mvsf.Degrain10(clip, super, bvn(1), fvn(1), bvn(2), fvn(2), bvn(3), fvn(3), bvn(4), fvn(4), bvn(5), fvn(5), bvn(6), fvn(6), bvn(7), fvn(7), bvn(8), fvn(8), bvn(9), fvn(9), bvn(10), fvn(10), thsad=thsad, plane=plane, limit=limit, thscd1=thscd1, thscd2=thscd2)
    elif tr == 11:
       dgn       = core.mvsf.Degrain11(clip, super, bvn(1), fvn(1), bvn(2), fvn(2), bvn(3), fvn(3), bvn(4), fvn(4), bvn(5), fvn(5), bvn(6), fvn(6), bvn(7), fvn(7), bvn(8), fvn(8), bvn(9), fvn(9), bvn(10), fvn(10), bvn(11), fvn(11), thsad=thsad, plane=plane, limit=limit, thscd1=thscd1, thscd2=thscd2)
    elif tr == 12:
       dgn       = core.mvsf.Degrain12(clip, super, bvn(1), fvn(1), bvn(2), fvn(2), bvn(3), fvn(3), bvn(4), fvn(4), bvn(5), fvn(5), bvn(6), fvn(6), bvn(7), fvn(7), bvn(8), fvn(8), bvn(9), fvn(9), bvn(10), fvn(10), bvn(11), fvn(11), bvn(12), fvn(12), thsad=thsad, plane=plane, limit=limit, thscd1=thscd1, thscd2=thscd2)
    elif tr == 13:
       dgn       = core.mvsf.Degrain13(clip, super, bvn(1), fvn(1), bvn(2), fvn(2), bvn(3), fvn(3), bvn(4), fvn(4), bvn(5), fvn(5), bvn(6), fvn(6), bvn(7), fvn(7), bvn(8), fvn(8), bvn(9), fvn(9), bvn(10), fvn(10), bvn(11), fvn(11), bvn(12), fvn(12), bvn(13), fvn(13), thsad=thsad, plane=plane, limit=limit, thscd1=thscd1, thscd2=thscd2)
    elif tr == 13:
       dgn       = core.mvsf.Degrain13(clip, super, bvn(1), fvn(1), bvn(2), fvn(2), bvn(3), fvn(3), bvn(4), fvn(4), bvn(5), fvn(5), bvn(6), fvn(6), bvn(7), fvn(7), bvn(8), fvn(8), bvn(9), fvn(9), bvn(10), fvn(10), bvn(11), fvn(11), bvn(12), fvn(12), bvn(13), fvn(13), thsad=thsad, plane=plane, limit=limit, thscd1=thscd1, thscd2=thscd2)
    elif tr == 14:
       dgn       = core.mvsf.Degrain14(clip, super, bvn(1), fvn(1), bvn(2), fvn(2), bvn(3), fvn(3), bvn(4), fvn(4), bvn(5), fvn(5), bvn(6), fvn(6), bvn(7), fvn(7), bvn(8), fvn(8), bvn(9), fvn(9), bvn(10), fvn(10), bvn(11), fvn(11), bvn(12), fvn(12), bvn(13), fvn(13), bvn(14), fvn(14), thsad=thsad, plane=plane, limit=limit, thscd1=thscd1, thscd2=thscd2)
    elif tr == 15:
       dgn       = core.mvsf.Degrain15(clip, super, bvn(1), fvn(1), bvn(2), fvn(2), bvn(3), fvn(3), bvn(4), fvn(4), bvn(5), fvn(5), bvn(6), fvn(6), bvn(7), fvn(7), bvn(8), fvn(8), bvn(9), fvn(9), bvn(10), fvn(10), bvn(11), fvn(11), bvn(12), fvn(12), bvn(13), fvn(13), bvn(14), fvn(14), bvn(15), fvn(15), thsad=thsad, plane=plane, limit=limit, thscd1=thscd1, thscd2=thscd2)
    elif tr == 16:
       dgn       = core.mvsf.Degrain16(clip, super, bvn(1), fvn(1), bvn(2), fvn(2), bvn(3), fvn(3), bvn(4), fvn(4), bvn(5), fvn(5), bvn(6), fvn(6), bvn(7), fvn(7), bvn(8), fvn(8), bvn(9), fvn(9), bvn(10), fvn(10), bvn(11), fvn(11), bvn(12), fvn(12), bvn(13), fvn(13), bvn(14), fvn(14), bvn(15), fvn(15), bvn(16), fvn(16), thsad=thsad, plane=plane, limit=limit, thscd1=thscd1, thscd2=thscd2)
    elif tr == 17:
       dgn       = core.mvsf.Degrain17(clip, super, bvn(1), fvn(1), bvn(2), fvn(2), bvn(3), fvn(3), bvn(4), fvn(4), bvn(5), fvn(5), bvn(6), fvn(6), bvn(7), fvn(7), bvn(8), fvn(8), bvn(9), fvn(9), bvn(10), fvn(10), bvn(11), fvn(11), bvn(12), fvn(12), bvn(13), fvn(13), bvn(14), fvn(14), bvn(15), fvn(15), bvn(16), fvn(16), bvn(17), fvn(17), thsad=thsad, plane=plane, limit=limit, thscd1=thscd1, thscd2=thscd2)
    elif tr == 18:
       dgn       = core.mvsf.Degrain18(clip, super, bvn(1), fvn(1), bvn(2), fvn(2), bvn(3), fvn(3), bvn(4), fvn(4), bvn(5), fvn(5), bvn(6), fvn(6), bvn(7), fvn(7), bvn(8), fvn(8), bvn(9), fvn(9), bvn(10), fvn(10), bvn(11), fvn(11), bvn(12), fvn(12), bvn(13), fvn(13), bvn(14), fvn(14), bvn(15), fvn(15), bvn(16), fvn(16), bvn(17), fvn(17), bvn(18), fvn(18), thsad=thsad, plane=plane, limit=limit, thscd1=thscd1, thscd2=thscd2)
    elif tr == 19:
       dgn       = core.mvsf.Degrain19(clip, super, bvn(1), fvn(1), bvn(2), fvn(2), bvn(3), fvn(3), bvn(4), fvn(4), bvn(5), fvn(5), bvn(6), fvn(6), bvn(7), fvn(7), bvn(8), fvn(8), bvn(9), fvn(9), bvn(10), fvn(10), bvn(11), fvn(11), bvn(12), fvn(12), bvn(13), fvn(13), bvn(14), fvn(14), bvn(15), fvn(15), bvn(16), fvn(16), bvn(17), fvn(17), bvn(18), fvn(18), bvn(19), fvn(19), thsad=thsad, plane=plane, limit=limit, thscd1=thscd1, thscd2=thscd2)
    elif tr == 20:
       dgn       = core.mvsf.Degrain20(clip, super, bvn(1), fvn(1), bvn(2), fvn(2), bvn(3), fvn(3), bvn(4), fvn(4), bvn(5), fvn(5), bvn(6), fvn(6), bvn(7), fvn(7), bvn(8), fvn(8), bvn(9), fvn(9), bvn(10), fvn(10), bvn(11), fvn(11), bvn(12), fvn(12), bvn(13), fvn(13), bvn(14), fvn(14), bvn(15), fvn(15), bvn(16), fvn(16), bvn(17), fvn(17), bvn(18), fvn(18), bvn(19), fvn(19), bvn(20), fvn(20), thsad=thsad, plane=plane, limit=limit, thscd1=thscd1, thscd2=thscd2)
    elif tr == 21:
       dgn       = core.mvsf.Degrain21(clip, super, bvn(1), fvn(1), bvn(2), fvn(2), bvn(3), fvn(3), bvn(4), fvn(4), bvn(5), fvn(5), bvn(6), fvn(6), bvn(7), fvn(7), bvn(8), fvn(8), bvn(9), fvn(9), bvn(10), fvn(10), bvn(11), fvn(11), bvn(12), fvn(12), bvn(13), fvn(13), bvn(14), fvn(14), bvn(15), fvn(15), bvn(16), fvn(16), bvn(17), fvn(17), bvn(18), fvn(18), bvn(19), fvn(19), bvn(20), fvn(20), bvn(21), fvn(21), thsad=thsad, plane=plane, limit=limit, thscd1=thscd1, thscd2=thscd2)
    elif tr == 22:
       dgn       = core.mvsf.Degrain22(clip, super, bvn(1), fvn(1), bvn(2), fvn(2), bvn(3), fvn(3), bvn(4), fvn(4), bvn(5), fvn(5), bvn(6), fvn(6), bvn(7), fvn(7), bvn(8), fvn(8), bvn(9), fvn(9), bvn(10), fvn(10), bvn(11), fvn(11), bvn(12), fvn(12), bvn(13), fvn(13), bvn(14), fvn(14), bvn(15), fvn(15), bvn(16), fvn(16), bvn(17), fvn(17), bvn(18), fvn(18), bvn(19), fvn(19), bvn(20), fvn(20), bvn(21), fvn(21), bvn(22), fvn(22), thsad=thsad, plane=plane, limit=limit, thscd1=thscd1, thscd2=thscd2)
    elif tr == 23:
       dgn       = core.mvsf.Degrain23(clip, super, bvn(1), fvn(1), bvn(2), fvn(2), bvn(3), fvn(3), bvn(4), fvn(4), bvn(5), fvn(5), bvn(6), fvn(6), bvn(7), fvn(7), bvn(8), fvn(8), bvn(9), fvn(9), bvn(10), fvn(10), bvn(11), fvn(11), bvn(12), fvn(12), bvn(13), fvn(13), bvn(14), fvn(14), bvn(15), fvn(15), bvn(16), fvn(16), bvn(17), fvn(17), bvn(18), fvn(18), bvn(19), fvn(19), bvn(20), fvn(20), bvn(21), fvn(21), bvn(22), fvn(22), bvn(23), fvn(23), thsad=thsad, plane=plane, limit=limit, thscd1=thscd1, thscd2=thscd2)
    elif tr == 24:
       dgn       = core.mvsf.Degrain24(clip, super, bvn(1), fvn(1), bvn(2), fvn(2), bvn(3), fvn(3), bvn(4), fvn(4), bvn(5), fvn(5), bvn(6), fvn(6), bvn(7), fvn(7), bvn(8), fvn(8), bvn(9), fvn(9), bvn(10), fvn(10), bvn(11), fvn(11), bvn(12), fvn(12), bvn(13), fvn(13), bvn(14), fvn(14), bvn(15), fvn(15), bvn(16), fvn(16), bvn(17), fvn(17), bvn(18), fvn(18), bvn(19), fvn(19), bvn(20), fvn(20), bvn(21), fvn(21), bvn(22), fvn(22), bvn(23), fvn(23), bvn(24), fvn(24), thsad=thsad, plane=plane, limit=limit, thscd1=thscd1, thscd2=thscd2)
    else:
       raise ValueError("DegrainN: dude, tr gotta be an int between 1-24, try something less wild maybe?")
    return dgn
