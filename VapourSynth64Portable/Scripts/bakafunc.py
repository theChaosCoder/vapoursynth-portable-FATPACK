import vapoursynth as vs
import functools
import math
import havsfunc as haf
	
    
    # Baka's Anti-aliasing script
    #
    # Basic Anti-Aliasing script (get it?!? ;) ) with supersampling and edge masking.
    # Takes any kind of bitdepth since processing happens in 8 bit, 
    # results are merged back into the original bitdepth with an edge mask as an attempt at preventing
    # precision loss in near flat, texture or gradient areas.
    # 99.9% of all edges don't benefit from high bitdepth precision anyway.
    #
    # Possible modes are: "nnedi3", "eedi3", "eedi2", "sangnom2", "eedi2+sangnom2" and "eedi3+sangnom2".
    # Sangnom2, nnedix and eedix are essentially just single field deinterlacers but nowadays they're mostly used in anti-aliasing scripts.
    #
    # Sangnom2 is rather strong, can create artefacts quite easily but fast!
    # That's why it's recommended to supersample sangnom2 in order to limit the strong results and prevent artefacts.
    #
    # Nnedi3 is neural network based deinterlacer.
    # "it takes in a frame, throws away one field, and then interpolates the missing pixels using only information from the kept field".
    # Subjectively I'd classify nnedi3 as a medium strong anti-aliasing kernel as it doesn't create that much (if any!) artefacts compared to eedix or sangnom2. 
    # Probably the most used kernel for anti-aliasing scripts.
    #
    # Eedi3 stands for Enhanced Edge Directed Interpolation 3.
    # "The cost is based on neighborhood similarity (favor connecting regions that look similar),
    # the vertical difference created by the interpolated values (favor small differences), the interpolation directions (favor short connections vs long),
    # and the change in interpolation direction from pixel to pixel (favor small changes)."
    # Eedi3 in my opinion can do wonders when we're talking about connecting edges but is also prone to making some nasty artefacts from time to time
    # But do not fear that's why we have things like supersampling!
    #
    # Eedi2 is basically just here for legacy sake since some people prefer it over eedi3 because it's faster.
    # Go read up on the difference between eedi3 and eedi2 if you're feeling adventurous.
    #
    # Lastly we have a combination of eedix and sangnom2, it's an incredibly strong kernel for anti-aliasing intended for bad sources like CG and bad upscales.
    # it's basically using eedix as supersampled clip for sangnom2.
    #
    # Note: sangnom2 only takes mod16 resolutions so if your resolution isn't mod16 I suggest setting ss to something different from 1 
    # (1.1 or even 1,0001, it's just in order to trigger the resize function which will make the resolution mod16 by default). 

def Baa(c, aa = "sangnom2", ss = None, mask=True, mthr = 30, blur=5, expand = 1, chroma = False):

    core = vs.get_core() 
    
    aac = 48 if chroma == True else 0
	
    if not isinstance(c, vs.VideoNode):
        raise ValueError('ediaaclip: This is not a clip')
        
    edibits=c.format.bits_per_sample

    if edibits > 8:
        a8 = core.fmtc.bitdepth(c, bits=8)
         
    else :
        a8 = c
        
    if mask == True:
        mask = core.generic.Prewitt(a8, mthr, mthr)
        mask = BlurLoop(ExpandLoop(mask, expand), blur)
        
    if ss == None:
        ss = 2 if aa == "sangnom2" else 1.1
         
    if ss != 1:
        a8 = Resize(a8, m16(c.width * ss), m16(c.height * ss))
         
    if aa == "nnedi3":
        a8 = Resize(core.nnedi3.nnedi3(a8, field=1,dh=True).std.Transpose().nnedi3.nnedi3(field=1,dh=True).std.Transpose(), a8.width, a8.height, -0.5,-0.5,2*a8.width+.001,2*a8.height+.001)
            
    elif aa == "eedi3":
        a8 = Resize(core.eedi3.eedi3(a8, field=1,dh=True).std.Transpose().eedi3.eedi3(field=1,dh=True).std.Transpose(), a8.width, a8.height, -0.5,-0.5,2*a8.width+.001,2*a8.height+.001)
            
    elif aa == "eedi2":
        a8 = Resize(core.eedi2.EEDI2(a8, field=1,dh=True).std.Transpose().eedi2.EEDI2(field=1,dh=True).std.Transpose(), a8.width, a8.height, -0.5,-0.5,2*a8.width+.001,2*a8.height+.001)
        
        #To-Do rewrite eedix+sangnom2 combo to not throw away good field
    elif aa == "eedi2+sangnom2":
        a8 = Resize(core.std.Transpose(core.sangnom.SangNomMod(core.std.Transpose(core.sangnom.SangNomMod(core.std.Transpose(core.eedi2.EEDI2(core.std.Transpose(core.eedi2.EEDI2(a8, field=1,dh=True)), field=1,dh=True)),aac=aac)),aac=aac)), a8.width, a8.height, -0.5,-0.5,2*a8.width+.001,2*a8.height+.001)
        
    elif aa == "eedi3+sangnom2":
        a8 = Resize(core.std.Transpose(core.sangnom.SangNomMod(core.std.Transpose(core.sangnom.SangNomMod(core.std.Transpose(core.eedi3.eedi3(core.std.Transpose(core.eedi3.eedi3(a8, field=1,dh=True)), field=1,dh=True)),aac=aac)),aac=aac)), a8.width, a8.height, -0.5,-0.5,2*a8.width+.001,2*a8.height+.001)
        
    else:
        a8 = core.sangnom.SangNomMod(a8,aac=aac).std.Transpose().sangnom.SangNomMod(aac=aac).std.Transpose()
  
    if ss != 1:
        a8 = Resize(a8, c.width, c.height)

    if edibits > 8:
        a8 = core.fmtc.bitdepth(a8,bits=edibits)
        if isinstance(mask, vs.VideoNode):
            mask = core.fmtc.bitdepth(mask,bits=edibits)


    if isinstance(mask, vs.VideoNode):
        return core.std.MaskedMerge(c,a8,mask)
        
		 
    return a8
    
    
    # As the name suggests it's a somewhat lazy but effective way to repair chroma bleed.
    # The chroma is warped according to the luma edges hence the chroma "sticks" to luma edges.
    
def LazyChromaBleedFix(c, depth = 16, ss=1, mthr = 30, blur=5, expand = 1):

    core = vs.get_core() 

    bits=c.format.bits_per_sample
    
    if bits > 8:
        a8 = core.fmtc.bitdepth(c, bits=8)
    else:
        a8 = c

    if ss != 1:
        a8 = Resize(a8, m16(c.width * ss), m16(c.height * ss))
         
    a8 = core.warp.AWarpSharp2(a8, depth = depth,planes=[1,2])
 
    if ss != 1:
        a8 = Resize(a8, c.width, c.height)
        
    if bits > 8:
        return RestoreDepth(a8,c,planes=[1,2])
    else:
        return a8
    
    # A workaround for preserving high bitdepth when functions only work in 8 bit.
    # This function is only really effective if the 8 bit functions doesn't alter banding or anything that really needs the high bitdepth, like edges.
    # The functions calculates the difference between the original high bitdepth clip and the new low bitdepth clip.
    # Afterwards Pastes the altered low bitdepth pixels back to the high bitdepth clip.
    # It's important to use the same dithering method you used to get the low bitdepth clip.
    # LazyChromaBleedFix is a great usage example of RestoreDepth.
    
def RestoreDepth(LowDepth, HighDepth, diffmask=None, dmode=None, planes=[0, 1, 2]):
    
    core = vs.get_core() 
    
    lowbits = LowDepth.format.bits_per_sample
    highbits = HighDepth.format.bits_per_sample
    
    DownscaledDepth = core.fmtc.bitdepth(HighDepth,bits=lowbits,dmode=dmode)
    
    Yexpr = ("x y - abs 0 > " + str(pow(2,lowbits)-1) + " 0 ?") if 0 in planes else ""
    Uexpr = ("x y - abs 0 > " + str(pow(2,lowbits)-1) + " 0 ?") if 1 in planes else ""
    Vexpr = ("x y - abs 0 > " + str(pow(2,lowbits)-1) + " 0 ?") if 2 in planes else ""
    
    if isinstance(diffmask, vs.VideoNode) == False:
        diffmask = core.std.Expr(clips=[LowDepth, DownscaledDepth], expr=[Yexpr, Uexpr, Vexpr])
    
    UpscaledDepth = core.fmtc.bitdepth(LowDepth,bits=highbits)
    UpscaledMask = core.fmtc.bitdepth(diffmask,bits=highbits)
    
    return core.std.MaskedMerge(HighDepth, UpscaledDepth, UpscaledMask,planes=planes)
    
	# original script by Torchlight and Firesledge(?)
	# port by BakaProxy
	# bob and qtgmc from HavsFunc is used 
    # what is this used for again?    

def dec_txt60mc (src,frame_ref, srcbob=False,draft=False,tff=None):

	core = vs.get_core()

	field_ref = frame_ref if srcbob else frame_ref * 2
	field_ref =      field_ref  % 5
	invpos    = (5 - field_ref) % 5
	pel       = 1 if draft else 2
 
	if srcbob:
		last = src
	elif draft:
		last = haf.Bob(src,tff=tff)  
	else:
		last = haf.QTGMC(src,SourceMatch=3, Lossless=2, TR0=1, TR1=1, TR2=1,TFF=tff)  

     	
	if invpos > 3:
		clean  = core.std.AssumeFPS(core.std.Trim(last, 0, 0)+core.std.SelectEvery(last,5, 8 - invpos), fpsnum=12000, fpsden=1001)
	else:
		clean = core.std.SelectEvery(last,5, 4 - invpos)
	if invpos > 1:
		jitter = core.std.AssumeFPS(core.std.Trim(last, 0, 0)+core.std.SelectEvery(last,5, [6 - invpos, 5 - invpos]), fpsnum=24000, fpsden=1001)
	else:
		jitter = core.std.SelectEvery(last,5, [1 - invpos, 2 - invpos])
		
	jsup   = core.mv.Super(jitter,pel=pel)
	vect_f = core.mv.Analyse (jsup,isb=False, delta=1, overlap=4)
	vect_b = core.mv.Analyse (jsup,isb=True,  delta=1, overlap=4)
	comp   = core.mv.FlowInter (jitter,jsup, vect_b, vect_f, time=50, thscd1=400)
	fixed  = core.std.SelectEvery (comp,2, 0)
	last   = core.std.Interleave ([fixed, clean])
	return last[invpos // 3:]
    
    
### Helper Functions
    
    
def m16(x):
    return 16 if x < 16 else int(round(x / 16) * 16)

def BlurLoop(c, x, planes=[0]):
    
    if c.format.color_family != vs.GRAY:
        rg = [0,0,0]
        if 0 in planes:
            rg[0] = 12
        else:
            rg[0] = 0
            
        if 1 in planes:
            rg[1] = 12
        else:
            rg[1] = 0
            
        if 2 in planes:
            rg[2] = 12
        else:
            rg[2] = 0
    else:
        rg = [12]
        
        
    core = vs.get_core()
    
    for y in range(0, x):
        c = core.rgvs.RemoveGrain(c, rg)
        
    return c
    
def ExpandLoop(c, x, planes=[0]):
        
    core = vs.get_core()
    
    for y in range(0, x):
        c = core.std.Maximum(c,planes=planes)
        
    return c

def InpandLoop(c, x, planes=[0]):
        
    core = vs.get_core()
    
    for y in range(0, x):
        c = core.std.Minimum(c, planes=planes)
        
    return c
    
    #From havsfunc, it's just a wrapper anyway.
def Resize(src, w, h, sx=None, sy=None, sw=None, sh=None, kernel=None, taps=None, a1=None, a2=None, invks=None, invkstaps=None, css=None, planes=None,
           center=None, cplace=None, cplaces=None, cplaced=None, interlaced=None, interlacedd=None, tff=None, tffd=None, flt=None, noring=False,
           bits=None, fulls=None, fulld=None, dmode=None, ampo=None, ampn=None, dyn=None, staticnoise=None, patsize=None):
    core = vs.get_core()
    
    if not isinstance(src, vs.VideoNode):
        raise TypeError('Resize: This is not a clip')
    
    if bits is None:
        bits = src.format.bits_per_sample
    
    sr_h = w / src.width
    sr_v = h / src.height
    sr_up = max(sr_h, sr_v)
    sr_dw = 1 / min(sr_h, sr_v)
    sr = max(sr_up, sr_dw)
    assert(sr >= 1)
    
    # Depending on the scale ratio, we may blend or totally disable the ringing cancellation
    thr = 2.5
    nrb = sr > thr
    nrf = sr < thr + 1 and noring
    if nrb:
        nrr = min(sr - thr, 1)
        nrv = round((1 - nrr) * 255)
        nrv = [nrv * 256 + nrv] * src.format.num_planes
    
    main = core.fmtc.resample(src, w, h, sx, sy, sw, sh, kernel=kernel, taps=taps, a1=a1, a2=a2, invks=invks, invkstaps=invkstaps, css=css, planes=planes, center=center,
                              cplace=cplace, cplaces=cplaces, cplaced=cplaced, interlaced=interlaced, interlacedd=interlacedd, tff=tff, tffd=tffd, flt=flt)
    
    if nrf:
        nrng = core.fmtc.resample(src, w, h, sx, sy, sw, sh, kernel='gauss', taps=taps, a1=100, invks=invks, invkstaps=invkstaps, css=css, planes=planes, center=center,
                                  cplace=cplace, cplaces=cplaces, cplaced=cplaced, interlaced=interlaced, interlacedd=interlacedd, tff=tff, tffd=tffd, flt=flt)
        
        # To do: use a simple frame blending instead of Merge
        last = core.rgvs.Repair(main, nrng, 1)
        if nrb:
            nrm = core.std.BlankClip(main, color=nrv)
            last = core.std.MaskedMerge(main, last, nrm)
    else:
        last = main
    
    return core.fmtc.bitdepth(last, bits=bits, fulls=fulls, fulld=fulld, dmode=dmode, ampo=ampo, ampn=ampn, dyn=dyn, staticnoise=staticnoise, patsize=patsize)
