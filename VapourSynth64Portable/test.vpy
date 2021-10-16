import sys, os
import functools
import vapoursynth as vs

import mvsfunc as mvf
import muvsfunc as muf

core = vs.core

time = 2.0

reta = muf.ColorBarsHD().std.DuplicateFrames([0] * 120)
retb = reta.std.Invert().std.FlipVertical().grain.Add(var = 200)

# Animation powered by vcmohans awesome vctrans plugin
# Source: http://www.avisynth.nl/users/vcmohan/vctrans/vctrans.html

ret = core.trans.Accord(retb, clipb = reta, overlap = time, twin = 1, dir = 0, open = 0)
vid = core.text.Text(ret, "Accord", 2)
ret = core.trans.Bubbles(reta, clipb = retb, overlap = time, static = 0)
ret = ret + core.trans.Bubbles(retb, clipb = reta, overlap = time, static = 1)
ret = core.text.Text(ret, "Bubbles", 2)
vid = vid + ret
ret = core.trans.Central(reta, clipb = retb, overlap = time, nturns = -5, emerge = 0)
ret = ret + core.trans.Central(retb, clipb = reta, overlap = time, nturns = 5, emerge = 1)
ret = ret + core.trans.Central(reta, clipb = retb, overlap = time, nturns = 0, emerge = 0, resize = 0)
ret = core.text.Text(ret, "Central", 2)
vid = vid + ret
ret = core.trans.Crumple(retb, clipb = reta, overlap = time, crumple = 1, emerge = 1)
ret = ret + core.trans.Crumple(reta, clipb = retb, overlap = time, crumple = 1, emerge = 0)
ret = ret + core.trans.Crumple(retb, clipb = reta, overlap = time, crumple = 0, emerge = 1)
ret = ret + core.trans.Crumple(reta, clipb = retb, overlap = time, crumple = 0, emerge = 0)
ret = core.text.Text(ret, "Crumple", 2)
vid = vid + ret
ret = core.trans.Disco(retb, clipb = reta, overlap = time, nturns = -4, emerge = 0)
ret = ret + core.trans.Disco(reta, clipb = retb, overlap = time, nturns = 0, emerge = 1)
ret = ret + core.trans.Disco(retb, clipb = reta, overlap = time, nturns = 5, emerge = 1)
ret = core.text.Text(ret, "Disco", 2)
vid = vid + ret
ret = core.trans.Door(reta, clipb = retb, overlap = time, vert = 0, open = 1)
ret = core.text.Text(ret, "Door", 2)
vid = vid + ret
ret = core.trans.FlipPage(retb, clipb = reta, overlap = time, left = 1)
ret = ret + core.trans.FlipPage(reta, clipb = retb, overlap = time, left = 0)
ret = core.text.Text(ret, "FlipPage", 2)
vid = vid + ret
ret = core.trans.Funnel(retb, clipb = reta, overlap = time, dir = 1)
ret = ret + core.trans.Funnel(reta, clipb = retb, overlap = time, dir = 2)
ret = ret + core.trans.Funnel(retb, clipb = reta, overlap = time, dir = 3)
ret = ret + core.trans.Funnel(reta, clipb = retb, overlap = time, dir = 4)
ret = core.text.Text(ret, "Funnel", 2)
vid = vid + ret 
ret = core.trans.Paint(retb, clipb = reta, overlap = time, style = 1)
ret = ret + core.trans.Paint(reta, clipb = retb, overlap = time, style = 2)
ret = ret + core.trans.Paint(retb, clipb = reta, overlap = time, style = 3)
ret = ret + core.trans.Paint(reta, clipb = retb, overlap = time, style = 4)
ret = core.text.Text(ret, "Paint", 2)
vid = vid + ret

ret = core.trans.Push(retb, clipb = reta, overlap = time, dir = 1)
ret = ret + core.trans.Push(reta, clipb = retb, overlap = time, dir = 2)
ret = ret + core.trans.Push(retb, clipb = reta, overlap = time, dir = 3)
ret = ret + core.trans.Push(reta, clipb = retb, overlap = time, dir = 4)
ret = core.text.Text(ret, "Push", 2)
vid = vid + ret
ret = core.trans.Ripple(retb, clipb = reta, overlap = time,origin = 0, amp = 12, wlength = 14, merge = 1) 
ret = ret + core.trans.Ripple(reta, clipb = retb, overlap = time,origin = 1, amp = 2, wlength = 8, merge = 1)
ret = ret + core.trans.Ripple(retb, clipb = reta, overlap = time,origin = 2, amp = 4, wlength = 12, merge = 1)
ret = ret + core.trans.Ripple(reta, clipb = retb, overlap = time,origin = 3, amp = 8, wlength = 16, merge = 1)
ret = ret + core.trans.Ripple(retb, clipb = reta, overlap = time,origin = 4, amp = 10, wlength = 20)
ret = ret + core.trans.Ripple(reta, clipb = retb, overlap = time,origin = 5, amp = 12, wlength = 30)
ret = ret + core.trans.Ripple(retb, clipb = reta, overlap = time,origin = 6, amp = 3, wlength = 40)
ret = ret + core.trans.Ripple(reta, clipb = retb, overlap = time,origin = 7, amp = 1, wlength = 50)
ret = ret + core.trans.Ripple(retb, clipb = reta, overlap = time,origin = 8, amp = 5, wlength = 20, merge = 1)
ret = core.text.Text(ret, "Ripple", 2)
vid = vid + ret
ret = core.trans.Roll(reta, clipb = retb, overlap = time, dir = 1)
ret = ret + core.trans.Roll(retb, clipb = reta, overlap = time, dir = 1, rollin = 0)
ret = ret + core.trans.Roll(reta, clipb = retb, overlap = time, dir = 3)
ret = ret + core.trans.Roll(retb, clipb = reta, overlap = time, dir = 3, rollin = 0)
ret = ret + core.trans.Roll(reta, clipb = retb, overlap = time, dir = 5)
ret = ret + core.trans.Roll(retb, clipb = reta, overlap = time, dir = 5, rollin = 0)
ret = ret + core.trans.Roll(reta, clipb = retb, overlap = time, dir = 7)
ret = ret + core.trans.Roll(retb, clipb = reta, overlap = time, dir = 7, rollin = 0)

ret = ret + core.trans.Roll(reta, clipb = retb, overlap = time, dir = 2)
ret = ret + core.trans.Roll(retb, clipb = reta, overlap = time, dir = 2, rollin = 0)
ret = ret + core.trans.Roll(reta, clipb = retb, overlap = time, dir = 4)
ret = ret + core.trans.Roll(retb, clipb = reta, overlap = time, dir = 4, rollin = 0)
ret = ret + core.trans.Roll(reta, clipb = retb, overlap = time, dir = 6)
ret = ret + core.trans.Roll(retb, clipb = reta, overlap = time, dir = 6, rollin = 0)
ret = ret + core.trans.Roll(reta, clipb = retb, overlap = time, dir = 8)
ret = ret + core.trans.Roll(retb, clipb = reta, overlap = time, dir = 8, rollin = 0)
ret = core.text.Text(ret, "Roll", 2)
vid = vid + ret
ret = core.trans.Scratch(reta, clipb = retb, overlap = time, style = 1)
ret = ret + core.trans.Scratch(retb, clipb = reta, overlap = time, style = 2)
ret = ret + core.trans.Scratch(reta, clipb = retb, overlap = time, style = 3)
ret = ret + core.trans.Scratch(retb, clipb = reta, overlap = time, style = 4)
ret = ret + core.trans.Scratch(reta, clipb = retb, overlap = time, style = 5)
ret = ret + core.trans.Scratch(retb, clipb = reta, overlap = time, style = 6)
ret = ret + core.trans.Scratch(reta, clipb = retb, overlap = time, style = 7)
ret = core.text.Text(ret, "Scratch", 2)
vid = vid + ret
ret = core.trans.Shuffle(retb, clipb = reta, overlap = time, dir = 1)
ret = ret + core.trans.Shuffle(reta, clipb = retb, overlap = time, dir = 2)
ret = ret + core.trans.Shuffle(retb, clipb = reta, overlap = time, dir = 3)
ret = ret + core.trans.Shuffle(reta, clipb = retb, overlap = time, dir = 4)
ret = core.text.Text(ret, "Shuffle", 2)
vid = vid + ret

ret = core.trans.Slide(retb,clipb = reta, overlap = time, dir = 0, slidein = 0)
ret = ret + core.trans.Slide(reta, clipb = retb, overlap = time, dir = 1, slidein = 0)
ret = ret + core.trans.Slide(retb, clipb = reta, overlap = time, dir = 2, slidein = 0)
ret = ret + core.trans.Slide(reta, clipb = retb, overlap = time, dir = 3, slidein = 0)
ret = ret + core.trans.Slide(retb, clipb = reta, overlap = time, dir = 4, slidein = 0)
ret = ret + core.trans.Slide(reta, clipb = retb, overlap = time, dir = 5, slidein = 0)
ret = ret + core.trans.Slide(retb, clipb = reta, overlap = time, dir = 6, slidein = 0)
ret = ret + core.trans.Slide(reta, clipb = retb, overlap = time, dir = 7, slidein = 0)
ret = ret + core.trans.Slide(retb, clipb = reta, overlap = time, dir = 8, slidein = 0)
ret = ret + core.trans.Slide(reta, clipb = retb, overlap = time, dir = 1, slidein = 1)
ret = ret + core.trans.Slide(retb, clipb = reta, overlap = time, dir = 2, slidein = 1)
ret = ret + core.trans.Slide(reta, clipb = retb, overlap = time, dir = 3, slidein = 1)
ret = ret + core.trans.Slide(retb, clipb = reta, overlap = time, dir = 4, slidein = 1)
ret = ret + core.trans.Slide(reta, clipb = retb, overlap = time, dir = 5, slidein = 1)
ret = ret + core.trans.Slide(retb, clipb = reta, overlap = time, dir = 6, slidein = 1)
ret = ret + core.trans.Slide(reta, clipb = retb, overlap = time, dir = 7, slidein = 1)
ret = ret + core.trans.Slide(retb, clipb = reta, overlap = time, dir = 8, slidein = 1)
ret = ret + core.trans.Slide(reta, clipb = retb, overlap = time, dir = 0, slidein = 1)
ret = core.text.Text(ret, "Slide", 2)
vid = vid + ret
ret = core.trans.Sprite(retb, clipb = reta, overlap = time, dir = 1)
ret = ret + core.trans.Sprite(reta, clipb = retb, overlap = time, dir = 2)
ret = ret + core.trans.Sprite(retb, clipb = reta, overlap = time, dir = 3)
ret = ret + core.trans.Sprite(reta, clipb = retb, overlap = time, dir = 4)
ret = core.text.Text(ret, "Sprite", 2)
vid = vid + ret
ret = core.trans.Swing(retb, clipb = reta, overlap = time, dir = 1,open = 0, ndoors = 1, corner = 1)
ret = ret + core.trans.Swing(reta, clipb = retb, overlap = time, dir = 0,open = 0, ndoors = 1, corner = 2)
ret = ret + core.trans.Swing(retb, clipb = reta, overlap = time, dir = 0,open = 1, ndoors = 1, corner = 3)
ret = ret + core.trans.Swing(reta, clipb = retb, overlap = time, dir = 1,open = 1, ndoors = 1, corner = 4)
ret = ret + core.trans.Swing(retb, clipb = reta, overlap = time, dir = 0,open = 1, ndoors = 2, corner = 1)
ret = ret + core.trans.Swing(reta, clipb = retb, overlap = time, dir = 0,open = 1, ndoors = 2, corner = 2)
ret = ret + core.trans.Swing(retb, clipb = reta, overlap = time, dir = 0,open = 1, ndoors = 2, corner = 3)
ret = ret + core.trans.Swing(reta, clipb = retb, overlap = time, dir = 0,open = 1, ndoors = 2, corner = 4)
ret = ret + core.trans.Swing(retb, clipb = reta, overlap = time, dir = 1,open = 1, ndoors = 4, corner = 1)
ret = ret + core.trans.Swing(reta, clipb = retb, overlap = time, dir = 1,open = 1, ndoors = 4, corner = 2)
ret = ret + core.trans.Swing(retb, clipb = reta, overlap = time, dir = 1,open = 1, ndoors = 4, corner = 3)
ret = ret + core.trans.Swing(reta, clipb = retb, overlap = time, dir = 1,open = 1, ndoors = 4, corner = 4)
ret = core.text.Text(retb, "Swing", 2)
vid = vid + ret
ret = core.trans.Swirl(retb, clipb = reta, overlap = time, dir = 1, qr = 5)
ret = ret + core.trans.Swirl(reta, clipb = retb, overlap = time, dir = -1, qr = 2)
ret = core.text.Text(ret, "Swirl", 2)
vid = vid + ret

ret = core.trans.VenitianBlinds(retb, clipb = reta, overlap = time, style = 1)
ret = ret + core.trans.VenitianBlinds(reta, clipb = retb, overlap = time, style = 2)
ret = ret + core.trans.VenitianBlinds(retb, clipb = reta, overlap = time, style = 3)
ret = ret + core.trans.VenitianBlinds(reta, clipb = retb, overlap = time, style = 4)
ret = ret + core.trans.VenitianBlinds(retb, clipb = reta, overlap = time, style = 1, open = 0)
ret = ret + core.trans.VenitianBlinds(reta, clipb = retb, overlap = time, style = 2, open = 0)
ret = ret + core.trans.VenitianBlinds(retb, clipb = reta, overlap = time, style = 3, open = 0)
ret = ret + core.trans.VenitianBlinds(reta, clipb = retb, overlap = time, style = 4, open = 0)
ret = core.text.Text(ret, "VenitianBlinds", 2)
vid = vid + ret
ret = core.trans.Weave(reta, clipb = retb, overlap = time, style = 1)
ret = ret + core.trans.Weave(retb, clipb = reta, overlap = time, style = 2)
ret = ret + core.trans.Weave(reta, clipb = retb, overlap = time, style = 3)
ret = ret + core.trans.Weave(retb, clipb = reta, overlap = time, style = 4)
ret = core.text.Text(ret, "Weave", 2)
vid = vid + ret
ret = core.trans.Wipe(reta, clipb = retb, overlap = time, dir = 1)
ret = ret + core.trans.Wipe(retb, clipb = reta, overlap = time, dir = 2)
ret = ret + core.trans.Wipe(reta, clipb = retb, overlap = time, dir = 3)
ret = ret + core.trans.Wipe(retb, clipb = reta, overlap = time, dir = 4)
ret = ret + core.trans.Wipe(reta, clipb = retb, overlap = time, dir = 5)
ret = ret + core.trans.Wipe(retb, clipb = reta, overlap = time, dir = 6)
ret = ret + core.trans.Wipe(reta, clipb = retb, overlap = time, dir = 7)
ret = ret + core.trans.Wipe(retb, clipb = reta, overlap = time, dir = 8)
ret =  core.text.Text(ret, "wipe", 2)
vid = vid + ret
#format conversions to suit the player
ret = core.resize.Bicubic(vid,format=vs.YUV420P8)
#ret = core.resize.Bicubic(ret,height = 480, width = 720,format=vs.YUV444P8)
#ret = core.fmtc.bitdepth(vid,flt = 0)
# set the clip to be output
ret.set_output()