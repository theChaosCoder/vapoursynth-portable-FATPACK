"""
colorfade v1.0 by Tormaid

A simple Vapoursynth script for doing bi-directional, mathematically-correct linear blending of a single frame over a specified range with a target color. 
Useful for re-creating static fades. 
Internal dissolve function is based in part on kageru's Crossfade.

Requires YUV input.

colorfade(clip, start, end, direction='forward', YUV=[])

    start, end [int]: Determine the beginning and end of the fade range.

    direction [str, default='forward']:
        forward = Fades to the specified color
        backward = Fades from the specified color

    YUV [lst]: The YUV color code of the color you wish to fade to/from. Scale is based on the bit-depth of the input clip.
	eg. [76, 84, 255] (Red, 8-bit Scale)
	Hint: Use the color picker in the VapourSynth Editor.

"""

import vapoursynth as vs
from functools import partial

core = vs.core

def colorfade(clip, start=None, end=None, direction='forward', YUV=None):

	fade_length = clip[start:(end + 1)].num_frames
	
	def dissolve(clip_a, clip_b):
		def merge_var(n, clip_a, clip_b): #clip_a and clip_b must be the same length!
			return core.std.Merge(clip_a, clip_b, weight=n / (fade_length - 1))

		return core.std.FrameEval(clip_a, partial(merge_var, clip_a=clip_a, clip_b=clip_b))
		
	color = core.std.BlankClip(format=clip.format, width=clip.width, height=clip.height, length=fade_length, fpsnum=clip.fps_num, fpsden=clip.fps_den, color=YUV)

	if direction == 'forward':
		frozen = clip[start] * fade_length
		fade = dissolve(frozen, color)
	
	if direction == 'backward':
		frozen = clip[end] * fade_length
		fade = dissolve(color, frozen)
	
	if start != 0:
		top = clip[:start]
	
	if end != clip.num_frames - 1:
		tail = clip[(end + 1):]
		
	if start == 0:
		return fade + tail
	
	elif end == clip.num_frames - 1:
		return top + fade
		
	else:
		return top + fade + tail
