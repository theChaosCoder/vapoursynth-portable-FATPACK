#TemporalSoften2 wrapper module

'''
put this python script into Python3.x/Lib/site_package as temporalsoften2.py

usage:


from temporalsoften2 import Temporalsoften
core.std.LoadPlugin('/path/to/scenechange.dll')
core.std.LoadPlugin('/path/to/temporalsoften2.dll')
clip = something
clip = TemporalSoften(core).soften(clip, luma_threshold=4, ...)
'''

import vapoursynth as vs

class TemporalSoften(object):
    def __init__(self, core):
        self.modframe = core.std.ModifyFrame
        self.resize = core.resize.Point
        self.detect = core.scd.Detect
        self.tsoften = core.focus2.TemporalSoften2

    def set_props(self, n, f):
        fout = f[0].copy()
        fout.props._SceneChangePrev = f[1].props._SceneChangePrev
        fout.props._SceneChangeNext = f[1].props._SceneChangeNext
        return fout

    def set_scenechange(self, clip, threshold=15, log=None):
        sc = clip
        cf = clip.format.color_family
        if cf == vs.RGB:
            sc = self.resize(format=vs.GRAY8)
        sc = self.detect(sc, threshold)
        if cf == vs.RGB:
            sc = self.modframe([clip, sc], self.set_props)
        return sc

    def soften(self, clip, radius=4, luma_threshold=4, chroma_threshold=8,
                scenechange=15, mode=None, log=None):
        if scenechange:
            clip = self.set_scenechange(clip, scenechange)

        return self.tsoften(clip, radius, luma_threshold, chroma_threshold,
                             scenechange)

