#
# mod by thechaoscoder
# added argparse stuff
#
# original code by dubhater https://gist.github.com/dubhater/3a2c8a59841cae49ecae25cd47ff78d2 
#

# Usage:
# python3 seek-test.py file.mkv [start_frame end_frame]
#
# Change the source filter as needed.

# Note: this script is slow because making the source filter decode frames randomly is slow.


import vapoursynth as vs
import argparse
import sys, os, hashlib, random

choices_filter = ['ffms2', 'lsmas', 'd2v', 'avi', 'ffms2seek0']

parser = argparse.ArgumentParser(description='Reliability tester of VapourSynth Source Filters - seek test')
parser.add_argument('file', help='Video file to perform a seek test on')
parser.add_argument('start_frame', nargs='?', type=int, default='0', help='Start frame') 
parser.add_argument('end_frame', nargs='?', type=int, help='End frame')
parser.add_argument('-f', choices=choices_filter, dest='source_filter', help='Set source filter')
args = parser.parse_args()

c = vs.get_core(add_cache=False)

extension = os.path.splitext(args.file)[1]
if(extension == ".d2v"):
    args.source_filter = choices_filter[2] #d2v

if args.source_filter is None:
    print("Press 1 for FFMS2000")
    print("      2 for L-SMASH-Works")
    print("      3 for D2V Source")
    print("      4 for AVISource")
    print("      5 for FFMS2000(seekmode=0) [slow but more safe]")
    user_choice = int(input("Number: "))
    if(1 <= user_choice <= len(choices_filter)):
        args.source_filter = choices_filter[user_choice-1]
    else:
        sys.exit("wrong input")
	
if(args.source_filter == "ffms2"):
	clip = c.ffms2.Source(args.file)
if(args.source_filter == "ffms2seek0"):
	clip = c.ffms2.Source(args.file, seekmode=0)
if(args.source_filter == "lsmas"):
	clip = c.lsmas.LWLibavSource(args.file)
if(args.source_filter == "d2v"):
   clip = c.d2v.Source(args.file, rff=False)
if(args.source_filter == "avi"):
   clip = c.avisource.AVISource(args.file)

print(args.source_filter)


start_frame = 0
end_frame = clip.num_frames - 1
if not args.end_frame is None and args.end_frame > 0:
    start_frame = int(args.start_frame)
    end_frame = int(args.end_frame)

print("Clip has {} frames.".format(clip.num_frames))
if clip.num_frames < end_frame:
	end_frame = clip.num_frames - 1
	#print("[INFO] End Frame parameter exceeds clip length, correcting end_frame to clip length.")
	
clip = c.std.Trim(clip, start_frame, end_frame)

def hash_frame(frame):
    md5 = hashlib.md5()
    for plane in range(frame.format.num_planes):
        for line in frame.get_read_array(plane):
            md5.update(line)
    return md5.hexdigest()


reference = []
for i in range(clip.num_frames):
    reference.append(hash_frame(clip.get_frame(i)))

    if i % 100 == 0:
        print("Hashing: {}%".format(i * 100 // (end_frame)), end='\r')

print("\nClip hashed.\n")

test = list(range(clip.num_frames))
random.shuffle(test)

hasError = False
for i in test:
    try:
        result = hash_frame(clip.get_frame(i))
        if (result != reference[i]):
            hasError = True
            print("Requested frame {}, ".format(i), end='')
            try:
                print("got frame {}.".format(reference.index(result)))
            except ValueError:
                print("got new frame with hash {}.".format(result))
            print("    Previous requests:", end='')
            start = test.index(i) - 10
            if (start < 0):
                start = 0
            for j in range(start, test.index(i) + 1):
                print(" {}".format(test[j]), end='')
            print("")

        if test.index(i) % 100 == 0:
            print("Seeking: {}%".format(test.index(i) * 100 // (clip.num_frames - 1)), end='\r')

    except vs.Error as e:
        print("requesting frame {} broke d2vsource. total: {}".format(i, len(test)))
        raise e

if not hasError:
    print("Test complete. No seeking issues found :D")
    sys.exit(0)
else:
    print("")
    print("Test complete. Seeking issues found :-(")
    sys.exit(1)