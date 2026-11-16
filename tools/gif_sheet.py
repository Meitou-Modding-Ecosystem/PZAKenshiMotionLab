# PZA's Kenshi Motion Lab - contact sheet of a GIF (every k-th frame) for frame-by-frame review. Author: PZA. GPLv3.
#   python tools/gif_sheet.py <in.gif> <out.png> [frames=8] [cols=4]
import sys
from PIL import Image
im = Image.open(sys.argv[1]); n = im.n_frames
k = int(sys.argv[3]) if len(sys.argv) > 3 else 8; cols = int(sys.argv[4]) if len(sys.argv) > 4 else 4
idx = [round(i * (n - 1) / max(k - 1, 1)) for i in range(k)]
fr = []
for i in idx:
    im.seek(i); f = im.convert('RGB').copy(); fr.append(f)
w, h = fr[0].size; rows = (len(fr) + cols - 1) // cols
sheet = Image.new('RGB', (w * cols, h * rows), (255, 255, 255))
for j, f in enumerate(fr): sheet.paste(f, ((j % cols) * w, (j // cols) * h))
sheet.save(sys.argv[2]); print('frames', idx, 'of', n)
