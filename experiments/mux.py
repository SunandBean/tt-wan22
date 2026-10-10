"""frames .npy [F, H, W, 3] uint8 -> mp4 at 16 fps (host, PyAV from ComfyUI's venv)."""
import sys
import av
import numpy as np
arr = np.load(sys.argv[1])
with av.open(sys.argv[2], "w") as c:
    s = c.add_stream("libx264", rate=16)
    s.width, s.height, s.pix_fmt = arr.shape[2], arr.shape[1], "yuv420p"
    s.options = {"crf": "14"}
    for f in arr:
        for p in s.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
            c.mux(p)
    for p in s.encode():
        c.mux(p)
print(sys.argv[2], arr.shape)
