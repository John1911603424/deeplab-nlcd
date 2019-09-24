```python
import ctypes
import numpy as np

libchips = ctypes.CDLL("./libchips.so")
bands = np.array(range(1,9), dtype=np.int32)
bands_ptr = bands.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
libchips.start(1, b"../../mul.tif", b"../../mask.tif", 2, 1, 1, 32, 8, bands_ptr)
libchips.stop()
```
