```python
import ctypes
import numpy as np

libchips = ctypes.CDLL("./libchips.so")
bands = np.array(range(1,9), dtype=np.int32)
bands_ptr = bands.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
libchips.start(32, b"../../mul.tif", b"../../mask.tif", 2, 1, 1, 224, 8, bands_ptr)
raster_buffer = np.zeros((224, 224, 8), dtype=np.uint16)
raster_buffer_ptr = raster_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
label_buffer = np.zeros((224, 224), dtype=np.uint8)
label_buffer_ptr = label_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
libchips.get_next(raster_buffer_ptr, label_buffer_ptr)
libchips.stop()
```
