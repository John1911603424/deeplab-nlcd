```python
import ctypes
import numpy as np

libchips = ctypes.CDLL("./libchips/src/libchips.so")
bands = np.array(range(1,8), dtype=np.int32)
bands_ptr = bands.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
libchips.start(
    16,  # Number of threads
    b"/tmp/mul.tif",  # Image data
    b"/tmp/mask.tif",  # Label data
    6,  # Make all rasters float32
    5,  # Make all labels int32
    1,  # Training mode
    224,
    len(bands),
    bands_ptr)
raster_buffer = np.zeros((len(bands), 224, 224), dtype=np.float32)
raster_buffer_ptr = raster_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
label_buffer = np.zeros((224, 224), dtype=np.int32)
label_buffer_ptr = label_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
libchips.get_next(raster_buffer_ptr, label_buffer_ptr)
libchips.stop()
```
