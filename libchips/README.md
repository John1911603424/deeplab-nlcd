```python
import ctypes
import numpy as np

libchips = ctypes.CDLL("./libchips.so")
libchips.init()

bands = np.array(range(1,12+1), dtype=np.int32)
bands_ptr = bands.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
mus = np.ndarray(12, dtype=np.float64)
mus_ptr = mus.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
sigmas = np.ndarray(12, dtype=np.float64)
sigmas_ptr = sigmas.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

libchips.start(
    16,  # Number of threads
    256, # Number of threads
    b"../../mul.tif",  # Image data
    b"../../mask.tif",  # Label data
    6,  # Make all rasters float32
    5,  # Make all labels int32
    mus_ptr, # Pointer to array of means
    sigmas_ptr, # Pointer to array of standard deviations
    10000, # Typical radius of a component
    1,  # Training mode
    256,
    len(bands),
    bands_ptr)

raster_buffer = np.zeros((len(bands), 256, 256), dtype=np.float32)
raster_buffer_ptr = raster_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
label_buffer = np.zeros((256, 256), dtype=np.int32)
label_buffer_ptr = label_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
libchips.get_next(raster_buffer_ptr, label_buffer_ptr)

libchips.stop()
libchips.deinit()
```
