```python
from ctypes import *
import numpy as np

libchips = CDLL("./libchips.so")
arr = np.zeros((3,3), dtype=np.float64)
libchips.moop(arr.ctypes.data_as(POINTER(c_double)))
```
