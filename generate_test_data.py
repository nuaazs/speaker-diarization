import numpy as np
a = np.random.rand(100, 192)
a = a.reshape(-1)
# float32
a = a.astype(np.float32)
a.tofile('test_data.bin')