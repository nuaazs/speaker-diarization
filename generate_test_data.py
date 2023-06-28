import numpy as np
# a = random: value from 1 ~ 10,shape: (100,192)
a = np.random.randint(1, 10, (100, 192))
a = a.reshape(-1)
# float32
a = a.astype(np.float32)
a.tofile('test_data.bin')