import numpy as np
import matplotlib.pyplot as plt
import time

# Import .dat and zip raw and smoothed into DF
#data = np.fromfile('data-visualization/sample-data/amplifierdata.bin', dtype=float)

data = np.fromfile('data-visualization/sample-data/daq.bin', dtype=float)

print(data.shape)
window_end = 5000

plt.plot(data[0:window_end:4])
#plt.plot(data[1:window_end:4])
#plt.plot(data[2:window_end:4])
#plt.plot(data[3:window_end:4])

#plt.plot(data)

plt.show()



