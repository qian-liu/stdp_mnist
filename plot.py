
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys



input_size =28
output_num = 8
#im = plt.imshow(np.zeros((input_size,input_size)))
start = 100
stop = 350

output_num = int(sys.argv[1])
par_start = int(sys.argv[2])


for run_i in range(200):
    stdp_weight = np.load('weight%d/weight_%d.npy'%(par_start, run_i))
    print stdp_weight
    to_plot = np.reshape(stdp_weight[:,output_num], (input_size, input_size))
    if run_i == 0:
        plt.figure(figsize=(3,3))
        im = plt.imshow(to_plot,interpolation='none',cmap = cm.Greys_r)#,cmap = cm.Greys_r)#,interpolation='none')
        fig = plt.gcf()
        im.set_clim(0, 0.8)
        plt.colorbar(im, fraction=0.046, pad=0.04)
    else:
        im.set_data(to_plot)
        
    #im = plt.imshow(to_plot)
    plt.title(run_i)
    plt.draw()
    plt.pause(0.1)
#plt.close()


# In[ ]:



