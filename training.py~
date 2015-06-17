import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.io as sio
import pyNN.nest as p
import sys  

# In[18]:

def load_data(matfile):
    dataset = sio.loadmat(matfile)
    # Rescale from int to float [0 1.0], and then between [0 0.2] of max firing
    train_x = dataset['train_x']
    train_y = dataset['train_y']
    #test_x = dataset['test_x'] #/ 255.0 * 0.2
    #test_y = dataset['test_y'] #/ 255.0 * 0.2
    return train_x, train_y#, test_x, test_y


# In[19]:

def plot_spikes(spikes, title):
    if spikes is not None:
        plt.figure(figsize=(15, 5))
        plt.plot([i[1] for i in spikes], [i[0] for i in spikes], ".")
        plt.xlabel('Time/ms')
        plt.ylabel('spikes')
        plt.title(title)

    else:
        print "No spikes received"


# In[ ]:
par_start = int(sys.argv[1])
input_size = 28#14
MIN_rate = 1.0
teaching_rate = 50.0
num_train = 30
dur_train = 300 #ms
silence = 0 #ms
num_epo = 1
num_output = 10
#num_par = 2000
num_per_par = 100
cell_params_lif = {'cm': 0.25,
                   'i_offset': 0.0,
                   'tau_m': 20.0,
                   'tau_refrac': 2.0,
                   'tau_syn_E': 1.0,
                   'tau_syn_I': 1.0,
                   'v_reset': -70.0,
                   'v_rest': -65.0,
                   'v_thresh': -50.0
                   }
train_x, train_y = load_data('mnist_uint8.mat')
freq = np.linspace(MIN_rate, 50.0, 256)
train_x = freq[train_x]

for i in range(len(train_x)):
    train_x[i] = train_x[i]/sum(train_x[i])*2000.0

ImagePoission = list()
TeachingPoission = list()

# In[ ]:

for run_i in range(par_start, par_start+num_per_par):
    p.setup(timestep=1.0, min_delay=1.0, max_delay=3.0)
    pop_input = p.Population(input_size*input_size, p.IF_curr_exp, cell_params_lif)
    ImagePoission = []
    TeachingPoission = []
    sys.stderr.write("numer=%d\n"%run_i)

    for epo in range(num_epo):
        for i in range(num_train):
            ind = i + run_i*num_train
            pop = p.Population(input_size*input_size,
                                              p.SpikeSourcePoisson,
                                              {'rate' : MIN_rate,#test_x[i],
                                               'start' : (epo*num_epo+i)*(dur_train+silence),
                                               'duration' : dur_train})
            for j in range(input_size*input_size):
                temp_popv = p.PopulationView(pop, np.array([j]))
                temp_popv.set('rate', train_x[ind][j])
            ImagePoission.append(pop)
            '''
            #test poission
            pop = p.Population(input_size*input_size,
                                              p.SpikeSourcePoisson,
                                              {'rate' : MIN_rate,#test_x[i],
                                               'start' : ((epo+num_train)*num_epo+i)*(dur_train+silence),
                                               'duration' : dur_train})
            for j in range(input_size*input_size):
                temp_popv = p.PopulationView(pop, np.array([j]))
                temp_popv.set('rate', temp[j])
            ImagePoission.append(pop)
            '''

            #output poission
            pop = p.Population(num_output,p.SpikeSourcePoisson,
                                              {'rate' : MIN_rate,#test_x[i],
                                               'start' : (epo*num_epo+i)*(dur_train+silence),
                                               'duration' : dur_train})
            temp_popv = p.PopulationView(pop, np.nonzero(train_y[ind])[0])
            temp_popv.set('rate', teaching_rate)
            TeachingPoission.append(pop)
    #print ImagePoission[10].get('start')
    ee_connector = p.OneToOneConnector(weights=3.0)
    for i in range(num_train*num_epo*1):
        p.Projection(ImagePoission[i], pop_input, ee_connector, target='excitatory')
    pop_output = p.Population(num_output, p.IF_curr_exp, cell_params_lif)

    weight_max = 1.3
    stdp_model = p.STDPMechanism(
        timing_dependence=p.SpikePairRule(tau_plus=10., tau_minus=10.0),
        weight_dependence=p.MultiplicativeWeightDependence(w_min=0.0, w_max=weight_max,
                                                       A_plus=0.01, A_minus=0.01)
        #AdditiveWeightDependence
    )
    '''
    weight_distr = p.RandomDistribution(distribution='normal',parameters=[1,0.1])
    '''
    if run_i == 0:
        stdp_weight = 0.0
    else:
        stdp_weight = np.load('saved/weight_%d.npy'%(run_i-1))
    proj_stdp = p.Projection(
        #pop_input, pop_output, p.AllToAllConnector(weights = weight_distr),
        pop_input, pop_output, p.AllToAllConnector(weights = stdp_weight),
        synapse_dynamics=p.SynapseDynamics(slow=stdp_model)
    )

    for i in range(num_output):
        conn_list = list()
        for j in range(num_output):
            if i != j:
                conn_list.append((i, j, -18.1, 1.0))
        p.Projection(pop_output, pop_output, p.FromListConnector(conn_list), target='inhibitory')

    for i in range(num_train*num_epo):
        p.Projection(TeachingPoission[i], pop_output, ee_connector, target='excitatory')
    #pop_input.record()
    pop_output.record()
    pre = proj_stdp.getWeights(format='array',gather=False)
    #print pre
    p.run(1.0*num_epo*num_train*(dur_train+silence))
    post = proj_stdp.getWeights(format='array',gather=False)
    np.save('saved/weight_%d'%run_i,post)
    '''
    plt.figure(figsize=(18,6))
    for i in range(num_output):
        plt.subplot(2, 5, i+1)
        plt.title(i)

        to_plot = np.reshape(post[:,i], (input_size, input_size))
        #to_plot = to_plot - np.reshape(pre[:, i], (input_size, input_size))
        img = plt.imshow(to_plot)
        #img.set_clim(0, 0.5)#weight_max)
        plt.colorbar(img, fraction=0.046, pad=0.04)
    #print post[:,0]
    spikes = pop_output.getSpikes(compatible_output=True)
    #spikes = pop_input.getSpikes(compatible_output=True)
    
    plot_spikes(spikes, "output")
    '''
    #pop_output.printSpikes('saved/outspikes_%d.txt'%run_i, gather=False, compatible_output=True)
    
    p.end()
    


# In[ ]:

plt.figure(figsize=(18,6))
for i in range(num_output):
    plt.subplot(2, 5, i+1)
    plt.title(i)
    to_plot = np.reshape(post[:,i], (input_size, input_size))
    img = plt.imshow(to_plot)#cmap = cm.Greys_r)
    plt.colorbar(img, fraction=0.046, pad=0.04)
#spikes = pop_output.getSpikes(compatible_output=True)
#plot_spikes(spikes, "output")


# In[ ]:



