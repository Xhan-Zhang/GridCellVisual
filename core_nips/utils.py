import numpy as np
import os, errno
import scipy
import json
#from visualize import compute_ratemaps, plot_ratemaps
from scipy.linalg import block_diag

def generate_run_ID(arg):
    ''' 
    Create a unique run ID from the most relevant
    parameters. Remaining parameters can be found in 
    params.npy file. 
    '''
    # params = [
    #     'steps', str(options.sequence_length),
    #     'batch', str(options.batch_size),
    #     options.RNN_type,
    #     str(options.Ng),
    #     options.activation,
    #     'rf', str(options.place_cell_rf),
    #     'DoG', str(options.DoG),
    #     'periodic', str(options.periodic),
    #     'lr', str(options.learning_rate),
    #     'weight_decay', str(options.weight_decay),
    # ]

    params = [
        str(arg.Ng),
        str(arg.Np),
        'bs', str(arg.batch_size),
        #'we',str(arg.weight_entropy),
        #'wd',str(arg.weight_decay),
        #str(arg.Np),
        'sl',str(arg.sequence_length),
        'model',str(arg.model_idx),

        ]
    separator = '_'
    run_ID = separator.join(params)
    #run_ID = run_ID.replace('.', '')#

    return run_ID


def get_2d_sort(x1,x2):
    """
    Reshapes x1 and x2 into square arrays, and then sorts
    them such that x1 increases downward and x2 increases
    rightward. Returns the order.
    """
    n = int(np.sqrt(len(x1)))
    print('np.sqrt(len(x1))',np.sqrt(len(x1)))
    print('n',n)
    total_order = x1[:n*n].argsort()
    print('total_order',total_order.shape)
    total_order = total_order.reshape(n,n)
    for i in range(n):
        row_order = x2[total_order.ravel()].reshape(n,n)[i].argsort()
        total_order[i] = total_order[i,row_order]
    total_order = total_order.ravel()
    return total_order


def dft(N,real=False,scale='sqrtn'):
    if not real:
        return scipy.linalg.dft(N,scale)
    else:
        cosines = np.cos(2*np.pi*np.arange(N//2+1)[None,:]/N*np.arange(N)[:,None])
        sines = np.sin(2*np.pi*np.arange(1,(N-1)//2+1)[None,:]/N*np.arange(N)[:,None])
        if N%2==0:
            cosines[:,-1] /= np.sqrt(2)
        F = np.concatenate((cosines,sines[:,::-1]),1)
        F[:,0] /= np.sqrt(N)
        F[:,1:] /= np.sqrt(N/2)
        return F


def skaggs_power(Jsort):
    F = dft(int(np.sqrt(N)), real=True)
    F2d = F[:,None,:,None]*F[None,:,None,:]

    F2d_unroll = np.reshape(F2d, (N, N))

    F2d_inv = F2d_unroll.conj().T
    Jtilde = F2d_inv.dot(Jsort).dot(F2d_unroll)

    return (Jtilde[1,1]**2 + Jtilde[-1,-1]**2) / (Jtilde**2).sum()


def skaggs_power_2(Jsort):
    J_square = np.reshape(Jsort, (n,n,n,n))
    Jmean = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            Jmean += np.roll(np.roll(J_square[i,j], -i, axis=0), -j, axis=1)

#     Jmean[0,0] = np.max(Jmean[1:,1:])
    Jmean = np.roll(np.roll(Jmean, n//2, axis=0), n//2, axis=1)
    Jtilde = np.real(np.fft.fft2(Jmean))
    
    Jtilde[0,0] = 0
    sk_power = Jtilde[1,1]**2 + Jtilde[0,1]**2 + Jtilde[1,0]**2
    sk_power += Jtilde[-1,-1]**2 + Jtilde[0,-1]**2 + Jtilde[-1,0]**2
    sk_power /= (Jtilde**2).sum()
    
    return sk_power


def calc_err():
    inputs, _, pos = next(gen)
    pred = model(inputs)
    pred_pos = place_cells.get_nearest_cell_pos(pred)
    return tf.reduce_mean(tf.sqrt(tf.reduce_sum((pos - pred_pos)**2, axis=-1)))




def compute_variance(res, n_avg):
    
    activations, rate_map, g, pos = compute_ratemaps(model, data_manager, hp, res=res, n_avg=n_avg)

    counts = np.zeros([res,res])
    variance = np.zeros([res,res])

    x_all = (pos[:,0] + hp['box_width']/2) / hp['box_width'] * res
    y_all = (pos[:,1] + hp['box_height']/2) / hp['box_height'] * res
    for i in tqdm(range(len(g))):
        x = int(x_all[i])
        y = int(y_all[i])
        if x >=0 and x < res and y >=0 and y < res:
            counts[x, y] += 1
            variance[x, y] += np.linalg.norm(g[i] - activations[:, x, y]) / np.linalg.norm(g[i]) / np.linalg.norm(activations[:,x,y])

    for x in range(res):
        for y in range(res):
            if counts[x, y] > 0:
                variance[x, y] /= counts[x, y]
                
    return variance


def load_trained_weights(model,model_dir):
    ''' Load weights stored as a .npy file (for github)'''

    # Train for a single step to initialize weights
    #trainer.train(n_epochs=1, n_steps=1, save=False)

    # Load weights from npy array

    model.load(model_dir)

    # for name, param in model.named_parameters():
    #     print('%%%trained:',name, param)

    print('Loaded trained weights.')



def mkdir_p(path):
    """
    Portable mkdir -p

    """
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def elapsed_time(totaltime):


    hrs  = int(totaltime//3600)
    mins = int(totaltime%3600)//60
    secs = int(totaltime%60)

    return '{}h {}m {}s elapsed'.format(hrs, mins, secs)



def save_log(log):
    """Save the log file of model."""
    model_dir = log['model_dir']
    fname = os.path.join(model_dir, 'log.json')
    with open(fname, 'w') as f:
        json.dump(log, f)







def rotation(n, dims, angle):
    """
    Parameters
    ------------
    n : int
        dimension of the space
    dims : 2-tuple of ints
        the vector indices which form the plane to perform the rotation in
    angle : array_like of shape (M...,)
        broadcasting angle to rotate by

    Returns
    --------
    m : array_like of shape (M..., n, n)
        (stack of) rotation matrix
    """
    i= dims[0]
    j =dims[1]
    assert i != j
    c = np.cos(angle)
    s = np.sin(angle)
    arr = np.eye(n, dtype=c.dtype)
    print('arr',arr.shape)
    arr[..., i, i] = c
    arr[..., i, j] = -s
    arr[..., j, i] = s
    arr[..., j, j] = c
    return arr

def rotation_matrix(angle=np.pi/3):
    a1 = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    mat1 = a1
    matrix = [a1 for i in range(256)]

    for i in range(256-1):
        mat_temp = mat1
        mat1 = block_diag(mat_temp,a1)

    mat1 = mat1.copy()
    mat1[mat1==0]=0
    print('mat1',mat1.shape,mat1)

    return mat1






def Bc(i, boundary):
    """(int, int) -> int

    Checks boundary conditions on index
    """
    if i > boundary - 1:
        return i - boundary
    elif i < 0:
        return boundary + i
    else:
        return i

def circulant_matrix(N=0):
    diffMat = np.zeros([N, N])
    for i in np.arange(0, N, 1):
        diffMat[i, [Bc(i+1, N), Bc(i+2, N), Bc(i+2+(N-5)+1, N), Bc(i+2+(N-5)+2, N)]] = [2.0/3, -1.0/12, 1.0/12, -2.0/3]

    return diffMat