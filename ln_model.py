# %%
import scipy.io as sio
import numpy as np
import matplotlib.pylab as plt
from scipy.linalg import block_diag
from scipy.optimize import minimize
data = sio.loadmat('data_for_cell77')


def pos_map(pos, nbins, boxSize):
    # discretize the position into different bins
    # pos: nx2 matrix
    bins = np.arange(boxSize/nbins, boxSize, boxSize/nbins)
    xcoord = np.digitize(pos[:, 0], bins)
    ycoord = np.digitize(pos[:, 1], bins)
    # make x and y start form the top left corner
    coord = (xcoord)*nbins+(nbins-ycoord-1)

    output = np.zeros((pos.shape[0], nbins**2))
    output[np.arange(pos.shape[0]), coord] = 1

    return (output, bins)


def hd_map(posx, posx2, posy, posy2, nbins):
    direction = np.arctan2(posy2-posy, posx2-posx)+np.pi/2
    direction[direction < 0] = direction[direction < 0]+2*np.pi
    direction = direction.ravel()  # change to 1d array

    hd_grid = np.zeros((posx.shape[0], nbins))
    dirVec = np.arange(2*np.pi/nbins, 2*np.pi, 2*np.pi/nbins)
    idx = np.digitize(direction, dirVec)
    hd_grid[np.arange(posx.shape[0]), idx] = 1

    return (hd_grid, dirVec, direction)


def speed_map(posx, posy, nbins, sampleRate=50, maxSpeed=50):
    velx = np.diff(np.insert(posx, 0, posx[0]))
    vely = np.diff(np.insert(posy, 0, posy[0]))
    speed = np.sqrt(velx**2+vely**2)*sampleRate
    # send everything over 50 cm/s to 50 cm/s
    speed[speed > maxSpeed] = maxSpeed

    speedVec = np.arange(maxSpeed/nbins, maxSpeed, maxSpeed/nbins)
    speed_grid = np.zeros((posx.shape[0], nbins))

    idx = np.digitize(speed, speedVec)
    speed_grid[np.arange(speed.shape[0]), idx.ravel()] = 1

    return (speed_grid, speedVec, speed, idx)


# %%
posc = np.hstack([data['posx_c'], data['posy_c']])
(posgrid, posVec) = pos_map(posc, 20, 100)

(hdgrid, dirVec, direction) = hd_map(
    data['posx'], data['posx2'], data['posy'], data['posy2'], 18)

(speedgrid, speedVec, speed, idx) = speed_map(
    data['posx_c'], data['posy_c'], 10)

too_fast = np.where(speed >= 50)
posgrid = np.delete(posgrid, too_fast, axis=0)
speedgrid = np.delete(speedgrid, too_fast, axis=0)
hdgrid = np.delete(hdgrid, too_fast, axis=0)
spiketrain = np.delete(data['spiketrain'], too_fast, axis=0)

# %%
# Smooth firing rate


def gaussfilter(x, mu, sigma):
    a = np.exp(-(x-mu)**2/(2*sigma**2))
    a = a/a.sum(axis=0)
    return a


f = gaussfilter(np.arange(-4, 5), 0, 2)
post = data['post']
dt = post[3]-post[2]
fr = data['spiketrain'].ravel()/dt
print(fr.shape)
smooth_fr = np.convolve(fr, f, mode='same')

# %%
plt.plot(smooth_fr)

# %%


def rough_penalty_2d(param, beta):
    numParam = (param.ravel().shape[0])

    # make diagnoal matrix
    n = np.sqrt(numParam).astype(int)
    D1 = np.ones((n, 1))*np.array([-1, 1])
    D1 = np.diag(D1[:, 1])+np.diag(D1[1:, 0], -1)
    D1 = D1[1:, :]
    DD1 = D1.T@D1

    M1 = np.kron(np.eye(n), DD1)
    M2 = np.kron(DD1, np.eye(n))
    M = (M1+M2)
    J = beta*0.5*param.T@M@param
    J_g = beta*M@param.T
    J_h = beta*M
    
    return (J, J_g, J_h)


def rough_penalty_1d_circ(param, beta):
    n = (param.ravel().shape[0])

    # make diagnoal matrix
    D1 = np.ones((n, 1))*np.array([-1, 1])
    D1 = np.diag(D1[:, 1])+np.diag(D1[1:, 0], -1)
    D1 = D1[1:, :]
    DD1 = D1.T@D1

    # correct for smooth in first and last bin
    DD1[0, :] = np.roll(DD1[1, :], -1, axis=0)
    DD1[-1, :] = np.roll(DD1[-2, :], 1, axis=0)

    J = beta*0.5*param.T@DD1@param
    J_g = beta*DD1@param.T
    J_h = beta*DD1

    return (J, J_g, J_h)


def rough_penalty_1d(param, beta):
    n = param.ravel().shape[0]
    D1 = np.ones((n, 1))*np.array([-1, 1])
    D1 = np.diag(D1[:, 1])+np.diag(D1[1:, 0], -1)
    D1 = D1[1:, :]
    DD1 = D1.T@D1

    J = beta*0.5*param.T@DD1@param
    J_g = beta*DD1@param.T
    J_h = beta*DD1

    return (J, J_g, J_h)


rough_penalty_2d(np.array([1, 2, 3, 4, 1, 2, 3, 4, 1]), 0.2)
rough_penalty_1d_circ(np.array([1, 2, 3, 4, 1, 2, 3, 4, 1]), 0.2)
rough_penalty_1d(np.array([1, 2, 3, 4, 1, 2, 3, 4, 1]), 0.2)

# %%


def ln_poisson_model(param, X, Y, modelType=[1,1,1,0]):
    # X: navigation variables, Y: spiketrain
    u = X*np.matrix(param).T
    rate = np.exp(u)


    # compute f, the gradient, and the hessian
    # print(J_pos)
    # print(J_hd)
    # print(J_spd)
    (J_pos, J_hd, J_spd, J_theta) = get_rough_penalty(param)['J']
    f = np.sum(rate-np.multiply(Y, u)) + J_pos + J_hd + J_spd + J_theta
    return f

def get_rough_penalty(param, numPos=400, numHD=18, numSpd=10, numTheta=18):
    # roughness regularizer weight - note: these are tuned using the sum of f,
    # and thus have decreasing influence with increasing amounts of data
    b_pos = 8e0
    b_hd = 5e1
    b_spd = 5e1
    b_th = 5e1

     # initialize parameter-relevant variables
    J_pos = 0
    J_pos_g = np.array([])
    J_pos_h = None

    J_hd = 0
    J_hd_g = np.array([])
    J_hd_h = None

    J_spd = 0
    J_spd_g = np.array([])
    J_spd_h = None

    J_theta = 0
    J_theta_g = np.array([])
    J_theta_h = None

    # find parameters
    (param_pos, param_hd, param_spd, param_theta) = find_param(param, modelType,
                                                               numPos, numHD, numSpd, numTheta)

    # Compute the contribution for f, df, and the hessian
    if param_pos is not None:
        (J_pos, J_pos_g, J_pos_h) = rough_penalty_2d(param_pos, b_pos)

    if param_hd is not None:
        (J_hd, J_hd_g, J_hd_h) = rough_penalty_1d_circ(param_hd, b_hd)

    if param_spd is not None:
        (J_spd, J_spd_g, J_spd_h) = rough_penalty_1d(param_spd, b_spd)

    if param_theta is not None:
        (J_theta, J_theta_g, J_theta_h) = rough_penalty_1d_circ(
            param_theta, b_th)

    
    return {'J':(J_pos, J_hd, J_spd, J_theta), 'J_g':(J_pos_g, J_hd_g, J_spd_g, J_theta_g), 'J_h': (J_pos_h, J_hd_h, J_spd_h, J_theta_h)}


def ln_poisson_model_jac(param, X, Y, modelType=[1,1,1,0]):
    (J_pos_g, J_hd_g, J_spd_g, J_theta_g) = get_rough_penalty(param)['J_g']

    u = X*np.matrix(param).T
    rate = np.exp(u)
    J = np.hstack([J_pos_g, J_hd_g, J_spd_g, J_theta_g])
    df = np.real(X.T * (rate - Y) + J[:,None])
    return np.array(df).flatten()


def ln_poisson_model_hessian(param, X, Y,modelType=[1,1,1,0]):

    (J_pos_h, J_hd_h, J_spd_h, J_theta_h) = get_rough_penalty(param)['J_h']

    u = X*np.matrix(param).T
    rate = np.exp(u)
    rX = np.multiply(rate, X)
    diag = [J_pos_h, J_hd_h, J_spd_h, J_theta_h]
    diag = [d for d in diag if d is not None]
    hessian_glm = rX.T*X + block_diag(*diag)
    return hessian_glm


def find_param(param, modelType, numPos, numHD, numSpd, numTheta):

    param_pos = None
    param_hd = None
    param_spd = None
    param_theta = None

    if np.all(modelType == [1, 0, 0, 0]):
        param_pos = param
    elif np.all(modelType == [0, 1, 0, 0]):
        param_hd = param
    elif np.all(modelType == [0, 0, 1, 0]):
        param_spd = param
    elif np.all(modelType == [0, 0, 0, 1]):
        param_theta = param

    elif np.all(modelType == [1, 1, 0, 0]):
        param_pos = param[:numPos]
        param_hd = param[numPos:numPos+numHD]
    elif np.all(modelType == [1, 0, 1, 0]):
        param_pos = param[:numPos]
        param_spd = param[numPos:numPos+numSpd]
    elif np.all(modelType == [1, 0, 0, 1]):
        param_pos = param[:numPos]
        param_theta = param[numPos:numPos+numTheta]
    elif np.all(modelType == [0, 1, 1, 0]):
        param_hd = param[:numHD]
        param_spd = param[numHD:numHD+numSpd]
    elif np.all(modelType == [0, 1, 0, 1]):
        param_hd = param[:numHD]
        param_theta = param[numHD+1:numHD+numTheta]
    elif np.all(modelType == [0, 0, 1, 1]):
        param_spd = param[:numSpd]
        param_theta = param[numSpd+1:numSpd+numTheta]

    elif np.all(modelType == [1, 1, 1, 0]):
        param_pos = param[:numPos]
        param_hd = param[numPos:numPos+numHD]
        param_spd = param[numPos+numHD:numPos+numHD+numSpd]
    elif np.all(modelType == [1, 1, 0, 1]):
        param_pos = param[:numPos]
        param_hd = param[numPos:numPos+numHD]
        param_theta = param[numPos+numHD:numPos+numHD+numTheta]
    elif np.all(modelType == [1, 0, 1, 1]):
        param_pos = param[:numPos]
        param_spd = param[numPos:numPos+numSpd]
        param_theta = param[numPos+numSpd:numPos+numSpd+numTheta]
    elif np.all(modelType == [0, 1, 1, 1]):
        param_hd = param[:numHD]
        param_spd = param[numHD:numHD+numSpd]
        param_theta = param[numHD+numSpd:numHD+numSpd+numTheta]
    elif np.all(modelType == [1, 1, 1, 1]):
        param_pos = param[:numPos]
        param_hd = param[numPos:numPos+numHD]
        param_spd = param[numPos+numHD:numPos+numHD+numSpd]
        param_theta = param[numPos+numHD+numSpd:numPos+numHD+numSpd+numTheta]

    return (param_pos, param_hd, param_spd, param_theta)


X = np.matrix(np.hstack([hdgrid, speedgrid, posgrid]))
# X = np.matrix(hdgrid)
Y = np.matrix(spiketrain)
numCol = X.shape[1]
param = np.random.randn(numCol)*1e-3

modelType=[1,1,1,0]
y = ln_poisson_model(param, X, Y, modelType=modelType)
dy = ln_poisson_model_jac(param, X, Y, modelType=modelType)
hess = ln_poisson_model_hessian(param, X, Y, modelType=modelType)

sio.savemat('test//testdata.mat', {'X':X,'hdgrid': hdgrid, 'posgrid': posgrid,
                             'speedgrid': speedgrid, 'spiketrain': spiketrain, 'param': param})

print(y)
print(dy.shape)
print(hess.shape)


# %%
Xsmall = X[1:10000,:]
Ysmall = Y[1:10000,:]

result=minimize(ln_poisson_model,param,args=(X,Y,modelType),method='Newton-CG',jac=ln_poisson_model_jac, hess=ln_poisson_model_hessian)
print(result.message)    
print(result.fun)


#%%
print(result.x)

#%% Compute compute model drived tuning curve

def computeModelTuning(param, n_pos_bins, n_dir_bins, n_speed_bins):
    #Compute the neuronal tuning based on models

    #Extract each individual parameters
    startIdx = 0
    endIdx = n_pos_bins
    pos_param = param[startIdx:endIdx]

    startIdx = endIdx
    endIdx = startIdx + n_dir_bins
    hd_param = param[startIdx:endIdx]

    startIdx = endIdx
    endIdx = startIdx + n_speed_bins
    speed_param = param[startIdx:endIdx]

    #Calculate the scale factor, assuming the other variables are at their mean rate
    scale_factor_pos = np.exp(speed_param).mean()*np.exp(hd_param).mean()
    scale_factor_hd = np.exp(speed_param).mean()*np.exp(pos_param).mean()
    scale_factor_speed = np.exp(speed_param).mean()*np.exp(pos_param).mean()

    #calculate the response curve
    pos_response = scale_factor_pos* np.exp(pos_param)
    hd_response = scale_factor_hd*np.exp(hd_param)
    speed_response = scale_factor_speed * np.exp(speed_param)

    return (pos_response, hd_response, speed_response)

(pos_response,hd_response, speed_response) = computeModelTuning(result.x,
    400,18,10)


#%%
plt.subplot(131)
plt.plot(np.insert(dirVec,0,0),hd_response)
plt.subplot(132)
plt.plot(np.insert(speedVec,0,0), speed_response)
plt.subplot(133)
plt.imshow(np.reshape(pos_response,(20,20)))

#%%
