from BasePostProcessor import *
import numpy as np
from scipy.optimize import minimize
import pickle
import scipy.io as sio
from scipy.linalg import block_diag
import scipy.signal as signal

# helpful scripts are in the package
# core logic is in the class

def gaussfilter(x,mu,sigma):
    a=np.exp(-(x-mu)**2/(2*sigma**2))
    a = a/a.sum(axis=0)
    return a

def computeModelTuning(pos_param=None, hd_param=None, speed_param=None):
    #Compute the neuronal tuning based on models

    #Calculate the scale factor, assuming the other variables are at their mean rate
    scale_factor_pos = np.exp(speed_param).mean()*np.exp(hd_param).mean()
    scale_factor_hd = np.exp(speed_param).mean()*np.exp(pos_param).mean()
    scale_factor_speed = np.exp(speed_param).mean()*np.exp(pos_param).mean()

    #calculate the response curve
    pos_response = scale_factor_pos* np.exp(pos_param)
    hd_response = scale_factor_hd*np.exp(hd_param)
    speed_response = scale_factor_speed * np.exp(speed_param)

    return (pos_response, hd_response, speed_response)


def ln_poisson_model(param, X, Y, modelType=[1,1,1,0]):
    # X: navigation variables, Y: spiketrain
    u = X*np.matrix(param).T
    rate = np.exp(u)


    # compute f, the gradient, and the hessian
    # print(J_pos)
    # print(J_hd)
    # print(J_spd)
    (J_pos, J_hd, J_spd, J_theta) = get_rough_penalty(param,modelType)['J']
    f = np.sum(rate-np.multiply(Y, u)) + J_pos + J_hd + J_spd + J_theta
    return f

def get_rough_penalty(param,modelType, numPos=400, numHD=18, numSpd=10, numTheta=18):
    # roughness regularizer weight - note: these are tuned using the sum of f,
    # and thus have decreasing influence with increasing amounts of data
    b_pos = 8e0
    b_hd = 5e1
    b_spd = 5e2 #5e1
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
    (J_pos_g, J_hd_g, J_spd_g, J_theta_g) = get_rough_penalty(param,modelType)['J_g']

    u = X*np.matrix(param).T
    rate = np.exp(u)
    J = np.hstack([J_pos_g, J_hd_g, J_spd_g, J_theta_g])
    df = np.real(X.T * (rate - Y) + J[:,None])
    return np.array(df).flatten()


def ln_poisson_model_hessian(param, X, Y,modelType=[1,1,1,0]):

    (J_pos_h, J_hd_h, J_spd_h, J_theta_h) = get_rough_penalty(param,modelType)['J_h']

    u = X*np.matrix(param).T
    rate = np.exp(u)
    rX = np.multiply(rate, X)
    diag = [J_pos_h, J_hd_h, J_spd_h, J_theta_h]
    diag = [d for d in diag if d is not None]
    hessian_glm = rX.T*X + block_diag(*diag)
    return hessian_glm


def find_param(param, modelType, numPos, numHD, numSpd, numTheta):
    #TODO cater for different number of bins

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

def pos_1d_map(pos, nbins, boxSize):
    # discretize the position into different bins
    # pos: nx2 matrix
    bins = np.arange(boxSize/nbins, boxSize+1, boxSize/nbins) #make sure it covers the last bin
    coord = np.digitize(pos, bins)
    output = np.zeros((pos.shape[0], nbins))
    output[np.arange(pos.shape[0]),coord] = 1

    return (output, bins)

def theta_map(phase,nbins):
    """Discretize theta phase
    
    Arguments:
        phase {np.narray} -- phase of EEG
        nbins {int} -- number of bins
    
    Returns:
        (np.narray,np.narray) -- (binned output, bin used)
    """
    bins = np.arange(2*np.pi/nbins,np.pi, np.pi/nbins)
    coord = np.digitize(phase,bins,right=True)
    output = np.zeros((phase.shape[0],nbins))
    output[np.arange(phase.shape[0]),coord] = 1

    return (output,bins)

#filter at theta band
def extract_theta_phase(eeg, Wn, Fs):
    """Filter the EEG to theta band and extract its phase
    
    Arguments:
        eeg {np.narray} -- Raw EEG signal in form (time x channel)
        Wn {np.narray} -- upper and lower filter corner frequency
        Fs {float} -- samplig frequency
    
    Returns:
        np.narray -- phase of the signal
    """
    #Filter signal
    (b,a)=signal.butter(5,Wn/(Fs*2),'bandpass')
    eegFilt = signal.filtfilt(b,a,eeg,axis=0)

    #Hilbert transform to find instantaneous phase
    eegHilbert=signal.hilbert(eegFilt,axis=0)
    phase = np.arctan2(np.imag(eegHilbert),np.real(eegHilbert))
    for i in range(phase.shape[1]):
        ind = np.where(phase[:,i]<0)
        phase[ind,i] = phase[ind,i]+2*np.pi

    return phase

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

    speedVec = np.arange(maxSpeed/nbins, maxSpeed+1, maxSpeed/nbins)
    speed_grid = np.zeros((posx.shape[0], nbins))

    idx = np.digitize(speed, speedVec)
    speed_grid[np.arange(speed.shape[0]), idx.ravel()] = 1

    return (speed_grid, speedVec, speed, idx)

def speed_map_1d(pos, nbins, sampleRate=50, maxSpeed=50,removeWrap=False):
    """bin and map the speed into one-hot vector
    
    Arguments:
        pos {np.narray} -- position of the animal, 1d array
        nbins {int} -- number of bins
    
    Keyword Arguments:
        sampleRate {int} -- sampling rate of signal (default: {50})
        maxSpeed {int} -- maximum speed (default: {50})
        removeWrap {bool} -- whether to remove the point going from the end back to the start (default: {False})
    
    Returns:
        {tuple} -- (one hot vector, bin edges, speed )
    """
    vel = np.diff(np.insert(pos, 0, pos[0]))
    speed = vel*sampleRate
    # send everything over 50 cm/s to 50 cm/s
    speed[speed > maxSpeed] = maxSpeed
    if removeWrap:
        idx = np.where(speed<-maxSpeed)[0]
        for i in idx:
            speed[i] = speed[i-1] #assign the teleport speed right before the teleport

    speedVec = np.arange(maxSpeed/nbins, maxSpeed+1, maxSpeed/nbins)
    speed_grid = np.zeros((pos.shape[0], nbins))

    idx = np.digitize(speed, speedVec,right=True)
    speed_grid[np.arange(speed.shape[0]), idx.ravel()] = 1

    return (speed_grid, speedVec, speed)

def calculate_track_location(recorded_location, track_length):
    print('Converting raw location input to cm...')
    recorded_startpoint = np.min(recorded_location)
    recorded_endpoint = np.max(recorded_location)
    recorded_track_length = recorded_endpoint - recorded_startpoint
    distance_unit = recorded_track_length/track_length  # Obtain distance unit (cm) by dividing recorded track length to actual track length
    location_in_cm = (recorded_location - recorded_startpoint) / distance_unit
    return location_in_cm # fill in dataframe

def make_binary_bin(spiketrain, binEdge):
    '''
    convert spike time into binary bin, specified by the binSize
    '''
    idx = np.digitize(spiketrain,binEdge)
    spiketrain_bin = np.zeros((binEdge.shape[0],1))
    for i in range(idx.shape[0]):
        spiketrain_bin[idx[i]] +=1
    return spiketrain_bin

def average_in_bin(position,binEdge,binSize):
    p = position[:binEdge[-1]]
    p = p.reshape(int(p.shape[0]//binSize),int(binSize)) #the reshape begins at the last indices
    p = p.mean(axis=1)
    
    return p
    
def plotRawTuningCurve(ax,datagrid,spiketrain_bin,response, vec):
    loc = np.argmax(datagrid, axis=1)
    count=np.zeros((vec.shape[0],))
    std = np.zeros((vec.shape[0],))

    for i in range(count.shape[0]):
        count[i] = np.mean(spiketrain_bin[i==loc])
        std[i] = np.std(spiketrain_bin[i==loc])
#     plt.errorbar(vec,count,std,fmt='ro')
    
    ax.plot(vec,count,'ro')
    ax.plot(vec,response)
    ax.legend(['Testing set','Model'])

        
def get_smooth_fr(spiketrain,halfWidth,dt):
    f = GLMPostProcessor.gaussfilter(np.arange(-halfWidth+1,halfWidth),0,10)
    fr = np.array(spiketrain).flatten()/dt
    fr_smooth=np.convolve(fr,f,mode='same')
    return fr_smooth


def findTeleportPoint(position,maxspeed=50):
    s = np.diff(position)
    return np.where(s<-maxspeed)[0]
        
def compare_model_performance(param,datagrid,spiketrain_bin,dt):
    spiketrain_hat = np.exp(datagrid*np.matrix(param).T)
    fr_param = get_smooth_fr(spiketrain_hat,5,dt)    
    fr = get_smooth_fr(spiketrain_bin,5,dt)
    
    #variance explained 
    sse = np.sum((fr_param-fr)**2)
    sst = np.sum((fr-np.mean(fr))**2)
    varExplain = 1-(sse/sst)
    
    #correlation
    (correlation,corr_p) = pearsonr(fr_param,fr)
    
    #log likelihood
    r_train = np.array(spiketrain_hat) #predicted
    n_train = np.array(spiketrain_bin) #true
    meanFR_train = np.mean(n_train)
    log_llh_train_model = np.sum(r_train-n_train*np.log(r_train)+np.log(factorial(n_train)))/np.sum(n_train)
    log_llh_train_mean = np.sum(meanFR_train-n_train*np.log(meanFR_train)+np.log(factorial(n_train)))/np.sum(n_train)
    log_llh_train = (-log_llh_train_model+log_llh_train_mean)*np.log(2)
    
    #mse
    mse = np.mean(((fr_param-fr)**2))
    
    return {'fr_param':fr_param,'fr':fr, 'varExplain':varExplain, 'correlation':correlation, 'corr_p':corr_p,
                'log_llh_train':log_llh_train, 'mse':mse}

#separate training set and testing set
def getTrainTestSet(x,trainPercent=0.7):
    split = int(x.shape[0]*0.7)
    
    if x.ndim==1 or x.shape[1]==1:
        return (x[:split],x[split:])
    else:
        return (x[:split,:],x[split:,:])
    

def plotTuningCurves(datagrid,spiketrain,result,dt,vec,tuningXLabel='Position'):
    performance=compare_model_performance(result.x,datagrid,spiketrain,dt)
    varExplain = performance['varExplain']
    correlation = performance['correlation']
    log_llh_train = performance['log_llh_train']
    mse = performance['mse']
    corr_p = performance['corr_p']
    
    response = np.exp(result.x)
    fig,ax=plt.subplots(1,3,figsize=(15,5))
    fig.suptitle(f'Neuron {clusterNo} (VarExplain: {varExplain:.2f}, corr: {correlation:.2f}, log_L: {log_llh_train:.2f}, MSE: {mse:.2f}) ')
    plotRawTuningCurve(ax[0],datagrid,spiketrain,response, vec)
    ax[0].set_ylabel('Mean spike')
    ax[0].set_xlabel(tuningXLabel)
    pltIdx = np.arange(0,5000)
    ax[1].plot(performance['fr_param'][pltIdx])
    ax[1].set_title('Model firing rate')
    ax[2].plot(performance['fr'][pltIdx])
    ax[2].set_title('Real firing rate')


def correct_for_restart(location):
    location[location <0.55] = 0.56 # deals with if the VR is switched off during recording - location value drops to 0 - min is usually 0.56 approx
    return location


class GLMPostProcessor(BasePostProcessor):
    def __init__(self,spiketrain, datagrid, modelType, speed=None, speedLimit=50):
        super().__init__()

        if speed is not None:
            too_fast = np.where(speed >= speedLimit)
            datagrid = np.delete(datagrid, too_fast, axis=0)
            spiketrain = np.delete(spiketrain, too_fast, axis=0)

        self.modelType = modelType
        self.datagrid = datagrid
        self.spiketrain = spiketrain


    def run(self):
        """Optimize on the LN poisson model
        
        Returns:
            optimization object -- result from scipy minimize function
        """
        X = self.datagrid 
        Y = self.spiketrain
        numCol = X.shape[1]
        param = np.random.randn(numCol)*1e-3

        self.result=minimize(ln_poisson_model,param,args=(X,Y,self.modelType),
            method='Newton-CG',jac=ln_poisson_model_jac, hess=ln_poisson_model_hessian)

        return self.result

    def cleanup(self):
        pickle.dump(self.result,open('GLM_tuning.pkl','wb'))


if __name__ =='__main__':
    data = sio.loadmat('data_for_cell77')

    posc = np.hstack([data['posx_c'], data['posy_c']])
    (posgrid, posVec) = pos_map(posc, 20, 100)

    (hdgrid, dirVec, direction) = hd_map(
        data['posx'], data['posx2'], data['posy'], data['posy2'], 18)

    (speedgrid, speedVec, speed, idx) = speed_map(
        data['posx_c'], data['posy_c'], 10)

    spiketrain = data['spiketrain']
    print(spiketrain.shape)
    datagrid = np.matrix(np.hstack([hdgrid, speedgrid, posgrid]))

    glmPostProcessor = GLMPostProcessor(spiketrain, datagrid, [1,1,1,0],speed)

    glmPostProcessor.run()
    glmPostProcessor.cleanup()


    




