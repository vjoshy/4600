import numpy as np
import matplotlib.pyplot as plt

class SDESimulator:
    def __init__(self, x0, drift_coef, diff_coef, t_span, dt = 0.001):
        
        # grab dimensions
        self.x0 = np.array(x0)
        if self.x0.ndim == 1:
            self.x0 = self.x0.reshape(-1, 1)
        self.n_samples, self.dim = self.x0.shape

        self.drift_coef = drift_coef
        self.diff_coef = diff_coef
        self.t0, self.T  = t_span
        self.dt = dt

        self.t = np.arange(self.t0, self.T + dt, dt)
        self.n_steps = len(self.t)

    def simulate(self, n_traj = 1):
        x = np.zeros((n_traj, self.n_steps, self.dim))
        x[:, 0, :] = self.x0
        
        # forward sde
        for i in range(self.n_steps - 1):
            t = self.t[i]
            drift = self.drift_coef(x[:,i], t)
            diffusion = self.diff_coef(t)

            dw = np.sqrt(self.dt) * np.random.normal(0, 1, (n_traj, self.dim))
            x[:, i + 1] = x[:,i] + (drift * self.dt) + (diffusion * dw)

        return x
    
class ReverseSDESimulator:
    def __init__(self, drift_coef, diff_coef, score_fn, t_span, dt = 0.001, dim = None):

        self.drift_coef = drift_coef
        self.diff_coef = diff_coef
        self.score_fn = score_fn

        self.t0, self.T  = t_span
        self.dt = dt
        self.dim  = dim
        
        # reverse time grid
        self.t = np.flip(np.arange(self.t0, self.T + dt, dt))
        self.n_steps = len(self.t)


    def simulate(self, prior, n_traj = 1):

        prior = np.array(prior)
        if prior.ndim == 1:
            prior = prior.reshape(-1, 1)
        
        x = np.zeros((n_traj, self.n_steps, prior.shape[1]))
        x[:, 0, :] = prior

        for i in range(self.n_steps - 1):
            t = self.t[i]
            drift = self.drift_coef(x[:,i, :], t)
            diffusion = self.diff_coef(t)
            score = self.score_fn(x[:,i, : ], t)              

            # brownian motion in reverse time
            dw_bar = np.sqrt(self.dt) * np.random.normal(0,1, (n_traj, prior.shape[1]))  
            reverse_drift = -drift + (diffusion**2)* score
            dx = (reverse_drift * self.dt) + (diffusion *  dw_bar)

            x[:, i + 1] = x[:,i] + dx
        return x
    
def drift(x, t):
    return 0

# time dependent noise
def beta(t):
    const = 20.0
    return  t * const


def diffusion(t):   
    return np.sqrt(beta(t))


# cumulative variance at time t ~ taking integral of beta(t) from 0 to t
def sigma(t):
    const = 20.0
    return np.sqrt((const/2) * t**2)

# score function of linear regression - not great
def score_fn(params, t, data):
    
    x_data, y_data = data
    w, b = params[:,0], params[:,1]
        
    # likelihood gradients
    y_pred = w.reshape(-1,1) * x_data + b.reshape(-1,1)
    error = y_pred - y_data
    
    grad_w = np.sum(error * x_data, axis=1) 
    grad_b = np.sum(error, axis=1)
    
    # Scale by the correct time-dependent variance
    var_t = sigma(t)**2  
    score = -np.column_stack([grad_w, grad_b]) / var_t 
    
    # clipping for numerical stability - score function explodes 
    return np.clip(score, -100, 100)


# setup
t_span = (0,1)
n_samples = 100

# true slope and intercept values
w_true = 2.0
b_true = 1.0

# current state
mu = np.array([w_true, b_true])

x_data = np.linspace(-5, 5, 100)
y_data = w_true * x_data + b_true + np.random.normal(0, 1, size=len(x_data))

# initial dist
x0 = np.repeat(mu.reshape(1,-1), n_samples, axis = 0)
#x0 = np.random.normal(mu, 0.1, (n_samples, len(mu)))

data = (x_data, y_data)

sde = SDESimulator(x0, drift, diffusion, t_span)
f_trajectories = sde.simulate(n_samples)
#print(f_trajectories[:, -1])

from functools import partial
# using partial to pass 'data' as parameter into score_fn
r_sde = ReverseSDESimulator(drift, diffusion, partial(score_fn, data=data), t_span, dim = len(mu))

r_trajectories = r_sde.simulate(f_trajectories[:, -1], n_samples)
#print(r_trajectories[:, 0])


# Plotting
plt.figure(figsize=(15, 5))


plt.subplot(131)
plt.plot(sde.t, f_trajectories[:,:,0].T, alpha=0.1, color='blue')
plt.plot(sde.t, f_trajectories[:,:,1].T, alpha=0.1, color='red')
plt.title('Forward SDE Trajectories')
plt.xlabel('Time')
plt.ylabel('Parameters (w,b)')

plt.subplot(132)
plt.plot(np.flip(r_sde.t), r_trajectories[:,:,0].T, alpha=0.1, color='blue', label='w')
plt.plot(np.flip(r_sde.t), r_trajectories[:,:,1].T, alpha=0.1, color='red', label='b')
plt.title('Reverse SDE Trajectories')
plt.xlabel('Time')
plt.ylabel('Parameters (w,b)')
#plt.legend()

# regression lines
plt.subplot(133)
for i in range(2):  # Plot first 2 samples
    w_new, b_new = r_trajectories[i,-1]
    plt.scatter(x_data, w_new*x_data + b_new, alpha=0.3, color = 'black', label = 'reconstructed')
plt.scatter(x_data, y_data, color='blue', alpha=0.5, label='Original Data')
plt.plot(x_data, w_true*x_data + b_true, 'r--', label='True')
plt.title('Recovered Regression Lines')
plt.legend()

plt.tight_layout()
plt.show()