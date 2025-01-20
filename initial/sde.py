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
            score = self.score_fn(x[:,i, : ], mu, t)              

            #  brownian motion in reverse time
            dw_bar = np.sqrt(self.dt) * np.random.normal(0,1, (n_traj, prior.shape[1]))  
            reverse_drift = -drift + (diffusion**2)* score
            dx = (reverse_drift * self.dt) + (diffusion *  dw_bar)

            x[:, i + 1] = x[:,i] + dx
        return x

# drift is set to zero
def drift(x, t):
    return 0
    
#Time-dependent diffusion coefficient - set to exp(t) follwing Song Y. article
def diffusion(t):
    return np.exp(t)

# score function
def score_fn(x, target_mean, t): 
    var_t = (np.exp(2*t) - 1)/2
    return -(x - target_mean) / (var_t + 1e-6)

# setup
t_span = (0,1)
n_samples = 1000


# current state
mu = np.array([20])

# initial dist
#   x0 = np.repeat(mu, n_samples, axis = 0)
x0 = np.random.normal(mu, 0.1, (n_samples, len(mu)))

sde = SDESimulator(x0, drift, diffusion, t_span)
f_trajectories = sde.simulate(n_samples)
#print(f_trajectories[:, -1])


r_sde = ReverseSDESimulator(drift, diffusion, score_fn, t_span, dim = len(mu))
r_trajectories = r_sde.simulate(f_trajectories[:, -1], n_samples)
#print(r_trajectories[:, 0])


plt.figure(figsize=(15, 5))

plt.subplot(131)
for i in range(n_samples):
    plt.plot(sde.t, f_trajectories[i, :, 0], alpha=0.5)
    #plt.plot(sde.t, f_trajectories[i, :, 1], alpha=0.5)
plt.title('Forward SDE Trajectories')
plt.xlabel('Time')
plt.ylabel('X(t)')

plt.subplot(132)
for i in range(n_samples):
    #print("\nreverse", i)
    #print(r_trajectories[i, :, 0])
    plt.plot(np.flip(r_sde.t), r_trajectories[i, :, 0], alpha=0.5)
    #plt.plot(r_sde.t, r_trajectories[i, :, 1], alpha=0.5)
plt.title('Reverse SDE Trajectories')
plt.xlabel('Time')
plt.ylabel('X(t)')
#plt.legend()

plt.subplot(133)
plt.hist(f_trajectories[:, 0], bins=10, alpha=1, label='Original X1', density=True)
plt.hist(r_trajectories[:, -1], bins=50, alpha=1, label='Recovered X1', density=True)
""" plt.hist(f_trajectories[:, 0, 1], bins=50, alpha=0.5, label='Original X2', density=True)
plt.hist(r_trajectories[:, -1, 1], bins=50, alpha=0.5, label='Recovered X2', density=True) """
plt.title('Distribution Comparison')
plt.legend()

plt.tight_layout()
plt.show()