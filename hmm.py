import pandas as pd
import numpy as np

_ = np.newaxis

# Add classes of HMM
#   Finite
#   Laplace
#   Multivariate


class HMM:
    def __init__(self, N=2):
        """
        Create a Hidden Markov Model.

        Parameters
        ----------
        N : int, optional
            Number of states of the Markov Model.
        linear: bool, optional
            Forces the Markov Model to have a linear topology.

        Returns
        -------
        out: HMM
            A Hidden Markov Model with N states, with linear topoloy and
            transition probability between spaces of 1%
        """
        if N < 1:
            raise ValueError("N has to be a positive integer")
        self.N = int(N)
        self.init = np.ones(self.N) / self.N
        self.fixed_trans = False
        self.set_trans(0.01)

    def set_obs(self, obs):
        """
        Sets the observed sequence.

        Parameters
        ----------
        obs : array_like
            Observed sequence.
        """
        if isinstance(obs, (pd.Series, pd.DataFrame)):
            obs = obs.values
        self.obs = obs

    def set_trans(self, T):
        """
        Sets the transition matrix of the Markov Model.

        Parameters
        ----------
        T : float or array_like
            Sets the transition matrix of the Markov Model
            If T is a float or an array in shape (N,), then the HMM is set to
            be linear, with diagonal equal to 1-T
            If T is an array with shape (N, N), then the HMM transition matrix
            is set to be equal to T
        """

        # Converts the argument to be a ndarray
        if isinstance(T, float):
            T = T * np.ones(self.N)
        elif isinstance(T, (list, tuple)):
            T = np.array(T)

        # Sets the transition matrix
        if T.ndim == 1:
            if not T.shape == (self.N,):
                raise ValueError("T has to have shape (N,)")
            d = T * .5
            d[0] = T[0]
            d[-1] = T[-1]
            self.T = np.diag(1 - T) + np.diag(d[:-1], k=1) + np.diag(d[1:], k=-1)
        elif T.ndim == 2:
            if not T.shape == (self.N, self.N):
                raise ValueError("T has to have shape (N, N)")
            if not (T.sum(1) == 1).all():
                raise ValueError(
                    "The sum of the elements of each line of T equals 1")
            self.T = T
        else:
            raise ValueError("Invalid input")

    def set_param(self, param):
        """
        Sets the parameters of the Markov Model.

        Parameters
        ----------
        param: array_like
            * If param has shape (N,) then it is assumed that there is only one
            parameter for each state.
            * If param has shape (P, N) then there is assumed that there are P
            parameters for each state.
            * If param has shape (P, 2) then there is assumed that there are P
            parameters for each state and the set param[p, 0] to be the
            parameters of the first state, param[p, -1] to be the parameters of
            the last state and the parameters of other states are set to be
            intermediary values of them.
        """
        if isinstance(param, (list, tuple)):
            param = np.array(param)

        # Only one parameter
        if param.ndim == 1:
            param = param[_, :]

        if param.ndim == 2:
            if param.shape[1] == 2:
                self.param = param[:, 0][:, _] + np.arange(self.N)[_, :] / (self.N - 1.0) * \
                    (param[:, 1] - param[:, 0])[:, _]
            elif param.shape[1] == self.N:
                self.param = param
            else:
                raise ValueError("param has to have shape[-1] == 2 or N")
        else:
            raise ValueError("param has to have one or twoo axis")

        # Coerced parameters, initially setted to be none.
        self.coerc = np.zeros(self.param.shape, np.bool)

    def _func(self, obs, param):
        raise NotImplementedError

    def likelihood(self):
        """
        Returns the likelihood.
        """
        self.f = self._func(self.obs, self.param)
        state_prob = self.init
        for t in xrange(1, self.obs.size):
            state_prob[:, t] = np.dot(self.T, state_prob[:, t - 1])
        return np.log((state_prob * self.f).sum(0)).sum()

    def step(self):
        self.f = self._func(self.obs, self.param)
        self.a = np.empty((self.N, self.obs.size))
        tmp = self.init * self.f[:, 0]
        self.scale = np.empty(self.obs.size)
        self.scale[0] = 1 / tmp.sum()
        self.a[:, 0] = self.scale[0] * tmp
        for t in xrange(1, self.obs.size):
            tmp = self.f[:, t] * (self.a[:, t - 1][:, _] * self.T).sum(0)
            self.scale[t] = 1 / tmp.sum()
            self.a[:, t] = self.scale[t] * tmp
        self.b = np.zeros((self.N, self.obs.size))
        self.b[:, -1] = 1
        for t in xrange(1, self.obs.size):
            tmp = (self.T * (self.f[:, -t] * self.b[:, -t])[_, :]).sum(1)
            self.b[:, -1 - t] = tmp / tmp.sum()
        self.gamma = self.a * self.b
        self.gamma /= self.gamma.sum(0)
        self.init = self.gamma[:, 0]
        self.state_prob = self.gamma.sum(-1) / self.gamma.sum()
        if not self.fixed_trans:
            self.xi = self.a[:, _, :-1] * \
                self.T[:, :, _] * (self.f * self.b)[_, :, 1:]
            self.xi /= self.xi.sum((0, 1))
            self.T = self.xi.sum(-1) / self.gamma[:, :-1].sum(-1)[:, _]

    def run(self, e=1e-5, max_steps=100):
        """
        Applies the Expectation-Maximization Algorithm to find the best
        parameters that adjust to the observed sequence.

        Parameters
        ----------
        e : float, optional
            Tolerance for termination.
        max_steps : int, optional
            Maximum number of steps that the algorithm will perform.

        Returns
        -------
        out : boolean
            Returns True if the algorithm converges, otherwise returns
            False.
        """
        old_var = list(self.param) + [self.T - np.diag(self.T)]
        norm = [np.linalg.norm(old_var[i]) for i in xrange(len(old_var))]
        for k in xrange(max_steps):
            self.step()
            print self.param
            new_var = list(self.param) + [self.T - np.diag(self.T)]
            d_norm = [np.linalg.norm(new_var[i] - old_var[i])
                      for i in xrange(len(old_var))]
            if sum(map(lambda x: x[0] / x[1], zip(d_norm, norm))) < e:
                return True  # Converged within max_steps
            old_var = new_var
            norm = [np.linalg.norm(old_var[i]) for i in xrange(len(old_var))]
        return False  # Didn't converge within max_steps

    def update(self, new_obs):
        """
        Set a new observation sequence and adjust the parameters.

        Parameters
        ----------
        new_obs : array_like
            Observed sequence.

        Returns
        -------
        out : boolean
            Returns True if the algorithm converges, otherwise returns
            False.        
        """
        self.obs = np.append(self.obs, new_obs)
        return self.run()


class HMM_normal(HMM):
    def __init__(self, N):
        HMM.__init__(self, N)
        self.set_param(np.array([1, -1]) * 0.0001, np.array([.01, .05]))

    def set_param(self, u, s):
        HMM.set_param(self, [u, s])

    def _func(self, obs, param):
        z = (obs[_, :] - param[0][:, _]) / param[1][:, _]
        return np.exp(-z * z / 2) / (param[1] * np.sqrt(2 * np.pi))[:, _]

    def step(self):
        HMM.step(self)
        self.param[0][~self.coerc[0]] = (
            (self.gamma * self.obs[_, :]).sum(-1) /
             self.gamma.sum(-1))[~self.coerc[0]]
        self.param[1][~self.coerc[1]] = np.sqrt(
            (self.gamma *
             (self.obs[_, :] - self.param[0][:, _]) *
             (self.obs[_, :] - self.param[0][:, _])
             ).sum(-1) / self.gamma.sum(-1))[~self.coerc[1]]


class HMM_bernoulli(HMM):
    def __init__(self, N):
        HMM.__init__(self, N)
        self.set_param(np.sort(np.random.rand(self.N)))

    def _func(self, obs, param):
        p = param[0]
        return obs[_, :] * p[:, _] + (1 - obs)[_, :] * (1 - p)[:, _]

    def step(self):
        HMM.step(self)
        self.param[0][~self.coerc[0]] = (
            (self.gamma * self.obs[_, :]).sum(-1) /
             self.gamma.sum(-1))[~self.coerc[0]]


class HMM_exponential(HMM):
    def __init__(self, N):
        HMM.__init__(self, N)
        self.set_param(np.sort((1 / np.rand.random(2) - 1)))

    def _func(self, obs, param):
        return param[0][:, _] * np.exp(-param[0][:, _] * obs[_, :])

    def step(self):
        HMM.step(self)
        self.param[0][~self.coerc[0]] = self.gamma.sum(-1) / \
            (self.gamma * self.obs[_, :]).sum(-1)[~self.coerc[0]]


def model(states, kind='Normal'):
    """
    Returns an Hidden Markov Model object

    Parameters
    ----------
    states : int
        The number of states of the Hidden Markov Model
    kind : {'Normal', 'Bernoulli', 'Exponential'}, optional
        The familly of probability distributions for each state

    Returns
    -------
    out : HMM object 
    """
    kind = kind.lower()
    if kind in ['normal','gaussian']:
        return HMM_normal(states)
    elif kind in ['bernoulli']:
        return HMM_bernoulli(states)
    elif kind in ['exp', 'exponential']:
        return HMM_exponential(states)
    else:
        raise ValueError
