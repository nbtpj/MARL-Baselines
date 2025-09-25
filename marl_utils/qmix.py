"""
This implementation is signiricantly different from author implementation:
- I try to support continuous action and make it is consistent with the discrete action, so policy network is always separated from critics.
- This leads to another different: there will be loss $\pi \gets \arg\max_{\pi\in \Pi} E_{s, a~\pi} Q(s, a)$ for each policy
- The network architecture is not hard code (man attention here is the hyper-network).
- Policy execution is truly decentralised via multi-threading.
Therefore, this is NOT FINAL YET: tests on smac is being conducted to compare with author implementation!
"""
from .utils import *
import torch
from torch import nn
import torch.nn.functional as F
from gymnasium import spaces
from gymnasium.spaces.utils import flatten_space, flatten, unflatten
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import trange
import itertools

import copy

def batch_flatten(space: spaces.Space, xs: np.ndarray)->np.ndarray:
    return np.array([flatten(space, x) for x in xs])

def batch_seq_flatten(space: spaces.Space, xs: np.ndarray)->np.ndarray:
    return np.array([[flatten(space, x) for x in _xs] for _xs in xs])

def batch_unflatten(space: spaces.Space, xs: np.ndarray)->np.ndarray:
    return np.array([unflatten(space, x) for x in xs])

def batch_seq_unflatten(space: spaces.Space, xs: np.ndarray)->np.ndarray:
    return np.array([[unflatten(space, x) for x in _xs] for _xs in xs])

def Polyak_update(model: nn.Module, target_model: nn.Module, tau):
    for param, target_param in zip(model.parameters(), target_model.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

def Hard_update(model: nn.Module, target_model: nn.Module, tau):
    for param, target_param in zip(model.parameters(), target_model.parameters()):
        target_param.data.copy_(param.data)

    
TARGET_UPDATE = {
    'Polyak_update': Polyak_update,
    'Hard_update': Hard_update
}


def build_mlp(hidden_layers, activation=nn.ReLU):
    print(hidden_layers)
    layers = []
    prev_dim = hidden_layers[0]

    # Add hidden layers
    for hidden_dim in hidden_layers[1:-1]:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(activation())
        prev_dim = hidden_dim

    # Add final output layer
    layers.append(nn.Linear(prev_dim, hidden_layers[-1]))

    return nn.Sequential(*layers)

def stack_action_spaces(*action_spaces):

    if isinstance(action_spaces[0], spaces.Discrete):
        return spaces.MultiDiscrete([sp.n for sp in action_spaces])
    elif isinstance(action_spaces[0], spaces.Box):
        min_, max_ = [], []
        for sp in action_spaces:
            min_.append(sp.low)
            max_.append(sp.high)
        min_ = np.concatenate(min_).reshape(-1)
        max_ = np.concatenate(max_).reshape(-1)
        return spaces.Box(low=min_, high=max_)
    raise RuntimeError(f"Can stack {action_spaces}")

def mask_before_last(mask: np.ndarray) -> np.ndarray:
    idx = mask.cumsum(1).argmax(1)      # first 1 (start of sequence)
    lengths = mask.sum(1)               # lengths
    before_last = lengths - 2           # index of before-last
    out = np.zeros_like(mask)
    out[np.arange(len(mask)), before_last] = 1
    return out

class Policy(nn.Module):
    """
    """
    def __init__(self, 
                 observation_space: spaces.Box, 
                 action_space: spaces.Space, 
                 hidden_layers: List[int],
                 activation = nn.ReLU,
                 input_tau_i = False,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert isinstance(observation_space, spaces.Box) and len(observation_space.shape)==1, "The observation space must be box and flatten"
        self.action_space = action_space
        self.observation_space = observation_space
        self.input_dim = flatten_space(self.observation_space).shape[0]
        self.output_dim = flatten_space(self.action_space).shape[0]


        self.use_gaussian = isinstance(self.action_space, spaces.Box)
        self.is_categorical = isinstance(self.action_space, spaces.Discrete)
        if self.use_gaussian:
            self.output_dim *=2
        assert self.use_gaussian or self.is_categorical, "Only continuous (as learnable gaussian) or categorical is support!"
        hidden_layers = [self.input_dim, *hidden_layers, self.output_dim]

        self.gru = None
        
        if input_tau_i:
            # this is recurrent policy implementation
            self.gru = nn.GRU(input_size=hidden_layers[0], hidden_size=hidden_layers[1], batch_first=True)
            hidden_layers = hidden_layers[2:]
         
        
        self.neural = build_mlp(hidden_layers=hidden_layers, 
                                activation=activation)

    def forward(self, obs: torch.Tensor, 
                last_state = None, 
                action_mask=None, 
                sample=True,
                **kwargs) -> Tuple:
        if self.gru is not None:
            obs, last_state = self.gru.forward(obs, last_state)
        
        nn_output = self.neural(obs)
        if self.use_gaussian:
            mean, log_var = torch.chunk(nn_output, dim=-1, chunks=2)
            random = torch.randn_like(mean) if sample else torch.zeros_like(mean)

            var = log_var.clamp(-6, 10).exp()
            std = torch.exp(0.5 * log_var.clamp(-6, 10))
            actions = mean + random * std
            log_probs = -0.5 * (torch.log(2 * torch.pi) + log_var + (actions - mean)**2 / var)
            log_probs = log_probs.sum(dim=-1, keepdims=True)
        else:
            logits = nn_output if action_mask is None else  nn_output+ (action_mask.float() + 1e-45).log()
            log_probs = F.log_softmax(logits, dim=-1)
            if sample:
                dist = torch.distributions.Categorical(probs=log_probs.exp())
                actions = dist.sample()
            else:
                actions = log_probs.argmax(dim=-1, keepdim=False)
            # if action_mask is not None:
            #     # after computing `actions`
            #     assert torch.all(action_mask.gather(-1, actions.unsqueeze(-1)).squeeze(-1) > 0), \
            #         "Sampled an invalid action!"

            actions = F.one_hot(actions, num_classes=log_probs.shape[-1]).float()
            log_probs = (actions * log_probs.exp()).sum(dim=-1, keepdims=True)
        return actions, log_probs, last_state
        
    def get_log_probs(self, obs: torch.Tensor, 
                      actions: torch.Tensor,
                last_state = None, 
                action_mask=None, 
                **kwargs) -> torch.Tensor:
        if self.gru is not None:
            obs, last_state = self.gru.forward(obs, last_state)
        
        nn_output = self.neural(obs)
        if self.use_gaussian:
            mean, log_var = torch.chunk(nn_output, dim=-1, chunks=2)
            # random = torch.randn_like(mean) if sample else torch.zeros_like(mean)

            var = log_var.clamp(-6, 10).exp()
            # std = torch.exp(0.5 * log_var.clamp(-6, 10))
            log_probs = -0.5 * (torch.log(2 * torch.pi) + log_var + (actions - mean)**2 / var)
            log_probs = log_probs.sum(dim=-1, keepdims=True)
        else:
            logits = nn_output if action_mask is None else  nn_output+ (action_mask.float() + 1e-45).log()
            log_probs = F.log_softmax(logits, dim=-1)
            # if actions.shape[-1] != 
            # actions = F.one_hot(actions, num_classes=log_probs.shape[-1]).float()
            log_probs = (actions.float() * log_probs.exp()).sum(dim=-1, keepdims=True)
        return log_probs
        
    def reset_state(self, last_state:Any, done: Union[np.ndarray, torch.Tensor, None]):
            
        if isinstance(last_state, torch.Tensor) \
            and done is not None and len(done.shape)==1 \
            and last_state.shape[-2] == done.shape[0]:
            last_state[:, done].zero_()
            return last_state
        return None
        
class Critic(nn.Module):
    """
    """
    def __init__(self, 
                 observation_space: spaces.Box, 
                 action_space: spaces.Space, 
                 hidden_layers: List[int],
                 activation = nn.ReLU,
                 input_tau_i = False,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert isinstance(observation_space, spaces.Box) and len(observation_space.shape)==1, "The observation space must be box and flatten"
        self.action_space = action_space
        self.observation_space = observation_space

        self.is_discrete = isinstance(self.action_space, spaces.Discrete) or isinstance(self.action_space, spaces.MultiDiscrete) 
        if not self.is_discrete:
            self.input_dim = flatten_space(self.observation_space).shape[0] + flatten_space(self.action_space).shape[0]
            self.output_dim = 1
        else:
            self.input_dim = flatten_space(self.observation_space).shape[0]
            self.output_dim = flatten_space(self.action_space).shape[0]


        hidden_layers = [self.input_dim, *hidden_layers, self.output_dim]

        self.gru = None
        
        if input_tau_i:
            # this is recurrent policy implementation
            self.gru = nn.GRU(input_size=hidden_layers[0], hidden_size=hidden_layers[1])
            hidden_layers = hidden_layers[2:]
         
        
        self.neural = build_mlp(hidden_layers=hidden_layers, 
                                activation=activation)

    def forward(self, obs: torch.Tensor, 
                last_state = None, 
                act: torch.Tensor = None, 
                **kwargs) -> Tuple:
        if not self.is_discrete:
            assert act is not None
            oa = torch.cat([obs, act], dim=-1)
            if self.gru is not None:
                oa, last_state = self.gru.forward(oa, last_state)
            return self.neural(oa), None # last dim = 1
        else:
            if self.gru is not None:
                obs, last_state = self.gru.forward(obs, last_state)
            Qs = self.neural(obs)# last dim = # actions
            if act is not None:
                return (Qs * act).sum(dim=-1, keepdims=True), Qs.amax(dim=-1, keepdims=True)
            else:
                return Qs, Qs.amax(dim=-1, keepdims=True)

class HyperMix(nn.Module):
    def __init__(self, state_dim, n_agents, hidden_layers=[256,], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.neural = build_mlp(
            [self.state_dim, *hidden_layers, self.n_agents]
        )

    def forward(self, state: torch.Tensor, q_values: torch.Tensor = None):
        """q_values must be stacked on the last dim"""
        w = torch.abs(self.neural(state))

        if q_values is None:
            return w
        # print(w.shape)
        # print(q_values.shape)   

        q_tol = (w * q_values).sum(dim=-1, keepdim=True)
        return w, q_tol


@dataclass
class Config:
    # a config of all hyper-params
    seed: int = 0
    n_rollout_worker: int = 10
    n_evalrollout_worker: int = 10
    each_rollout_worker_resources: dict = field(default_factory=lambda: {"n_cpus": 2})
    devices: Optional[List[str]] = None
    hidden_layers: list = field(default_factory=lambda: [256,256])

    lr: float = 3e-5
    gamma: float = 0.99
    need_state: bool = True
    train_sample: str = 'sample_trajectories:8' # can be "sample_transitions:32"
    discover: str = 'eps_greedy:0.05'
     
class LearnerGroup:
    agents: List[str]
    global_step: int

    def load_state_dict(self, state_dict):
        raise NotImplementedError
    
    def state_dict(self,):
        raise NotImplementedError
    
    def update_inference_state(self, last_state: Optional[Dict] = None, Done: Optional[MulBatched] = None):
        return {agent_id: None for agent_id in self.agents}
    
    def reset_last_state(self, LastState: dict, batch_id: int, agent_id):
        # overwrite for each rnn type if needed
        if isinstance(LastState[agent_id][batch_id], torch.Tensor):
            LastState[agent_id][batch_id].zero_()
        return LastState

    def inference(self, obs: MulObs, sample: bool, last_state: Any, **kwargs)-> Tuple[MulAction, MulBatched, Any]:
        raise NotImplementedError

    def optimize_step(self, *args, **kwargs)->dict:
        raise NotImplementedError

class Algorithm:
    policy: LearnerGroup
    env_kwargs: dict
    train_rollout_group: RolloutGroup
    eval_rollout_group: RolloutGroup

    rollout_scale: int
    agents: List[str]
    env_construct_fn: Callable[..., ParallelEnv]
    env_kwargs: dict
    config: Config


    def __init__(self, 
                 config: Config,
                env_construct_fn: Callable[..., ParallelEnv],
                env_kwargs: dict = {}) -> None:
        self.env_construct_fn = env_construct_fn
        self.config = config
        self.env_kwargs = env_kwargs
        self.train_rollout_group = self.construct_rollout_group(need_buffer=True)
        self.eval_rollout_group = self.construct_rollout_group(need_buffer=False, is_eval=True)
        self.agents = self.train_rollout_group.agents
        self.learner_group = self.construct_learner_group()
        self._Obs = None
        self._Info = None
        self._LastState = None
        self._Done = None

    def construct_learner_group(self, )-> LearnerGroup:
        raise NotImplementedError()
    
  
    @torch.no_grad
    def eval_fn(self, 
                 n_episode: int=32,
                 verbose: Optional[Callable[[int], Any]] = None,) -> Dict:
        """
        Perform one-episode evaluation in an auto reset scheme
        """
        import math
        import copy
        eval_rollout_group = self.eval_rollout_group
        inference_fn = self.learner_group.inference
        LastIState = self.learner_group.update_inference_state(None) # for rnn policy
        Obs, Info = eval_rollout_group.reset()

        Done = {
            agent_id: (1-Obs[agent_id].batch_mask).astype(bool)
            for agent_id in eval_rollout_group.agents
        } # batched all done signal

        total_eps = 0
        AllRew = {
            agent_id: []
            for agent_id in eval_rollout_group.agents
        }
        AllLastInfor = {
            agent_id: []
            for agent_id in eval_rollout_group.agents
        }
        AllEpsLen = {
            agent_id: []
            for agent_id in eval_rollout_group.agents
        }

        TempRew = {
            agent_id: np.zeros_like(Obs[agent_id].batch_mask).astype(float)
            for agent_id in eval_rollout_group.agents
        }
        TempLastInfor = {
            agent_id: [None for _ in range(eval_rollout_group.n_worker)]
            for agent_id in eval_rollout_group.agents
        }
        TempEpsLen = {
            agent_id: np.zeros_like(Obs[agent_id].batch_mask).astype(float)
            for agent_id in eval_rollout_group.agents
        }


        while not (total_eps>= n_episode):
            if verbose is not None:
                verbose.update(1)
            Act, LogProb, LastIState = inference_fn(eval_rollout_group=eval_rollout_group, 
                                                obs=Obs, 
                                                sample=False,
                                                last_state=LastIState)
            Obs, Rew, Ter, Trunc, _ = eval_rollout_group.step(actions=Act, log_probs=LogProb)
            _, Info = eval_rollout_group.get_last_b4_reset()
           
            ThisStepDone = {
                agent_id: np.logical_and(np.logical_or(Ter[agent_id].data.astype(bool), Trunc[agent_id].data.astype(bool)),
                                         Ter[agent_id].batch_mask.astype(bool)) 
                for agent_id in eval_rollout_group.agents
            }

            # we add not done yet info:
            for agent_id in eval_rollout_group.agents:
                TempRew[agent_id][~Done[agent_id]] += Rew[agent_id][~Done[agent_id]]
                TempEpsLen[agent_id][~Done[agent_id]] += 1
                for idx in np.flatnonzero(~Done[agent_id]):
                    TempLastInfor[agent_id][idx.item()] = Info[idx.item()][agent_id]

            # per agent done
            Done = {
                agent_id: np.logical_or(Done[agent_id].astype(bool), 
                                        ThisStepDone[agent_id].astype(bool)) 
                for agent_id in eval_rollout_group.agents
            }
            AllDone = np.array([all(Done[agent_id][worker_id] for agent_id in eval_rollout_group.agents)
                        for worker_id in range(eval_rollout_group.n_worker)])
            
            # now we reset doned episodes
            for idx in np.flatnonzero(AllDone):
                for agent_id in eval_rollout_group.agents:
                    # add    
                    AllRew[agent_id].append(TempRew[agent_id][idx].item())
                    AllLastInfor[agent_id].append(copy.copy(TempLastInfor[agent_id][idx.item()]))
                    AllEpsLen[agent_id].append(TempEpsLen[agent_id][idx])

                    # reset
                    TempRew[agent_id][idx] = 0.0
                    TempLastInfor[agent_id][idx.item()] = None
                    TempEpsLen[agent_id][idx] = 0.0
                    Done[agent_id][idx] = False
            total_eps += AllDone.astype(int).sum()

        AllRew = {agent_id: AllRew[agent_id][:n_episode] for agent_id in eval_rollout_group.agents}  
        AllEpsLen = {agent_id: AllEpsLen[agent_id][:n_episode] for agent_id in eval_rollout_group.agents}  
        AllLastInfor = {agent_id: AllLastInfor[agent_id][:n_episode] for agent_id in eval_rollout_group.agents}  

        return {
            'return': AllRew, 'eps_len': AllEpsLen, "last_info": AllLastInfor
        }

    def construct_rollout_group(self, need_buffer=True, is_eval=False) -> RolloutGroup:
        rollout_group = RolloutGroup(
            env_construct_fn=self.env_construct_fn,
            contain_buffer=need_buffer,
            scale=self.config.n_rollout_worker if not is_eval else self.config.n_evalrollout_worker,
            seed=self.config.seed,
            need_state=self.config.need_state,
            each_resource=self.config.each_rollout_worker_resources,
            env_kwargs=self.env_kwargs)
        return rollout_group
    
    def get_sample(self)-> Dict:
        method, batch_size = self.config.train_sample.split(':')
        batch_size = int(batch_size)
        return getattr(self.train_rollout_group, method)(batch_size)


    def collect_random_n_step(self, n: int, 
                      verbose: Optional[Callable[[int], Any]] = None,):
        if self._Obs is None:
            # is first step
            self._Obs, self._Info = self.train_rollout_group.reset()

        for step in range(n):
            # do one rollout step
            Act, LogProb = self.train_rollout_group.sample_action()
            Obs, Rew, Ter, Trunc, Info = self.train_rollout_group.step(actions=Act,
                                                                      log_probs=LogProb)
            if verbose is not None:
                verbose.update(1)

            # prepare for next step
            self._Obs, self._Info = Obs, Info

    def optimize_n_steps(self, n: int, 
                      verbose: Optional[Callable[[int], Any]] = None,
                      **kwargs) -> dict:
        train_metrics = []
        for step in range(n):
            train_metric = self.learner_group.optimize_step(**self.get_sample())
            train_metrics.append(train_metric)
            if verbose is not None:
                verbose.update(1)
        return {k: np.sum([v[k] for v in train_metrics]) \
                      for k in train_metrics[0]}
    
    def wrap_with_discovery(self, Acts: MulAction, LogProbs: MulBatched, ):
        current_eps = 0.0
        if 'linear_epsgreedy' in self.config.discover:
            _, start_eps, end_eps, start_step, end_step = self.config.discover.split(':')
            start_eps, end_eps, start_step, end_step = float(start_eps), float(end_eps), float(start_step), float(end_step)
            current_r = (self.learner_group.global_step - start_step)/(end_step - start_step)
            current_r = max(0.0, min(current_r, 1.0))
            current_eps = start_eps + current_r * (end_eps - start_eps)
        elif 'epsgreedy' in self.config.discover:
            _, current_eps = self.config.discover.split(':')
            current_eps = float(current_eps)
        if 1 > current_eps > 0:
            random_Acts, random_LogProbs = self.train_rollout_group.sample_action()
            for agent_id in Acts.keys():
                if self.train_rollout_group.random_generator.random()<= current_eps:
                    Acts[agent_id] = random_Acts[agent_id]
                    LogProbs[agent_id] = random_LogProbs[agent_id]

        return Acts, LogProbs
        



    
    def optimize_while_rollout(self, n: int, 
                      verbose: Optional[Callable[[int], Any]] = None,
                      **kwargs) -> dict:
        """ Return the off-policy training for n steps """
        if self._Obs is None:
            # is first step
            self._Obs, self._Info = self.train_rollout_group.reset()
        if self._LastState is None:
            self._LastState = self.learner_group.update_inference_state(None) # for rnn policy

        train_metrics = []
        for step in range(n):
            # do one rollout step
            Act, LogProb, self._LastState = self.learner_group.inference(self._Obs, 
                                                                   sample=False,
                                                                   last_state=self._LastState)
            Act, LogProb = self.wrap_with_discovery(Act, LogProb)
            Obs, Rew, Ter, Trunc, Info = self.train_rollout_group.step(actions=Act,
                                                                      log_probs=LogProb)
            train_metric = self.learner_group.optimize_step(**self.get_sample())
            train_metrics.append(train_metric)
            if verbose is not None:
                verbose.update(1)

            # prepare for next step
            self._Obs, self._Info = Obs, Info
            Done = MulBatched({
                agent_id: BatchBasic(
                    data=np.logical_or(Ter[agent_id].data.astype(bool), Trunc[agent_id].data.astype(bool)),
                    batch_mask=np.logical_or(Ter[agent_id].batch_mask.astype(bool), Trunc[agent_id].batch_mask.astype(bool))
                ) for agent_id in self.agents
            })
            self._LastState = self.learner_group.update_inference_state(last_state=self._LastState, Done=Done)
                    
            # we reset the state to 0 for new episode, because rollout group is auto reset
        return {k: np.sum([v[k] for v in train_metrics]) \
                      for k in train_metrics[0]}

    
    def state_dict(self) -> dict:
        return {}
    
    def load_state_dict(self, state_dict: dict)-> bool:
        return True
    
class QmixLearner(LearnerGroup):
    def __init__(self, 
                 observation_spaces: Dict[str, spaces.Space],
                 action_spaces: Dict[str, spaces.Space],
                 devices: Dict[str, str],
                 state_dim: int,
                 target_update: str,
                 input_tau_i: bool = False,
                 gamma: float = 0.99,
                 lr: float = 3e-5,
                 hyper_hidden_layers: List = [256,],
                 **kwargs) -> None:
        super().__init__()
        self.input_tau_i = input_tau_i
        self.agents = list(action_spaces.keys())
        self.lr = lr
        self.devices = devices

        self.policies: Dict[str, Policy] = {}
        self.critics: Dict[str, Critic] = {}
        self.observation_spaces: Dict[str, spaces.Space] = {}
        self.target_policies: Dict[str, Policy] = {}
        self.target_critics: Dict[str, Critic] = {}

        t_update_fn, interval, tau = target_update.split(':')
        self.t_update_fn = TARGET_UPDATE[t_update_fn]
        self.t_update_interval = int(interval)
        self.tau = float(tau)

        self.optimizers = {}
        self.action_spaces = action_spaces


        for agent_id in self.agents:
            device = devices[agent_id]
            if isinstance(observation_spaces[agent_id], spaces.Dict)\
                and "observation" in observation_spaces[agent_id].spaces:
                # observation_spaces[agent_id]: spaces.Dict
                flatten_obs_space = flatten_space(observation_spaces[agent_id].spaces["observation"])
            elif isinstance(observation_spaces[agent_id], dict)\
                and "observation" in observation_spaces[agent_id]:
                flatten_obs_space = flatten_space(observation_spaces[agent_id]["observation"])
            else:
                # print('-'*30)
                # print(isinstance(observation_spaces[agent_id], spaces.Dict) \
                # and "observation" in observation_spaces[agent_id].spaces)
                flatten_obs_space = flatten_space(observation_spaces[agent_id])
            self.observation_spaces[agent_id] = flatten_obs_space
            self.policies[agent_id] = Policy(observation_space=flatten_obs_space,
                                              action_space=action_spaces[agent_id],
                                              input_tau_i=input_tau_i, **kwargs).to(device)
            self.critics[agent_id] = Critic(observation_space=flatten_obs_space,
                                              action_space=action_spaces[agent_id],
                                              input_tau_i=input_tau_i, **kwargs).to(device)
            self.target_policies[agent_id] = copy.deepcopy(self.policies[agent_id])
            self.target_critics[agent_id] = copy.deepcopy(self.critics[agent_id])
            self.optimizers[agent_id] = torch.optim.Adam(
                (*self.policies[agent_id].parameters(), *self.critics[agent_id].parameters()),
                lr=self.lr
            )

        self.hyper_net = HyperMix(state_dim=state_dim, 
                                  n_agents=len(self.agents), 
                                  hidden_layers=hyper_hidden_layers).to(devices['hyper_net'])
        
        self.optimizers['hyper_net'] = torch.optim.Adam(
            self.hyper_net.parameters(),
                lr=self.lr
            )
        self.target_hyper_net = copy.deepcopy(self.hyper_net)

        self.U_space = stack_action_spaces(*[self.action_spaces[agent_id] for agent_id in self.agents])
        self.is_discrete = isinstance(self.U_space, spaces.MultiDiscrete)
        self.gamma = gamma
        self.global_step = 0

    
    def load_state_dict(self, state_dict)-> bool:
        raise NotImplementedError

    def update_inference_state(self, last_state: Optional[Dict[str, torch.Tensor]] = None, 
                               Done: Optional[MulBatched] = None):

        if not self.input_tau_i or Done is None or last_state is None:
            return {agent_id: None for agent_id in self.agents}
    
        done_or_wait = {agent_id: np.logical_or(Done[agent_id].data.astype(bool), \
                                                (1-Done[agent_id].batch_mask).astype(bool)
                                                ).reshape(-1) 
                for agent_id in Done}

        return {agent_id: self.policies[agent_id].reset_state(last_state=last_state[agent_id], 
                                                              done=done_or_wait[agent_id]) 
                for agent_id in last_state} # zero out done or wait

    def inference(self, 
                  obs: MulObs, 
                  sample: bool, 
                  last_state: Any,
                  **kwargs)-> Tuple[MulAction, MulBatched, Optional[Dict]]:
        
        batch_mask = {
            agent_id: np.logical_and(o.batch_mask, o.action_mask.sum(axis=-1)>0) \
                if o.action_mask is not None else o.batch_mask
            for agent_id, o in obs.items() 
        }
        unflatten_fn = batch_unflatten # just batch unflatten
        unflatten_log_prob = lambda x: x
        if self.input_tau_i:
            # recurrent policy, need to augment the sequence dim
            obs = MulObs({
                agent_id: BatchedObs(data=o.data[:, None, ...],
                                     batch_mask=o.batch_mask[:, None, ...],
                                     is_sequence=True,
                                     action_mask=o.action_mask[:, None, ...] if o.action_mask is not None else None,
                                     position_mask=o.batch_mask[:, None, ...]) 
                for agent_id, o in obs.items()
            })
            unflatten_fn = lambda spc, x: batch_seq_unflatten(spc, x)[:, 0, ...] # unflatten seq and take the first position
            unflatten_log_prob = lambda x: x[:, 0, ...]


        dec_results = {}
        def run_decentralized_fw(agent_id):
            policy = self.policies[agent_id]
            _obs = obs[agent_id].data
            action_mask = obs[agent_id].action_mask
            _last_state = last_state[agent_id] if last_state is not None and agent_id in last_state else None
            def to_torch(any):
                return torch.as_tensor(any, device=self.devices[agent_id], dtype=torch.float32)
            with torch.no_grad():
                _obs = to_torch(_obs)
                action_mask = to_torch(action_mask) if action_mask is not None else None
                _last_state = to_torch(_last_state) if _last_state is not None else None
                actions, log_probs, _last_state = policy(obs=_obs, 
                                                        last_state=_last_state,
                                                        sample=sample, 
                                                        action_mask=action_mask)

            return dict(actions=actions.cpu().numpy(), 
                        agent_id=agent_id,
                        log_probs=log_probs.cpu().numpy(), 
                        last_state=_last_state)
        
        with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
            futures = [executor.submit(run_decentralized_fw, agent_id)
                   for agent_id in obs]
            for f in as_completed(futures):
                _r = f.result()
                dec_results[_r['agent_id']] = _r


        Act = MulAction({
            agent_id: BatchBasic(
                data=unflatten_fn(self.action_spaces[agent_id],
                                      dec_results[agent_id]['actions']),
                batch_mask=batch_mask[agent_id]
            ) 
            for agent_id in dec_results
        })
        LogProb = MulBatched({
            agent_id: BatchBasic(
                data=unflatten_log_prob(dec_results[agent_id]['log_probs']),
                batch_mask=batch_mask[agent_id]
            ) 
            for agent_id in dec_results
        })
        LastIState = {
            agent_id: dec_results[agent_id]['last_state'] for agent_id in dec_results
        }

        return Act, LogProb, LastIState

    # def forward_policy(self, agent_id, obs: np.ndarray)
    def optimize_step(self, 
                      obs: MulObs,
                      state: BatchBasic,
                      act: MulAction,
                      log_prob: MulBatched,
                      rew:  MulBatched,
                      ter: MulBatched,
                  *args, **kwargs):
        # rl_info = {}
        # loss = torch.zeros(1)
        if self.global_step % self.t_update_interval ==0:
            self.t_update_fn(self.hyper_net, self.target_hyper_net, self.tau)


        def forward_policy_and_q(agent_id):

            policy, critic = self.policies[agent_id],  self.critics[agent_id]

            def to_torch(any):
                return torch.as_tensor(any, device=self.devices[agent_id], dtype=torch.float32)
            _obs = to_torch(obs[agent_id].data)
            _flatten_act = to_torch(batch_seq_flatten(self.action_spaces[agent_id], act[agent_id].data))
            _mask = to_torch(obs[agent_id].position_mask)[..., None]
            _action_mask = to_torch(obs[agent_id].action_mask) if obs[agent_id].action_mask is not None else None

            policy_loss = None
            log_probs = policy.get_log_probs(_obs, _flatten_act, action_mask=_action_mask)
            Q, max_val = critic(obs=_obs, act=_action_mask)
            # we mul with each-agent mask here
            # print(_flatten_act.shape)
            # print(Q.shape)
            # print(log_probs.shape)
            # print(_mask.shape)
            policy_loss = -(log_probs * Q * _mask)/(_mask.sum() + 1e-4)
            policy_loss = policy_loss.sum()

            if self.global_step % self.t_update_interval == 0:
                self.t_update_fn(self.policies[agent_id], self.target_policies[agent_id], self.tau)
                self.t_update_fn(self.critics[agent_id], self.target_critics[agent_id], self.tau)

            return dict(Q=Q * _mask, 
                        max_val=max_val * _mask, 
                        log_probs=log_probs, 
                        mask=_mask, 
                        policy_loss=policy_loss, agent_id=agent_id)
        

        def get_V_next_step(agent_id):
            def to_torch(any):
                return torch.as_tensor(any, device=self.devices[agent_id], dtype=torch.float32)
            with torch.no_grad():
                policy, critic = self.target_policies[agent_id],  self.target_critics[agent_id]
                _obs = to_torch(obs[agent_id].data)
                _action_mask = to_torch(obs[agent_id].action_mask) if obs[agent_id].action_mask is not None else None

                if critic.is_discrete:
                    _, max_val = critic(obs=_obs, act=None)
                else:
                    # use target policy to estimate in continuous space
                    flatten_act, _ , _ = policy.forward(_obs, action_mask=_action_mask, sample=True)
                    max_val, _ = critic(obs=obs, act=flatten_act)

            return dict(max_val=max_val, agent_id=agent_id)

        dec_results = {}
        target_dec_results = {}

        

        with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
            futures = [executor.submit(forward_policy_and_q, agent_id)
                   for agent_id in self.agents]
            
            futures2 = [executor.submit(get_V_next_step, agent_id)
                   for agent_id in self.agents]
            
            for  f in  as_completed(futures):
                _r = f.result()
                dec_results[_r['agent_id']] = _r

            for f in as_completed(futures2):
                _r = f.result()
                target_dec_results[_r['agent_id']] = _r

        # now we stack for the mix

        to_torch = lambda x: torch.as_tensor(x, device=self.devices['hyper_net'], dtype=torch.float32)
        State = to_torch(state.data)
        # If any non-mask
        Mask = to_torch(state.position_mask)[..., None].float() # sequence-size mask
        AllTer = np.zeros_like(state.position_mask).astype(bool)
        R = np.zeros_like(state.position_mask).astype(float)

        for agent_id in ter.keys():
            AllTer = np.logical_or(AllTer, 
                                   (ter[agent_id].data.astype(float) * ter[agent_id].position_mask.astype(float)).astype(bool))
            R += rew[agent_id].data.astype(float) * rew[agent_id].position_mask.astype(float)
        AllTer = AllTer.astype(float)
        AllTer = to_torch(np.cumsum(AllTer[:, ::-1], axis=-1)[:, ::-1] == 1)[..., None] # get the last terminal position
        R = to_torch(R)[..., None]


        Q = torch.cat([dec_results[agent_id]['Q'].to(device=self.devices['hyper_net']) for agent_id in self.agents], dim=-1)
        w, Q_tol = self.hyper_net(state=State, q_values=Q)

        with torch.no_grad():
            Q_tar = torch.cat([target_dec_results[agent_id]['max_val'].to(device=self.devices['hyper_net'])
                                for agent_id in self.agents], dim=-1)
            _, Q_tol_tar = self.target_hyper_net(state=State, q_values=Q_tar)
            Q_target = R[:, :-1, :] + self.gamma * AllTer[:, :-1, :] * Q_tol_tar[:, 1:, :]

        Td_error = (Q_tol[:, :-1, :] - Q_target)**2 * Mask[:, :-1, :]/(Mask[:, :-1, :].sum() + 1e-4)
        Td_error = Td_error.mean()

        policy_loss = [dec_results[agent_id]['policy_loss'].to(device=self.devices['hyper_net']) for agent_id in self.agents]
        policy_loss = torch.stack(policy_loss).mean()

        total_loss = Td_error + policy_loss

        for k in self.optimizers:
            self.optimizers[k].zero_grad()
        total_loss.backward()
        for k in self.optimizers:
            self.optimizers[k].step()

        self.global_step += 1
        return {
            'loss': total_loss.item(),
            'Td_error': Td_error.item(),
            'policy_loss': policy_loss.item(),
            'total_transition': Mask.sum().item(),
        }

@dataclass
class QMixConfig(Config):
    state_dim: int = 0
    input_tau_i: bool = True
    target_update: str = 'Polyak_update:32:0.005'
    hyper_hidden_layers: list = field(default_factory=lambda: [256,])

class QMix(Algorithm):
    config: QMixConfig
        
    def construct_learner_group(self,):
        if torch.cuda.is_available() and self.config.devices is None:
            devices = itertools.cycle([f'cuda:{i}' for i in range(torch.cuda.device_count())])
        else:
            devices = itertools.cycle(self.config.devices)
        self.config.devices = {k: next(devices) for k in [*self.agents, 'hyper_net']}
        self.config.state_dim = self.train_rollout_group.state_dim
        learner_group = QmixLearner(observation_spaces=self.train_rollout_group.observation_spaces[0],
                                    action_spaces=self.train_rollout_group.action_spaces[0],
                                    devices=self.config.devices,
                                    target_update=self.config.target_update,
                                    state_dim = self.config.state_dim,
                 input_tau_i = self.config.input_tau_i,
                 gamma = self.config.gamma,
                 lr = self.config.lr,
                 hyper_hidden_layers = self.config.hyper_hidden_layers,
                hidden_layers = self.config.hidden_layers,)
        return learner_group


    
    def state_dict(self)-> dict:
        return {}
    
    def load_state_dict(self, state_dict: dict)-> bool:
        return True
    
if __name__=='__main__':
    from .hparams import HPARAMS
    from .register_env import ENV, READ_RESULT
    from pathlib import Path
    import argparse
    import os
    import json
    from tqdm import tqdm
    import math


    algo = Path(__file__).stem
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, help="Environment name in register_env.py", default="smac3m")
    parser.add_argument('--seed', type=int, help="Random seed", default=42)
    parser.add_argument('--pre-collect', type=int, help="Pre-trained # data-collecting steps", default=500)
    parser.add_argument('--n', type=int, help="Total # training steps", default=2_000_000)
    args = parser.parse_args()

    haprams = HPARAMS.get((args.env, algo), {})
    haprams['seed'] = args.seed
    env_construct_fn = ENV[args.env]
    on_result_callback = READ_RESULT[args.env] if args.env in READ_RESULT else lambda x: x

    qmix = QMix(
        config= QMixConfig(**haprams),
        env_construct_fn=env_construct_fn,
    )
    # qmix.eval_fn(verbose=tqdm("Rollouting..."))
    if args.pre_collect > 0:
        qmix.collect_random_n_step(args.pre_collect, verbose=trange(args.pre_collect,desc="Pre-collecting data ..."))

    def eval_and_save(step, file_path):
        mode = "a" if os.path.exists(file_path) else "w"
        eval_metrics = on_result_callback(qmix.eval_fn(n_episode=32, verbose=tqdm(desc="Evaluating...")))
        eval_metrics["step"] = step
        with open(file_path, mode, encoding="utf-8") as f:
            f.write(json.dumps(eval_metrics, ensure_ascii=False) + "\n")
        return eval_metrics
    
    result_files = f"{args.env}_{algo}_results.jsonl"
    
    eval_metrics = eval_and_save(0, result_files)
    print(eval_metrics)

    tbar = trange(args.n,desc="Training...")

    for i in range(1, math.ceil(args.n/1000)+1):
        qmix.optimize_while_rollout(1000, verbose=tbar)
        eval_metrics = eval_and_save(i*1000, result_files)
        print(eval_metrics)
    


