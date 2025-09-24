
from typing import *
import ray
from pettingzoo import ParallelEnv
from gymnasium import spaces
import torch
import numpy as np
from dataclasses import dataclass, field
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
import inspect


from gymnasium.spaces.utils import flatten_space, flatten, unflatten



def batch_unflatten(space: spaces.Space, xs: np.ndarray)->np.ndarray:
    return np.array([unflatten(space, x) for x in xs])



@ray.remote
class RolloutWorker:
    # we flat the obs space here
    def __init__(self, 
                 seed: int,
                 env_construct_fn: Callable[..., ParallelEnv], 
                env_kwargs: dict = {}) -> None:
        self.random_generator = np.random.default_rng(seed=seed)
        self.env: ParallelEnv = env_construct_fn(**env_kwargs)
        self._last_obs = None
        self._last_info = None
        self.agents = self.env.agents
        self._ter = None
        self._trunc = None

    def get_state_dim(self,) -> tuple:
        return self.get_state().shape
    
    def _flatten_obs_space(self, space: spaces.Space):
        if isinstance(space, spaces.Dict)\
            and "observation" in space.spaces\
            and "action_mask" in space.spaces:
            return spaces.Dict({
                "observation": flatten_space(space.spaces["observation"]),
                "action_mask": space.spaces["action_mask"],
            })
        else:
            return flatten_space(space)
        
    def _flatten_obs(self, obs):

        def _flat_each(_obs, agent_id):
            if isinstance(_obs, dict)\
                and "observation" in _obs\
                and "action_mask" in _obs:
                return {
                    "observation": flatten(self.env.observation_space(agent=agent_id)["observation"], _obs["observation"]),
                    "action_mask": _obs["action_mask"]
                }
            else:
                return flatten(self.env.observation_space(agent=agent_id), _obs)
        return {
            agent_id: _flat_each(obs[agent_id], agent_id) for agent_id in obs
        }

    def get_obs_spaces(self)-> Dict[str, spaces.Space]:
        return {
            agent_id: self._flatten_obs_space(self.env.observation_space(agent=agent_id)) 
            for agent_id in self.get_possible_agents()
        }
    
    def get_act_spaces(self)-> Dict[str, spaces.Space]:
        return {
            agent_id: self.env.action_space(agent=agent_id) 
            for agent_id in self.get_possible_agents()
        }

    def get_agents(self):
        return self.agents
    
    def get_possible_agents(self):
        return self.env.possible_agents
    
    def get_last_obs(self):
        return self._last_obs
        
    def state_dict(self,):
        return {'random_state': self.random_generator.bit_generator.state}
    
    def load_state_dict(self, state_dict):
        if 'random_state' not in state_dict:
            return False
        try:
            self.random_generator.bit_generator.state = state_dict['random_state']
            return True
        except:
            return False
    
    def reset(self, *args, **kwargs):
        if 'seed' in kwargs:
            del kwargs['seed']
        seed = self.random_generator.integers(int(1e9))
        self._ter   = {agent: False for agent in self.env.agents}
        self._trunc = {agent: False for agent in self.env.agents}
        obs, infor = self.env.reset(seed=seed, *args, **kwargs)
        self._last_obs = obs

        return self._flatten_obs(obs), infor 

    def done(self) -> dict:
        return {agent: self._ter[agent] or self._trunc[agent] for agent in self.agents}
    
    def all_done(self):
        return all(self.done().values())
    
    def step(self, actions):
        obs, rewards, truncs, ters, infos = self.env.step(actions)
        self._last_obs = obs
        self._last_info = infos
        self._ter.update(ters)
        self._trunc.update(truncs)

        return self._flatten_obs(obs), rewards, truncs, ters, infos
    
    def get_state(self):
        return self.env.state()

    def __del__(self):
        self.env.close()

def to_torch(data, **to_torch_kwargs):
    """
    Recursively convert a nested dict/list/tuple of numpy arrays
    (or other array-like objects) to torch.Tensors.

    Args:
        data: Nested structure (dict, list, tuple, np.ndarray, torch.Tensor, scalar).
        **to_torch_kwargs: Arguments passed to torch.as_tensor, e.g. dtype, device.

    Returns:
        Nested structure with same shape, but all arrays -> torch.Tensor.
    """
    if isinstance(data, dict):
        return {k: to_torch(v, **to_torch_kwargs) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        if isinstance(data[0], dict):
            try:
                data = collate_nested_dicts(data)
                return to_torch(data)
            except:
                pass
        if isinstance(data[0], list) and isinstance(data[0][0], dict):
            return to_torch([to_torch(_data) for _data in data])
        try:
            data = np.array(data)
            return torch.as_tensor(data, **to_torch_kwargs)
        except:
            pass
        converted = [to_torch(v, **to_torch_kwargs) for v in data]
        return tuple(converted) if isinstance(data, tuple) else converted
    elif isinstance(data, (np.ndarray, list, float, int)):
        try:
            return torch.as_tensor(data, **to_torch_kwargs)
        except:
            return data
    elif torch.is_tensor(data):
        # Already a tensor → move/cast if kwargs specify
        return data.to(**to_torch_kwargs) if to_torch_kwargs else data
    else:
        # Fallback: leave unchanged
        return data

def to_np(data, **to_np_kwargs):

    if isinstance(data, dict):
        return {k: to_np(v, **to_torch_kwargs) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        if isinstance(data[0], dict):
            try:
                data = collate_nested_dicts(data)
                return to_np(data)
            except:
                pass
        if isinstance(data[0], list) and isinstance(data[0][0], dict):
            return to_np([to_np(_data) for _data in data])
        try:
            return np.array(data)
        except:
            pass
        converted = [to_np(v, **to_np_kwargs) for v in data]
        return tuple(converted) if isinstance(data, tuple) else converted
    elif isinstance(data, (np.ndarray, list, float, int)):
        try:
            return np.asarray(data)
        except:
            return data
    elif torch.is_tensor(data):
        # Already a tensor → move/cast if kwargs specify
        return data.cpu().detach().numpy()
    else:
        # Fallback: leave unchanged
        return data
     
def get_shape(data):
    if isinstance(data, dict):
        return {k: get_shape(v) for k, v in data.items()}
    elif hasattr(data, 'shape'):
        return data.shape
    elif isinstance(data, (list, tuple)):
        return (len(data),)
    return 'no-shape'

def collate_nested_dicts(dict_list, ):
    """
    Collates data from a list of nested dictionaries, preserving structure.'
    This helps the neuralnet forward
    
    Args:
        dict_list (list of dict): List of nested dictionaries.
    
    Returns:
        dict: Nested dictionary with lists of values at the leaves.
    """
    def merge(d1, d2):
        for key, value in d2.items():
            if key in d1:
                if isinstance(value, dict) and isinstance(d1[key], dict):
                    merge(d1[key], value)
                else:
                    # Convert to list or append to existing list
                    if not isinstance(d1[key], list):
                        d1[key] = [d1[key]]
                    d1[key].append(value)
            else:
                d1[key] = value if not isinstance(value, dict) else merge({}, value)
        return d1

    result = {}
    for d in dict_list:
        merge(result, d)
    return result

def collate_with_mask(dict_list: List[Dict[str, Any]], 
                      list_ids: List[int],
                      total_id: int,
                      agents: List[str]):

    collated = {
        agent_id: [None for _  in range(total_id)] 
        for agent_id in agents
    }
    mask = {
        agent_id: [0, ] * total_id
        for agent_id in agents
    }
    fill_value = {}
    for data_dict, list_id in zip(dict_list, list_ids):
        for agent_id in data_dict.keys():
            if agent_id not in fill_value:
                fill_value[agent_id] = data_dict[agent_id]
            collated[agent_id][list_id] = data_dict[agent_id]
            mask[agent_id][list_id] = 1
    for agent_id in agents:
        for i in range(len(collated[agent_id])):
            if collated[agent_id][i] is None:
                collated[agent_id] = fill_value[agent_id]
    

    return collated, mask

def decolate_with_mask(
    collated: Dict[str, List[Any]],
    mask: Dict[str, List[int]]
) -> Tuple[List[Dict[str, Any]], List[int]]:
    """
    Reverse the operation of collate_with_mask.
    
    Args:
        collated: dict mapping agent_id -> list of values (with fill_value where missing).
        mask: dict mapping agent_id -> list of 0/1 indicating original presence.
    
    Returns:
        dict_list: List[Dict[str, Any]]
        list_ids: List[int]
    """
    total_id = len(next(iter(collated.values())))
    dict_list: List[Dict[str, Any]] = []
    list_ids: List[int] = []

    for idx in range(total_id):
        cur_dict: Dict[str, Any] = {}
        present = False
        for agent_id in collated.keys():
            if mask[agent_id][idx] == 1:
                cur_dict[agent_id] = collated[agent_id][idx]
                present = True
        if present:
            dict_list.append(cur_dict)
            list_ids.append(idx)

    return dict_list, list_ids


def print_shape(obj, indent=0, key=None):
    """
    Recursively print the shape/structure of a nested dict/list/tuple/ndarray/torch.Tensor.
    Shows variable name, types, and shapes in a tree-like format.
    """

    # Header (only for the top-level call)
    if indent == 0:
        # Try to extract the variable name from caller
        frame = inspect.currentframe().f_back
        call_line = inspect.getframeinfo(frame).code_context[0]
        var_name = call_line.split("print_shape(")[1].split(")")[0].split(",")[0].strip()
        print("\n" + "="*80)
        print(f"Structure of: {var_name}")
        print("="*80)

    prefix = " " * indent
    name = f"{key} : " if key is not None else ""

    if isinstance(obj, dict):
        print(f"{prefix}{name}dict ({len(obj)} keys)")
        for k, v in obj.items():
            print_shape(v, indent + 2, key=k)

    elif isinstance(obj, (list, tuple)):
        print(f"{prefix}{name}{type(obj).__name__} (length={len(obj)})")
        for i, item in enumerate(obj):
            print_shape(item, indent + 2, key=f"[{i}]")

    else:
        # Handle arrays / tensors
        if hasattr(obj, 'shape') and hasattr(obj, 'dtype'):
            print(f"{prefix}{name}{type(obj).__name__} shape={tuple(obj.shape)} dtype={obj.dtype}")
        elif hasattr(obj, 'shape'):
            print(f"{prefix}{name}{type(obj).__name__} shape={tuple(obj.shape)}")
        else:
            print(f"{prefix}{name}{type(obj).__name__}")



#========== data class for infererence ===========
# because we aim to infer in batch, we need mask some done agents until all done


AnyArray = Union[np.ndarray, torch.Tensor, List]
@dataclass
class BatchBasic:
    # the first dim is always the batch size (worker id)
    # shape: Tuple[int]
    data: np.ndarray
    batch_mask: np.ndarray
    position_mask: Optional[np.ndarray] = None
    is_sequence: bool = False

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)
    
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def dtype(self):
        return {"data": self.data.dtype, "batch_mask": self.batch_mask.dtype}

@dataclass
class BatchedObs(BatchBasic):
    action_mask: Optional[np.ndarray] = None

class MulBatched(dict):
    def debatch(self, batch_id: int) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        # print({agent: batch.batch_mask[batch_id] for agent, batch in self.items()})
        return {agent: batch.data[batch_id] for agent, batch in self.items() \
                if batch.batch_mask[batch_id]}
    
    def __getitem__(self, key: Any) -> BatchBasic:
        return super().__getitem__(key)
    
class MulObs(dict):
    def debatch(self, batch_id: int) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        output = {}
        for agent_id, batch in self.items():
            if batch.batch_mask[batch_id]:
                if batch.action_mask is None:
                   output[agent_id] = batch.data[batch_id]
                else:
                     output[agent_id] = {
                         "observation": batch.data[batch_id],
                         "action_mask": batch.action_mask[batch_id],
                     }
        return output
    
    def __getitem__(self, key: Any) -> BatchedObs:
        return super().__getitem__(key)

MulAction = MulBatched


def mulbatch_from_list(dict_list: List[Dict[str, Union[Dict[str, np.ndarray], np.ndarray]]],
                    list_ids: List[int],
                    total_id: int,
                    is_obs: bool,
                    fill_value: Dict) \
                    -> Union[MulBatched, MulObs]:
    # for list of single transitions, and update the fill value
    agents = fill_value.keys() # we only care the controlable agents
    collated = {
        agent_id: [None for _  in range(total_id)] 
        for agent_id in agents
    }
    optional_action_mask = {
        agent_id: [None for _  in range(total_id)] 
        for agent_id in agents
    }
    mask = {
        agent_id: [0, ] * total_id
        for agent_id in agents
    }
    useoptional_action_mask = {k: False for k in agents}
    for list_id in range(total_id):
        if list_id in list_ids:
            data_dict = dict_list[list_ids.index(list_id)]
        else:
            data_dict = {}
        for agent_id in agents:
            data_dict_or_iterable = data_dict.get(agent_id, fill_value[agent_id])
            mask[agent_id][list_id] = 1 if agent_id in data_dict else 0
            useoptional_action_mask[agent_id] = useoptional_action_mask[agent_id] or isinstance(data_dict_or_iterable, dict)
            if not isinstance(data_dict_or_iterable, dict):
                collated[agent_id][list_id] = data_dict_or_iterable
            else:
                collated[agent_id][list_id] = data_dict_or_iterable["observation"]
                optional_action_mask[agent_id][list_id] = data_dict_or_iterable["action_mask"]


    output = dict()

    for agent_id in agents:
        if useoptional_action_mask[agent_id] or is_obs:
            output[agent_id] = BatchedObs(data=np.array(collated[agent_id]),
                                          action_mask=np.array(optional_action_mask[agent_id]) \
                                            if useoptional_action_mask[agent_id] else None,
                                          batch_mask=np.array(mask[agent_id]),
                                          is_sequence=False)
        else:
            output[agent_id] = BatchBasic(data=np.array(collated[agent_id]),
                                          batch_mask=np.array(mask[agent_id]),
                                          is_sequence=False)
    

    return MulObs(output) if is_obs else MulBatched(output)

class RolloutGroup:
    observation_spaces: List[Dict[str, spaces.Space]]
    action_spaces: List[Dict[str, spaces.Space]]

    """
    A group of rollout workers to collect experiences in parallel.
    The LearnerGroup.inference is perform here then actions are distributed to each worker async.
    This works as a vectorized env.
    We keep the buffer here
    """
    def __init__(self, 
                env_construct_fn: Callable[..., ParallelEnv],
                contain_buffer: bool, # maybe the evaluator is not nessesarry
                scale: int, 
                seed: int,
                need_state: bool,
                each_resource: dict = {'num_cpus':2},
                env_kwargs: dict = {}) -> None:
        self._fill_value = None

        self.contain_buffer = contain_buffer
        self.need_state = need_state
        self.random_generator = np.random.default_rng(seed=seed)
        self.trajectory_ids = list(range(scale))
        self.step_ids = [0,] * scale
        self.n_worker = scale
        subseeds = self.random_generator.integers(int(1e9), size=(scale,))
        self._workers = [RolloutWorker.options(**each_resource).remote(env_construct_fn=env_construct_fn,
                                            seed=seed,
                                            env_kwargs=env_kwargs) 
                                            for seed in subseeds]
        self.agents = ray.get(self._workers[0].get_possible_agents.remote())
        self.observation_spaces = ray.get([worker.get_obs_spaces.remote() for worker in self._workers])
        self.action_spaces = ray.get([worker.get_act_spaces.remote() for worker in self._workers])
        # _ = ray.get(self._workers) # wait until all subworkers initialized.
        self.buffer = None
        self.states = None
        self._last_obs, self._last_info = None, None # they are to save the last (obs, info) of the current action for the buffer
        self._last_obs_, self._last_info_ = None, None # they are to save the last true (obs, info) from the auto reset
        self.renew_buffer()
        # print(self.fill_value)
        _ = self.reset()
        if need_state:
            self.state_dim = np.prod(ray.get(self._workers[0].get_state_dim.remote()))
            self.state_dim = self.state_dim.item()
            if self._fill_value is not None:
                self._fill_value.update({
                    'state': np.zeros((self.state_dim)) if self.need_state else None,
                    'next_state': np.zeros((self.state_dim)) if self.need_state else None,
                })


    @property
    def fill_value(self,)->Dict[str, Dict[str, Any]]:
        if self._fill_value is None:
            self._fill_value  = {
                'obs': {agent_id: space.sample() for agent_id, space in self.observation_spaces[0].items()},
                'next_obs': {agent_id: space.sample() for agent_id, space in self.observation_spaces[0].items()},
                'rew': {agent_id: 0 for agent_id, space in self.observation_spaces[0].items()},
                'ter': {agent_id: False for agent_id, space in self.observation_spaces[0].items()},
                'log_prob':  {agent_id: np.array([0.0,]) for agent_id, space in self.observation_spaces[0].items()},
                'trunc': {agent_id: False for agent_id, space in self.observation_spaces[0].items()},
                'act': {agent_id: space.sample() for agent_id, space in self.action_spaces[0].items()},

                # 'agent_id':
                # 'traj_id': 
                # ,'traj_id','step_id', 'mask'
            }
        return self._fill_value

    def get_last_b4_reset(self):
        return self._last_obs_, self._last_info_


    def renew_buffer(self):
        if self.contain_buffer:
            self.buffer = []
            self.states = []

    def sample_action(self,) -> Tuple[MulAction, MulBatched]:
        if self._last_obs is not None:
            output_actions = {}
            output_logprobs = {}
            for agent_id in self._last_obs:
                actions = []
                batch_masks = []
                logprobs = []
                if self._last_obs[agent_id].action_mask is not None:
                    for i, batch_msk in enumerate(self._last_obs[agent_id].batch_mask):
                        mask_i = self._last_obs[agent_id].action_mask[i]
                        if mask_i.sum() > 0 and batch_msk:
                            valid = np.nonzero(mask_i)[0]
                            chosen = np.random.choice(valid)
                            actions.append(chosen)
                            batch_masks.append(1)
                            logprobs.append(np.array([-np.log(len(valid))]))  # uniform log-prob
                        else:
                            actions.append(self.fill_value['act'][agent_id])
                            batch_masks.append(0)
                            logprobs.append(np.array([0.0]))
                else:
                    for i in range(self.n_worker):
                        chosen = self.action_spaces[i][agent_id].sample()
                        actions.append(chosen)
                        batch_masks.append(1)
                        logprobs.append(np.array([0.0]))
                output_actions[agent_id] = BatchBasic(
                    data=np.array(actions), batch_mask=np.array(batch_masks)
                )
                output_logprobs[agent_id] = BatchBasic(
                    data=np.array(logprobs), batch_mask=np.array(batch_masks)
                )
            return MulAction(output_actions), MulBatched(output_logprobs)

        else:
            output_actions = {}
            output_logprobs = {}
            for agent_id in self.agents:
                actions = []
                batch_masks = []
                logprobs = []
                for i in range(self.n_worker):
                    chosen = self.action_spaces[i][agent_id].sample()
                    actions.append(chosen)
                    batch_masks.append(1)
                    logprobs.append(0.0)
                output_actions[agent_id] = BatchBasic(
                    data=np.array(actions), batch_mask=np.array(batch_masks)
                )
                output_logprobs[agent_id] = BatchBasic(
                    data=np.array(logprobs), batch_mask=np.array(batch_masks)
                )
            return MulAction(output_actions), MulBatched(output_logprobs)


    def reset(self, *args, **kwargs) -> Tuple[MulObs, List]:
        # call a reset uniformly across all
        jobs = [worker.reset.remote(*args, **kwargs) for worker in self._workers]
        _last_obs = []
        self._last_info = []
        for (obs, info) in ray.get(jobs):
            _last_obs.append(obs)
            self._last_info.append(info)
        if self.need_state:
            states = ray.get([worker.get_state.remote()
                               for worker in self._workers])
            self._last_state = [state.reshape(-1) for state in states]

        self._last_obs = mulbatch_from_list(dict_list=_last_obs,
                    list_ids=list(range(self.n_worker)),
                    total_id=self.n_worker,
                    is_obs=True,
                    fill_value=self.fill_value['obs'])
        self._last_obs_, self._last_info_ = self._last_obs, self._last_info
        return self._last_obs, self._last_info
    
    def step(self, 
             actions: MulAction, log_probs: MulBatched) \
            -> Tuple[MulObs, MulBatched, MulBatched, MulBatched, List]:
        jobs = []
        reset_ids = []
        results = ray.get([worker.step.remote(actions.debatch(i)) 
                            for i, worker in enumerate(self._workers)])
        states = None
        if self.need_state:
            states = ray.get([worker.get_state.remote()
                               for worker in self._workers])
            states = [state.reshape(-1) for state in states]
        
        if self.contain_buffer:
            for i, (obs, rew, ter, trunc, info) in enumerate(results):
                for agentid in rew.keys():
                    record = {
                        'obs': self._last_obs.debatch(i)[agentid],
                        'act': actions.debatch(i)[agentid],
                        'log_prob': log_probs.debatch(i)[agentid],
                        'rew': rew[agentid],
                        'next_obs': obs[agentid],
                        'ter': ter[agentid],
                        'trunc': trunc[agentid],
                        'info': info[agentid],
                        'agent_id': agentid,
                        'traj_id': self.trajectory_ids[i],
                        'step_id': self.step_ids[i],
                        # 'state': None if states is None else states[i]
                    }
                    self.buffer.append(record)
                if self._last_state is not None:
                    self.states.append({
                        'traj_id': self.trajectory_ids[i],
                        'step_id': self.step_ids[i],
                        'state': self._last_state[i],
                        'next_state': states[i],
                    })
        if self.need_state:
            self._last_state = states
        self.step_ids = [step + 1 for step in self.step_ids]
        results = [list(result) for result in results]

        # before auto reset
        Obs = [obs for (obs, rew, ter, trunc, info) in results]
        Info = [info for (obs, rew, ter, trunc, info) in results]
        
        Obs = mulbatch_from_list(dict_list=Obs,
                                    list_ids=list(range(self.n_worker)),
                                    total_id=self.n_worker,
                                    is_obs=True,
                                    fill_value=self.fill_value['obs']) 
        self._last_obs_, self._last_info_ = Obs, Info[:]

        need_resets = ray.get([worker.all_done.remote() for worker in self._workers])

        for worker_id, need_reset in enumerate(need_resets):
            if need_reset:
                jobs.append(self._workers[worker_id].reset.remote())
                reset_ids.append(worker_id)
                self.trajectory_ids[worker_id] = max(*self.trajectory_ids) + 1
                self.step_ids[worker_id] = 0

        
                
        for reset_id, (reset_obs, reset_info) in zip(reset_ids, ray.get(jobs)):
            # _last_obs[reset_id] = reset_obs
            # _last_info[reset_id] = reset_info

            results[reset_id][0] = reset_obs
            results[reset_id][-1] = reset_info


        Obs = [obs for (obs, rew, ter, trunc, info) in results]
        Rew = [rew for (obs, rew, ter, trunc, info) in results]
        Ter = [ter for (obs, rew, ter, trunc, info) in results]
        Trunc = [trunc for (obs, rew, ter, trunc, info) in results]
        Info = [info for (obs, rew, ter, trunc, info) in results]
        
        Obs, Rew, Ter, Trunc = [mulbatch_from_list(dict_list=Val,
                                    list_ids=list(range(self.n_worker)),
                                    total_id=self.n_worker,
                                    is_obs=is_obs,
                                    fill_value=self.fill_value[val]) 
                    for Val, val, is_obs in zip([Obs, Rew, Ter, Trunc], ['obs', 'rew', 'ter', 'trunc'], \
                                        [True, False, False, False])]
      
        self._last_obs = Obs

        return Obs, Rew, Ter, Trunc, Info
    
    def sample_transitions(self, batch_size: int) -> Dict[str, Union[MulBatched, MulObs, Dict]]:
        # Uniformly sample transitions
        # return dict of { [agent id]: {[record key]: [list (batch) of record values (data shape)]}}
        a_result = {}
        result_keys = ['obs', 'act', 'log_prob', 'rew', 'next_obs', \
                      'ter', 'trunc', 'info', 'agent_id', 'traj_id', \
                        'step_id', 'state', 'next_state']
        agent_ids = np.array([rec['agent_id'] for rec in self.buffer])
        if self.need_state:
            state_indices = {
                'traj_id': np.array([r['traj_id'] for r in self.states]),
                'step_id': np.array([r['step_id'] for r in self.states])
            }
        for agent_id in self.agents:
            is_agent = np.argwhere(agent_ids==agent_id).reshape(-1)
            if len(is_agent)==0:
                continue
            selected_ids = self.random_generator.choice(is_agent, 
                                                        size=batch_size, 
                                                        replace=True)
            result = {k: [] for k in result_keys}
            for _id in selected_ids:
                rec = self.buffer[_id]
                for k, v in rec.items():
                    result[k].append(v)
                if self.need_state:
                    state_id = np.argwhere(np.logical_and(
                        state_indices['step_id']==rec['step_id'],
                        state_indices['traj_id']==rec['traj_id'],
                    )).reshape(-1)[0]
                    result['state'].append(self.states[state_id]['state'])
                    result['next_state'].append(self.states[state_id]['next_state'])

            a_result[agent_id] = result
        output = dict({k: {agent_id: None for agent_id in a_result.keys()} for k in result_keys})
        for agent_id, data_dict in a_result.items():
            for k in result_keys:
                if 'obs' in k and  isinstance(data_dict[k][0], dict):
                    output[k][agent_id] = BatchedObs(data=np.array([d['observation'] for d in data_dict[k]]),
                                                    action_mask=np.array([d['action_mask'] for d in data_dict[k]]),
                                                    batch_mask=np.ones((batch_size,)))
                elif isinstance(data_dict[k][0], (np.ndarray, list, int, float, bool, np.number)):
                    output[k][agent_id] = BatchBasic(data=np.array(data_dict[k]),
                                                    batch_mask=np.ones((batch_size,)))
        agent_id = list(a_result.keys())[0]
        for k in result_keys:
            if isinstance(output[k][agent_id], BatchedObs):
                output[k] = MulObs(output[k])
            elif k=='act':
                output[k] = MulAction(output[k])
            elif isinstance(output[k][agent_id], BatchBasic):
                output[k] = MulBatched(output[k])

        return output
    
    def sample_trajectories(self, batch_size, indices=None) -> Dict[str, Dict[str, Dict[str, list]]]:
        # Uniformly sample trajectories
        # return dict of { [agent id]: {[record key]: [list (batch) of list (sequence len) of recorded values (data shape)]}}
        # there is no next-obs or next-state, they will be cated in the end of the sequences
        # be carefull with the mask: it involve the the next!
        a_result = {}
        a_mask = {}
        rec_indices = {
            'traj_id': np.array([r['traj_id'] for r in self.buffer]),
            'step_id': np.array([r['step_id'] for r in self.buffer]),
            'agent_id': np.array([r['agent_id'] for r in self.buffer]),
        }
        selected_ids = self.random_generator.choice(np.unique(rec_indices['traj_id']), 
                                                    size=batch_size, 
                                                    replace=True) if indices is None else indices
        max_seq_len = np.max(rec_indices['step_id'][np.isin(rec_indices['traj_id'], selected_ids)]) + 2 # +1 for the len and + 1 for the last position
        buffer_index = {}
        for i, rec in enumerate(self.buffer):
            buffer_index.setdefault((rec['traj_id'], rec['agent_id']), []).append(i)
        if self.need_state:        
            state_index = {(s['traj_id'], s['step_id']): i for i, s in enumerate(self.states)}
        result_keys = [
                'obs','act','log_prob','rew','ter','trunc','info',
                'traj_id','step_id', 
                # 'mask'
            ]

        def each_agent_worker(agent_id):
            # preallocate result with full-length sequences
            result = {
                **{k: [[self.fill_value[k][agent_id] if k in self.fill_value else None] * max_seq_len \
                          for _ in range(batch_size)] for k in result_keys},     
                **{f'{k}_mask': [[0] * max_seq_len for _ in range(batch_size)] for k in result_keys},
                }
        
            # cache to avoid recomputing same trajectory
            traj_cache = {}
        
            for traj_order, traj_id in enumerate(selected_ids):
                cache_key = (traj_id, agent_id)

                # reuse if seen before
                if cache_key in traj_cache:
                    cached = traj_cache[cache_key]
                    for k in result:
                        result[k][traj_order] = cached[k]
                    continue
        
                # lookup buffer indices (faster with dict)
                if (traj_id, agent_id) not in buffer_index:
                    continue
                in_buff_ids = buffer_index[(traj_id, agent_id)]
                records = [self.buffer[idx] for idx in in_buff_ids]
     
                # assign actual 2 steps the current and the next one
                for rec in records:
                    step = rec['step_id']
                    # for k, v in rec.items():
                    for k in result_keys:
                        if k in rec and not result[f'{k}_mask'][traj_order][step]: # not assigned
                            result[k][traj_order][step] = rec[k]
                        if f'next_{k}' in rec and not result[f'{k}_mask'][traj_order][step+1]:
                            result[f'{k}_mask'][traj_order][step+1] = 1
                            result[k][traj_order][step+1] = rec[f'next_{k}']
                    
                    result[f'{k}_mask'][traj_order][step] = 1

                # save in cache
                traj_cache[cache_key] = {k: result[k][traj_order] for k in result}
                
            return agent_id, None, result
        
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(each_agent_worker, agent_id) for agent_id in self.agents]
            for f in as_completed(futures):
                agent_id, _ , result = f.result()
                a_result[agent_id] = result
        states = None
        state_masks = None
        if self.need_state:
            state_fill_vale = self.fill_value['state']
            states = [[state_fill_vale,]*max_seq_len for _ in range(batch_size)]
            state_masks = [[0,]*max_seq_len for _ in range(batch_size)]
            for traj_order, traj_id in enumerate(selected_ids):
                for step_id in range(max_seq_len):
                    if (traj_id, step_id) in state_index:
                        states[traj_order][step_id] = self.states[state_index[(traj_id, step_id)]]['state']
                        state_masks[traj_order][step_id] = 1
                        if not state_masks[traj_order][step_id+1]:
                            states[traj_order][step_id+1] = self.states[state_index[(traj_id, step_id)]]['next_state']
                            state_masks[traj_order][step_id+1] = 1

            state_masks = np.array(state_masks)
            states = BatchBasic(
                data=np.array(states),
                batch_mask=state_masks.sum(axis=-1)>0,
                is_sequence=True,
                position_mask=state_masks
            )

        output = dict({k: {agent_id: None for agent_id in a_result.keys()} for k in result_keys})
        
        for agent_id, data_dict in a_result.items():
            for k in result_keys:
                position_mask = np.array(data_dict[f'{k}_mask'])
                if 'obs' in k and isinstance(data_dict[k][0][0], dict):
                    output[k][agent_id] = BatchedObs(data=np.array([[each['observation']  for each in each_list]\
                                                                    for each_list in data_dict[k]]),
                                                     action_mask=np.array([[each['action_mask'] for each in each_list]\
                                                                    for each_list in data_dict[k]]),
                                                     is_sequence=True,
                                                     position_mask=position_mask,
                                                     batch_mask=position_mask.sum(axis=-1)>0)
                elif k in self.fill_value.keys():
                    output[k][agent_id] = BatchBasic(data=np.array(data_dict[k]),
                                                     is_sequence=True,
                                                     position_mask=position_mask,
                                                     batch_mask=position_mask.sum(axis=-1)>0)

        agent_id = list(a_result.keys())[0]
        for k in result_keys:
            if isinstance(output[k][agent_id], BatchedObs):
                output[k] = MulObs(output[k])
            elif k=='act':
                output[k] = MulAction(output[k])
            elif isinstance(output[k][agent_id], BatchBasic):
                output[k] = MulBatched(output[k])

        return {"state": states, **output} 
  
    def state_dict(self,):
        random_states = {
            'worker_states': ray.get([worker.state_dict.remote() for worker in self._workers]),
            'self_state': self.random_generator.bit_generator.state
            }
        return {
            'random_states': random_states,
            'buffer': self.buffer,
            'states': self.states,
        }

    
    def load_state_dict(self, state_dict):
        try:
            self.random_generator.bit_generator.state = state_dict['random_states']['self_state']
            loaded_worker = all(ray.get([worker.load_state_dict.remote(state_dict['random_states']['worker_states'][i])
                                                                  for i, worker in enumerate(self._workers)]))
            self.buffer = state_dict['buffer']
            self.states = state_dict['states']
            return loaded_worker
        except:
            return False

