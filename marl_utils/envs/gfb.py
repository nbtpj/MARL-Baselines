from pettingzoo import ParallelEnv
import gfootball.env as football_env
import gymnasium as gym
import copy
import numpy as np

class GFootBall(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "name": "GFootBall"}

    def __init__(self, env_name='',
               stacked=False,
               render_mode='rgb_array',
               representation='simple115v2',
               rewards='scoring',
               write_goal_dumps=False,
               write_full_episode_dumps=False,
               render=False,
               write_video=False,
               dump_frequency=1,
               logdir='',
               extra_players=None,
               number_of_left_players_agent_controls=1,
               number_of_right_players_agent_controls=0,
               max_cycles = None,
               channel_dimensions=(
                   football_env.observation_preprocessing.SMM_WIDTH,
                   football_env.observation_preprocessing.SMM_HEIGHT),
               **other_config_options):
        self.max_cycles = max_cycles
        self._env = football_env.create_environment(
            env_name=env_name,
            stacked=stacked,
            representation=representation,
               rewards=rewards,
               write_goal_dumps=write_goal_dumps,
               write_full_episode_dumps=write_full_episode_dumps,
               render=render,
               write_video=write_video,
               dump_frequency=dump_frequency,
               logdir=logdir,
               extra_players=extra_players,
               number_of_left_players_agent_controls=number_of_left_players_agent_controls,
               number_of_right_players_agent_controls=number_of_right_players_agent_controls,
               channel_dimensions=channel_dimensions,
               other_config_options=other_config_options
        )
        self.render_mode = render_mode
        n_agents = self._env.action_space.shape[0]
        self.agents = list(range(n_agents))
        self.possible_agents = self.agents
        self.observation_spaces = {
            i: gym.spaces.Box(low=self._env.observation_space.low[i],
                              high=self._env.observation_space.high[i],
                             dtype=self._env.observation_space.dtype) for i in self.agents
        }
        self.action_spaces = {
            i: self._env.action_space[i] for i in self.agents
        }
        self._cycle = None
    def step(self, actions):
        _actions = [actions[i] for i in self.agents]
        obs, rew, trunc, done, info = self._env.step(_actions)
        _obs, _rew, _trunc, _done, _info = dict(), dict(), dict(), dict(), dict()
        self._cycle +=1
        for i in self.agents:
            _obs[i] = obs[i]
            _rew[i] = rew[i]
            _trunc[i] = trunc or self._cycle >= self.max_cycles if self.max_cycles is not None else trunc
            _done[i] = done
            _info[i] = info
        return copy.deepcopy(_obs), copy.deepcopy(_rew), \
            copy.deepcopy(_trunc), copy.deepcopy(_done), copy.deepcopy(_info)

    def state(self):
        env_state_dict = self._env.unwrapped._env.observation()
        arr = []
        for k, v in env_state_dict.items():
            if k=='frame':
                continue
            if isinstance(v, np.ndarray):
                arr.append(v.astype(np.float32).reshape(-1))
            elif isinstance(v, (int, bool, float)):
                arr.append(np.array([float(v)]))
        return np.concatenate(arr, axis=0)
    
    def render(self):
        env_state_dict = self._env.unwrapped._env.observation()
        if 'frame' in env_state_dict:
            return env_state_dict['frame']
        return self._env.unwrapped.render(mode=self.render_mode)
    def reset(self, *args, **kwargs):
        obs, info = self._env.reset(*args, **kwargs)
        _obs,  _info = dict(), dict()
        for i in self.agents:
            _obs[i] = obs[i]
            _info[i] = info
        self._cycle = 0 
        return copy.deepcopy(_obs), copy.deepcopy(_info)
    
    def close(self):
        self._env.close()

    # @property
    # def unwrapped(self):
    #     return self._env.unwrapped

if __name__ == "__main__":
    import numpy as np
    import imageio
    from tqdm import tqdm

    frames = []

    env = GFootBall(
        env_name="11_vs_11_stochastic",
        representation="simple115v2",
        stacked = True,
        channel_dimensions=(72, 96),
        number_of_left_players_agent_controls=5,
        number_of_right_players_agent_controls=5,
        # render=True
    )


    # Assuming your ParallelEnv is called `env`
    env.reset()
    assert env.agents == env.possible_agents
    done = {agent: False for agent in env.agents}

    bar = tqdm()
    while not all(done.values()):
        # Take random actions for each agent (replace with your policy)
        actions = {agent: env.action_space(agent).sample() for agent in env.agents if not done[agent]}
        
        # Step the environment
        obs, rewards, truncs, dones, infos = env.step(actions)
        bar.update(1)
        # Update done flags
        done.update(dones)
        
        # Render the environment
        img = env.render()
        frames.append(img)
    print(env.state())
    env.close()
    imageio.mimsave("game_football_pettingzoo.mp4", frames, fps=30)
# plt.imshow(env.render())