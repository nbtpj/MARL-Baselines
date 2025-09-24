import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
from smacv2.env import StarCraft2Env
from gymnasium.utils import EzPickle, seeding
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv
from pettingzoo.utils.conversions import parallel_to_aec
from pettingzoo.utils import wrappers
import numpy as np

max_cycles_default = 1000


# def parallel_env(max_cycles=max_cycles_default, **smac_args):
#     return PSSmac(max_cycles, **smac_args)


# def raw_env(max_cycles=max_cycles_default, **smac_args):
#     # provide an AEC wrapper if desired
#     return parallel_to_aec(parallel_env(max_cycles, **smac_args))


# def make_env(raw_env):
#     def env_fn(**kwargs):
#         env = raw_env(**kwargs)
#         env = wrappers.AssertOutOfBoundsWrapper(env)
#         env = wrappers.OrderEnforcingWrapper(env)
#         return env

#     return env_fn


class smacPSSmac(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "name": "sc2"}

    def __init__(self, env, max_cycles, render_mode=None):
        self.max_cycles = max_cycles
        self.env = env
        self.render_mode = render_mode
        # ensure SMAC internal reset before constructing agent lists
        # self.env.full_restart()
        # self.env._episode_count = 1
        self.env.reset()
        self.reset_flag = 0
        self.agents, self.action_spaces = self._init_agents()
        self.possible_agents = self.agents[:]

        observation_size = env.get_obs_size()
        self.observation_spaces = {
            name: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=-1,
                        high=1,
                        shape=(observation_size,),
                        dtype=np.float32,
                    ),
                    "action_mask": spaces.Box(
                        low=0,
                        high=1,
                        shape=(self.action_spaces[name].n,),
                        dtype=np.int8,
                    ),
                }
            )
            for name in self.agents
        }
        self._reward = 0

    def render(self):
        return self.env.render(self.render_mode)

    def _init_agents(self):
        agents = []
        action_spaces = {}
        self.agents_id = {}
        # simply generate stable unique agent names (user requested IDs only)
        for idx, (agent_id, agent_info) in enumerate(self.env.agents.items()):
            agent_name = f"agent_{idx}"
            agents.append(agent_name)
            self.agents_id[agent_name] = agent_id
            action_spaces[agent_name] = spaces.Discrete(self.env.get_total_actions() - 1)
        return agents, action_spaces

    def seed(self, seed=None):
        # gymnasium seeding helper
        if seed is None:
            self.env._seed = seeding.create_seed(seed, max_bytes=4)
        else:
            self.env._seed = seed
        # use SMACv2 full_restart if available
        if hasattr(self.env, "full_restart"):
            self.env.full_restart()
        else:
            # fallback
            self.env.reset()

    # def render(self, mode="human"):
    #     self.env.render(mode)

    def close(self):
        self.env.close()

    def state(self) -> np.ndarray:
        return self.env.get_state()


    def reset(self, seed=None, options=None):
        # follow PettingZoo ParallelEnv API: reset(seed=None, options=None) -> (observations, infos)
        if seed is not None:
            try:
                self.seed(seed)
            except Exception:
                # best-effort seeding; ignore if SMAC doesn't accept
                pass

        # reset SMAC internals
        self.env._episode_count = 1
        self.env.reset()

        self.agents = self.possible_agents[:]
        self.frames = 0
        # infos must contain an entry for each agent
        infos = {agent: {} for agent in self.possible_agents}
        return self._observe_all(), infos

    def get_agent_smac_id(self, agent):
        return self.agents_id[agent]

    def _all_rewards(self, reward):
        # uniform reward to all agents (keeps old behavior)
        return {agent: reward for agent in self.agents}

    def _observe_all(self):
        all_obs = []
        for agent in self.agents:
            agent_id = self.get_agent_smac_id(agent)
            obs = self.env.get_obs_agent(agent_id)
            action_mask = self.env.get_avail_agent_actions(agent_id)
            # drop no-op (first action) to match previous behaviour
            action_mask = action_mask[1:]
            action_mask = np.array(action_mask).astype(np.int8)
            obs = np.asarray(obs, dtype=np.float32)
            all_obs.append({"observation": obs, "action_mask": action_mask})
        return {agent: obs for agent, obs in zip(self.agents, all_obs)}

    def _compute_term_trunc(self, env_terminated: bool, env_truncated: bool):
        """
        Returns two dicts: terminations and truncations keyed by agent.
        - termination=True if the agent died (health==0) OR the env reported termination.
        - truncation=True if the time limit (or other truncation) occurred.
        """
        terminations = {}
        truncations = {}
        for agent in self.agents:
            agent_id = self.get_agent_smac_id(agent)
            agent_info = self.env.get_unit_by_id(agent_id)
            dead = agent_info.health == 0
            terminations[agent] = bool(env_terminated or dead)
            truncations[agent] = bool(env_truncated)
        return terminations, truncations

    def step(self, all_actions):
        """
        PettingZoo Parallel API expects:
        observations, rewards, terminations, truncations, infos
        Each is a dict keyed by agent name.
        """
        # build SMAC action list (SMAC expects ints + 1 for real actions)
        action_list = [0] * self.env.n_agents
        for agent in self.agents:
            agent_id = self.get_agent_smac_id(agent)
            if agent in all_actions:
                act = all_actions[agent]
                action_list[agent_id] = 0 if act is None else int(act) + 1

        # env.step returns (reward, terminated, info) in your SMACv2 wrapper
        self._reward, env_terminated, smac_info = self.env.step(action_list)
        self.frames += 1
        env_truncated = self.frames >= self.max_cycles

        # infos: provide an entry per agent (attach smac_info if dict)
        all_infos = {agent: {} for agent in self.agents}
        if isinstance(smac_info, dict):
            for agent in all_infos:
                # shallow copy of smac_info into each agent's info; adjust as needed
                all_infos[agent].update(smac_info)

        # compute per-agent termination and truncation dicts
        terminations, truncations = self._compute_term_trunc(env_terminated, env_truncated)

        # observations/rewards
        all_observes = self._observe_all()
        all_rewards = self._all_rewards(self._reward)

        # remove agents that are done (either terminated or truncated) from active list
        self.agents = [agent for agent in self.agents if (not terminations.get(agent, False)) and (not truncations.get(agent, False))]

        return all_observes, all_rewards, terminations, truncations, all_infos

    def __del__(self):
        try:
            self.env.close()
        except Exception:
            pass


# env = make_env(raw_env)


class PSSmac(smacPSSmac, EzPickle):
    metadata = {"render_modes": ["human", "rgb_array"], "name": "sc2"}

    def __init__(self, max_cycles, render_mode='rgb_array', **smac_args):
        EzPickle.__init__(self, max_cycles, render_mode=render_mode, 
                          **smac_args)
        env = StarCraft2Env(**smac_args)
        super().__init__(env, max_cycles, render_mode=render_mode)




# def test_env():
#     env = parallel_env(max_cycles=50, **smac_args)
#     print("Possible agents:", env.possible_agents)

#     observations, infos = env.reset(seed=42)
#     print("Initial obs keys:", list(observations.keys()))
#     print("Initial infos keys:", list(infos.keys()))

#     for step in range(10):
#         # sample actions for *currently active* agents
#         actions = {agent: env.action_spaces[agent].sample() for agent in env.agents}
#         obs, rewards, terminations, truncations, infos = env.step(actions)

#         print(f"\nStep {step+1}")
#         print("Active agents:", env.agents)
#         print("Actions sent:", actions)
#         print("Rewards:", rewards)
#         print("Terminations:", terminations)
#         print("Truncations:", truncations)

#         # stop if all agents are done (either terminated or truncated)
#         if not env.agents:
#             print("All agents finished (env.agents is empty).")
#             break

#     env.close()
# if __name__=='__main__':
#     import numpy as np
#     from pettingzoo.utils import parallel_to_aec

#     # Example SMAC2 map (replace with one you have installed, e.g. "3m" or "2s3z")
#     smac_args = {"map_name": "3m"}  # Requires SMACv2 maps
#     test_env()
        
if __name__ == "__main__":
    import numpy as np
    import imageio

    frames = []


    env = PSSmac(max_cycles=max_cycles_default,
        render_mode='rgb_array',
        # capability_config=distribution_config,
        map_name="3m",
        # debug=True,
        # conic_fov=False,
        # obs_own_pos=True,
        # use_unit_ranges=True,
        # min_attack_range=2,
    )
    # print(f"{env.action_spaces=}")
    # print(f"{env.observation_spaces=}")


    # Assuming your ParallelEnv is called `env`
    obs, infor = env.reset()
    done = {agent: False for agent in env.agents}


    while not all(done.values()):
        # Take random actions for each agent (replace with your policy)
        msks = {agent: np.nonzero(obs[agent]["action_mask"])[0] \
                         for agent in env.agents if not done[agent]}
        valid_actions = {agent: np.random.choice(msk) for agent, msk in msks.items()}
        # actions = {agent: env.action_space(agent).sample() for agent in env.agents if not done[agent]}
        # print(f"{valid_actions=}")
        # print(f"{msks=}")

        # Step the environment
        obs, rewards, truncs, ters, infos = env.step(valid_actions)
        dones = {agent: ters[agent] or truncs[agent] for agent in ters}
        # Update done flags
        done.update(dones)
        
        # Render the environment
        img = env.render()
        frames.append(img)
        print(img.shape)
    print(env.state())
    print(env.agents)
    print(env.possible_agents)
    env.close()
    
    imageio.mimsave("game_smacv1_pettingzoo.mp4", frames, fps=30)

