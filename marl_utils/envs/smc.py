import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
from smacv2.env import StarCraft2Env
# from smac.env import StarCraft2Env smac v1 is deprecated!
import gymnasium as gym
from gymnasium.utils.ezpickle import EzPickle
from gymnasium.utils import seeding
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv
from pettingzoo.utils.conversions import parallel_to_aec
from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper

# from pettingzoo.utils.conversions import from_parallel_wrapper
from pettingzoo.utils import wrappers
import numpy as np

max_cycles_default = 1000


class PSmac(ParallelEnv, EzPickle):
    # This is SMACv2 with ability wrapper. In short, every reset will result in a change in agents. 
    # we design this wrapper to blend with PettinZoo api

    metadata = {"render_modes": ["human", "rgb_array"], "name": "PSmac"}

    def __init__(self, max_cycles, 
                 render_mode='rgb_array',
                 **smac_args):
        EzPickle.__init__(self, max_cycles, **smac_args)

        self._agent_norm_name_map = None
        self.render_mode = render_mode
        self.max_cycles = max_cycles
        self.env = StarCraftCapabilityEnvWrapper(**smac_args)
        self.env.reset()
        self.reset_flag = 0
        self._agents, self._action_spaces = self._init_agents()
        self._possible_agents = self._agents[:]


        observation_size = self.env.get_obs_size()

        self.agents = list(range(len(self._agents)))
        self.possible_agents = self.agents[:]


        self.action_spaces = {
            name: self._action_spaces[self._agents[name]]
            for name in self.agents
        }
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


    def get_normed_name(self, real_agent_id: str) -> int:
        return self._possible_agents.index(real_agent_id)
    
    def get_real_name(self, normed_agent_id: int) -> str:
        return self._possible_agents[normed_agent_id]
        
    def _init_agents(self):
        last_type = ""
        agents = []
        action_spaces = {}
        self._agents_id = {}
        i = 0
        for agent_id, agent_info in self.env.agents.items():
            unit_action_space = spaces.Discrete(
                self.env.get_total_actions() - 1
            )  # no-op in dead units is not an action
            if agent_info.unit_type == self.env.marine_id:
                agent_type = "marine"
            elif agent_info.unit_type == self.env.marauder_id:
                agent_type = "marauder"
            elif agent_info.unit_type == self.env.medivac_id:
                agent_type = "medivac"
            elif agent_info.unit_type == self.env.hydralisk_id:
                agent_type = "hydralisk"
            elif agent_info.unit_type == self.env.zergling_id:
                agent_type = "zergling"
            elif agent_info.unit_type == self.env.baneling_id:
                agent_type = "baneling"
            elif agent_info.unit_type == self.env.stalker_id:
                agent_type = "stalker"
            elif agent_info.unit_type == self.env.colossus_id:
                agent_type = "colossus"
            elif agent_info.unit_type == self.env.zealot_id:
                agent_type = "zealot"
            else:
                raise AssertionError(f"agent type {agent_type} not supported")

            if agent_type == last_type:
                i += 1
            else:
                i = 0

            agents.append(f"{agent_type}_{i}")
            self._agents_id[agents[-1]] = agent_id
            action_spaces[agents[-1]] = unit_action_space
            last_type = agent_type

        return agents, action_spaces

    def render(self):
        return self.env.render(mode=self.render_mode)

    def _norm_dict(self, obj:dict)-> dict:
        return {self._possible_agents.index(k): obj[k] for k in obj }
    
    def _denorm_dict(self, obj:dict)->dict:
        return {self._possible_agents[k]: obj[k] for k in obj }


    def reset(self, *, seed = None, options = None):
        if seed is None:
            rng, seed = seeding.np_random(seed)
            safe_seed = rng.integers(0, 2**18 - 1)
            self.env._seed = safe_seed
        else:
            self.env._seed = seed

        self.env.full_restart()
        self.env._episode_count = 1
        self.env.reset()

        self._agents = self._possible_agents[:]
        self.frames = 0
        self.all_dones = {agent: False for agent in self._possible_agents}
        obs = self._observe_all()
        return self._norm_dict(obs), {}

    def get_agent_smac_id(self, agent):
        return self._agents_id[agent]

    def _all_rewards(self, reward):
        all_rewards = [reward] * len(self._agents)
        return {
            agent: reward for agent, reward in zip(self._agents, all_rewards)
        }

    def _observe_all(self):
        all_obs = []
        for agent in self._agents:
            agent_id = self.get_agent_smac_id(agent)
            obs = self.env.get_obs_agent(agent_id)
            action_mask = self.env.get_avail_agent_actions(agent_id)
            action_mask = action_mask[1:]
            action_mask = np.array(action_mask).astype(np.int8)
            obs = np.asarray(obs, dtype=np.float32)
            all_obs.append({"observation": obs, "action_mask": action_mask})
        return {agent: obs for agent, obs in zip(self._agents, all_obs)}

    def _all_dones(self, step_done=False):
        dones = [True] * len(self._agents)
        if not step_done:
            for i, agent in enumerate(self._agents):
                agent_done = False
                agent_id = self.get_agent_smac_id(agent)
                agent_info = self.env.get_unit_by_id(agent_id)
                if agent_info.health == 0:
                    agent_done = True
                dones[i] = agent_done
        return {agent: bool(done) for agent, done in zip(self._agents, dones)}

    def step(self, all_actions):
        all_actions = self._denorm_dict(all_actions)
        action_list = [0] * self.env.n_agents
        for agent in self._agents:
            agent_id = self.get_agent_smac_id(agent)
            if agent in all_actions:
                if all_actions[agent] is None:
                    action_list[agent_id] = 0
                else:
                    action_list[agent_id] = all_actions[agent] + 1
        self._reward, terminated, smac_info = self.env.step(action_list)
        self.frames += 1
        done = terminated
        trunc = {agent: self.frames >= self.max_cycles for agent in self._agents}

        all_infos = {agent: smac_info for agent in self._agents}
        # all_infos.update(smac_info)
        all_dones = self._all_dones(done)
        all_rewards = self._all_rewards(self._reward)
        all_observes = self._observe_all()

        self._agents = [agent for agent in self._agents if not all_dones[agent]]

        return self._norm_dict(all_observes), self._norm_dict(all_rewards), self._norm_dict(trunc), self._norm_dict(all_dones), self._norm_dict(all_infos)

    def state(self) -> np.ndarray:
        return self.env.get_state()
    
    def close(self):
        self.env.close()



if __name__ == "__main__":
    import numpy as np
    import imageio

    frames = []

    distribution_config = {
        "n_units": 5,
        "n_enemies": 5,
        "team_gen": {
            "dist_type": "weighted_teams",
            "unit_types": ["marine", "marauder", "medivac"],
            "exception_unit_types": ["medivac"],
            "weights": [0.45, 0.45, 0.1],
            "observe": True,
        },
        "start_positions": {
            "dist_type": "surrounded_and_reflect",
            "p": 0.5,
            "n_enemies": 5,
            "map_x": 32,
            "map_y": 32,
        },
    }

    env = PSmac(max_cycles=max_cycles_default,
        render_mode='rgb_array',
        capability_config=distribution_config,
        map_name="10gen_terran",
        debug=True,
        conic_fov=False,
        obs_own_pos=True,
        use_unit_ranges=True,
        min_attack_range=2,
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
        obs, rewards, truncs, dones, infos = env.step(valid_actions)
        
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
    
    imageio.mimsave("game_smac_pettingzoo.mp4", frames, fps=30)





