from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
from smac.env.pettingzoo import StarCraft2PZEnv as StarCraft1
# from smacv2.env.pettingzoo import StarCraft2PZEnv as StarCraft1
from smacv2.env.pettingzoo import StarCraft2PZEnv as StarCraft2

from pprint import pprint

from gfootball.env.petting_zoo import GFootBall

import json
from tqdm import tqdm




def smac_random_policy(obs, env, done):
    # Take random actions for each agent (replace with your policy)
    msks = {agent: np.nonzero(obs[agent]["action_mask"])[0] \
                    for agent in obs if not done[agent]}
    valid_actions = {agent: np.random.choice(msk) for agent, msk in msks.items()}

    return valid_actions

def football_policy(obs, env, done):
    actions = {agent: env.action_space(agent).sample() for agent in obs if not done[agent]}
    return actions


def test_parallel_env(prefix, 
                      env, 
                      policy_fn,
                      episodes=10, save_video=False):
    """
    Runs an env object with random actions.
    """
    total_reward = 0
    frames = []
    done = False
    completed_episodes = 0
    tabr1 = tqdm(desc=f"{prefix} Episode")
    tabr = tqdm(desc=f"{prefix} Total Step")
    agents = []
    while completed_episodes < episodes:
        obs, infor = env.reset(seed=len(agents))
        agents.append(env.agents)
        done = {agent: False for agent in env.agents}

        while not all(done.values()):
            tabr.update(1)
            actions = policy_fn(obs, env, done)

            # Step the environment
            obs, rewards, truncs, ters, infos = env.step(actions)
            if save_video:
                frames.append(env.render())
            total_reward += sum(rewards.values())
            dones = {agent: ters[agent] or truncs[agent] or done[agent] for agent in ters}
            # Update done flags
            done.update(dones)
        tabr1.update(1)
        completed_episodes+=1

    env.close()

    print(f"{prefix} Average total reward", total_reward / episodes)
    print(f"{prefix} Agent List: ")
    pprint(agents, indent=4, compact=True)
    return frames


if __name__ == "__main__":
    try:
        prefix = '[SMAC2 (ver1, fixed agents)]'
        env = StarCraft1.parallel_env(max_cycles=1000, map_name='8m', render_mode='rgb_array')
        episodes = 10
        policy_fn = smac_random_policy
        test_parallel_env(prefix, env, policy_fn, episodes, save_video=False)
    except Exception as e:
        print(prefix, e)

    try:
        prefix = '[SMAC2 (ver2, fixed agents)]'
        env = StarCraft2.parallel_env(max_cycles=1000, map_name='8m_v2', render_mode='rgb_array')
        episodes = 10
        policy_fn = smac_random_policy
        test_parallel_env(prefix, env, policy_fn, episodes, save_video=False)
    except Exception as e:
        print(prefix, e)


    prefix = '[SMAC2 (ver2, flex agents)]'    
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
    try:
        env = StarCraft2.parallel_env21(max_cycles=1000, 
                                    capability_config=distribution_config,
                                    map_name='10gen_terran_v2', 
                                    render_mode='rgb_array', 
                                    full_reset=True)
        episodes = 10
        policy_fn = smac_random_policy
        test_parallel_env(prefix, env, policy_fn, episodes, save_video=False)
    except Exception as e:
        print(prefix, e)

    try:
        prefix = '[SMAC2 (ver2, flex-norm agents)]'    
        env = StarCraft2.parallel_env22(max_cycles=1000, 
                                    capability_config=distribution_config,
                                    map_name='10gen_terran_v2', 
                                    render_mode='rgb_array', 
                                    full_reset=True)
        episodes = 10
        policy_fn = smac_random_policy
        test_parallel_env(prefix, env, policy_fn, episodes, save_video=False)
    except Exception as e:
        print(prefix, e)

    try:
        prefix = '[GFootBall]'    
        env = GFootBall(
            env_name="11_vs_11_stochastic",
            representation="simple115v2",
            stacked = True,
            channel_dimensions=(72, 96),
            number_of_left_players_agent_controls=5,
            number_of_right_players_agent_controls=5,
            max_cycles=100,
            # render=True
        )
        episodes = 10
        policy_fn = football_policy
        test_parallel_env(prefix, env, policy_fn, episodes, save_video=False)
    except Exception as e:
        print(prefix, e)
