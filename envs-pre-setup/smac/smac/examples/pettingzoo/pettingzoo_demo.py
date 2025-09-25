from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
from smac.env.pettingzoo import StarCraft2PZEnv


def main():
    """
    Runs an env object with random actions.
    """
    env = StarCraft2PZEnv.env()
    episodes = 10

    total_reward = 0
    done = False
    completed_episodes = 0
    from tqdm import tqdm
    tabr = tqdm()
    while completed_episodes < episodes:
        env.reset()
        for agent in env.agent_iter():
            tabr.update(1)
            env.render()

            obs, reward, terms, truncs, _ = env.last()
            total_reward += reward
            if terms or truncs:
                action = None
            elif isinstance(obs, dict) and "action_mask" in obs:
                action = random.choice(np.flatnonzero(obs["action_mask"]))
            else:
                action = env.action_spaces[agent].sample()
            env.step(action)

        completed_episodes += 1

    env.close()

    print("Average total reward", total_reward / episodes)


if __name__ == "__main__":
    main()
