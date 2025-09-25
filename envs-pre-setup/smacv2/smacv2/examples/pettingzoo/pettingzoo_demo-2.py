from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
from smacv2.env.pettingzoo import StarCraft2PZEnv


def main():
    """
    Runs an env object with random actions.
    """
    env = StarCraft2PZEnv.parallel_env(max_cycles=1000,)
    episodes = 10

    total_reward = 0
    done = False
    completed_episodes = 0
    from tqdm import tqdm
    tabr1 = tqdm(desc="EPS")
    tabr = tqdm()
    while completed_episodes < episodes:
        obs, infor = env.reset()
        done = {agent: False for agent in env.agents}

        while not all(done.values()):
            tabr.update(1)
            # Take random actions for each agent (replace with your policy)
            msks = {agent: np.nonzero(obs[agent]["action_mask"])[0] \
                            for agent in env.agents if not done[agent]}
            valid_actions = {agent: np.random.choice(msk) for agent, msk in msks.items()}

            # Step the environment
            obs, rewards, truncs, ters, infos = env.step(valid_actions)
            total_reward += sum(rewards.values())
            dones = {agent: ters[agent] or truncs[agent] or done[agent] for agent in ters}
            # Update done flags
            done.update(dones)
        tabr1.update(1)
        completed_episodes+=1

    env.close()

    print("Average total reward", total_reward / episodes)


if __name__ == "__main__":
    main()
