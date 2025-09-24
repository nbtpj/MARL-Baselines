from .envs import *
import numpy as np
import pandas as pd
from typing import Dict, Any
# try:
def init_gfootball():
    return GFootBall(
        env_name="11_vs_11_stochastic",
        representation="simple115v2",
        stacked = True,
        channel_dimensions=(72, 96),
        number_of_left_players_agent_controls=5,
        number_of_right_players_agent_controls=0,
        # render=True
    )


def init_smacv2():
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
    max_cycles_default = 1000
    env = PSmac(max_cycles=max_cycles_default,
        render_mode='rgb_array',
        capability_config=distribution_config,
        map_name="10gen_terran",
        debug=True,
        conic_fov=False,
        obs_own_pos=True,
        use_unit_ranges=True,
        min_attack_range=2,
        seed=2,
    )
    return env

def init_smac():
    max_cycles_default = 1000
    env = PSSmac(max_cycles=max_cycles_default,
        render_mode='rgb_array',
        map_name="3m",
    )
    return env



def summarize_smac_results(result_dict: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """
    Compute mean & std for return, eps_len, and game information
    selecting the info with the highest dead_allies per episode.

    Parameters
    ----------
    result_dict : dict
        Dictionary containing:
            - 'return': {agent_id: np.ndarray[episodes]}
            - 'eps_len': {agent_id: np.ndarray[episodes]}
            - 'last_info': {agent_id: List[List[dict]]}
              Each inner list contains one or more dicts per episode.
              Only the dict with max dead_allies is used.

    Returns
    -------
    summary : dict
        {
          'return':   pd.DataFrame  # mean & std per agent and overall
          'eps_len':  pd.DataFrame
          'game_info': pd.DataFrame # dead_allies, dead_enemies, battle_won
        }
    """
    # ---- Helper to compute mean/std ----
    def stats(arr):
        return np.mean(arr), np.std(arr, ddof=1) if len(arr) > 1 else 0.0

    agents = result_dict['return'].keys()
    ret_stats, len_stats = {}, {}
    allies_stats, enemies_stats, win_stats = {}, {}, {}

    for agent in agents:
        # --- Return & episode length ---
        mean_r, std_r = stats(result_dict['return'][agent])
        mean_l, std_l = stats(result_dict['eps_len'][agent])
        ret_stats[agent] = (mean_r, std_r)
        len_stats[agent] = (mean_l, std_l)

        # --- Game info: choose dict with highest dead_allies per episode ---
        info = []
        for ep_info in result_dict['last_info'][agent]:
            if isinstance(ep_info, list):
                best = max(ep_info, key=lambda x: x['dead_allies'])
            else:
                best = ep_info  # already a single dict
            info.append(best)

        dead_allies = [x['dead_allies'] for x in info]
        dead_enemies = [x['dead_enemies'] for x in info]
        wins = [float(x['battle_won']) for x in info]

        allies_stats[agent]  = stats(dead_allies)
        enemies_stats[agent] = stats(dead_enemies)
        win_stats[agent]     = stats(wins)

    # ---- Overall statistics ----
    all_returns = np.concatenate(list(result_dict['return'].values()))
    all_lengths = np.concatenate(list(result_dict['eps_len'].values()))
    all_allies  = np.concatenate([[x['dead_allies'] for x in lst]
                                  for lst in result_dict['last_info'].values()])
    all_enemies = np.concatenate([[x['dead_enemies'] for x in lst]
                                  for lst in result_dict['last_info'].values()])
    all_wins    = np.concatenate([[float(x['battle_won']) for x in lst]
                                  for lst in result_dict['last_info'].values()])

    ret_stats['overall'] = stats(all_returns)
    len_stats['overall'] = stats(all_lengths)
    allies_stats['overall']  = stats(all_allies)
    enemies_stats['overall'] = stats(all_enemies)
    win_stats['overall']     = stats(all_wins)

    # ---- Build DataFrames for easy reading ----
    return_df = pd.DataFrame(ret_stats, index=['mean','std']).T
    epslen_df = pd.DataFrame(len_stats, index=['mean','std']).T
    game_df = pd.DataFrame({
        'dead_allies_mean':  [v[0] for v in allies_stats.values()],
        'dead_allies_std' :  [v[1] for v in allies_stats.values()],
        'dead_enemies_mean':[v[0] for v in enemies_stats.values()],
        'dead_enemies_std' :[v[1] for v in enemies_stats.values()],
        'win_rate_mean'   :[v[0] for v in win_stats.values()],
        'win_rate_std'    :[v[1] for v in win_stats.values()],
    }, index=list(allies_stats.keys()))

    return {'return': return_df.to_json(), 'eps_len': epslen_df.to_json(), 'game_info': game_df.to_json()}

ENV = {
    'gfootball':init_gfootball,
    'smacv2': init_smacv2,
    'smac': init_smac,
}

READ_RESULT = {
    'smacv2': summarize_smac_results,
    'smac': summarize_smac_results,
}