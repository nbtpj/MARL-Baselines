# coding=utf-8
# Modified to run headless and save video

from __future__ import absolute_import, division, print_function
import os
import imageio
from absl import app, flags, logging

from gfootball.env import config
from gfootball.env import football_env

# Force SDL to use dummy driver for headless rendering
os.environ["SDL_VIDEODRIVER"] = "dummy"

FLAGS = flags.FLAGS

flags.DEFINE_string('players', 'keyboard:left_players=1',
                    'Semicolon separated list of players, single keyboard '
                    'player on the left by default')
flags.DEFINE_string('level', '', 'Level to play')
flags.DEFINE_enum('action_set', 'default', ['default', 'full'], 'Action set')
flags.DEFINE_bool('real_time', True,
                  'If true, environment will slow down so humans can play.')
flags.DEFINE_bool('render', True, 'Whether to do game rendering.')
flags.DEFINE_string('video_path', 'game.mp4', 'Path to save the output video')
flags.DEFINE_integer('fps', 30, 'Frames per second for output video')


def main(_):
    players = FLAGS.players.split(';') if FLAGS.players else ''
    assert not any(['agent' in player for player in players]), (
        "Player type 'agent' cannot be used with play_game."
    )

    cfg = config.Config({
        'action_set': FLAGS.action_set,
        'dump_full_episodes': True,
        'players': players,
        'real_time': FLAGS.real_time,
    })

    if FLAGS.level:
        cfg['level'] = FLAGS.level

    env = football_env.FootballEnv(cfg)
    frames = []

    # Capture the first frame
    if FLAGS.render:
        frame = env.render(mode='rgb_array')
        frames.append(frame)

    obs = env.reset()

    try:
        done = False
        while True:
            _, _, trunc, done, _ = env.step([])
            if FLAGS.render:
                frame = env.render(mode='rgb_array')
                frames.append(frame)
            if done:
                obs = env.reset()
                break  # Stop after one episode
    except KeyboardInterrupt:
        logging.warning('Game interrupted, saving video...')

    # Save all captured frames to video
    if FLAGS.render and frames:
        imageio.mimsave(FLAGS.video_path, frames, fps=FLAGS.fps)
        logging.info(f"Video saved to {FLAGS.video_path}")

    # Write dump as before
    env.write_dump('shutdown')


if __name__ == '__main__':
    app.run(main)
