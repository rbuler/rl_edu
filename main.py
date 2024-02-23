import gymnasium as gym
from comet_ml import Experiment
from comet_ml.integration.gymnasium import CometLogger

experiment = Experiment(
  project_name="rl-edu",
  workspace="rafalbuler"
)

mode = 'offline'

if mode == 'offline':
    render_mode = 'human'
    disable_logger = True
elif mode == 'online':
    render_mode = 'rgb_array'
    disable_logger = False

env = gym.make("LunarLander-v2", render_mode=render_mode)
env = gym.wrappers.RecordVideo(env, 'rendered_videos', disable_logger=disable_logger)
env = CometLogger(env, experiment)

episodes = 2

for _ in range(episodes):
    observation, info = env.reset(seed=42)
    truncated = False
    terminated = False
    while not (truncated or terminated):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()

env.close()
