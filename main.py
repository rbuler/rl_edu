import gymnasium as gym
from comet_ml import Experiment
from comet_ml.integration.gymnasium import CometLogger

experiment = Experiment(
  project_name="rl-edu",
  workspace="rafalbuler"
)

env = gym.make("LunarLander-v2", render_mode="rgb_array")

# env = gym.wrappers.RecordVideo(env, 'rendered_videos')
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
