import gymnasium as gym
from comet_ml import Experiment
from comet_ml.integration.gymnasium import CometLogger

experiment = Experiment(
  project_name="rl-edu",
  workspace="rafalbuler"
)

mode = 'show-only'

if mode == 'show-only':
    render_mode = 'human'
    env = gym.make("LunarLander-v2", render_mode=render_mode)
    env = CometLogger(env, experiment)
    experiment.add_tag(mode)

elif mode == 'save-only':
    render_mode = 'rgb_array'
    disable_logger = False
    env = gym.make("LunarLander-v2", render_mode=render_mode)
    env = gym.wrappers.RecordVideo(env, 'rendered_videos', disable_logger=disable_logger)
    env = CometLogger(env, experiment)
    experiment.add_tag(mode)

episodes = 5

test_metric = 0
experiment.log_metric ('test_int', test_metric)

for e in range(episodes):
    observation, info = env.reset(seed=42)
    truncated = False
    terminated = False
    while not (truncated or terminated):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()

    test_metric = e
    experiment.log_metric ('test_int', test_metric)


env.close()
