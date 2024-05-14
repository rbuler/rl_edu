import gymnasium as gym
from comet_ml import Experiment
from comet_ml.integration.gymnasium import CometLogger
from utils import mc_prediction, td, ntd, td_lambda
from utils import mc_control, sarsa, q_learning, double_q_learning


def main():
    # environment = 'LunarLander-v2'
    # environment = 'FrozenLake-v1'
    environment = 'FrozenLake-v1'
    mode = 'show-only'

    if mode == 'show-only':
        experiment = None
        render_mode = 'human'
        env = gym.make(environment, render_mode=render_mode)
        # env = CometLogger(env, experiment)
        # experiment.add_tag(mode)

    elif mode == 'save-only':
        experiment = Experiment(
            project_name="rl-edu",
            workspace="rbuler"
        )
        render_mode = 'rgb_array'
        disable_logger = False
        env = gym.make(environment, render_mode=render_mode)
        env = gym.wrappers.RecordVideo(env, 'rendered_videos', disable_logger=disable_logger)
        env = CometLogger(env, experiment)
        experiment.add_tag(mode)

    test_metric = 0
    if experiment is not None:
        experiment.log_metric('test_int', test_metric)

    # READY     mc_prediction, td, ntd, td_lambda, mc_control, sarsa, q_learning, double_q_learning
    # TODO      dqn, a2c, ppo, ddpg, sac, trpo, her 



    # prediction problem
    if 1:
        def pi(s):
            actions = [0, 3, 0, 0, 0, 0, 2, 0, 3, 1, 0, 0, 0, 2, 1, 0]
            return actions[s]
        # pi = None
        V, V_track = mc_prediction(env, pi=pi, n_episodes=10)

    # control problem
    if 0:
        Q, V, pi, Q_track, pi_track = mc_control(env, n_episodes=3000)
        print(Q)
        print(V)
        print(pi)
        print(Q_track)
        print(pi_track)

if __name__ == '__main__':
    main()
