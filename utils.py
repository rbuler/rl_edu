import tqdm
import numpy as np

def decay_schedule(init_value, min_value,
                   decay_ratio, max_steps,
                   log_start=-2, log_base=10):
    
    decay_steps = int(max_steps * decay_ratio)
    rem_steps = max_steps - decay_steps

    values = np.logspace(log_start, 0, decay_steps,
                         base=log_base, endpoint=True)[::-1]
    
    values = (values - values.min()) / (values.max() - values.min())
    values = (init_value - min_value) * values + min_value
    values = np.pad(values, (0, rem_steps), 'edge')
    return values


def generate_trajectory(env, pi=None, max_steps=20):

    trajectory = []
    truncated = False
    terminated = False
    state, info = env.reset(seed=42)

    while not (truncated or terminated):
        if pi is not None:
            action = pi(state)  # policy action select
        else:
            action = env.action_space.sample()  # random action sample
        
        next_state, reward, terminated, truncated, info = env.step(action)
        experience = (state, action, reward, next_state, terminated, truncated)
        trajectory.append(experience)

        if len(trajectory) >= max_steps:
            trajectory = []
            break
        state = next_state
    return trajectory


def mc_prediction(env,
                  pi=None,
                  gamma=1.0,
                  init_alpha=0.5,
                  min_alpha=0.01,
                  alpha_decay_ratio=0.3,
                  n_episodes=500,
                  max_steps=100,
                  first_visit=True):
    
    nS = env.observation_space.n
    discounts = np.logspace(0, max_steps, num=max_steps,
                            base=gamma, endpoint=False)
    alphas = decay_schedule(init_alpha, min_alpha,
                            alpha_decay_ratio, n_episodes)
    
    V = np.zeros(nS)
    V_track = np.zeros((n_episodes, nS))

    for e in tqdm(range(n_episodes)):
        trajectory = generate_trajectory(env, pi, max_steps)
        visited = np.zeros(nS, dtype=np.bool_)
        
        for t, (state, _, reward, _, _, _) in enumerate(trajectory):
            if visited[state] and first_visit:
                continue
            visited[state] = True
            n_steps = len(trajectory[t:])
            G = np.sum(discounts[:n_steps] * trajectory[t:, 2])
            V[state] = V[state] + alphas * (G - V[state])

        V_track[e] = V
    return V.copy(), V_track


def td(env,
       pi=None,
       gamma=1.0,
       init_alpha=0.5,
       min_alpha=0.01,
       alpha_decay_ratio=0.3,
       n_episodes=500):
    
    nS = env.observation.n
    V = np.zeros(nS)
    V_track = np.zeros((n_episodes, nS))
    alphas = decay_schedule(init_alpha, min_alpha,
                            alpha_decay_ratio, n_episodes)
    
    for e in tqdm(range(n_episodes)):
        truncated = False
        terminated = False
        state, info = env.reset(seed=42)

        while not (truncated or terminated):
            if pi is not None:
                action = pi(state)  # policy action select
            else:
                action = env.action_space.sample()  # random action sample
            next_state, reward, terminated, truncated, info = env.step(action)

            td_target = reward + gamma * V[next_state] * (1 - terminated)
            td_error = td_target - V[state]
            V[state] = V[state] + alphas[e] * td_error

            state = next_state

        V_track[e] = V
    return V, V_track