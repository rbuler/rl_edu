from tqdm import tqdm
import numpy as np

def decay_schedule(init_value, min_value,
                   decay_ratio, max_steps,
                   log_start=-2, log_base=10):
    
    """Compute decaying values as specified in the function arguments"""

    decay_steps = int(max_steps * decay_ratio)
    rem_steps = max_steps - decay_steps

    values = np.logspace(log_start, 0, decay_steps,
                         base=log_base, endpoint=True)[::-1]
    eps = 1e-8
    if values.max() - values.min() > eps:
        values = (values - values.min()) / (values.max() - values.min())
    values = (init_value - min_value) * values + min_value
    values = np.pad(values, (0, rem_steps), 'edge')
    return values


def generate_trajectory_mc_pred(env, pi=None, max_steps=20):

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
    return np.array(trajectory, np.object_)


def generate_trajectory(select_action, Q,
                        epsilon, env, max_steps=20):

    """Roll out the policy in the environment for a full episode"""

    trajectory = []
    truncated = False
    terminated = False
    state, _ = env.reset(seed=42)

    while not (truncated or terminated):
        action = select_action(state, Q, epsilon)

        next_state, reward, terminated, truncated, _ = env.step(action)
        experience = (state, action, reward, next_state, terminated, truncated)
        trajectory.append(experience)

        if terminated:
            break

        if len(trajectory) >= max_steps:
            trajectory = []
            state, _ = env.reset(seed=42)
            break

        state = next_state
    return np.array(trajectory, np.object_)


# Prediction algorithms


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
        trajectory = generate_trajectory_mc_pred(env, pi, max_steps)
        visited = np.zeros(nS, dtype=np.bool_)
        
        for t, (state, _, reward, _, _, _) in enumerate(trajectory):
            if visited[state] and first_visit:
                continue
            visited[state] = True
            n_steps = len(trajectory[t:])
            G = np.sum(discounts[:n_steps] * trajectory[t:, 2])
            V[state] = V[state] + alphas[e] * (G - V[state])

        V_track[e] = V
    return V.copy(), V_track


def td(env,
       pi=None,
       gamma=1.0,
       init_alpha=0.5,
       min_alpha=0.01,
       alpha_decay_ratio=0.3,
       n_episodes=500):
    
    nS = env.observation_space.n
    V = np.zeros(nS)
    V_track = np.zeros((n_episodes, nS))
    alphas = decay_schedule(init_alpha, min_alpha,
                            alpha_decay_ratio, n_episodes)
    for e in range(n_episodes):
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


def ntd(env,
        pi=None,
        gamma=1.0,
        init_alpha=0.5,
        min_alpha=0.01,
        alpha_decay_ratio=0.3,
        n_step=3,
        n_episodes=500):
    
    nS = env.observation_space.n
    V = np.zeros(nS)
    V_track = np.zeros((n_episodes, nS))
    alphas = decay_schedule(init_alpha, min_alpha,
                            alpha_decay_ratio, n_episodes)
    discounts = np.logspace(0, n_step+1, num=n_step+1,
                            base=gamma, endpoint=False)
    
    for e in tqdm(range(n_episodes)):
        truncated = False
        terminated = False
        state, info = env.reset(seed=42)
        path = []
        while not (truncated or terminated) or path is not None:
            path = path[1:]
            while not (truncated or terminated) and len(path) < n_step:
                if pi is not None:
                    action = pi(state)  # policy action select
                else:
                    action = env.action_space.sample()  # random action sample
                next_state, reward, terminated, truncated, info = env.step(action)
                experience = (state, action, reward, next_state, terminated, truncated)
                path.append(experience)
                state = next_state
                if terminated or truncated:
                    break
            n = len(path)
            est_state = path[0][0]
            rewards = np.array(path)[:,2]
            partial_return = discounts[:n] * rewards
            bs_val = discounts[-1] * V[next_state] * (1 - terminated)

            ntd_target = np.sum(np.append(partial_return, bs_val))
            ntd_error = ntd_target - V[est_state]
            V[est_state] = V[est_state] + alphas[e] * ntd_error

            if len(path) == 1 and path[0][4]:  # if only terminal state in path then set path
                path = None                    # to none to break out of the episode loop

        V_track[e] = V
    return V, V_track


def td_lambda(env,
              pi=None,
              gamma=1.0,
              init_alpha=0.5,
              min_alpha=0.01,
              alpha_decay_ratio=0.3,
              lambda_=0.3,
              n_episodes=500):
    
    nS = env.observation_space.n
    V = np.zeros(nS)
    V_track = np.zeros((n_episodes, nS))
    E = np.zeros(nS)
    alphas = decay_schedule(init_alpha, min_alpha,
                            alpha_decay_ratio, n_episodes)

    for e in tqdm(range(n_episodes)):
        E.fill(0)
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
            E[state] = E[state] + 1
            V = V + alphas[e] * td_error * E
            E = gamma * lambda_ * E
            state = next_state
        V_track[e] = V
    return V, V_track


# Control algorithms


def mc_control(env, gamma=1.0,
               init_alpha=0.5,
               min_alpha=0.01,
               alpha_decay_ratio=0.5,
               init_epsilon=1.0,
               min_epsilon=0.1,
               epsilon_decay_ratio=0.9,
               n_episodes=3000,
               max_steps=200,
               first_visit=True):
    
    nS, nA = env.observation_space.n, env.action_space.n

    discounts = np.logspace(0, max_steps, num=max_steps,
                            base=gamma, endpoint=False)
    alphas = decay_schedule(init_alpha, min_alpha,
                            alpha_decay_ratio, n_episodes)
    
    epsilons = decay_schedule(init_epsilon, min_epsilon,
                              epsilon_decay_ratio, n_episodes)
    
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)
    
    select_action = lambda state, Q, epsilon: np.argmax(Q[state]) \
        if np.random.rand() > epsilon else np.random.choice(nA)

    for e in tqdm(range(n_episodes)):
        trajectory = generate_trajectory(select_action, Q,
                                         epsilons[e], env, max_steps)
        visited = np.zeros((nS, nA), dtype=np.bool_)
        
        for t, (state, action, reward, _, _, _) in enumerate(trajectory):
            if visited[state][action] and first_visit:
                continue
            visited[state][action] = True
            n_steps = len(trajectory[t:])
            G = np.sum(discounts[:n_steps] * trajectory[t:, 2])
            Q[state][action] = Q[state][action] + alphas[e] * (G - Q[state][action])

        Q_track[e] = Q
        pi_track.append(np.argmax(Q, axis=1))
    V = np.max(Q, axis=1)
    pi = lambda s: {s:a for s, a in enumerate(pi_track[-1])}[s]
    return Q, V, pi, Q_track, pi_track


def sarsa(env,
          gamma=1.0,
          init_alpha=0.5,
          min_alpha=0.01,
          alpha_decay_ratio=0.5,
          init_epsilon=1.0,
          min_epsilon=0.1,
          epsilon_decay_ratio=0.9,
          n_episodes=3000):

    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []

    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)
    
    select_action = lambda state, Q, epsilon: np.argmax(Q[state]) \
        if np.random.rand() > epsilon else np.random.choice(nA)

    alphas = decay_schedule(init_alpha, min_alpha,
                            alpha_decay_ratio, n_episodes)
    epsilons = decay_schedule(init_epsilon, min_epsilon,
                              epsilon_decay_ratio, n_episodes)
    
    for e in tqdm(range(n_episodes)):
        truncated = False
        terminated = False
        state, _ = env.reset(seed=42)
        action = select_action(state, Q, epsilons[e])

        while not (truncated or terminated):
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_action = select_action(next_state, Q, epsilons[e])

            td_target = reward + gamma * Q[next_state][next_action] * (1 - terminated)
            td_error = td_target - Q[state][action]
            Q[state][action] = Q[state][action] + alphas[e] * td_error

            state, action = next_state, next_action

        Q_track[e] = Q
        pi_track.append(np.argmax(Q, axis=1))
    V = np.max(Q, axis=1)
    pi = lambda s: {s:a for s, a in enumerate(pi_track[-1])}[s]
    return Q, V, pi, Q_track, pi_track


def q_learning(env,
               gamma=1.0,
               init_alpha=0.5,
               min_alpha=0.01,
               alpha_decay_ratio=0.5,
               init_epsilon=1.0,
               min_epsilon=0.1,
               epsilon_decay_ratio=0.9,
               n_episodes=3000):

    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []

    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)

    select_action = lambda state, Q, epsilon: np.argmax(Q[state]) \
        if np.random.rand() > epsilon else np.random.choice(nA)
    
    alphas = decay_schedule(init_alpha, min_alpha,
                            alpha_decay_ratio, n_episodes)
    epsilons = decay_schedule(init_epsilon, min_epsilon,
                              epsilon_decay_ratio, n_episodes)
    
    for e in tqdm(range(n_episodes)):
        truncated = False
        terminated = False
        state, _ = env.reset(seed=42)

        while not (truncated or terminated):
            action = select_action(state, Q, epsilons[e])
            next_state, reward, terminated, truncated, _ = env.step(action)

            td_target = reward + gamma * Q[next_state].max() * (1 - terminated)
            td_error = td_target - Q[state][action]
            Q[state][action] = Q[state][action] + alphas[e] * td_error

            state = next_state

        Q_track[e] = Q
        pi_track.append(np.argmax(Q, axis=1))
    V = np.max(Q, axis=1)
    pi = lambda s: {s:a for s, a in enumerate(pi_track[-1])}[s]
    return Q, V, pi, Q_track, pi_track
    

def double_q_learning(env,
                      gamma=1.0,
                      init_alpha=0.5,
                      min_alpha=0.01,
                      alpha_decay_ratio=0.5,
                      init_epsilon=1.0,
                      min_epsilon=0.1,
                      epsilon_decay_ratio=0.9,
                      n_episodes=3000):

    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []

    Q1 = np.zeros((nS, nA), dtype=np.float64)
    Q2 = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)

    select_action = lambda state, Q1, Q2, epsilon: np.argmax(Q1[state] + Q2[state]) \
        if np.random.rand() > epsilon else np.random.choice(nA)
    
    alphas = decay_schedule(init_alpha, min_alpha,
                            alpha_decay_ratio, n_episodes)
    epsilons = decay_schedule(init_epsilon, min_epsilon,
                              epsilon_decay_ratio, n_episodes)
    
    for e in tqdm(range(n_episodes)):
        truncated = False
        terminated = False
        state, _ = env.reset(seed=42)

        while not (truncated or terminated):
            action = select_action(state, Q1, Q2, epsilons[e])
            next_state, reward, terminated, truncated, _ = env.step(action)

            if np.random.rand() < 0.5:
                td_target = reward + gamma * Q2[next_state].max() * (1 - terminated)
                td_error = td_target - Q1[state][action]
                Q1[state][action] = Q1[state][action] + alphas[e] * td_error
            else:
                td_target = reward + gamma * Q1[next_state].max() * (1 - terminated)
                td_error = td_target - Q2[state][action]
                Q2[state][action] = Q2[state][action] + alphas[e] * td_error

            state = next_state

        Q_track[e] = Q1 + Q2
        pi_track.append(np.argmax(Q1 + Q2, axis=1))
    V = np.max(Q1 + Q2, axis=1)
    pi = lambda s: {s:a for s, a in enumerate(pi_track[-1])}[s]
    return Q1, V, pi, Q_track, pi_track