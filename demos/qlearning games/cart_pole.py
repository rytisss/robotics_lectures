### Code inspired by https://github.com/johnnycode8/gym_solutions, but refactored

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle


def run(
    is_training=True,
    render=False,
    alpha=0.1,               # Learning rate
    gamma=0.99,              # Discount factor
    epsilon_start=1.0,       # Initial exploration rate
    epsilon_decay=0.00001,   # Epsilon decay rate
    reward_threshold=1000,   # Mean reward threshold to stop training
    max_steps_per_episode=10000,  # Max steps allowed per episode
    render_frequency=100,    # Frequency of rendering during evaluation
    pickle_file='cartpole.pkl'  # File to save/load Q-table
):
    """Runs the CartPole-v1 simulation with Q-learning training or evaluation."""
    env = gym.make('CartPole-v1', render_mode='human' if render else None)

    # Define state space discretization
    pos_space = np.linspace(-2.4, 2.4, 10)
    vel_space = np.linspace(-4, 4, 10)
    ang_space = np.linspace(-0.2095, 0.2095, 10)
    ang_vel_space = np.linspace(-4, 4, 10)

    # Initialize Q-table
    if is_training:
        q_table = np.zeros((
            len(pos_space) + 1,
            len(vel_space) + 1,
            len(ang_space) + 1,
            len(ang_vel_space) + 1,
            env.action_space.n
        ))
    else:
        with open(pickle_file, 'rb') as f:
            q_table = pickle.load(f)

    # Exploration rate
    epsilon = epsilon_start
    rng = np.random.default_rng()  # Random number generator

    rewards_per_episode = []

    # Training loop
    episode = 0
    while True:
        state = env.reset()[0]
        state_indices = [
            np.digitize(state[0], pos_space),
            np.digitize(state[1], vel_space),
            np.digitize(state[2], ang_space),
            np.digitize(state[3], ang_vel_space)
        ]

        terminated = False
        total_reward = 0

        while not terminated and total_reward < max_steps_per_episode:
            # Select an action using epsilon-greedy policy
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()  # Random action
            else:
                action = np.argmax(q_table[tuple(state_indices)])

            # Perform the action
            new_state, reward, terminated, _, _ = env.step(action)
            new_state_indices = [
                np.digitize(new_state[0], pos_space),
                np.digitize(new_state[1], vel_space),
                np.digitize(new_state[2], ang_space),
                np.digitize(new_state[3], ang_vel_space)
            ]

            # Update Q-table if training
            if is_training:
                current_q = q_table[tuple(state_indices) + (action,)]
                max_future_q = np.max(q_table[tuple(new_state_indices)])
                q_table[tuple(state_indices) + (action,)] = current_q + alpha * (
                    reward + gamma * max_future_q - current_q
                )

            # Update state and accumulate rewards
            state_indices = new_state_indices
            total_reward += reward

            # Render occasionally during evaluation
            if not is_training and total_reward % render_frequency == 0:
                print(f'Episode: {episode}  Rewards: {total_reward}')

        # Log rewards and decay epsilon
        rewards_per_episode.append(total_reward)
        mean_reward = np.mean(rewards_per_episode[-100:])
        if is_training and episode % 100 == 0:
            print(f'Episode: {episode}  Rewards: {total_reward}  Epsilon: {epsilon:.2f}  Mean Rewards: {mean_reward:.1f}')

        if mean_reward > reward_threshold:
            break

        epsilon = max(epsilon - epsilon_decay, 0)
        episode += 1

    env.close()

    # Save Q-table if training
    if is_training:
        with open(pickle_file, 'wb') as f:
            pickle.dump(q_table, f)

    # Plot and save mean rewards
    mean_rewards = [
        np.mean(rewards_per_episode[max(0, t - 100):(t + 1)]) for t in range(episode)
    ]
    plt.plot(mean_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Mean Reward (Last 100 Episodes)')
    plt.title('Training Progress')
    plt.savefig('cartpole.png')


if __name__ == '__main__':
    run(
        is_training=False,
        render=True,
        alpha=0.1,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_decay=0.00001,
        reward_threshold=1000,
        max_steps_per_episode=10000,
        render_frequency=100,
        pickle_file='cartpole.pkl'
    )