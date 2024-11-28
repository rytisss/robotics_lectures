### Code inspired by https://github.com/johnnycode8/gym_solutions, but refactored

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle


def run(
    episodes,
    is_training=True,
    render=False,
    alpha=0.9,               # Learning rate
    gamma=0.9,               # Discount factor
    epsilon_start=1.0,       # Initial exploration rate
    reward_threshold=-200,   # Reward threshold for success
    epsilon_decay_factor=2,  # Factor for calculating epsilon decay rate
    max_steps_per_episode=1000,  # Max steps allowed per episode
    pickle_file='mountain_car.pkl'  # File to save/load Q-table
):
    """Runs the MountainCar-v0 simulation with Q-learning training or evaluation."""
    env = gym.make('MountainCar-v0', render_mode='human' if render else None)

    # Define state space discretization
    pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20)  # -1.2 to 0.6
    vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20)  # -0.07 to 0.07

    # Initialize Q-table
    if is_training:
        q_table = np.zeros((len(pos_space), len(vel_space), env.action_space.n))
    else:
        with open(pickle_file, 'rb') as f:
            q_table = pickle.load(f)

    # Exploration rate
    epsilon = epsilon_start
    epsilon_decay_rate = epsilon_decay_factor / episodes
    rng = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes)

    for episode in range(episodes):
        state = env.reset()[0]
        state_indices = [
            np.digitize(state[0], pos_space),
            np.digitize(state[1], vel_space)
        ]

        terminated = False
        total_reward = 0

        while not terminated and total_reward > -max_steps_per_episode:
            # Select an action using epsilon-greedy policy
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()  # Random action
            else:
                action = np.argmax(q_table[tuple(state_indices)])

            # Perform the action
            new_state, reward, terminated, _, _ = env.step(action)
            new_state_indices = [
                np.digitize(new_state[0], pos_space),
                np.digitize(new_state[1], vel_space)
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

        # Decay epsilon and record rewards
        epsilon = max(epsilon - epsilon_decay_rate, 0)
        rewards_per_episode[episode] = total_reward

        # Print training progress every 100 episodes
        if is_training and episode % 100 == 0:
            mean_rewards = np.mean(rewards_per_episode[max(0, episode - 100):(episode + 1)])
            print(f'Episode: {episode}  Total Reward: {total_reward}  Epsilon: {epsilon:.2f}  Mean Reward: {mean_rewards:.1f}')

    env.close()

    # Save Q-table to file
    if is_training:
        with open(pickle_file, 'wb') as f:
            pickle.dump(q_table, f)

    # Plot and save mean rewards
    mean_rewards = [
        np.mean(rewards_per_episode[max(0, t - 100):(t + 1)]) for t in range(episodes)
    ]
    plt.plot(mean_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Mean Reward (Last 100 Episodes)')
    plt.title('Training Progress')
    plt.savefig('mountain_car.png')


if __name__ == '__main__':
    run(
        episodes=100,
        is_training=False,
        render=True,
        alpha=0.9,
        gamma=0.9,
        epsilon_start=1.0,
        reward_threshold=-200,
        epsilon_decay_factor=2,
        max_steps_per_episode=1000,
        pickle_file='mountain_car.pkl'
    )
