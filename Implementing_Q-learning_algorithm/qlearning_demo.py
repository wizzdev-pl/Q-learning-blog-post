"""
File: qlearning_demo.py
Author: Milosz Gajewski
Company: WizzDev P.S.A.
Date: 07.2024
Description: Classic Q-learning implementation demo using gymnasium framework.
"""

from tqdm import tqdm
import numpy as np
import gymnasium as gym


def main():
    env = gym.make("Taxi-v3")  # Create environment
    print(f"Available actions: {env.action_space.n}")
    print(f"Available observations: {env.observation_space.n}")

    # Q learning matrix will have dimensions observations x actions
    n_observations = env.observation_space.n
    n_actions = env.action_space.n
    Q_matrix = np.zeros((n_observations, n_actions))

    # Learning constants
    lr = 0.4
    discount_factor = 0.6
    exploration_factor = 0.7
    min_exploration_factor = 0.2
    # First we explore a lot and over time we explore exponentially less and less
    decay_rate = 1e-4

    print("--- Training ---")
    for e in tqdm(range(10000)):  # Steps of the simulation
        current_state, _ = env.reset()  # Reset the environment
        # env.render()  # Turn off rendering for training
        done = False
        while not done:
            if np.random.uniform(0, 1) < exploration_factor:
                # Choose action (here: sample/random action)
                action = env.action_space.sample()
            else:
                # Get column with the est action
                action = np.argmax(Q_matrix[current_state])
            next_state, reward, done, _, _ = env.step(
                action)  # Execute the action
            # Update Q matrix
            Q_matrix[current_state, action] = (1.0 - lr) * Q_matrix[
                current_state, action
            ] + lr * (reward + discount_factor * max(Q_matrix[next_state]))

            current_state = next_state

        exploration_factor = max(
            min_exploration_factor, np.exp(-decay_rate * e))
    print("Training ended with Q matrix:")
    print(Q_matrix)

    print("--- Testing ---")
    epochs = 50
    total_reward = []
    env = gym.make("Taxi-v3", render_mode="human")
    for _ in tqdm(range(epochs)):
        done = False
        epoch_reward = 0
        current_state, _ = env.reset()
        while not done:
            action = np.argmax(Q_matrix[current_state])
            current_state, reward, done, _, _ = env.step(action)
            epoch_reward += reward
        total_reward.append(epoch_reward)
    print(f"Total reward: {total_reward}")
    env.close()  # Close the environment


if __name__ == "__main__":
    main()
