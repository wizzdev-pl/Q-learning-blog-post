"""
File: gymnasium_demo.py
Author: Milosz Gajewski
Company: WizzDev P.S.A.
Date: 07.2024
Description: Simple demo of gymnasium framework.
"""

import gymnasium as gym


def main():
    # Environment: https://gymnasium.farama.org/environments/classic_control/cart_pole/
    env = gym.make("CartPole-v1", render_mode="human")  # Create environment
    env.reset()  # Reset the environment to base state
    for _ in range(1000):  # Steps of the simulation
        action = env.action_space.sample()  # Choose action (here: sample/random action)
        env.render()  # Render GUI
        observation, reward, done, tr, info = env.step(
            action)  # Execute the action
        if done:
            env.reset()
    env.close()  # Close the environment


if __name__ == "__main__":
    main()
