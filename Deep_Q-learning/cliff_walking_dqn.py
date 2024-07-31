"""
File: cliff_walking_dqn.py
Author: Milosz Gajewski
Company: WizzDev P.S.A.
Date: 07.2024
Description: Deep Q-learning implementation form CliffWalking Gymnasim environment.
"""

import random
from typing import List, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from replay_memory import ReplayMemory
from nn_model import DQN


def state_to_dqn_input(state: int, num_states: int) -> torch.Tensor:
    """
    Convert a state (represented by int) to a tensor representation.
    """
    input_tensor = torch.zeros(num_states)
    input_tensor[state] = 1
    return input_tensor


class CliffWalkingDQN:
    """
    Cliff Walking Deep Q-learning implememation
    """

    def __init__(self):
        # Hyperparameters
        self.learning_rate = 0.001  # learning rate (alpha)
        self.discount_factor = 0.9  # discount factor (gamma)
        self.network_sync_rate = 10  # number of steps the agent takes before syncing the policy and target network
        self.replay_memory_size = 1000  # size of replay memory
        self.mini_batch_size = 32  # size of the training sampled from the replay memory

    def train(self, episodes: int = 1000, render: bool = False):
        # Prepare environment
        env = gym.make("CliffWalking-v0", render_mode="human" if render else None)
        num_states = env.observation_space.n
        num_actions = env.action_space.n

        epsilon = 1  # exploration_factor - 1 = 100% random actions, explore at the beginning
        memory = ReplayMemory(self.replay_memory_size)

        # Create policy and target neural networks
        policy_dqn = DQN(in_states=num_states, out_actions=num_actions)
        target_dqn = DQN(in_states=num_states, out_actions=num_actions)

        # Be sure that both models have the same weights
        target_dqn.load_state_dict(policy_dqn.state_dict())

        # Training components
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate)

        rewards_per_episode = np.zeros(episodes)
        epsilon_history = []
        reward = None
        step_count = 0

        print("--- Training ---")
        for i in tqdm(range(episodes)):
            state = env.reset()[0]  # Reset the environment
            terminated = False
            truncated = False

            while not terminated and not truncated:
                # Select action based on epsilon-greedy
                if random.random() < epsilon:
                    # Select random action
                    action = env.action_space.sample()
                else:
                    # Select best action
                    with torch.no_grad():
                        action = policy_dqn(state_to_dqn_input(state, num_states)).argmax().item()

                # Execute action
                new_state, action_reward, terminated, _, _ = env.step(action)

                # Process reward and observations
                terminated, reward = self._process_epizode(action_reward, new_state)

                # Save experience into memory
                memory.append((state, action, new_state, reward, terminated))

                # Move to the next state
                state = new_state

                # Increment step counter
                step_count += 1

            # Keep track of the rewards collected per episode.
            if reward == 1:
                rewards_per_episode[i] = 1

            # Check if enough experience has been collected and if at least 1 reward has been collected
            if len(memory) > self.mini_batch_size and np.sum(rewards_per_episode) > 0:
                mini_batch = memory.sample(self.mini_batch_size)
                self._optimize(num_states, mini_batch, policy_dqn, target_dqn)

                # Decay exploration_factor
                epsilon = max(epsilon - (1 / episodes), 0)
                epsilon_history.append(epsilon)

                # Synchronize policy network to target network after a certain number of steps
                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0

        # Close environment
        env.close()

        # Save policy
        torch.save(policy_dqn.state_dict(), "cliff_walking_dql.pt")
        print("--- Training ended ---")

    def _process_epizode(self, reward: int, new_state: int) -> List[Union[bool, int]]:
        terminated = False
        output_reward = 0

        if new_state == 47:
            # Reached the goal! Episode ends
            terminated = True
            output_reward = 1
        elif reward == -100:
            # Slipped off the cliff - end of episode
            terminated = True
            output_reward = 0

        return [terminated, output_reward]

    def _optimize(self, num_states: int, mini_batch: List, policy_dqn: DQN, target_dqn: DQN):
        # Prepare placeholders for target and actual q values
        current_q_list = []
        target_q_list = []

        # Iterate over saved memory
        for state, action, new_state, reward, terminated in mini_batch:
            if terminated:
                # Agent either reached goal (reward = 1) or slipped off the cliff (reward = 0)
                # Target q value should be set to the reward.
                target = torch.tensor([reward])
            else:
                # The agent neither fell nor reached his goal
                with torch.no_grad():
                    target = (
                        torch.tensor(reward)
                        + self.discount_factor * target_dqn(state_to_dqn_input(new_state, num_states)).max()
                    )

            # Get the current set of Q values
            current_q = policy_dqn(state_to_dqn_input(state, num_states))
            current_q_list.append(current_q)

            # Get the target set of Q values
            target_q = target_dqn(state_to_dqn_input(state, num_states))
            target_q_list.append(target_q)

            # Associate action with target
            target_q[action] = target

        # --- Back propagation ---
        # Compute loss for the whole minibatch
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def test(self, episodes: int = 100):
        # Prepare environment
        env = gym.make("CliffWalking-v0", render_mode="human")
        num_states = env.observation_space.n
        num_actions = env.action_space.n

        # Load learned policy
        policy_dqn = DQN(in_states=num_states, out_actions=num_actions)
        policy_dqn.load_state_dict(torch.load("cliff_walking_dql.pt"))
        policy_dqn.eval()  # switch model to evaluation mode

        num_success = 0
        print("--- Test ---") 
        for _ in tqdm(range(episodes)):
            state = env.reset()[0]  # Initialize to state 0
            terminated = False
            truncated = False

            step_count = 0
            while not terminated and not truncated:
                # Select best action
                with torch.no_grad():
                    action = policy_dqn(state_to_dqn_input(state, num_states)).argmax().item()
                # Execute action
                state, reward, terminated, truncated, _ = env.step(action)
                step_count += 1
                if reward == -100:
                    # Agent slipped off the cliff
                    break
                if state == 47:
                    # Agent reached goal!
                    num_success += 1
                    break
                if step_count > 50:
                    # Agent did to many moves
                    break
        env.close()
        print(f"Success rate: {num_success / episodes * 100} %")


if __name__ == "__main__":
    torch.set_default_device("cpu")
    cliff_walking = CliffWalkingDQN()
    cliff_walking.train(3000)
    cliff_walking.test()
