from abc import ABC, abstractmethod
import numpy as np
import gymnasium as gym


class TabularAgent(ABC):
    """
    Agent abstract base class.
    """

    def __init__(self, state_space: gym.spaces.Discrete, action_space: gym.spaces.Discrete, learning_rate=0.1,
                 discount_rate=0.9):
        """
        Agent Base Class constructor.
        Assumes discrete gymnasium spaces.
        You may want to make these attributes private.
        :param state_space: state space of gymnasium environment.
        :param action_space: action space of gymnasium environment.
        :param learning_rate: of the underlying algorithm.
        :param discount_rate: discount factor (`gamma`).
        """
        self.q_table = np.zeros([state_space.n, action_space.n])
        self.env_action_space = action_space
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate

    @abstractmethod
    def update(self, trajectory: tuple) -> None:
        """
        Where the update rule is applied
        :param trajectory: (S, A, S, R) (Q-learning), (S, A, S, R, A) (SARSA).
        """
        pass

    @abstractmethod
    def behavior_policy(self, state):
        """
        This is where you would do action selection.
        For epsilon greedy you can opt to make a separate object for epsilon greedy
        action selection and use composition.
        :param state:
        :return an action
        """
        pass
    
    @abstractmethod
    def target_policy(self, state):
        pass
    
class EpsilonGreedy:
    decaying_epsilon = 1
    
    @abstractmethod
    def select_action(state, state_space, q_table, epsilon = 0.1, decay=True):
        if decay:
            EpsilonGreedy.decaying_epsilon *= 0.99
            epsilon = EpsilonGreedy.decaying_epsilon
        else:
            EpsilonGreedy.decaying_epsilon = 1
        action = None
        if np.random.uniform(0, 1) < epsilon:
            action = state_space.sample()
        else:
            action = np.argmax(q_table[state, :])
        return action


class SarsaAgent(TabularAgent):
    def __init__(self, state_space: gym.spaces.Discrete, action_space: gym.spaces.Discrete, learning_rate=0.1,
                 discount_rate=0.9):
        super().__init__(state_space, action_space, learning_rate, discount_rate)
        
        
    def update(self, trajectory: tuple) -> None:
        """
        Where the update rule is applied
        :param trajectory: (S, A, S, R) (Q-learning), (S, A, S, R, A) (SARSA).
        """
        state_t, action_t, reward_t1, state_t1 = trajectory
        action_t1 = self.behavior_policy(state_t1)
        target_q_value = reward_t1 + self.discount_rate * self.q_table[state_t1, action_t1]
        self.q_table[state_t, action_t] = self.q_table[state_t, action_t] + self.learning_rate * (target_q_value - self.q_table[state_t, action_t])
        
    def behavior_policy(self, state):
        """
        This is where you would do action selection.
        For epsilon greedy you can opt to make a separate object for epsilon greedy
        action selection and use composition.
        :param state:
        :return an action
        """
        action = EpsilonGreedy.select_action(state, self.env_action_space, self.q_table)
        return action
    
    def target_policy(self, state):
        return self.behavior_policy(state)
    
class QLearningAgent(TabularAgent):
    def __init__(self, state_space: gym.spaces.Discrete, action_space: gym.spaces.Discrete, learning_rate=0.1,
                 discount_rate=0.9):
        super().__init__(state_space, action_space, learning_rate, discount_rate)
        
        
    def update(self, trajectory: tuple) -> None:
        """
        Where the update rule is applied
        :param trajectory: (S, A, S, R) (Q-learning), (S, A, S, R, A) (SARSA).
        """
        state_t, action_t, reward_t1, state_t1 = trajectory
        target_q_value = reward_t1 + self.discount_rate * self.q_table[state_t1, self.target_policy(state_t1)]
        self.q_table[state_t, action_t] += self.learning_rate * (target_q_value - self.q_table[state_t, action_t])
        
    def behavior_policy(self, state):
        """
        This is where you would do action selection.
        For epsilon greedy you can opt to make a separate object for epsilon greedy
        action selection and use composition.
        :param state:
        :return an action
        """
        action = EpsilonGreedy.select_action(state, self.env_action_space, self.q_table)
        return action
    
    def target_policy(self, state):
        return np.argmax(self.q_table[state, :])
    
class DoubleQLearningAgent(TabularAgent):
    def __init__(self, state_space: gym.spaces.Discrete, action_space: gym.spaces.Discrete, learning_rate=0.1,
                 discount_rate=0.9):
        super().__init__(state_space, action_space, learning_rate, discount_rate)
        self.q_1 = self.q_2 = self.q_table
        
        
    def update(self, trajectory: tuple) -> None:
        """
        Where the update rule is applied
        :param trajectory: (S, A, S, R) (Q-learning), (S, A, S, R, A) (SARSA).
        """
        state_t, action_t, reward_t1, state_t1 = trajectory        
        if np.random.uniform(0, 1) < 0.5:
            target_q_value = reward_t1 + self.discount_rate * self.q_2[state_t1, np.argmax(self.q_1[state_t1, :])]
            self.q_1[state_t, action_t] += self.learning_rate * (target_q_value - self.q_1[state_t, action_t])
        else:
            target_q_value = reward_t1 + self.discount_rate * self.q_1[state_t1, np.argmax(self.q_2[state_t1, :])]
            self.q_2[state_t, action_t] += self.learning_rate * (target_q_value - self.q_2[state_t, action_t])
        
    def behavior_policy(self, state):
        """
        This is where you would do action selection.
        For epsilon greedy you can opt to make a separate object for epsilon greedy
        action selection and use composition.
        :param state:
        :return an action
        """
        action = EpsilonGreedy.select_action(state, self.env_action_space, self.q_1 + self.q_2)
        return action
    
    def target_policy(self, state):
        combined = self.q_1 + self.q_2
        return np.argmax(combined[state, :])