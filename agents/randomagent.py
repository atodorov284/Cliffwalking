from agents.tabularagent import TabularAgent


class RandomAgent(TabularAgent):
    def update(self, trajectory: tuple) -> None:
        pass

    def behavior_policy(self, state):
        return self.env_action_space.sample()
    
    def target_policy(self, state):
        return self.env_action_space.sample()
