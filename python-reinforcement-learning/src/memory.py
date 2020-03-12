
class Memory():
    def __init__(self):
       self.reset_memory()

    def reset_memory(self):
        self.actions = []
        self.observations = []
        self.rewards = []
        self.selected_action_idx = []
        self.terminals = []

    def append(self, observation, action, selected_action_idx, reward, is_terminal_state):
        self.observations.append(observation)
        self.actions.append(action)
        self.selected_action_idx.append(selected_action_idx)
        self.rewards.append(reward)
        self.terminals.append(is_terminal_state)