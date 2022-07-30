# David Levin , 316554641

import sys
import random
import os.path
import numpy as np
from prettytable import PrettyTable

from pddlsim.executors.executor import Executor
from pddlsim.local_simulator import LocalSimulator
from customized_valid_actions import CustomizedValidActions

# read arguments
input_flag = sys.argv[1]
domain_path = sys.argv[2]
problem_path = sys.argv[3]
policy_file_path = sys.argv[4]


class QLearningExecutor(Executor):
    def __init__(self, policy_file):
        super(QLearningExecutor, self).__init__()

        self.lr = 0.8  # The 'alpha' from Bellman's equation
        self.epsilon = 1  # The indicator for epsilon greedy method
        self.gamma = 0.6  # The 'gamma' from Bellman's equation

        self.q_table = None  # Q Table
        self.state_space = []  # State Space in the given domain
        self.action_space = []  # Action Space in the given domain
        self.policy_file = policy_file
        self.policy_file_helper = policy_file + "_HELPER"

    """
    Function role is to initialize the services and the policy files.
    """
    def initialize(self, services):
        self.services = services
        self.initialize_Q_table()  # Init/Import policy file
        self.print_Q_table_to_file()  # Create the policy file
        self.print_Q_table_helper_to_file()  # Create the policy helper file

    """
    Function role is to initialize the Q table with zeros: 
    # ROWS = the possible states will be the rows in the Q table.
    # COLUMNS = the possible actions will be the columns in the Q table.
    """
    def initialize_Q_table(self):
        for action in self.services.parser.actions:  # Save all the possible actions of the given domain
            self.action_space.append(action)

        for object in self.services.parser.objects:  # Save all the possible objects of the given domain
            if "food" not in object and "person" not in object:
                self.state_space.append(object)

        if os.path.isfile((os.getcwd() + "/" + self.policy_file)):  # If the policy file exist - import it
            self.q_table = self.merge_Q_table()
        else:  # If there isn't policy file - create it
            self.q_table = np.zeros((len(self.state_space), len(self.action_space)))

    """
    Function role is merge the Q_table if it unnecessary (= if this is not the first iteration of the agent).
    """
    def merge_Q_table(self):
        values = []
        lines = open(self.policy_file).read().splitlines()  # Import all the value from the policy file
        for i, line in enumerate(lines):  # Save the values
            values.append(np.fromstring(line, dtype=float, sep=' '))

        matrix = np.array(values)
        return matrix

    """
    Function role is to print the Q table to the policy file.
    """
    def print_Q_table_to_file(self):
        matrix = np.matrix(self.q_table)
        with open(self.policy_file, 'wb') as f:
            for row in matrix:
                np.savetxt(f, row, fmt='%.5f')
            f.close()

    """
    Function role is to create the "pretty policy table". 
    The difference between the original policy file, is that in this file we will print the columns and rows values.
    """
    def print_Q_table_helper_to_file(self):
        t_headers = list(self.action_space)  # Create the columns names
        t_headers.insert(0, "State")

        table = PrettyTable([header for header in t_headers])

        for i, n_row in enumerate(self.q_table):  # Import the values from the policy file
            row = list([col for col in n_row])
            row.insert(0, self.state_space[i])
            table.add_row([col for col in row])

        with open(self.policy_file_helper, 'w') as w:
            w.write(str(table))
        w.close()

    """
    Function role is to define the reward function of the agent.
    """
    @staticmethod
    def get_reward(action):
        if "pick-food" in action[0]:  # The agent is in the state of the food
            return 10
        else:
            return -1

    def next_action(self):
        if self.services.goal_tracking.reached_all_goals():
            return None

        exp_exp_tradeoff = random.uniform(0, 1)  # Exploitation & Exploration tradeoff

        if exp_exp_tradeoff > self.epsilon:  # Agent => Exploitation
            action = self.get_action_by_exploitation()
        else:  # Agent => Exploration
            action = self.get_action_by_exploration()

        self.print_Q_table_to_file()  # Print the Q-table to the policy file
        self.print_Q_table_helper_to_file()  # Print the Q-table to the policy helper file
        self.epsilon *= 0.8

        return action

    """
    Function role is to return a tuple: (agent current state , state index in Q table).
    """
    def get_state_values(self, state_info):
        state = list(state_info['at'])[0][1]
        state_index = self.state_space.index(state)

        return state, state_index

    """
    Function role is to return the best action for out agent, using Epsilon Greedy algorithm:
    Given a specific state, we iterate over all the valid actions and choose the biggest Q value for this state.
    """
    def get_action_values(self, state_index, actions):
        first = True
        action_value = 0
        best_action = None
        action_index = None
        random.shuffle(actions)
        for action in actions:  # Iterate over all the valid actions
            if first:
                first = False
                best_action = action
                act = action.replace("(", "").replace(")", "").split(" ")[0]
                action_index = self.action_space.index(act)
                action_value = self.q_table[state_index, action_index]
            act = action.replace("(", "").replace(")", "").split(" ")[0]  # Extract the action name
            matrix_column = self.action_space.index(act)
            if action_value <= self.q_table[state_index, matrix_column]:  # Save the action by the biggest Q value
                best_action = action
                action_index = matrix_column
                action_value = self.q_table[state_index, matrix_column]

        return best_action, action_index

    """
    Function role is to check if the given action succeed or not (it's a probability action):
    Check if the "new" agent state is like we expected from the given action.
    """
    @staticmethod
    def is_action_affected(action, new_state):
        action_l = action.split()
        for _, agent_at in new_state['at']:
            if len(action_l) == 4 and action_l[-1].replace("(", "").replace(")", "") == agent_at:
                return True
        return False

    """
    Function role is to the execute the EXPLOITATION process: take the biggest Q value for the current state.
    """
    def get_action_by_exploitation(self):
        current_state = self.services.perception.get_state()
        actions = CustomizedValidActions.get(CustomizedValidActions(self.services.parser, self.services.perception),
                                             current_state)

        state = self.get_state_values(current_state)                # Get the agent location
        action = self.get_action_values(state[1], actions)          # Get the best valid action for the agent
        reward = self.get_reward(action)                            # Get the reward for the selected action

        # Execute the selected action from the current state & get the new agent location
        next_state = self.services.parser.copy_state(current_state)
        self.services.parser.apply_action_to_state(action[0], next_state, check_preconditions=False)
        new_state = self.get_state_values(next_state)

        # Update Q(s,a):= (1 - alpha) * Q(s,a) + alpha * [R(s,a) + gamma * max Q(s',a')]
        self.q_table[state[1], action[1]] = (1 - self.lr) * self.q_table[state[1], action[1]] + \
                                            self.lr * (reward + self.gamma * np.max(self.q_table[new_state[1], :]))

        if not self.is_action_affected(action[0], next_state):
            self.q_table[state[1], action[1]] -= 0.1

        return action[0]

    """
    Function role is to the execute the EXPLORATION process: choose a random action.
    """
    def get_action_by_exploration(self):
        current_state = self.services.perception.get_state()
        actions = CustomizedValidActions.get(CustomizedValidActions(self.services.parser, self.services.perception),
                                             current_state)

        state = self.get_state_values(current_state)                            # Get the agent location
        random_action = random.choice(actions)                                  # Get a random action
        act = random_action.replace("(", "").replace(")", "").split(" ")[0]     # Extract the action name
        matrix_column = self.action_space.index(act)                            # Extract the relevant column in Q table
        action = (act, matrix_column)
        reward = self.get_reward(action)                                        # Get the reward for the selected action

        # Execute the selected action from the current state & get the new agent location
        next_state = self.services.parser.copy_state(current_state)
        self.services.parser.apply_action_to_state(random_action, next_state, check_preconditions=False)
        new_state = self.get_state_values(next_state)

        # Update Q(s,a):= (1 - alpha) * Q(s,a) + alpha * [R(s,a) + gamma * max Q(s',a')]
        self.q_table[state[1], action[1]] = (1 - self.lr) * self.q_table[state[1], action[1]] + \
                                            self.lr * (reward + self.gamma * np.max(self.q_table[new_state[1], :]))

        if not self.is_action_affected(random_action, next_state):
            self.q_table[state[1], action[1]] -= 0.05

        return random_action


class LearnedPolicyExecutor(Executor):
    def __init__(self, policy_file):
        super(LearnedPolicyExecutor, self).__init__()
        self.policy_file = policy_file
        self.policy_file_helper = policy_file + "_HELPER"

    def initialize(self, services):
        self.services = services

    def next_action(self):
        if self.services.goal_tracking.reached_all_goals():
            return None
        return self.get_best_policy_action()

    @staticmethod
    def is_float(num):
        try:
            float(num)
            return True
        except ValueError:
            return False

    """
    Function role is to return the index of the best action, according to the calculated policy
    (agent_state_values = the relevant values according the agent position from the table, actions = legal actions, 
    action_names = actions names by the table content).
    """
    def get_best_action_index(self, agent_state_values, legal_actions, action_names):
        options = []
        max_value = -sys.maxsize - 1
        for i, val in enumerate(agent_state_values):
            if self.is_float(val) and float(val) >= max_value and action_names.split("|")[i].strip() in legal_actions:
                max_value = float(val)
                options.append(i)
        option = random.choice(options)                         # If there are several legal options - choose randomly
        return option

    """
    Function role is to return a the best action according to the calculated policy file.
    """
    def get_best_action(self, state_info, actions):
        state = list(state_info['at'])[0][1]
        n_actions = [action.replace("(", "").split(" ")[0] for action in actions]

        lines = open(self.policy_file_helper).read().splitlines()
        for i, line in enumerate(lines):                                            # Iterate over the lines in FILE
            if "|" in line and str(state) == line.split("|")[1].strip():            # Find the 'at' state in the FILE
                n_line = [col.replace(" ", "") for col in line.split("|")]
                best_action_index = self.get_best_action_index(n_line, n_actions, lines[1])
                best_action = lines[1].split("|")[best_action_index]
                return best_action.strip()

    """
    Function role is to return the best action according to the calculated policy.
    """
    def get_best_policy_action(self):
        current_state = self.services.perception.get_state()
        actions = CustomizedValidActions.get(CustomizedValidActions(self.services.parser, self.services.perception),
                                             current_state)

        best_action = self.get_best_action(current_state, actions)
        for action in actions:
            if best_action in action:
                return action


# Parse the input flag and determine which executor to run
if input_flag == '-L':
    print LocalSimulator().run(domain_path, problem_path, QLearningExecutor(policy_file_path))
elif input_flag == '-E':
    print LocalSimulator().run(domain_path, problem_path, LearnedPolicyExecutor(policy_file_path))
else:
    raise NameError("Wrong input flag")
