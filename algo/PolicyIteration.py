import numpy as np
import gym
from gym.envs.toy_text import  discrete

class PolicyIteration:

    def __init__(self,env):
        self.env = env
        self.gamma = 0.99

        if (isinstance(env, discrete.DiscreteEnv)):
            self.is_gym_discrete_env = True
        else:
            self.is_gym_discrete_env = False
            #for discrete gym env

        if (self.is_gym_discrete_env):
            self.values = np.zeros((env.nS,))
            self.policy = np.zeros(env.nS)
            for i in range(env.nS):
                transition_dict = env.P[i]
                self.policy[i] = (np.random.choice((list)(transition_dict.keys())))

            # print(self.policy)
            # print()

            #self.policy = np.random.choice(self.env.actions, size=(self.env.num_states))
        # if (hasattr(self.env, 'nS')):
        #     self.values = np.zeros((self.env.nS,))
        #     self.policy = np.random.choice(self.env.actions, size=(self.env.num_states))
        else:
            self.values = np.zeros((self.env.num_states,))
            self.policy = np.random.choice(self.env.actions, size=(self.env.num_states))

    
    def policy_evaluation(self,num_iters=10,gamma=0.99):
        for i in range(num_iters):
            if (self.is_gym_discrete_env):
                transition_probs = np.zeros((self.env.nS, self.env.nS))
                rewards = np.zeros(self.env.nS)
                for j in range(self.env.nS):
                    p_list = self.env.P[j][self.policy[j]]
                    for k in p_list:
                        prob = k[0]
                        next_s = k[1]
                        value_s = k[2]
                        transition_probs[j][next_s] = prob

                        rewards[next_s] = value_s

                for j in range(self.env.nS):
                    value = 0
                    p_list = self.env.P[j][self.policy[j]]
                    #policyが確率的な場合は，ここのループが一個増える
                    for k in p_list:
                        prob = k[0]
                        next_s = k[1]
                        reward = k[2]

                        value += prob * (reward + gamma * self.values[next_s])

                    self.values[j] = value

                # pass
                # print()
                # print(rewards)
                # print()
                # transition_probs[j]
            else:
                transition_probs = np.zeros((self.env.num_states,self.env.num_states))

                #print(transition_probs.shape)
                for j in range(self.env.num_states):
                    transition_probs[j] = self.env.get_transition_probabilities(j,self.policy[j])
                # print(transition_probs)
                self.values = self.env.get_rewards() + gamma * np.dot(transition_probs, self.values)
                # print(self.values)
                # print()

    def policy_iteration(self,num_iters=10):
        #print(self.values)
        for i in range(num_iters):
            if (self.is_gym_discrete_env):
                self.policy_evaluation()
                for s in range(self.env.nS):
                    Q_values = np.zeros(len(self.env.P[s]))
                    for q_values in range(len(Q_values)):
                        Q_value = 0
                        p_list = self.env.P[s][q_values]
                        for transition_p in p_list:
                            prob = transition_p[0]
                            next_s = transition_p[1]
                            reward = transition_p[2]

                            if (reward >= 0):
                                pass

                            Q_value += prob * (reward + self.gamma * self.values[next_s])
                            #print()
                        Q_values[q_values] = Q_value

                    self.policy[s] = np.argmax(Q_values)
            else:
                self.policy_evaluation()
                self.policy = self.env.take_greedy_action(self.values)


        return self.policy

    def get_feature(self, state):
        if (self.is_gym_discrete_env):
            feature_vec = [0] * self.env.nS
            feature_vec[state] = 1
        else:
            feature_vec = [0] * self.num_states
            feature_vec[state] = 1

        return feature_vec

    def generate_trajectory(self, policy=None, num_trajectories=10):

        if policy is None:
            if (self.is_gym_discrete_env):
                #values = np.zeros((self.env.nS,))
                policy = np.zeros(self.env.nS)
                for i in range(self.env.nS):
                    transition_dict = self.env.P[i]
                    policy[i] = (np.random.choice((list)(transition_dict.keys())))
            else:
                policy = np.random.choice(self.actions, size=(self.num_states))

        trajectories = []

        for i in range(num_trajectories):
            trajectory = []
            if (self.is_gym_discrete_env):
                current_state = self.env.reset()
                while (len(trajectory) < self.env.nrow * 3):
                    trajectory.append(self.get_feature(current_state))
                    next_s = self.env.step(policy[current_state])
                    #done check
                    if (next_s[2]):
                        trajectory.append(self.get_feature(next_s[0]))
                        break
                    else:
                        current_state = next_s[0]

                trajectories.append(np.array(trajectory))

            else:
                current_state = np.random.randint(self.num_states)
                #print("current_state:", current_state)
                while current_state != self.goal_state and len(trajectory) < self.grid_size * 3:

                    # if (current_state > self.num_states):
                    #     print("state num error at:")
                    #     print(current_state)
                    #     sys.exit()
                    #
                    # if (type(current_state) is not int):
                    #     print("An error has occurred! at:")
                    #     print(current_state)
                    #     sys.exit()

                    trajectory.append(self.get_feature(current_state))
                    current_state = self.result_of_action(current_state, policy[current_state])
                if current_state == self.goal_state:
                    trajectory.append(self.get_feature(self.goal_state))
                trajectories.append(np.array(trajectory))

        return np.array(trajectories)
