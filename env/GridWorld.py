import numpy as np
import sys

np.set_printoptions(suppress=True)

class GridWorld:
    def __init__(self,GRID_SIZE=50):
        self.grid_size = GRID_SIZE
        self.num_states = self.grid_size**2
        self.rewards = np.random.choice([0,-1],p=[0.67,0.33],size=(self.grid_size,self.grid_size))
        self.rewards[0,self.grid_size-1] = 1
        self.goal_state = self.grid_size-1
        self.actions = np.array(["up","down","left","right"])
        
    def get_feature(self,state):
        _ = [0]*self.num_states
        a = _


        # print("_: \n", _)
        # sys.exit()

        #print("a:", a)

        if (state >= len(a)):
            print("stete:\n", state)
            print("_: \n", len(_))
            sys.exit()
        _[state] = 1
        return _
        
    def get_rewards(self):
        aaa = self.rewards.flatten()
        #print()
        return self.rewards.flatten()
    
    def result_of_action(self,state,action):
        state_coords = ((int)(state/self.grid_size), state%self.grid_size)
        next_states = [(max(0,state_coords[0]-1),state_coords[1]),(min(self.grid_size-1,state_coords[0]+1),state_coords[1]),\
                      (state_coords[0],max(0,state_coords[1]-1)),(state_coords[0],min(self.grid_size-1,state_coords[1]+1))]
        transition_probs = 0.1*np.ones((len(self.actions)))
        transition_probs[np.where(self.actions == action)[0][0]] = 0.7
        next_state = next_states[np.random.choice(list(range(len(next_states))),p=transition_probs)]
        return (int)(next_state[0]*self.grid_size+next_state[1])
    
    def generate_trajectory(self,policy=None,num_trajectories=10):
        if policy is None:
            policy = np.random.choice(self.actions,size=(self.num_states))
        trajectories = []
        for i in range(num_trajectories):
            trajectory = []
            current_state = np.random.randint(self.num_states)
            print("current_state:", current_state)
            while current_state != self.goal_state and len(trajectory) < self.grid_size*3:

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
                current_state = self.result_of_action(current_state,policy[current_state])
            if current_state == self.goal_state:
                trajectory.append(self.get_feature(self.goal_state))
            trajectories.append(np.array(trajectory))
        return np.array(trajectories)
        
    
    def get_transition_probabilities(self,state,action):
        '''While calculating the transition probabilities, we make the assumption that if you were in a cell along
        the border, and you tried to make a transition outside the border with probability p, you end up not
        moving with the same probability p.'''



        transition_probs = np.zeros((self.grid_size,self.grid_size))
        state_coords = ((int)(state/self.grid_size), state%self.grid_size)
        #
        # print("transition prob")
        # print(transition_probs)
        # print("state coord")
        # print(state_coords)
        # print(state_coords[0])
        # print(transition_probs[state_coords[0]])
        # print(max(0, state_coords[1] - 1))
        # print(transition_probs[state_coords[0], max(0, state_coords[1] - 1)])

        if action == "up":
            transition_probs[max(0,state_coords[0]-1),state_coords[1]] += 0.7 # up
            transition_probs[state_coords[0],max(0,state_coords[1]-1)] += 0.1 # left
            transition_probs[min(self.grid_size-1,state_coords[0]+1),state_coords[1]] += 0.1 # down
            transition_probs[state_coords[0],min(self.grid_size-1,state_coords[1]+1)] += 0.1 #right
        elif action == "down":
            transition_probs[max(0,state_coords[0]-1),state_coords[1]] += 0.1
            transition_probs[state_coords[0],max(0,state_coords[1]-1)] += 0.1
            transition_probs[min(self.grid_size-1,state_coords[0]+1),state_coords[1]] += 0.7
            transition_probs[state_coords[0],min(self.grid_size-1,state_coords[1]+1)] += 0.1
        elif action == "left":
            transition_probs[max(0,state_coords[0]-1),state_coords[1]] += 0.1
            transition_probs[state_coords[0],max(0,state_coords[1]-1)] += 0.7
            transition_probs[min(self.grid_size-1,state_coords[0]+1),state_coords[1]] += 0.1
            transition_probs[state_coords[0],min(self.grid_size-1,state_coords[1]+1)] += 0.1
        elif action == "right":
            transition_probs[max(0,state_coords[0]-1),state_coords[1]] += 0.1
            transition_probs[state_coords[0],max(0,state_coords[1]-1)] += 0.1
            transition_probs[min(self.grid_size-1,state_coords[0]+1),state_coords[1]] += 0.1
            transition_probs[state_coords[0],min(self.grid_size-1,state_coords[1]+1)] += 0.7
        return transition_probs.flatten()
    
    def take_greedy_action(self,values):
        values = values.reshape(self.grid_size,self.grid_size)
        policy = np.repeat("random",self.num_states)
        for i in range(self.num_states):
            state_coords = ((int)(i/self.grid_size),i%self.grid_size)
            policy[i] = self.actions[np.argmax([values[max(0,state_coords[0]-1),state_coords[1]],
                                                values[min(self.grid_size-1,state_coords[0]+1),state_coords[1]],
                                                values[state_coords[0],max(0,state_coords[1]-1)],
                                                values[state_coords[0],min(self.grid_size-1,state_coords[1]+1)]])]
        return policy