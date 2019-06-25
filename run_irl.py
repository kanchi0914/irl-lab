from env.GridWorld import GridWorld
from algo.PolicyIteration import PolicyIteration
from algo.RelEntIRL import RelEntIRL
from my_env.envs.deep_sea_test import DeepSeaEnv
import gym
import numpy as np



# Obtain the optimal policy for the environment to generate expert demonstrations
#


if (False):
    env = gym.make("DeepSeaEnv-v0")
    pi = PolicyIteration(env)

    optimal_policy = pi.policy_iteration(2000)
    expert_demos = pi.generate_trajectory(optimal_policy)
    nonoptimal_demos = pi.generate_trajectory()

    print("expert:\n", expert_demos[0].shape)

    for i in range(expert_demos[0].shape[0]):
        print("state:", (list)(expert_demos[0][i]).index(1))

    # expert_demos = gw.generate_trajectory(optimal_policy)
    # nonoptimal_demos = gw.generate_trajectory()

    relent = RelEntIRL(expert_demos, nonoptimal_demos)

    # Train the model with default hyperparameters.
    relent.train()
    print(relent.weights.reshape(env.nrow, env.ncol))

else:

    grid_size = 4
    gw = GridWorld(grid_size)

    pi = PolicyIteration(gw)

    optimal_policy = pi.policy_iteration(2000)
    expert_demos = gw.generate_trajectory(optimal_policy)
    nonoptimal_demos = gw.generate_trajectory()

    print(gw.rewards)

    #print("expert:\n", expert_demos[0].shape)
    for i in range(expert_demos[0].shape[0]):
        print("state:", (list)(expert_demos[0][i]).index(1))

    relent = RelEntIRL(expert_demos, nonoptimal_demos)

    # Train the model with default hyperparameters.
    relent.train()
    print(relent.weights.reshape(grid_size, grid_size))


