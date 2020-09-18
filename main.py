import SARSA as sarsa
import QLearning as qlearning
import DeepQLearning as deepQLeaning
import Random as random
import DQN_Continuo as DQNCon
import DDQN_Continuo as DDQNCon
import A2C_Discreto as A2C_D
import A2C_Continuo as A2C_C

import Games as games

seed = 1234
env = games.Tiger(seed)
A2C_D.TrainEpisodic(env, seed, gamma=0.75, lmbda=0.95, alpha=0.001, beta=0.001, clip=[0.1,0.9], normalizar=True, Max_episodes = 10000)
A2C_D.TrainOneStep(env, seed, gamma=0.75, lmbda=0.95, alpha=0.001, beta=0.001, clip=[0.1,0.9], normalizar=True, Max_episodes = 10000)
A2C_D.Test(env, seed, 100, "episodic")
A2C_D.Test(env, seed, 100, "one-step")

random.Random(env, seed, 100, env.NAME)

seed = 1234
env = games.MountainCarContinuous(seed)

for i in [0.1, 0.5, 1.0]:
	A2C_D.TrainEpisodic(env, seed, gamma=0.75, lmbda=0.95, alpha=0.001, beta=0.001, noise=i, Max_episodes = 5000)
	A2C_D.TrainOneStep(env, seed, gamma=0.75, lmbda=0.95, alpha=0.001, beta=0.001, noise=i, Max_episodes = 5000)
	A2C_D.Test(env, seed, 100, "episodic")
	A2C_D.Test(env, seed, 100, "one-step")

random.Random(env, seed, 100, env.NAME)