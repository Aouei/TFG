import numpy as np

def Random(env, seed, EPISODES, NAME):
	np.random.seed(seed)
	env.config()

	R = []
	RMean = []

	file = open("Resultados/Random "+NAME+"_.txt","a")

	for i in range(EPISODES):
		s = env.reset()
		done = False
		total_reward = 0

		while not done:
			action = env.action_space.sample()
			s, reward, done, _ = env.step(action)
			total_reward += reward

		file.write(str(total_reward)+" \n")
		R.append(total_reward)

	file.write("Mean: "+str(np.mean(R))+" \n")
	file.write("Std: "+str(np.std(R))+" \n")
	file.close()