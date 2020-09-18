import numpy as np

def AddState(Q, state, actions):
	if not state in Q:
		Q[state] = [0 for i in range(actions)]
	return Q

def Load(filename, actions):
	Q = {}
	with open(filename) as file:
		for line in file:
			aux = line.split(")")

			state = []
			for i in aux[0].split(","):
				i = i.replace('(', '')
				i = i.replace(',', '')
				state.append(round(float(i)))

			action = round(float(aux[1]))
			state = tuple(state)

			Q[state] = [0 for i in range(actions)]
			Q[state][action] = 1

	return Q

def QLearning(env, seed, LEARNINGRATE, FIRSTEPSILON, FINALEPSILON, GAMMA, EPISODES, MAXSTEPS, STEPS, NAME, exploration_type="", load=False):
	np.random.seed(seed)
	env.config()

	q_table = {}

	R = []
	RMean = []

	learning_rate = LEARNINGRATE
	epsilon = FIRSTEPSILON
	min_epsilon = FINALEPSILON
	decay = (epsilon - min_epsilon)/STEPS
	discount_factor = GAMMA

	if load:
		file = open("Resultados/Test QLearning "+NAME+"|a"+str(learning_rate)+"|"+exploration_type+" e"+str(FINALEPSILON)+"|g"+str(discount_factor)+".txt","a")
		q_table = Load("Resultados/QLearning table"+NAME+"|a"+str(learning_rate)+"|"+exploration_type+" e"+str(FINALEPSILON)+"|g"+str(discount_factor)+".txt", env.action_space.n)
	else:
		file = open("Resultados/QLearning "+NAME+"|a"+str(learning_rate)+"|"+exploration_type+" e"+str(FINALEPSILON)+"|g"+str(discount_factor)+".txt","a")

	for i in range(EPISODES):
		s = env.reset()
		total_reward = 0
		q_table = AddState(q_table, s, env.action_space.n)

		for j in range(MAXSTEPS):
			action = env.action_space.sample() if np.random.random() < epsilon else int(np.argmax(q_table[s]))
			s_, reward, done, _ = env.step(action)
			q_table = AddState(q_table, s_, env.action_space.n)
			total_reward += reward

			action_ = int(np.argmax(q_table[s_]))
			#action_ = np.random.randint(env.action_space.n)

			if not load:
				q_table[s][action] += learning_rate * (reward + discount_factor * q_table[s_][action_] - q_table[s][action])
			s = s_

			epsilon = max(min_epsilon, epsilon - decay)

			if done:
				break

		file.write(str(total_reward)+" \n")
		if (i+1) % 1000 == 0:
			print("Q Learning Episode",i,"G",total_reward,"Alfa",learning_rate,"Gamma",discount_factor,"Epsilon",epsilon)
		R.append(total_reward)
		RMean.append(np.mean(R))
	if load:
		file.write("Mean: "+str(np.mean(R))+" \n")
		file.write("Std: "+str(np.std(R))+" \n")
	file.close()

	if not load:
		file = open("Resultados/QLearning table"+NAME+"|a"+str(learning_rate)+"|"+exploration_type+" e"+str(epsilon)+"|g"+str(discount_factor)+".txt","a")
		for i in q_table:
			file.write(str(i)+" "+str(np.argmax(q_table[i]))+" \n")
		file.close()