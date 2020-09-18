from keras.models import Sequential, clone_model, load_model
from keras.layers import Dense
from keras.optimizers import Adam
from tensorflow.keras.losses import Huber
import numpy as np
from collections import deque
import random

class QAgent():
	def __init__(self, input_dim, epsilon=1.0, gamma = 0.99):

		self.model = Sequential()
		self.model.add(Dense(units=32, activation='relu', input_dim=input_dim))
		self.model.add(Dense(units=32, activation='relu'))
		self.model.add(Dense(units=50, activation='linear'))
		self.model.compile(loss="mse", optimizer=Adam(lr=0.001))

		self.epsilon = epsilon
		self.gamma = gamma

	def getAction(self,state):
		if np.random.rand() < self.epsilon:
			return np.random.randint(50)
		return np.argmax(self.model.predict(state)[0])

	def remember(self, state, action, reward, newstate, done):
		self.memory.append((state,action,reward,newstate,done))

	def learn(self, state, action, reward, newstate, done):
		TARGET = []
		
		Q = self.model.predict(state)[0]
		Q[action] = reward + self.gamma * np.amax(self.model.predict(newstate)[0])*(1-done)
		TARGET.append(Q)
		
		self.model.fit(state,np.array(TARGET),verbose=0)

	def load(self,filename):
		self.model = load_model(filename)


def TrainDQN(env, seed, epsilon, gamma, Maxruns):
	agent = QAgent(env.observation_space_n, epsilon, gamma)
	
	env.config()
	np.random.seed(seed)
	
	Best = -9999
	runs = 0
	
	file = open("Resultados/Train DQN "+env.NAME+"|e="+str(epsilon)+"|g="+str(gamma)+".txt","a")

	while runs < Maxruns:
		runs += 1
		s = env.reset()
		d = False
		total = 0
		while not d:
			a = agent.getAction(np.array([s]))
			s_, r, d, _ = env.step([(env.LIM_DIST/49)*a - env.LIM_MIN])
			total += r
			agent.learn(np.array([s]), a, r, np.array([s_]), int(d))
			s = s_
			if d:
				file.write(str(total)+" \n")

		if runs % 1000 == 0:
			print("Train DQN Episode:",runs,"G:",total,"Epsilon",round(agent.epsilon,4),"Gamma",gamma)
				
			current = 0

			for nn in range(100):
				s = env.reset()
				d = False
				total = 0

				while not d:
					a = agent.getAction(np.array([s]))
					s_, r, d, _ = env.step([(env.LIM_DIST/49)*a - env.LIM_MIN])
					total += r
					if d:
						break
				current += total
			current /= 100

			if Best < current:
				Best = current
				agent.model.save("Resultados/DQN "+env.NAME+"|e="+str(epsilon)+"|g="+str(gamma)+".h5")
			print("Current",current,"Best",Best)

	file.close()

def TestDQN(env, seed, epsilon, gamma, Maxruns):
	agent = QAgent(env.observation_space_n, epsilon, gamma)
	agent.load("Resultados/DQN "+env.NAME+"|e="+str(epsilon)+"|g="+str(gamma)+".h5")

	env.config()
	np.random.seed(seed)

	runs = 0

	file = open("Resultados/Test DQN "+env.NAME+"|e="+str(epsilon)+"|g="+str(gamma)+".txt","a")

	while runs < Maxruns:
		runs += 1
		s = env.reset()
		d = False
		total = 0
		while not d:
			a = agent.getAction(np.array([s]))
			s_, r, d, _ = env.step([(env.LIM_DIST/49)*a - env.LIM_MIN])
			total += r
			s = s_
			if d:
				file.write(str(total)+" \n")
	file.close()