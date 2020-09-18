from keras.models import Sequential, clone_model, load_model
from keras.layers import Dense
from keras.optimizers import Adam
from tensorflow.keras.losses import Huber
import numpy as np
from collections import deque
import random

class QAgent():
	def __init__(self, input_dim, epsilon=1.0, gamma = 0.99, exp_replay=2000):
		self.memory = deque(maxlen=exp_replay)
		self.model = Sequential()
		self.model.add(Dense(units=32, activation='relu', input_dim=input_dim))
		self.model.add(Dense(units=32, activation='relu'))
		self.model.add(Dense(units=50, activation='linear'))
		self.model.compile(loss="mse", optimizer=Adam(lr=0.001))

		self.target = clone_model(self.model)

		self.epsilon = epsilon
		self.gamma = gamma

	def getAction(self,state):
		if np.random.rand() < self.epsilon:
			return np.random.randint(50)
		return np.argmax(self.model.predict(state)[0])

	def remember(self, state, action, reward, newstate, done):
		self.memory.append((state,action,reward,newstate,done))

	def trainBatch(self, size):
		STATE = []
		TARGET = []
		minibatch = random.sample(self.memory, size)
		for state, action, reward, newstate, done in minibatch:
			Q = self.model.predict(state)[0]
			Q[action] = reward + self.gamma * np.amax(self.target.predict(newstate)[0])*(1-done)
			STATE.append(state[0])
			TARGET.append(Q)
		
		self.model.fit(np.array(STATE),np.array(TARGET),verbose=0,epochs=10).history["loss"]

	def update(self):
		self.target.set_weights(self.model.get_weights())

	def load(self,filename):
		self.model = load_model(filename)

def TrainDDQN(env, seed, epsilon, gamma, exp_replay, Maxruns):
	agent = QAgent(env.observation_space_n, epsilon, gamma, exp_replay)
	
	env.config()
	np.random.seed(seed)
	
	Best = -9999
	runs = 0
	
	file = open("Resultados/Train DDQN ER"+env.NAME+"|e="+str(epsilon)+"|g="+str(gamma)+".txt","a")

	for i in range(100):
		s = env.reset()
		d = False
		while not d:
			a = np.random.randint(50)
			s_, r, d, _ = env.step([(env.LIM_DIST/49)*a - env.LIM_MIN])
			agent.remember(np.array([s]), a, r, np.array([s_]), int(d))
			s = s_


	while runs < Maxruns:
		runs += 1
		s = env.reset()
		d = False
		total = 0
		step = 0
		while not d:
			step += 1
			a = agent.getAction(np.array([s]))
			s_, r, d, _ = env.step([(env.LIM_DIST/49)*a - env.LIM_MIN])
			total += r
			agent.remember(np.array([s]), a, r, np.array([s_]), int(d))

			if step % 4 == 0:
				agent.trainBatch(32)

			s = s_
			if d:
				file.write(str(total)+" \n")

		if runs % 5 == 0:
			agent.update()

		if runs % 1000 == 0:
			print("Train DDQN ER Episode:",runs,"G:",total,"Epsilon",round(agent.epsilon,4),"Gamma",gamma)
				
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
				agent.model.save("Resultados/DDQN ER"+env.NAME+"|e="+str(epsilon)+"|g="+str(gamma)+".h5")
			print("Current",current,"Best",Best)

	file.close()

def TestDDQN(env, seed, epsilon, gamma, exp_replay, Maxruns):
	agent = QAgent(env.observation_space_n, epsilon, gamma, exp_replay)
	agent.load("Resultados/DDQN ER"+env.NAME+"|e="+str(epsilon)+"|g="+str(gamma)+".h5")

	env.config()
	np.random.seed(seed)

	runs = 0

	file = open("Resultados/Test DDQN ER"+env.NAME+"|e="+str(epsilon)+"|g="+str(gamma)+".txt","a")

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