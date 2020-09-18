from keras.models import Sequential, clone_model, load_model
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
from collections import deque
import random

class Topology():
	def __init__(self, type_, learning_rate, in_dim, out_dim, error="mse"):
		self.model = Sequential()
		if type_ == "Lineal":
			self.model.add(Dense(units=out_dim, activation='linear', input_dim=in_dim))
		elif type_ == "2Capas":
			self.model.add(Dense(units=32, activation='relu', input_dim=in_dim))
			self.model.add(Dense(units=32, activation='relu'))
			self.model.add(Dense(units=out_dim, activation='linear'))
		self.model.compile(loss=error, optimizer=Adam(lr=learning_rate))
		self.model.summary()
		
class DeepQAgent():
	def __init__(self, env, learning_rate, gamma, first_epsilon, final_epsilon, episodes_decay, type_, exp_replay=0, ddqn=0,exploration_type=""):
		self.n = env.action_space.n
		self.env = env
		self.model = Topology(type_, learning_rate, env.observation_space_n, env.action_space.n)
		self.target = Topology(type_, learning_rate, env.observation_space_n, env.action_space.n)
		self.learning_rate = learning_rate
		self.gamma = gamma
		self.epsilon = first_epsilon
		self.epsilon_min = final_epsilon
		self.epsilon_decay = (self.epsilon - self.epsilon_min)/episodes_decay
		self.type_ = type_
		self.memory = deque(maxlen=exp_replay)
		self.ddqn = ddqn
		self.exploration_type = exploration_type

	def getAction(self,state):
		if np.random.random() < self.epsilon:
			return self.env.action_space.sample()
		return np.argmax(self.model.model.predict(np.array([state]))[0])

	def decay(self):
		self.epsilon = max(self.epsilon_min, self.epsilon -  self.epsilon_decay)

	def learn(self,state,action,reward,newstate,done):
		Q = self.model.model.predict(np.array([state]))[0]
		Qnew = self.model.model.predict(np.array([newstate]))[0]
		newaction = np.argmax(Qnew)
		Q[action] = reward + self.gamma * Qnew[newaction]*(1-done)
		self.model.model.fit(np.array([state]),np.array([Q]),verbose=0)
		
	def remember(self,state,action,reward,newstate,done):
		self.memory.append([state, action, reward, newstate, done])

	def batch(self, size):
		if len(self.memory) >= size:
			STATE = []
			TARGET = []
			minibatch = random.sample(self.memory, size)
			for state, action, reward, newstate, done in minibatch:
				if not self.ddqn:
					Q = self.model.model.predict(np.array([state]))[0]
					Q[action] = reward + self.gamma * np.amax(self.model.model.predict(np.array([newstate]))[0])*(1-int(done))
				else:
					Q = self.model.model.predict(np.array([state]))[0]
					Qnew = self.target.model.predict(np.array([newstate]))[0]
					newaction = np.argmax(self.model.model.predict(np.array([newstate]))[0])
					Q[action] = reward + self.gamma * Qnew[newaction] * (1-int(done))
				STATE.append(state)
				TARGET.append(Q)
			
			self.model.model.fit(np.array(STATE),np.array(TARGET),verbose=0).history["loss"]

	def update(self):
		self.target.model.set_weights(self.model.model.get_weights())

	def load(self):
		self.model.model = load_model("Resultados/DeepQLearning "+self.env.NAME+"|a"+str(self.learning_rate)+"|"+self.exploration_type+" e"+str(self.epsilon_min)+"|g"+str(self.gamma)+"|t"+self.type_+".h5",compile=False)

	def save(self):
		self.model.model.save("Resultados/DeepQLearning "+self.env.NAME+"|a"+str(self.learning_rate)+"|"+self.exploration_type+" e"+str(self.epsilon_min)+"|g"+str(self.gamma)+"|t"+self.type_+".h5")

def DeepQLearning(env, seed, LEARNINGRATE, FIRSTEPSILON, FINALEPSILON, GAMMA, EPISODES, MAXSTEPS, STEPS, NAME, TOPOLOGY, exploration_type="", ER=[0,0], DDQN=False,load=False):
	np.random.seed(seed)
	env.config()

	R = []

	agent = DeepQAgent(env, LEARNINGRATE, GAMMA, FIRSTEPSILON, FINALEPSILON, STEPS, TOPOLOGY, ER[0], DDQN, exploration_type)

	if load:
		file = open("Resultados/Test DeepQLearning "+NAME+"|a"+str(LEARNINGRATE)+"|"+exploration_type+" e"+str(FINALEPSILON)+"|g"+str(GAMMA)+"|t"+TOPOLOGY+".txt","a")
		agent.load()	
	else:
		file = open("Resultados/DeepQLearning "+NAME+"|a"+str(LEARNINGRATE)+"|"+exploration_type+" e"+str(FINALEPSILON)+"|g"+str(GAMMA)+"|t"+TOPOLOGY+".txt","a")

	if ER[0] != 0:
		for i in range(10):
			s = env.reset()
			for j in range(200):
				a = env.action_space.sample()
				s_, r, d, _ = env.step(a)
				agent.remember(s,a,r,s_,d)
				s = s_
				if d:
					break

	best_test = -10000

	for i in range(EPISODES):
		s = env.reset()
		total_reward = 0

		for j in range(MAXSTEPS):
			action = agent.getAction(s)
			s_, reward, done, _ = env.step(action)
			total_reward += reward

			if not load:
				if ER[0] == 0:
					agent.learn(s, action, reward, s_, done or (j+1)==MAXSTEPS)
				else:
					agent.remember(s, action, reward, s_, done or (j+1)==MAXSTEPS)
					if (j+1) % ER[1] == 0:
						agent.batch(32)
				agent.decay()
			s = s_

			if done or (j+1)==MAXSTEPS:
				R.append(total_reward)
				print("Game",NAME,"Deep Q Learning Episode",i,"G",round(total_reward,4),"Mean",round(np.mean(R),4),"Alfa",LEARNINGRATE,"Gamma",GAMMA,"Epsilon",round(agent.epsilon,4),"Topology",TOPOLOGY,"DDQN",DDQN)
				break

		if DDQN and (i+1) % 5 == 0 and not load:
			agent.update()

		if (i+1) % 10 == 0 and not load:
			current_epsilon = agent.epsilon
			current_test = test(env, agent, MAXSTEPS, FINALEPSILON)
			agent.epsilon = current_epsilon

			if best_test < current_test:
				best_test = current_test
				agent.save()
			print("Test:",current_test,"Best:",best_test)

		file.write(str(total_reward)+" \n")
	if load:
		file.write("Mean: "+str(np.mean(R))+" \n")
		file.write("Std: "+str(np.std(R))+" \n")
	file.close()

def test(env, agent, MAXSTEPS, epsilon):
	agent.epsilon = epsilon
	R = []
	for i in range(100):
		s = env.reset()
		total_reward = 0

		for j in range(MAXSTEPS):
			action = agent.getAction(s)
			s_, reward, done, _ = env.step(action)
			total_reward += reward
			s = s_

			if done:
				break
		R.append(total_reward)

	return np.mean(R)