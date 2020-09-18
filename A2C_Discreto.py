from keras.models import Model, load_model
from keras.layers import Dense, Input
from keras.optimizers import Adam
import keras.backend as K
import numpy as np
from collections import deque
import random

class ActorCritic(object): 
	def __init__(self, input_dim, actions, gamma=1.0, lmbda=0.95, alpha=1.0, beta=1.0, clip=[0,1], normalizar=False):
		super(ActorCritic, self).__init__()
		self.EPS = np.finfo(np.float32).eps.item()
		self.n = actions
		self.gamma = gamma
		self.lmbda = lmbda
		self.alpha = alpha
		self.normalizar = normalizar
		self.beta = beta
		self.clip = clip
		self.create_model(input_dim)

	def create_model(self, input_dim):
		inputlayer = Input(shape=(input_dim,),name="InputLayer")
		delta = Input(shape=[1],name="Delta")
		hidden = Dense(32,name="Input",activation="relu")(inputlayer)
		hidden1 = Dense(32,name="Input1",activation="relu")(hidden)
		outputt = Dense(self.n, activation='softmax',name="Output")(hidden1)
		values = Dense(1,activation="linear")(hidden1)

		def custorerror(y_true,y_pred):
			out = K.clip(y_pred,self.clip[0],self.clip[1])
			log = y_true*K.log(out)
			return K.sum(-log*delta)

		self.actor = Model(inputs=[inputlayer,delta],outputs=outputt)
		self.actor.compile(optimizer=Adam(lr=self.alpha),loss=custorerror)
		self.actor.summary()

		self.critic = Model(inputs=[inputlayer],outputs=values)
		self.critic.compile(optimizer=Adam(lr=self.beta),loss="mse")
    
		self.policy = Model(inputs=[inputlayer],outputs=outputt)

		self.states = []
		self.values = []
		self.actions = []
		self.rewards = []
		self.dones = []

	def getAction(self,state):
		state = np.array([state])
		return np.random.choice(self.n,p=self.actor.predict({"InputLayer":state,"Delta":np.zeros(1)})[0])

	def remember(self,state,action,reward,done):
		self.states.append(state)
		actions = [ 0. for i in range(self.n)]
		actions[action] = 1.
		self.values.append( self.critic.predict(np.array([state]))[0] )
		self.actions.append(actions)
		self.rewards.append(reward)
		self.dones.append(not done)

	def get_advantages(self,values, dones, rewards):
	    returns = []
	    gae = 0
	    for i in reversed(range(len(rewards))):
	    	delta = rewards[i] + self.gamma * values[i + 1] * dones[i] - values[i]
	    	gae = delta + self.gamma * self.lmbda * dones[i] * gae
	    	returns.insert(0, gae + values[i])

	    remember = np.array(returns)
	    if self.normalizar:
		    returns = (returns - np.mean(returns)) / (np.std(returns) + self.EPS)
	    adv = np.array(returns) - values[:-1]
	    return returns, adv

	def train_one_shot(self, state, action, reward, newstate, done):
		actions = [ 0.0 for i in range(self.n)]
		actions[action] = 1.0
		returns = reward + self.gamma*self.critic.predict(np.array([newstate]))[0]*(1-done)
		real_advantage = returns - self.critic.predict(np.array([state]))[0]
		self.actor.fit([np.array([state]), np.array([real_advantage])], np.array([actions]), verbose=0)
		self.critic.fit(np.array([state]), np.array([returns]), verbose=0)

	def train_episodic(self, state):
		self.values.append( self.critic.predict(np.array([state]))[0] )

		returns, advantage = self.get_advantages(self.values,self.dones,self.rewards)

		self.actions = np.array(self.actions)
		self.states = np.array(self.states)
		real_advantage = []
		for i in advantage:
			real_advantage.append(i[0])
		real_advantage = np.array(real_advantage)

		self.actor.fit([self.states,real_advantage], self.actions, verbose=0)
		self.critic.fit(self.states, returns, verbose=0)

		self.states = []
		self.values = []
		self.actions = []
		self.rewards = []
		self.dones = []

	def save(self,filename):
		self.actor.save(filename + "Actor.h5")

	def load(self,filename):
		self.actor =  load_model(filename + "Actor.h5",compile=False)

def TrainEpisodic(env, seed, gamma=0.99, lmbda=0.95, alpha=0.0001, beta=0.001, clip=[0,1], normalizar=False, Max_episodes = 5000):
  np.random.seed(seed)
  agent = ActorCritic(env.observation_space_n, env.action_space.n, gamma, lmbda, alpha, beta, clip, normalizar)

  file = open("Resultados/Train A2C episodic "+env.NAME+" .txt","a")
  M = []
  BEST = -9999
  for i in range(Max_episodes):
    s = env.reset()
    d = False
    t = 0

    while not d:
      a = agent.getAction(s)
      s_, r, d, _ = env.step(a)
      t += r
      agent.remember(s, a, r, d)
      s = s_
    agent.train_episodic(s)
    file.write(str(t)+"\n")
    M.append(t)
    print("Episodio",i,"G",t,"Media",np.mean(M))

    if (i+1) % 1000 == 0:
      current = 0
      for j in range(100):
        s = env.reset()
        d = False
        while not d:
          a = agent.getAction(s)
          s, r, d, _ = env.step(a)
          current += r
      current /= 100

      if current > BEST:
        BEST = current
        agent.save("Resultados/A2C episodic "+env.NAME)
      print("Best",BEST,"Current",current)
  file.close()

def TrainOneStep(env, seed, gamma=0.99, lmbda=0.95, alpha=0.0001, beta=0.001, clip=[0,1], normalizar=False, Max_episodes = 5000):
  np.random.seed(seed)
  agent = ActorCritic(env.observation_space_n, env.action_space.n, gamma, lmbda, alpha, beta, clip, normalizar)

  file = open("Resultados/Train A2C one-step "+env.NAME+" .txt","a")
  M = []
  BEST = -9999
  for i in range(Max_episodes):
    s = env.reset()
    d = False
    t = 0

    while not d:
      a = agent.getAction(s)
      s_, r, d, _ = env.step(a)
      t += r
      agent.train_one_shot(s, a, r, s_, d)
      s = s_
    file.write(str(t)+"\n")
    M.append(t)
    print("Episodio",i,"G",t,"Media",np.mean(M))

    if (i+1) % 1000 == 0:
      current = 0
      for j in range(100):
        s = env.reset()
        d = False
        while not d:
          a = agent.getAction(s)
          s, r, d, _ = env.step(a)
          current += r
      current /= 100

      if current > BEST:
        BEST = current
        agent.save("Resultados/A2C one-step "+env.NAME)
      print("Best",BEST,"Current",current)
  file.close()

def Test(env, seed, Max_episodes = 5000, typee="episodic"):
  np.random.seed(seed)
  agent = ActorCritic(env.observation_space_n, env.action_space.n)
  agent.load("Resultados/A2C "+typee+" "+env.NAME)
  file = open("Resultados/Test A2C "+typee+" "+env.NAME+" .txt","a")
  M = []
  for i in range(Max_episodes):
    s = env.reset()
    d = False
    t = 0

    while not d:
      a = agent.getAction(s)
      s , r, d, _ = env.step(a)
      t += r
    file.write(str(t)+"\n")
    M.append(t)
    print("Episodio",i,"G",t,"Media",np.mean(M))
  file.close()