import gym
import gym_maze
import numpy as np
import random

class Mountain():
	def __init__(self, seed, n):
		self.NAME = "MountainCar-v0"
		self.env = gym.make(self.NAME)
		self.action_space = self.env.action_space
		self.seed = seed
		self.config()
		self.posContainer = np.linspace(-1.2,.6,n)
		self.velContainer = np.linspace(-.07,.07,n)
		self.observation_space_n = self.env.observation_space.shape[0]

	def config(self):
		self.env.seed(self.seed)
		self.env.action_space.seed(self.seed)

	def step(self, action):
		s, r, d, _ = self.env.step(action)
		s = self.preprocess(s)
		return s, r, d, _

	def reset(self):
		s = self.env.reset()
		s = self.preprocess(s)
		return s

	def preprocess(self, state):
		pos, vel = state
		pos = np.digitize(pos,self.posContainer)
		vel = np.digitize(vel,self.velContainer)
		return (pos,vel)

	def render(self):
		return self.env.render()

class CartPole():
	def __init__(self, seed, n):
		self.NAME = "CartPole-v0"
		self.env = gym.make(self.NAME)
		self.action_space = self.env.action_space
		self.seed = seed
		self.config()
		self.posContainer = np.linspace(-2.4,2.4,n)
		self.velContainer = np.linspace(-5,5,n)
		self.angleContainer = np.linspace(-0.418,0.418,n)
		self.vel1Container = np.linspace(-5,5,n)
		self.observation_space_n = self.env.observation_space.shape[0]

	def config(self):
		self.env.seed(self.seed)
		self.env.action_space.seed(self.seed)

	def step(self, action):
		s, r, d, _ = self.env.step(action)
		s = self.preprocess(s)
		return s, r, d, _

	def reset(self):
		s = self.env.reset()
		s = self.preprocess(s)
		return s

	def preprocess(self, state):
		pos, vel, angle, vel1 = state
		pos = np.digitize(pos,self.posContainer)
		vel = np.digitize(vel,self.velContainer)
		angle = np.digitize(angle,self.angleContainer)
		vel1 = np.digitize(vel1,self.vel1Container)
		return (pos,vel,angle,vel1)

	def render(self):
		return self.env.render()

class Maze():
	def __init__(self, seed, n):
		self.NAME = "maze-sample-"+str(n)+"x"+str(n)+"-v0"
		self.env = gym.make(self.NAME)
		self.action_space = self.env.action_space
		self.seed = seed
		self.config()
		self.observation_space_n = self.env.observation_space.shape[0]
		self.MAXSTEP = n*n
		self.steps = 0

	def step(self, action):
		self.steps += 1
		s, r, d, _ = self.env.step(int(action))
		d = d or self.steps >= self.MAXSTEP
		return tuple(s), r, d, _

	def reset(self):
		self.steps = 0
		s = self.env.reset()
		return tuple(s)

	def render(self):
		return self.env.render()

	def config(self):
		self.env.seed(self.seed)
		self.env.action_space.seed(self.seed)

class action_space():
	def __init__(self,seed,n):
		self.n = n
		self.seed(seed)

	def seed(self,seed):
		np.random.seed(seed)

	def sample(self):
		return np.random.choice(self.n)

class Cliff():
	def __init__(self, seed):
		self.NAME = "Cliff"
		self.seed = seed
		n = -100
		self.Map = [ [-1, -1, -1, -1, -1],
		        	 [-1, -1, -1, -1, -1],
		        	 [-1, -1, -1, -1, -1],
		        	 [-1, n, n, n, 1],
		]
		self.action_space = action_space(seed,4)
		self.observation_space_n = 2
		self.n = 0

	def step(self, action):
		self.n += 1
		if action == 0: #UP
			self.state[0] = max(0, self.state[0] - 1)
		elif action == 1: #DOWN
			self.state[0] = min(3, self.state[0] + 1)
		elif action == 2: #LEFT
			self.state[1] = max(0, self.state[1] - 1)
		elif action == 3: #RIGHT
			self.state[1] = min(4, self.state[1] + 1)
		x, y = self.state
		done = x == 3 and y > 0 or self.n >= 100
		return tuple(self.state), self.Map[x][y], done, {}

	def reset(self):
		self.n = 0
		self.state = [3,0]
		return tuple(self.state)

	def config(self):
		self.action_space.seed(self.seed)

class Tiger():
	def __init__(self, seed):
		self.NAME = "Tiger"
		self.observation_space_n = 1
		self.seed = seed
		self.action_space = action_space(seed, 3)
		self.config()

	def step(self, action):
		rand = np.random.random()
		self.n += 1
		if action == 0: #Escuchar
			if self.state == 0: #Izquierda
				if rand < 0.15:
					return 1, -1, (self.n >= 100), 1 #Derecha
				else:
					return 0, -1, (self.n >= 100), 1 #Izquierda
			else: #Derecha
				if rand < 0.15: 
					return 0, -1, (self.n >= 100), 1 #Izquierda
				else:
					return 1, -1, (self.n >= 100), 1 #Derecha
		elif action == 1: #Abrir Izquierda
			if self.state == 0: #Izquierda
				n = 0
				if rand > 0.5:
					n = 1
				return n, -100, True, 1
			else: #Derecha
				n = 1
				if rand > 0.5: #Izquierda
					n = 0
				return n, 10, True, 1
		else:
			if self.state == 0: #Izquierda
				n = 0
				if rand > 0.5: #Izquierda
					n = 1
				return n, 10, True, 1
			else: #Derecha
				n = 1
				if rand > 0.5: #Izquierda
					n = 0
				return n, -100, True, 1

	def reset(self):
		self.n = 0
		self.state = np.random.randint(2)
		
		rand = np.random.random()
		if self.state == 0:
			if rand < 0.15:
				return (1)
			else:
				return (0)
		else:
			if rand < 0.15:
				return (0)
			else:
				return (1)

	def config(self):
		np.random.seed(self.seed)
		self.action_space.seed(self.seed)
		
class GridWorld(object):
	"""docstring for GridWorld"""
	def __init__(self, seed):
		self.NAME = "GridWorld"
		self.seed = seed
		self.action_space = action_space(seed, 4)
		self.observation_space_n = 1
		self.config()
		R = -0.04
		self.Map = [
			[+R, +R, +R, +1],
			[+R, +R, +R, -1],
			[+R, +R, +R, +R],
		]

	def step(self, action):
		self.n += 1
		rand = np.random.random()

		if action == 0: #UP
			if rand < 0.8: #UP
				self.state[0] -= 1
			elif rand < 0.9: #left
				self.state[1] -= 1
			else: #right
				self.state[1] += 1
		elif action == 1: #Down
			if rand < 0.8: #down
				self.state[0] += 1
			elif rand < 0.9: #left
				self.state[1] -= 1
			else: #right
				self.state[1] += 1
		elif action == 2: #left:
			if rand < 0.8: #left
				self.state[1] -= 1
			elif rand < 0.9: #up
				self.state[0] -= 1
			else: #down
				self.state[0] += 1
		elif action == 3: #right:
			if rand < 0.8: #right
				self.state[1] += 1
			elif rand < 0.9: #up
				self.state[0] -= 1
			else: #down
				self.state[0] += 1

		self.borders()

		done = (self.state[1] == 3 and self.state[0] != 2) or self.n >= 100

		return self.obs(), self.Map[self.state[0]][self.state[1]], done, 1

	def borders(self):
		self.state[0] = np.clip(self.state[0], 0, 2)
		self.state[1] = np.clip(self.state[1], 0, 3)

	def reset(self):
		self.n = 0
		self.state = [2,3]
		return self.obs()

	def config(self):
		np.random.seed(self.seed)
		self.action_space.seed(self.seed)

	def obs(self):
		return ( (self.state[0]*4+self.state[1])/11 )

class Pendulum():
	def __init__(self, seed):
		self.NAME = "Pendulum-v0"
		self.env = gym.make(self.NAME)
		self.action_space = self.env.action_space
		self.seed = seed
		self.config()
		self.observation_space_n = self.env.observation_space.shape[0]
		self.LIM_MIN = -2
		self.LIM_MAX = 2
		self.LIM_DIST = 4

	def config(self):
		self.env.seed(self.seed)
		self.env.action_space.seed(self.seed)

	def step(self, action):
		s, r, d, _ = self.env.step(action)
		return s, r, d, _

	def reset(self):
		s = self.env.reset()
		return s

class MountainCarContinuous():
	def __init__(self, seed):
		self.NAME = "MountainCarContinuous-v0"
		self.env = gym.make(self.NAME)
		self.action_space = self.env.action_space
		self.seed = seed
		self.config()
		self.observation_space_n = self.env.observation_space.shape[0]
		self.LIM_MIN = -1
		self.LIM_MAX = 1
		self.LIM_DIST = 2

	def config(self):
		self.env.seed(self.seed)
		self.env.action_space.seed(self.seed)

	def step(self, action):
		s, r, d, _ = self.env.step(action)
		return s, r, d, _

	def reset(self):
		s = self.env.reset()
		return s