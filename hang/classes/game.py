import retro

class Game:
	def __init__(self,env_name):
		self.env = retro.make(game=env_name)
	def restart(self):
		obs = self.env.reset()
		return obs
	def render(self):
		self.env.render()
		
	def Run(self):
		act = [0,0,0,0,0,0,0,1,0]
		return act
	def Stop(self):
		act = [0,0,0,0,0,0,1,1,0]
		return act
	def Jump(self):
		act = [0,0,0,0,0,0,0,1,1]
		return act

	def get_score(self, act):
		obs, rew, done, info = self.env.step(act)
		
		return obs, rew, done, info
	
	def get_action_sample(self):
		act = self.env.action_space.sample()
		return act
	
	def end(self):
		self.env.close()