import retro
from classes.game import Game

class clown_agent:
	def __init__(self,game): #takes game as input for taking actions
		self._game = game
		
	def Run(self):
		self._game.Run()
	def Jump(self):
		self._game.Jump()
	def Stop(self):
		self._game.Stop()
