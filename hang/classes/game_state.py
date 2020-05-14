import retro
import cv2 #opencv

from classes.game import Game
from classes.clown_agent import clown_agent as agent


def process_img(image):
	
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #RGB to Grey Scale
	image = image[35:135, 30:150] #Crop Region of Interest(ROI)
	image = cv2.resize(image, (80,80))
	return  image

def show_img(graphs = False):
	"""
	Show images in new window
	"""
	while True:
		screen = (yield)
		window_title = "logs" if graphs else "game_play"
		cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)        
		imS = cv2.resize(screen, (150, 150)) 
		cv2.imshow(window_title, imS)
		if (cv2.waitKey(1) & 0xFF == ord('q')):
			cv2.destroyAllWindows()
			break

class Game_state_:
	def __init__(self,agent,game):
		self._agent = agent
		self._game = game
		self._display = show_img() #display the processed image on screen using openCV, implemented using python coroutine 
		self._display.__next__() # initiliaze the display coroutine 
	def get_state(self,actions,actions_df,scores_df,loss_df):

		image, reward, is_over, info = self._game.get_score(actions)
		
		self._game.render()
		
		reward_1 = 0.2 
		
		image_ap = process_img(image)
		
		actions_df.loc[len(actions_df)] = actions[8] # storing actions in a dataframe
		
		self._display.send(image_ap) #display the image on screen
		
		if reward > 50:
			reward_1 = reward_1 + 8
		elif actions[8]==1:
			reward_1 = reward_1 - 0.2
		
		#delay = 200
		score_base = 8180
		
		if info['dead'] < 100:
			score = info['score'] - score_base
			reward_1 = -10
			is_over = True
			scores_df.loc[len(scores_df)] = score # log the score when game is over
			self._game.restart()

		return image_ap, reward_1, is_over, info #return the Experience tuple