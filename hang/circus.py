import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import retro

import numpy as np
from PIL import Image
import io
import time
import pandas as pd
import numpy as np
from IPython.display import clear_output
from random import randint
import random
import os

#from selenium import webdriver
#from selenium.webdriver.chrome.options import Options
#from selenium.webdriver.common.keys import Keys

import random
from io import BytesIO
import base64
import json

from keras.optimizers import SGD, Adam

#class imports
from classes.game import Game
from classes.clown_agent import clown_agent
from classes.game_state import Game_state_
from classes.parameters import Parameters
import models,utils

para = Parameters()

#game parameters
ACTIONS = para.ACTIONS # possible actions: jump, do nothing
Key_num = para.Key_num
GAMMA = para.GAMMA # decay rate of past observations original 0.99
OBSERVATION = para.OBSERVATION # timesteps to observe before training
EXPLORE = para.EXPLORE # frames over which to anneal epsilon
FINAL_EPSILON = para.FINAL_EPSILON # final value of epsilon
INITIAL_EPSILON = para.INITIAL_EPSILON # starting value of epsilon
REPLAY_MEMORY = para.REPLAY_MEMORY # number of previous transitions to remember
BATCH = para.BATCH # size of minibatch
FRAME_PER_ACTION = para.FRAME_PER_ACTION
LEARNING_RATE = para.LEARNING_RATE

loss_file_path = para.loss_file_path
actions_file_path = para.actions_file_path
q_value_file_path = para.q_value_file_path
scores_file_path = para.scores_file_path

loss_df = utils.loadCSV(loss_file_path,'loss')
scores_df = utils.loadCSV(scores_file_path,'scores')
actions_df = utils.loadCSV(actions_file_path,'actions')
q_values_df =utils.loadCSV(q_value_file_path,'qvalues')

''' 
main training module
Parameters:
* model => Keras Model to be trained
* game_state => Game State module with access to game environment and dino
* observe => flag to indicate wherther the model is to be trained(weight updates), else just play
'''
def trainNetwork(model,game_state,observe=False):
	last_time = time.time()
	# store the previous observations in replay memory
	D = utils.load_obj("D") #load from file system
	# get the first state by doing nothing
	do_nothing = np.zeros(Key_num)
	#do_nothing[0] =1 #0 => do nothing,
					 #1=> jump
	
	x_t, r_0, terminal = game_state.get_state(do_nothing,actions_df,scores_df,loss_df) # get next step after performing the action
	
	s_t = np.stack((x_t, x_t, x_t, x_t), axis=2) # stack 4 images to create placeholder input
		
	s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  #1*20*40*4
	
	initial_state = s_t 

	if observe :
		OBSERVE = 9999999    #We keep observe, never train
		epsilon = FINAL_EPSILON
		print ("Now we load weight")
		model.load_weights("model.h5")
		adam = Adam(lr=LEARNING_RATE)
		model.compile(loss='mse',optimizer=adam)
		print ("Weight load successfully")    
	else:                       #We go to training mode
		OBSERVE = OBSERVATION
		epsilon = utils.load_obj("epsilon") 
		model.load_weights("model.h5")
		adam = Adam(lr=LEARNING_RATE)
		model.compile(loss='mse',optimizer=adam)

	t = utils.load_obj("time") # resume from the previous time step stored in file system
	while (True): #endless running
		
		loss = 0
		Q_sa = 0
		action_index = 0
		r_t = 0 #reward at 4
		a_t = np.zeros(Key_num) # action at t
		#run, jump
		
		#choose an action epsilon greedy
		if t % FRAME_PER_ACTION == 0: #parameter to skip frames for actions
			if  random.random() <= epsilon: #randomly explore an action
				print("----------Random Action----------")
				#action_index = random.randrange(ACTIONS)
				a_t = random.choice([[0,0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,1,1]])
				
			else: # predict the output
				q = model.predict(s_t)       # input a stack of 4 images, get the prediction
				print(q)
				max_Q = np.argmax(q)         # chosing index with maximum q value
				action_index = max_Q
				a_t_all = [[0,0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,1,1]]
				a_t = a_t_all[action_index]       # run, jump
				
		#We reduced the epsilon (exploration parameter) gradually
		if epsilon > FINAL_EPSILON and t > OBSERVE:
			epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE 

		#run the selected action and observed next state and reward
		x_t1, r_t, terminal = game_state.get_state(a_t,actions_df,scores_df,loss_df)
				
		print('fps: {0}'.format(1 / (time.time()-last_time))) # helpful for measuring frame rate
		last_time = time.time()
		x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1) #1x20x40x1
		
		s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3) # append the new image to input stack and remove the first one
			
		# store the transition in D
		D.append((s_t, action_index, r_t, s_t1, terminal))
		if len(D) > REPLAY_MEMORY:
			D.popleft()

		#only train if done observing
		if t > OBSERVE: 
			
			#sample a minibatch to train on
			minibatch = random.sample(D, BATCH)
			inputs = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))   #32, 20, 40, 4
			targets = np.zeros((inputs.shape[0], ACTIONS))                         #32, 2

			#Now we do the experience replay
			for i in range(0, len(minibatch)):
				state_t = minibatch[i][0]    # 4D stack of images
				action_t = minibatch[i][1]   #This is action index
				reward_t = minibatch[i][2]   #reward at state_t due to action_t
				state_t1 = minibatch[i][3]   #next state
				terminal = minibatch[i][4]   #wheather the agent died or survided due the action
				

				inputs[i:i + 1] = state_t    

				targets[i] = model.predict(state_t)  # predicted q values
				 
				Q_sa = model.predict(state_t1)      #predict q values for next step
				
				if terminal:
					targets[i, action_t] = reward_t # if terminated, only equals reward
				else:
					targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)

			loss += model.train_on_batch(inputs, targets)
			loss_df.loc[len(loss_df)] = loss
			q_values_df.loc[len(q_values_df)] = np.max(Q_sa)
		s_t = initial_state if terminal else s_t1 #reset game to initial frame if terminate
		t = t + 1
		
		# save progress every 1000 iterations
		if t % 1000 == 0:
			print("Now we save model")
			#game_state._game.pause() #pause game while saving to filesystem
			model.save_weights("model.h5", overwrite=True)
			utils.save_obj(D,"D") #saving episodes
			utils.save_obj(t,"time") #caching time steps
			utils.save_obj(epsilon,"epsilon") #cache epsilon to avoid repeated randomness in actions
			loss_df.to_csv(loss_file_path,index=False)
			scores_df.to_csv(scores_file_path,index=False)
			actions_df.to_csv(actions_file_path,index=False)
			q_values_df.to_csv(q_value_file_path,index=False)
			with open("model.json", "w") as outfile:
				json.dump(model.to_json(), outfile)
			clear_output()
			#game_state._game.resume()
		# print info
		state = ""
		if t <= OBSERVE:
			state = "observe"
		elif t > OBSERVE and t <= OBSERVE + EXPLORE:
			state = "explore"
		else:
			state = "train"

		print("TIMESTEP", t, "/ STATE", state,             "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t,             "/ Q_MAX " , np.max(Q_sa), "/ Loss ", loss)

	print("Episode finished!")
	print("************************")


#main function
def main(observe=False):
	env_name = 'CircusCharlie-Nes'
	game = Game(env_name)
	game.restart()
	
	clown = clown_agent(game)
	game_state = Game_state_(clown,game)
	model = models.buildmodel(loss_file_path)
	
	try:
		trainNetwork(model,game_state,observe=observe)
	except StopIteration:
		game.end()

if __name__ == "__main__":
	main()