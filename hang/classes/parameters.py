class Parameters:
	def __init__(self):
		#game parameters
		self.ACTIONS = 2 # possible actions: jump, do nothing
		self.Key_num = 9
		self.GAMMA = 0.99 # decay rate of past observations original 0.99
		self.OBSERVATION = 100. # timesteps to observe before training
		self.EXPLORE = 100000  # frames over which to anneal epsilon
		self.FINAL_EPSILON = 0.0001 # final value of epsilon
		self.INITIAL_EPSILON = 0.1 # starting value of epsilon
		self.REPLAY_MEMORY = 50000 # number of previous transitions to remember
		self.BATCH = 16 # size of minibatch
		self.FRAME_PER_ACTION = 1
		self.LEARNING_RATE = 1e-4
		self.img_rows , self.img_cols = 80,80
		self.img_channels = 4 #We stack 4 frames
		self.loss_file_path = "./objects/loss_df.csv"
		self.actions_file_path = "./objects/actions_df.csv"
		self.q_value_file_path = "./objects/q_values.csv"
		self.scores_file_path = "./objects/scores_df.csv"