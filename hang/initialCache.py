from collections import deque

import utils
from classes.parameters import Parameters

para = Parameters()
INITIAL_EPSILON = para.INITIAL_EPSILON

#Call only once to init file structure

# training variables saved as checkpoints to filesystem to resume training from the same step
def init_cache():
	"""initial variable caching, done only once"""
	utils.save_obj(INITIAL_EPSILON,"epsilon")
	print("epsilon.pkl has been created")
	t = 0
	utils.save_obj(t,"time")
	print("time.pkl has been created")
	D = deque()
	utils.save_obj(D,"D")
	print("D.pkl has been created")

if __name__ == "__main__":
	init_cache()