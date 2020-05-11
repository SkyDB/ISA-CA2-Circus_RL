import pickle
import pandas as pd
import os

def save_obj(obj, name ):
	with open('./objects/'+ name + '.pkl', 'wb') as f: #dump files into objects folder
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
def load_obj(name ):
	with open('./objects/' + name + '.pkl', 'rb') as f:
		return pickle.load(f)


#Intialize log structures from file if exists else create new
def loadCSV(file_path, column_name):
	df = pd.read_csv(file_path) if os.path.isfile(file_path) else pd.DataFrame(columns =[column_name])
	return df