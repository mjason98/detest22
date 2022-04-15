'''
	This is a set of functions and classes to help the main prosses but they are pressindible.
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from progress.bar import  Bar

import os

def strToListF(cad:str, sep=' '):
	return [float(s) for s in cad.split(sep)]
def strToListI(cad:str, sep=' '):
	return [int(s) for s in cad.split(sep)]

def getSTime(time_sec):
	mon, sec = divmod(time_sec, 60)
	hr, mon = divmod(mon, 60)
	return "{:02.0f}:{:02.0f}:{:02.0f}".format(hr,mon,sec)

class MyBar(Bar):
	empty_fill = '.'
	fill = '='
	bar_prefix = ' ['
	bar_suffix = '] '
	value = 0
	hide_cursor = False
	width = 20
	suffix = '%(percent).1f%% - %(value).5f'
	
	def next(self, _v=None):
		if _v is not None:
			self.value = _v
		super(MyBar, self).next()

def colorizar(text):
	return '\033[91m' + text + '\033[0m'
def headerizar(text):
	return '\033[1m' + text + '\033[0m'

def projectData2D(data_path:str, save_name='2Data', drops = ['is_humor','humor_rating', 'id'], use_centers=False):
	'''
		Project the vetors in 2d plot

		data_path:str most be a cvs file
	'''
	from sklearn.manifold import TSNE 
	
	data = pd.read_csv(data_path)

	np_data = data.drop(drops, axis=1).to_numpy().tolist()
	np_data = [i for i in map(lambda x: [float(v) for v in x[0].split()], np_data)]
	np_data = np.array(np_data, dtype=np.float32) 

	L1, L2 = 0, 0
	if use_centers:
		# L1, L2 = [], []
		# P = [['neg_center.txt', L1], ['pos_center.txt', L2]]
		# for l,st in P:
		# 	with open(os.path.join('data', l), 'r') as file:
		# 		lines = file.readlines()
		# 		lines = np.array([[float(v) for v in x.split()] for x in lines], dtype=np.float32)
		# 		st.append(lines)
		# L1 = np.concatenate(L1, axis=0)
		# L2 = np.concatenate(L2, axis=0)	
		L1 = np.load(os.path.join('data', 'neg_center.npy'))
		L2 = np.load(os.path.join('data', 'pos_center.npy'))

		np_data = np.concatenate([np_data, L1, L2], axis=0)
		L1, L2 = L1.shape[0], L2.shape[0]
	print ('# Projecting', colorizar(os.path.basename(data_path)), 'in 2d vectors', end='')
	X_embb = TSNE(n_components=2).fit_transform(np_data)
	#X_embb = PCA(n_components=2, svd_solver='full').fit_transform(np_data)
	#X_embb = TruncatedSVD(n_components=2).fit_transform(np_data)
	print ('  Done!')
	del np_data
	
	D_1, D_2 = [], []
	fig , axes = plt.subplots()

	for i in range(len(data)):
		if int(data.loc[i, 'is_humor']) == 0:
			D_1.append([X_embb[i,0], X_embb[i,1]])
		else:
			D_2.append([X_embb[i,0], X_embb[i,1]])
	D_1, D_2 = np.array(D_1), np.array(D_2)
	axes.scatter(D_1[:,0], D_1[:,1], label=r'$N~class$', color=(255/255, 179/255, 128/255, 1.0))
	axes.scatter(D_2[:,0], D_2[:,1], label=r'$P~class$', color=(135/255, 222/255, 205/255, 1.0))
	
	if L1 > 0:
		X_embb_1 = X_embb[-(L1+L2):-L2]
		X_embb_2 = X_embb[-L2:]

		axes.scatter(X_embb_1[:,0], X_embb_1[:,1], label=r'$N_{Set}$', color=(211/255, 95/255, 95/255, 1.0), marker='s')
		axes.scatter(X_embb_2[:,0], X_embb_2[:,1], label=r'$P_{Set}$', color=(0/255, 102/255, 128/255, 1.0), marker='s')
		
		del X_embb_1
		del X_embb_2
	del X_embb

	fig.legend()
	fig.tight_layout()
	axes.axis('off')
	fig.savefig(os.path.join('out', save_name+'.png'))
	print ('# Image saved in', colorizar(os.path.join('out', save_name+'.png')))
	# plt.show()
	
	del fig
	del axes
	
	