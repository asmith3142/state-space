

#State-space EM code in python
#Anne Smith 2015
import pandas as pd	
import matplotlib.pyplot as plt
import numpy as np
import filtersandEM as ff
from   random import random
from   importlib import reload

reload(ff)  #helps with debugging


################################
def RunEM(df, p_init):

	startflag  = 0
	sigma2e    = 0.5**2 #starting estimate = guess
	sigma_init = sigma2e

	x_init     = 0  #assume state starts at zero, mu accounts for offset

	if(p_init > 0.01 and p_init < 0.99):
		mu = np.log(p_init/(1-p_init))
	elif(p_init<=0.01):
		mu = -3.0
	else:
		mu = 3.0

	print(sigma2e)
	x_post, sigma2_post, sigma2e, sigma_init, converge_flag =  ff.EM(df.y, mu, sigma2e, x_init, sigma_init)
	
	pmode, pmeanv, pll, pul       = ff.TransformToProb(x_post, sigma2_post, mu)

	fig         = plt.figure(0)
	plt.plot(pmode,  linestyle = '-', color= 'orange', alpha=0.9, lw=2)
	plt.plot(pmeanv, linestyle = '-', color= 'orange', alpha=0.9, lw=2)
	plt.plot(pll, linestyle = '-',    color= 'darkgray', alpha=0.9, lw=3)
	plt.plot(pul, linestyle = '-',    color= 'darkgray', alpha=0.9, lw=3)
	plt.plot([0, len(pmeanv)], [0.25, 0.25], 'k--', lw = 1, alpha=0.9)
	plt.plot(range(1,len(df) + 1), df['y'],'s', color = 'black')
	plt.ylabel('Probabilty of a Correct Response')
	plt.xlabel('Trial')
	plt.locator_params(axis = 'y', nbins = 3)
	plt.ylim([-0.01,1.01])
	#plt.xlim([1,len(pmeanv)+1])
	plt.show(block = False)
	return

###############################
def main():

	df = pd.DataFrame()

	resp_values = [1,0,0,0,0,0,1,0,0,0,1,1,1,1,0,1,0,0,1,0,0,0,0]
	resp_values.extend(np.ones(17))

	df['y'] = resp_values
	p_init  = 0.25
	RunEM(df, p_init)




if __name__ == '__main__':
  main()



