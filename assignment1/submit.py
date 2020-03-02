import numpy as np
import random as rnd
import time as tm

#function to choose random coordinate

def getRandpermCoord( currentCoord , n ):        
    global randperm, randpermInner
    if randpermInner >= n-1 or randpermInner < 0 or currentCoord < 0:
        randpermInner = 0
        randperm = np.random.permutation( n )
        return randperm[randpermInner]
    else:
        randpermInner = randpermInner + 1
        return randperm[randpermInner]


randperm = []
randpermInner = -1

################################
# Non Editable Region Starting #
################################
def solver( X, y, C, timeout, spacing ):
	(n, d) = X.shape
	t = 0
	totTime = 0
	
	# w is the normal vector and b is the bias
	# These are the variables that will get returned once timeout happens
	w = np.zeros( (d,) )
	b = 0
	tic = tm.perf_counter()
################################
#  Non Editable Region Ending  #
################################
	

	alpha = C * np.ones( (y.size,) )

	alphay = np.multiply( alpha, y )

	w = X.T.dot( alphay )

	b = alpha.dot( y )

	normSq = np.square( np.linalg.norm( X, axis = 1 ) ) + 1
	
	
	randperm = np.random.permutation( y.size )
	randpermInner = -1

	# We have not made any choice of coordinate yet
	i = -1
################################
# Non Editable Region Starting #
################################
	while True:
		t = t + 1
		if t % spacing == 0:
			toc = tm.perf_counter()
			totTime = totTime + (toc - tic)
			if totTime > timeout:
				return (w, b, totTime)
			else:
				tic = tm.perf_counter()
################################
#  Non Editable Region Ending  #
################################

		i = getRandpermCoord( i , n)
		
		x = X[i,:]

		newAlphai = (1 - y[i] * (x.dot(w) + b) + alpha[i] * normSq[i]) / (1/(2*C) + normSq[i])

		#project the value of alpha 
		if newAlphai < 0:
			newAlphai = 0

		w = w + (newAlphai - alpha[i]) * y[i] * x
		b = b + (newAlphai - alpha[i]) * y[i]

		alpha[i] = newAlphai
		
	return (w, b, totTime) # This return statement will never be reached
