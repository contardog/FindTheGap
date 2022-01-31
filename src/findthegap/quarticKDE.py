import numpy as np
import torch

from scipy.spatial import cKDTree

def quartic_kernel(ss):
	"""
	# The quartic kernel
	This kernel is twice differentiable everywhere and has support only inside
	a radius of 3. It is as close to a Gaussian as possible in shape (not
	amplitude).

	## inputs:
	- `ss`: a list of dimensionless Euclidean separations (square roots of metric-
		  scaled, squared components of displacement)

	## outputs:
	- evaluations of the kernel at those separations

	## notes:
	- Filled with magic coefficients. These coefficients are rigidly set to make
	the kernel twice-differentiable and cut off at a dimensionless distance of
	3. Don't mess with the coefficients.

	## bugs?:
	- IF you care about normalization or integrations of your density, this
	function needs to be placed within a context in which the amplitude or
	prefactor is set correctly. This will depend on the dimension d of the
	ambient space.
	"""
	s = ss / 3.
	foo = 1.0 - 6.0 * s ** 2 + 8.0 * s ** 3 - 3.0 * s ** 4
	foo[s >= 1.] = 0.
	return foo

def ln_quartic_kernel(ss):
	return torch.log(quartic_kernel(ss))

class QuarticKDE():
	def __init__(self, X, bw):
		"""
		## inputs:
		- `X` array `(n, d)`: `n` points with `d` dimensions to which KDE will be fit
		- `bw` numeric: inverse bandwidth (or spatial scale) for kernel

		
		"""
		self.X = X
		self.bw = bw
		self.n = X.shape[0]
		self.d = X.shape[-1]
		self.logn = np.log(self.n)
		self.prefactor = -0.5 * (np.log(2. * np.pi * self.bw ** 2) * self.d)
		self.tree = cKDTree(X)

	def score_samples_batch(self, Y, size_chunk=10000):
		"""
		Return densities estimate at points Y. 
		This function makes chunks of the Y to avoid memory overflow but computes
		query_ball_point on 'batches' of size_chunk. This might actually not help with speed...

		Input:
		- 'Y': torch tensor of size (n, d) of points

		Output:
		- torch tensor of size (n) 

		BUG: Use of self.prefactor needs to be audited?

		"""
		if len(list(Y.shape)) == 1:
			thisY = Y.unsqueeze(0) 
		else:
			thisY = Y
		m, d = thisY.shape
		assert d == self.d
		scores = torch.tensor(np.zeros(m), dtype=torch.float32)
		
		Y_chunks = thisY.split(size_chunk) ## Prevent overflow from query_ball....not very elegantly
		## IS there any optim done in query_ball_pt code that justify doing it like this?
		## There is another query function for large grp of pts but when audited it wasnt doing that much better...

		count_ = 0
		for y_ in Y_chunks:
			for i, idxs in enumerate(self.tree.query_ball_point(y_.detach().numpy(), 3. * self.bw)): # MAGIC 3 
				Xi = self.X[idxs]
				Yi = thisY[count_]
				## Handling case where Yi are from the data X, need to be removed 
				## from Xi to not break everything
				xi_yi = Xi==Yi
				Xi = Xi[np.where(~np.all(xi_yi.numpy(), axis=1))[0]]
				
				ssi = torch.sqrt((((Xi - Yi[None, :]) / self.bw) ** 2).sum(1))
				# ssi = ssi[np.where(ssi!=0)] ? Do we need that ??
				scores[count_] += quartic_kernel(ssi).sum()
				count_+=1

		return scores * np.exp(self.prefactor - self.logn)
	
	def score_samples_queryBallTree(self, Y, size_chunk=10000):
		"""
		Return densities estimate at points Y
		This function uses query_ball_tree (potential speed up?) on chunks

		Input:
		- 'Y': torch tensor of size (n, d) of points

		Output:
		- torch tensor of size (n) 

		BUG: Use of self.prefactor needs to be audited?

		This doesnt seem to be actually faster....??
		"""
		if len(list(Y.shape)) == 1:
			thisY = Y.unsqueeze(0)
		else:
			thisY = Y
		m, d = thisY.shape
		assert d == self.d
		scores = torch.tensor(np.zeros(m), dtype=torch.float32)
		
		Y_chunks = thisY.split(size_chunk) ## Prevent overflow from query_ball....not very elegantly
		## IS there any optim done in query_ball_pt code that justify doing it like this?
		## There is another query function for large grp of pts but when audited it wasnt doing that much better...

		count_ = 0
		for y_ in Y_chunks:
			## For each chunks, create a KDTree and call query_ball_tree instead for speedup?
			y_tree = cKDTree(y_.detach().numpy())
			# results of query ball_tree : list of lists
			#For each element self.data[i] of this tree, 
			# results[i] is a list of the indices of its neighbors in other.data.


			for i, idxs in enumerate(y_tree.query_ball_tree(self.tree, 3. * self.bw)): # MAGIC 3 
				Xi = self.X[idxs]
				Yi = y_[i]
				## Handling case where Yi are from the data X, need to be removed 
				## from Xi to not break everything
				xi_yi = Xi==Yi
				Xi = Xi[np.where(~np.all(xi_yi.numpy(), axis=1))[0]]
				
				ssi = torch.sqrt((((Xi - Yi[None, :]) / self.bw) ** 2).sum(1))
				# ssi = ssi[np.where(ssi!=0)] ? Do we need that ??
				scores[count_] += quartic_kernel(ssi).sum()
				count_+=1

		return scores * np.exp(self.prefactor - self.logn)
	

	def score_samples(self, Y):
		"""
		Return densities estimate at points Y.
		This function does not make 'batch' (process Ys one by one)

		I think in the end this one is more efficient in terms of speed?? And no memory problem(?)

		Input:
		- 'Y': torch tensor of size (n, d) of points

		Output:
		- torch tensor of size (n) 

		BUG: Use of self.prefactor needs to be audited?

		Fix: verify unsqueeze(1) or 0?
		"""
		if len(list(Y.shape)) == 1:
			thisY = Y.unsqueeze(0) 
		else:
			thisY = Y
		m, d = thisY.shape
		assert d == self.d
		scores = torch.tensor(np.zeros(m), dtype=torch.float32)
		
		## Memory not great but speed? not sure....
		thisY_numpy = thisY.detach().numpy()
		for i in range(m):
			idxs = self.tree.query_ball_point(thisY_numpy[i], 3. * self.bw) # MAGIC 3 
			Xi = self.X[idxs]
			Yi = thisY[i]
			## Handling case where Yi are from the data X, need to be removed 
			## from Xi to not break everything
			xi_yi = Xi==Yi
			Xi = Xi[np.where(~np.all(xi_yi.numpy(), axis=1))[0]]
			
			ssi = torch.sqrt((((Xi - Yi[None, :]) / self.bw) ** 2).sum(1))
			# ssi = ssi[np.where(ssi!=0)] ? Do we need that ??
			scores[i] += quartic_kernel(ssi).sum()
			
				
		return scores * np.exp(self.prefactor - self.logn)
	
	
	def log_score_samples(self, Y):
		return torch.log(self.score_samples(Y))
	
	def log_prob(self, Y):
		"""Returns the total log probability of one or more points, `Y`.

		Input:
		- Y : array (m, d)
		  `m` points with `d` dimensions for which the probability density will
		  be calculated

		Output:
		- log_prob : numeric
		  total log probability density for the queried points, `Y`

		Fix: audit this 
		"""
		m, d = Y.shape
		assert d == self.d
		Y_chunks = Y.split(1000) # magic 1000
		log_prob = 0
		for y in Y_chunks:
			log_prob += self.log_score_samples(y).sum(dim=0)
		return log_prob

