import numpy as np
import torch
import scipy.optimize as op
from gappy.quarticKDE import QuarticKDE


def check_boundary(pt, boundaries):
	## Return TRUE if OUTSIDE BOUNDARIES 
	
	## pt array of size N
	## boundaries array(N,2) (min max)
	
	
	low_ = pt < boundaries[:,0]
	sup_ = pt > boundaries[:,1]
   
	return (np.any(low_ >0) or np.any(sup_ > 0))   
  

class Gapper():
	
	def __init__(self, X, bw, bounds=None):
		"""
		## inputs:
		- `X` array `(n, d)`: `n` points with `d` dimensions to which KDE will be fit
		- `bw` numeric: inverse bandwidth (or spatial scale) for Gaussian kernel
		- `bounds` array `d, 2`: boundaries for which to evaluate
		"""
		self.X = X
		self.bw = bw
		self.dim = X.shape[1]
		self.kde = QuarticKDE(X=torch.tensor(self.X, dtype=torch.float32), bw=self.bw)
		
		if bounds is None:
			self.bounds = [[np.min(self.X[:,d]),np.max(self.X[:,d])] for d in range(self.dim)]
			self.bounds = np.array(self.bounds)
		else:
			self.bounds = bounds
			
		self.critical_points = None
		
		
	def get_density(self, points):
		"""
		Return the density estimate from QuarticKDE 

		Input:
		- points: numpy array `(k, d)` of k points with `d` dimensions to evaluate density on.
		
		Output:
		- 
		TODO : Add param to toggle requires_grad or not 
		"""
		
		data_t = torch.tensor(points, dtype=torch.float32, requires_grad=True)
		return self.kde.score_samples(data_t)
	
	def get_grad_density(self, points, dens_est=None, create_graph=False):
		"""
		Remove ?? OR TEST
		Set create_graph if plan on using this grad further down the line for higher order derivative
		"""
		data_t = torch.tensor(points, dtype=torch.float32, requires_grad=True)

		if dens_est == None:			
			dens_est = self.kde.score_samples(data_t)

		return torch.autograd.grad(dens_est, data_t, create_graph=create_graph) 

		

	def get_Hessian(self,point):
		"""
		Remove ??
		"""
		data_t = torch.tensor(point, dtype=torch.float32, requires_grad=True)

		_H = torch.autograd.functional.hessian(self.kde.score_samples, 
						data_t).detach().numpy()#, create_graph=True)
		
		return _H


	def compute_density_grid(self, gridding_size):
		"""
		Automatically compute the density on a grid over the data space (within the boundaries)
		- `griddind_size`: int number by which to divide the space 

		Outputs:
		- grid_data : (gridding_size ** d, d) grid points
		- grid_density : (gridding_size ** d) density estimate at those points

		TODO: gridding size is one number fo all dim....change this?
		"""
		bounds = self.bounds
		#gridding_size = ##80

		grid_linspace = [ np.linspace(bounds[d][0], bounds[d][1], gridding_size) for d in range(self.dim) ]


		meshgrid = np.meshgrid(*grid_linspace, indexing='ij')

		meshgrid_ravel = [ xi.ravel().reshape(-1,1) for xi in meshgrid]
		grid_data = np.hstack(meshgrid_ravel)
		
		self.grid_data = grid_data
		self.grid_density = self.kde.score_samples(torch.tensor(grid_data))
		
		return self.grid_data, self.grid_density
	
		
	def g2f(self, x):
		"""
			For optimization purpose: 
			Return squared  gradient of density estimate at point x, and its derivative
		"""
		a_point = torch.tensor(x, requires_grad=True)
		d = self.kde.score_samples(a_point)
		g1 = torch.autograd.grad(d, a_point, create_graph=True)[0]
		foo = torch.dot(g1, g1) # * torch.exp(2 * logd)
		gradfoo = torch.autograd.grad(foo, a_point)#, create_graph=True) #Is Create_graph true necessary here??
		gradfoo = gradfoo[0].detach().numpy()
		return foo.detach().numpy(), gradfoo
	
	

	def get_critical_point(self, x, max_iter=80):
		"""
		Optimize squared gradient of density estimate from x to find 
		critical point 
		
		This code remove critical points landing outside of boundaries defined at the begining

		bug: This code should also check that g2f(x)[0] is close to zero?
		todo: add hyper params for gtol and maxiter in op.minimize?
		Re-add recursvie for early stopping when out-of-boundaries? But might not happen?
		"""
		bounds = self.bounds
		res = op.minimize(self.g2f, x, jac=True, method="bfgs",
					options={"gtol": 1.e-5, "maxiter": max_iter})
		xx = res.x

		if check_boundary(xx, bounds):
			""" TODO: put warning?"""
			return None

		if res.success:
			return xx

		else:
			return None
	
	def compute_crit_points_setmesh(self, grid_mesh, distance_to_group=None, max_iter=80):
		"""
			Do the optimization search for critical points from a list of location (grid_mesh)
			Resulting points are then grouped into 'close points' as a function of bandwidth
			(if closer than .01 * (bw**2))

			Input: 
			- `grid_mesh` : TODO

			Output:
			- Array of resulting critical points (n, d)

			bug: Do we have opinion on the grouping distance choice??

		"""
		if distance_to_group==None:
			distance_to_group = 0.01 * self.bw * self.bw # magic

		bounds = self.bounds ## ?
				
		meshgrid = np.meshgrid(*grid_mesh)
		meshgrid_flatten = [xs.flatten() for xs in meshgrid]
		
		xs = np.vstack(meshgrid_flatten).T
		print("List of grid points for critical points: {}".format(xs.shape))
		# optimize all initial points to final points or Nones
		def gcp(x):
			return self.get_critical_point(x, max_iter)
		foo = list(map(gcp, xs))
		print("finished optimizations")

		# keep only the non-Nones
		bigxs = []
		for f in foo:
			if f is not None:
				bigxs.append(f)
		bigxs = np.array(bigxs)

		# group into close points, based on the bw
		groups = np.zeros(len(bigxs)).astype(int) - 1
		print(len(bigxs), len(groups))
		for i in range(len(bigxs)):
			if groups[i] >= 0:
				continue
			unmatched = groups < 0
			dists2 = np.sum((bigxs - bigxs[i][None, :]) ** 2, axis=1)
			#print()
			matched = dists2 < distance_to_group 
			groups[matched] = i

		gs = np.unique(groups)
		xs = np.zeros((len(gs), self.dim)) 
		for i, g in enumerate(gs):
			xs[i] = np.mean(bigxs[groups == g], axis=0)

		self.critical_points = xs
		return xs

	def compute_all_critical_points(self):
		"""
			This function creates a grid of starting points (spaced by 'bandwidth' space) within boundaries 
			and optimize to find an estimation of all the critical points of the density estimate.

			Output: List of critical points (n,d)

			TODO : add hyperparam for the dist grouping
		"""
		# make grid of initial points
		bounds = self.bounds
		grid_mesh = [ np.arange(bounds[d][0] + .5 * self.bw, bounds[d][1], self.bw) for d in range(self.dim) ]

		return self.compute_crit_points_setmesh(grid_mesh)
	


	def stepu2_gradient_descent(self, critical_pt, first_eps, eps_step, thresh_gg = 1e-7,
					  n_iter=1000, sign=1):
		
		"""
		 This function takes a critical point as input, takes one step in the direction of the XXXX eigen-vector of the hessian 
		 of the density estmate at that point and then perform gradient descent following the gradient of 
		 the estimate of the density. The gradient step follow a normalized gradient XXXX verify this

		 ##TODO : check if this could be replace with op.minimize if we can get the list of points?
		 Better to have evenly spaced whatever thing? 

		 Input:
		 - critical_pt: A critical point from which to compute the gradient descent 
		 - first_eps: Factor for first step in the XXX direction of 
		 - eps_step: Factor for the gradient descent step 
		 - thresh_gg: A threshold stop criterion
		 - n_iter: number of iterations for gradient descent / length of the paths
		 - sign: direction in which step is applied on eigen-vector XXX 

		 Output:
		 - list_pts : list of points of the 'path' following gradient of density estimate
		 - resolution: string indicating the reason for stopping:
				'outbound' = reach out of boundaries 
				'nangrad' = gradient is nan for the critical point or get to nan while doing the descent
				'lowg' = gradient threshold (thresh_gg) is reached 
				'maxiter' = number of iterations max (n_iter) is reached 

		 bug: Can critical points have nan grad?? Do we like that list_pts is nan if nangrad?
		"""

		list_pts = [critical_pt]

		iter_ = 0
		boundaries = np.array(self.bounds)
		
		#
		if check_boundary(list_pts[-1], boundaries):
			## This shouldn't happen?
			return list_pts[:-1], 'outbound'

		a_point = torch.tensor( list_pts[-1], dtype=torch.float32, requires_grad=True)
		dens_est = self.kde.score_samples(a_point)

		g = torch.autograd.grad(dens_est, a_point)#, create_graph=True) 


		if (np.isnan(g[0].detach().numpy()).any()):
			# This shouldn't happen either?
			""" TODO: put warning?"""
			return list_pts[:-1], 'nangrad'

		H = torch.autograd.functional.hessian(self.kde.score_samples, a_point).detach().numpy()#, create_graph=True)
		eig_val, eig_vec = np.linalg.eigh(H)

		id_sort = np.argsort(eig_val)[::-1]
		eig_val = eig_val[id_sort]
		eig_vec = eig_vec[:,id_sort]

		### TODO FIX THIS!!!
		## Are we still using u2 tho ?? or the lowest ?
		u2 = eig_vec[:,1]
		u1 = eig_vec[:,0]



		new_point = np.array(a_point.detach().numpy() - sign * first_eps * u2) 
		list_pts.append(new_point)

		## Then go downhill in g
		while (iter_ < n_iter):# 
			iter_+=1

			if check_boundary(list_pts[-1], boundaries):
				return list_pts[1:-1], 'outbound'

			a_point = torch.tensor( list_pts[-1], dtype=torch.float32, requires_grad=True)
			dens_est = self.kde.score_samples(a_point)

			g = torch.autograd.grad(dens_est, a_point)[0].detach().numpy() # create_graph=True


			if (np.isnan(g).any()):
				## Keep the point that causes problems?
				## Throw some warning ?
				""" TODO: put warning?"""
				return list_pts[1:], 'nangrad'

			
			norm_g = g / np.linalg.norm(g) ## check if g==0 first ?

			new_point = np.array(a_point.detach().numpy() - eps_step * norm_g) ## Change step here?
			list_pts.append(new_point)

			if np.sum(g * g) < thresh_gg:
				return list_pts[1:], 'lowg'


		return list_pts[1:], 'maxiter'
	
	

	def compute_path_of_a_critpt(self, crit_pt, first_eps=None, eps_step=None,
			thresh_gg = 1e-4,n_iter=1000):
		"""
			This function compute the `path' following the gradient flow of the density 
			from a critical point. 

			Input:
			- crit_pt : critical point to start from 
			- first_eps: Factor for first step in the XXX direction of XX 
					default: bandwidth
			- eps_step: Factor for the gradient descent step 
				default: bandwidth * .1
			- thresh_gg: A threshold stop criterion
				default: 1e-4
			- n_iter: number of iterations for gradient descent / length of the paths
				default = 1000

			Output:
			- list_pts: list of points in the path 
			- [feedback_pl, feedback_mn]: resolution of the gradient descent in both direction
		"""
		if first_eps==None:
			first_eps = self.bw 
		if eps_step==None:
			eps_step = self.bw * .1

		list_path_pl, feedback_pl = self.stepu2_gradient_descent(crit_pt, first_eps = first_eps, 
												  eps_step = eps_step, thresh_gg = thresh_gg,
												  n_iter=n_iter, sign=1)
		#feedback_u[feedback_pl]+=1
		list_path_mn, feedback_mn = self.stepu2_gradient_descent(crit_pt, first_eps = first_eps, 
												  eps_step = eps_step, thresh_gg = thresh_gg,
												  n_iter=n_iter, sign=-1)

		#feedback_d[feedback_mn]+=1


		list_path = np.array(list_path_pl + list_path_mn)
		return list_path, [feedback_pl, feedback_mn]


	def get_g_H_eigvH(self, pt):
	
		"""
			This function computes the density estimate, its gradient, 
			and the eigen value and eigen vector for a given point.

			Input: 
			- pt, a d-dimensional point within boundaries (ideally?)

			Output: 
			- _eig_val: eigenvalues (sorted in descending order) of the Hessian of density at that point
			- _eig_vec: eigenvectors (corresponding to eigenvalues) of the Hessian of density at that point
			- dens_est : density estimate at that point
			- g: gradient of density estimate at that point
			- _H: Hessian of density estimate at that point

			Bug: this should return Hessian too
			gradient nan ?
			Should be able to take an array?

		"""

		a_point = torch.tensor( pt, dtype=torch.float32, requires_grad=True)
				
		dens_est = self.kde.score_samples(a_point)

		g = torch.autograd.grad(dens_est, a_point) #, create_graph=True) 

		""" TODO: REPLACE THIS WITH ASSERT OR SOME ERROR"""

		# if (np.isnan(g[0].detach().numpy()).any()):
		# 	print("gradient is nan -- might need some fix")
		# 	#print("TODO Fix this")
		# 	return np.nan, np.nan, np.nan, np.nan, np.nan

		

		_H = torch.autograd.functional.hessian(self.kde.score_samples, a_point).detach().numpy()#, create_graph=True)
		
		_eig_val, _eig_vec = np.linalg.eigh(_H) ## or eig? 

		id_sort = np.argsort(_eig_val)[::-1]
		_eig_val = _eig_val[id_sort]
		_eig_vec = _eig_vec[:,id_sort]

		return _eig_val, _eig_vec, dens_est, g, _H


	def get_PiHPi(self, point, g=None, H=None):
		"""
		Return PiHPi matrix for a location 'point'

		Input: 
		- point: an array of size d
		- g: the gradient of density estimate at point 'point' (numpy array). If None, this function recomputes it.
		- H: Hessian of density estimate at point 'point' (numpy array). If None, this function recomputes it.

		Output:
		- pihpi: the (d,d) matrix PiHPi 

		TODO add an assert or some stuff that g H assumed numpy array
		"""
		a_point = torch.tensor(point, dtype=torch.float32, requires_grad=True)

		if g==None:
			dens_est = self.kde.score_samples(a_point)
			g = torch.autograd.grad(dens_est, a_point) #, create_graph=True) 			
			g = g[0].detach().numpy()

		if H==None:
			## Is it more computationally efficient to call grad on g or Hessian on score_sample?
			## But require create_graph....
			H = torch.autograd.functional.hessian(self.kde.score_samples, a_point).detach().numpy()

					

		if np.all(g==0):
			hatg = g
		else:
			hatg = g / np.linalg.norm(g) # check

		hatg = hatg.reshape(-1,1)
		hatgdot = hatg @ hatg.T

		Pi = np.identity( hatgdot.shape[0])  - hatgdot
		PiHPi = Pi.T @ H @ Pi

		return PiHPi




