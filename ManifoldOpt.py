import numpy as np
import warnings

class ManifoldOpt():

	is_invert = 0
	is_definite = 0

	def __init__(self, retraction, riemannian_gradient, cost_function, manifold=None, hessian=None):

		self.manifold = manifold
		self.cost_function = cost_function
		self.rGrad = riemannian_gradient
		self.retract = retraction
		self.H = self.check_hessian(hessian)
		"""
		cost_function : function handle
			Cost function being optimized.
		riemannian_gradient : function handle
			Riemannian Gradient of cost function, f.
		retraction : function handle
			Retraction or exponential mapping back onto the manifold.
		hessian : function handle
			Hessian-like linear operator or sufficiently positive-definite Hessian-like linear operator used to solve the sub-problem.
		manifold : ndarray
			Manifold to be optimized on. Currently impelmenting functions that utilize this.

		"""
	class HessianValue(UserWarning):
		pass

	@classmethod
	def check_hessian(cls, H):

		"""
		Checks to make sure the proposed hessian is sufficiently positive-definite.

		Extended description of function.

		Parameters
		----------
		H : function handle
			Proposed hessian.

		Returns
		-------
		H : function handle
			Returns H regardless but shows a warning if it is not sufficiently positive-definite or invertible.

		"""
		# 80% of cases must be positive definite
		threshold = 0.8

		# Checking to see if it is sufficiently positive definite and invertible for 50 test cases
		for i in range(50):

			x = np.random.rand(len(H(0)),1) #fix to account for constrained manifold and when user defines 0 to not be an input
			w, v = np.linalg.eigh(H(x))
			cls.is_invert += ~np.all(w == 0)/50
			cls.is_definite += ~np.any(w < 0)/50

		if (cls.is_invert < threshold):

			warnings.warn("Hessian operator is not invertible, cannot use Riemannian Newton Method.", ManifoldOpt.HessianValue)

		if (cls.is_definite < threshold):

			warnings.warn("Hessian operator is not positive-definite, Riemannian Newton and Trust-region models will not work well.", ManifoldOpt.HessianValue)

		return H

	def RGD(self, x, lam=1e-2, tolerance=1e-7, max_iter=1e4):

		"""
		Implementation of Riemannian Gradient Descent Method (first order).

		Extended description of function.

		Parameters
		----------
		x_0 : ndarray
			Input vector in R^n to initialize the model.
		gradR : function handle
			Riemannian Gradient of cost function, f.
		Retract : function_handle
			Retraction or exponential mapping back onto the manifold.
		f : function handle
			Cost function being optimized. If provided, then the model will use linesearch.
		max_iter : int (default=10000)
			Maximum iterations for Trust-region model.


		Returns
		-------
		x_arr : ndarray
			NumPy array consisting of every optimal value at each iterate
		error_arr : list
			List array consisting of every relative error (current iterate and previous iterate) based on l-2 norm

		"""

		error_arr = []
		x_arr = x
		x_prev = x

		# if there is an objective function, perform backtracking line search
		if self.cost_function != None:
			for i in range(int(max_iter)):

				# First compute Riemannian gradient
				r_grad = self.rGrad(x)

				# Perform linesearch
				lam = self.linesearch(x)

				# Calculate new iterate by moving along tangent space and then retracting back onto sphere
				x = self.retract(-lam*r_grad, x)

				# Checking Error
				error = np.linalg.norm(x - x_prev)/np.linalg.norm(x)
				error_arr.append(error)
				x_arr = np.concatenate((x_arr, x), axis=1)

				# Stopping criteria
				if error < tolerance:
					return x_arr, error_arr
				x_prev = x

			warnings.warn('Reached max iterations.')
			return x_arr, error_arr
		# Don't preform linesearch
		else:
			for i in range(int(max_iter)):

				# First compute Riemannian gradient
				r_grad = self.rGrad(x)

				# Calculate new iterate by moving along tangent space and then retracting back onto sphere
				x = self.retract(-lam*r_grad, x)

				# Checking Error
				error = np.linalg.norm(x - x_prev)/np.linalg.norm(x)
				err.append(error)
				x_arr = np.concatenate((x_arr, x), axis=1)

				# stopping criteria
				if error < tolerance:
					return x_arr, error_arr
				x_prev = x

			warnings.warn(f'Reached max iterations: {max_iter}')
			return x_arr, error_arr

	def linesearch(self, x):

		"""
		Backtracking linesearch used in Riemannian Gradient Descent.

		Extended description of function.

		Parameters
		----------
		x_0 : ndarray
			Input vector in R^n to initialize the model.
		gradR : function handle
			Riemannian Gradient of cost function, f.

		Returns
		-------
		t : float
			Learning rate calculated from linesearch
		"""

		# constants for linesearch
		r = 1e-4
		tau = 0.8
		t = 0.5

		fx = self.cost_function(x)
		i = 0

		# Armijo–Goldstein condition (check when it starts increasing)
		while (fx - self.cost_function(self.retract(-t * self.rGrad(x), x))) < (r * t * np.linalg.norm(self.rGrad(x))**2):
			t = tau * t
			i += 1
		return t

	def RNM(self, x_0, tolerance=1e-7, max_iter=1e4):

		"""
		Implementation of Riemannian Newton Method (second order).

		Extended description of function.

		Parameters
		----------
		x_0 : ndarray
			Input vector in R^n to initialize the model.
		tolerance : float (default=1e-7)
			Tolerance for stopping criteria. Compares previous iterate and current iterate difference in l-2 norm.
		max_iter : int (default=10000)
			Maximum iterations for Trust-region model.


		Returns
		-------
		x_arr : ndarray
			NumPy array consisting of every optimal value at each iterate.
		error_arr : list
			List array consisting of every relative error (current iterate and previous iterate) based on l-2 norm.

		"""

		# Initialize variables
		x = x_0
		x_prev = x
		error_arr = []
		x_arr = x

		# Checking to see if we have a hessian operator
		if self.H == None:
			raise ValueError('Please define a Hessian-like operator as an instance attribute.')

		# Iterate and find each newton step
		for i in range(int(max_iter)):

			# Find newton step by solving for s in H[s] = b
			s_k = np.linalg.inv(self.H(x)) @ - self.rGrad(x)

			# Retract
			x = self.retract(s_k, x)

			# Concatenating error
			error = np.linalg.norm(x - x_prev)/np.linalg.norm(x)
			error_arr.append(error)
			x_arr = np.concatenate((x_arr, x), axis=1)

			# Stopping criteria
			if error < tolerance:
				return x_arr, error_arr
			x_prev = x

		warnings.warn(f'Reached max iterations: {max_iter}')
		return x_arr, error_arr

	def RTR(self, x_0, radius_0, max_radius=3, threshold=0.1, max_iter=10000):

		"""
		Implementation of Riemannian Trust-region (second order) optimization method.

		Extended description of function.

		Parameters
		----------
		x_0 : ndarray
			Input vector in R^n to initialize the model.
		radius_0 : float
			Initial radius of the Trust-region model.
		max_radius : float (default=3)
			Maximum radius allowed for the model, typically set to sqrt(dim(M)), where M is the Manifold.
		threshold : threshold (default=1/10)
			Threshold for accepting a Trust-region's candidate iterate.
		max_iter : int (default=10000)
			Maximum iterations for Trust-region model.


		Returns
		-------
		x_arr : ndarray
			NumPy array consisting of every optimal value at each iterate.
		error_arr : list
			List array consisting of every relative error (current iterate and previous iterate) based on l-2 norm.

		"""

		# Initial values and defined model, m (evaluated at x_k with s being the variable we want to minimize)

		x = x_0
		radius = radius_0
		x_arr = x
		x_prev = x
		error_arr = []

		# Quadratic approximation
		m = lambda x, v: self.cost_function(x) +  self.rGrad(x).T @ v + 0.5 * v.T @ self.H(x).T @ v

		for i in range(int(max_iter)):

			# Approximately solving sub-problem with truncated CG (v is our new s_k)
			b = -self.rGrad(x)
			v, Hs = self.truncated_CG(self.H(x), b, radius)

			# Candidate iterate
			s = self.retract(x, v)

			# Normalizer for numerical stability
			del_k = max(1, abs(self.cost_function(x)))*1e-13

			# Quality quotient
			p_k = (self.cost_function(x) - self.cost_function(s) + del_k) / (- v.T @ self.rGrad(x) - 0.5*v.T @ Hs + del_k)

			# Accept or reject
			if p_k > threshold:

				# Set new iterate
				x = s

				# Checking Error
				error = np.linalg.norm(x - x_prev)/np.linalg.norm(x)
				error_arr.append(error)
				x_arr = np.concatenate((x_arr, x), axis=1)
			else:

				# Checking Error
				error = np.linalg.norm(x - x_prev)/np.linalg.norm(x)
				error_arr.append(error)
				x_arr = np.concatenate((x_arr, x), axis=1)

			# Update trust-region radius
			if p_k < 0.25:
				radius = 0.25 * radius
			elif ((p_k > 0.75) & (np.linalg.norm(v) == radius)): # if point lies on radius then reduce radius
				radius = min(2*radius, max_radius)
			else:
				radius = radius

			# Stopping criteria
			if np.linalg.norm(self.rGrad(x)) <= 1e-8 * np.linalg.norm(self.rGrad(x_0)):
				return x_arr, error_arr
			x_prev = x

		warnings.warn(f'Reached max iterations: {max_iter}')
		return x_arr, error_arr

	def truncated_CG(self, Hess, b, radius, K=0.1, theta=1):

		"""
		Implementation of Conjugate Gradient

		Extended description of function.

		Parameters
		----------
		Hess : function handle
			Hessian-like linear operator or sufficiently positive-definite Hessian-like linear operator used to solve the sub-problem.
		b : ndarray
			b = -grad(f(x)), where x is the current iterate of the Trust-region model.
		radius : float
			Current radius of the Trust-region iterate.
		K : float
			Stopping criteria parameter.
		theta : int
			Order of stopping criteria for convergence rate.

		Returns
		-------
		v : ndarray
			Solution to sub-problem for quadratic approximation.
		Hs : ndarray
			Left hand side of quadratic equation 'Hs=b,' where it can be used as a byproduct in the calculation of the quality quotient.

		"""

		# Starting values
		v = np.zeros((b.shape[0],1))
		r_0 = p = b
		r = r_0
		max_iter = 100 # enough to cover whole span of s due to linear independence

		# Checking to see if we have a hessian operator
		if self.H == None:
			raise ValueError('Please define a Hessian-like operator as an instance attribute.')

		for i in range(max_iter):

			# Computing constants (turn Hessian into linear operator!)
			x = Hess @ p
			y = p.T @ x

			# finding coefficients of our basis, p
			a = np.linalg.norm(r)**2/y
			v_plus = v + a * p

			# Checking to see if H is not positive definite or we left the trust region
			if ((y <= 0) or (np.linalg.norm(v_plus) >= radius)):

				# Find t s.t. we minimize our model, m, and move away from v along p
				t = self.solve_t_quad(v, p, radius)

				# Update v
				v = v + t*p

				# Update byproduct, Hs
				Hs = b - r + t * x
				return v, Hs
			else:
				v = v_plus

			# updating residual
			r_prev = r
			r = r_prev - a * x

			# Stopping criteria - that is, if the residual is small, then we will have faster convergence
			if np.linalg.norm(r) <= np.linalg.norm(r_0) * min(np.linalg.norm(r_0)**(theta), K):
				Hs = b - r
				return v, Hs

			# updating B and H-conjugate directions, p
			B = np.linalg.norm(r)**2/(np.linalg.norm(r_prev)**2)
			p = r + B * p

	def solve_t_quad(self, v, p, radius):

		"""
		Solve for the root of the quadratic || v_{n-1} + t*p_{n-1}||^2_x - radius^2 = 0.

		The positive root corresponds to exactly the amount one should move back along p_{n-1} such that the iterate stays within the trust region.

		Parameters
		----------
		v : ndarray
			Current iterate for an approximation of our newton step, s, in Hs = b.
		p : ndarray
			Current H-conjugate direction being utilized.
		radius : float
			Current radius of the Trust-region iterate.


		Returns
		-------
		t : float
			Positive root that solves for the quadratic framed in this problem.
		"""

		# Quadratic constants
		a = np.linalg.norm(p)**2
		b = 2* v.T @ p
		c = np.linalg.norm(v)**2 - radius**2

		# Use quadratic formula and select only positive root (explained in ABG_{07}, section 3)
		t1 = (-b + np.sqrt(b**2 - 4 * a * c)) / (2*a)
		t2 = (-b - np.sqrt(b**2 - 4 * a * c)) / (2*a)

		# Positive root
		t = max(t1, t2)

		return t

