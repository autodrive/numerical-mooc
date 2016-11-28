# -*- coding: utf8 -*-
from pylab import *
"""
###### Content under Creative Commons Attribution license CC-BY 4.0, code under MIT license (c)2014 L.A. Barba, C.D. Cooper, G.F. Forsyth.
"""
"""
# Reaction-diffusion model
"""
"""
This IPython Notebook presents the context and set-up for the coding assignment of Module 4: _Spreading out: Diffusion problems_, of the course [**"Practical Numerical Methods with Python"**](https://github.com/numerical-mooc/numerical-mooc) (a.k.a., numericalmooc).

So far in this module, we've studied diffusion in 1D and 2D. Now it's time to add in some more interesting physics.  You'll study a model represented by *reaction-diffusion* equations.  What are they?  The name says it all—it's a system that has the physics of diffusion but also has some kind of reaction that adds different behaviors to the solution.

We're going to look at the _Gray-Scott model_, which simulates the interaction of two generic chemical species reacting and ... you guessed it ... diffusing! Some amazing patterns can emerge with simple reaction models, eerily reminiscent of patterns formed in nature. It's fascinating! Check out this simulation by Karl Sims posted on You Tube ... it looks like a growing coral reef, doesn't it?
"""
from IPython.display import YouTubeVideo
YouTubeVideo('8dTmUr5qKvI')
####################

"""
## Gray-Scott model
"""
"""
The Gray-Scott model represents the reaction and diffusion of two generic chemical species, $U$ and $V$, whose concentration at a point in space is represented by variables $u$ and $v$. The model follows some simple rules.  

*  Each chemical _diffuses_ through space at its own rate.
*  Species $U$ is added at a constant feed rate into the system.
*  Two units of species V can 'turn' a unit of species U into V: $\; 2V+U\rightarrow 3V$
*  There's a constant kill rate removing species $V$.

This model results in the following system of partial differential equations for the concentrations $u(x,y,t)$ and $v(x,y,t)$ of both chemical species:
"""
"""
\begin{align}
\frac{\partial u}{\partial t} &= D_u \nabla ^2 u - uv^2 + F(1-u)\\
\frac{\partial v}{\partial t} &= D_v \nabla ^2 v + uv^2 - (F + k)v
\end{align}
"""
"""
You should see some familiar terms, and some unfamiliar ones.  On the left-hand side of each equation, we have the time rate of change of the concentrations. The first term on the right of each equation correspond to the spatial diffusion of each concentration, with $D_u$ and $D_v$ the respective rates of diffusion.

In case you forgot, the operator $\nabla ^2$ is the Laplacian:

$$
\nabla ^2 u = \frac{\partial ^2 u}{\partial x^2} + \frac{\partial ^2 u}{\partial y^2}
$$

The second term on the right-hand side of each equation corresponds to the reaction. You see that this term decreases $u$ while it increases $v$ in the same amount: $uv^2$.  The reaction requires one unit of $U$ and two units of $V$,  resulting in a reaction rate proportional to the concentration $u$ and to the square of the concentration $v$. This result derives from the _law of mass action_, which we can explain in terms of probability: the odds of finding one molecule of species $U$ at a point in space is proportional to the concentration $u$, while the odds of finding two molecules of $V$ is proportional to the concentration squared, $v^2$. We assume here a reaction rate constant equal to $1$, which just means that the model is non-dimensionalized in some way.

The final terms in the two equations are the "feed" and "kill" rates, respectively: $F(1-u)$ replenishes the species $U$ (which would otherwise run out, as it is being turned into $V$ by the reaction); $-(F+k)v$ is diminishing the species $V$ (otherwise the concentration $v$ would simply increase without bound). 

The values of $F$ and $k$ are chosen parameters and part of the fun of this assignment is to change these values, together with the diffusion constants, and see what happens. 
"""
"""
### Problem setup
"""
"""
The system is represented by two arrays, `U` and `V`, holding the discrete values of the concentrations $u$ and $v$, respectively.  We start by setting `U = 1` everywhere and `V = 0` everywhere, then introduce areas of difference, as initial conditions.  We then add a little noise to the whole system to help the $u$ and $v$ reactions along.  

Below is the code segment we used to generate the initial conditions for `U` and `V`.  

**NOTE**: *DO NOT USE THIS CODE IN YOUR ASSIGNMENT*.
We are showing it here to help you understand how the system is constructed.  However, you _must use the data we've supplied below_ as your starting condition or your answers will not match those that the grading system expects.

```[Python]
num_blocks = 30
randx = numpy.random.randint(1, nx-1, num_blocks)
randy = numpy.random.randint(1, nx-1, num_blocks)
U = numpy.ones((n,n))
V = numpy.zeros((n,n))

r = 10
U[:,:] = 1.0

for i, j in zip(randx, randy):
    U[i-r:i+r,j-r:j+r] = 0.50
    V[i-r:i+r,j-r:j+r] = 0.25

U += 0.05*numpy.random.random((n,n))
V += 0.05*numpy.random.random((n,n))
```
"""
"""
## Your assignment
"""
"""
*  Discretize the reaction-diffusion equations using forward-time/central-space and assume that $\Delta x = \Delta y = \delta$.

*  For your timestep, set 
$$\Delta t = \frac{9}{40}\frac{\delta^2}{\max(D_u, D_v)}$$

*  Use zero Neumann boundary conditions on all sides of the domain.

You should use the initial conditions and constants listed in the cell below.  They correspond to the following domain:

*  Grid of points with dimension `192x192` points
*  Domain is $5{\rm m} \times 5{\rm m}$
*  Final time is $8000{\rm s}$.
"""
import numpy
from matplotlib import pyplot
import matplotlib.cm as cm
#%matplotlib inline
####################

n = 192

Du, Dv, F, k = 0.00016, 0.00008, 0.035, 0.065 # Bacteria 1 

dh = 5/(n-1)

T = 8000

dt = .9 * dh**2 / (4*max(Du,Dv))

nt = int(T/dt)

####################

"""
### Initial condition data files
"""
"""
In order to ensure that you start from the same initial conditions as we do, please download the file [uvinitial.npz](https://github.com/numerical-mooc/numerical-mooc/blob/master/lessons/04_spreadout/data/uvinitial.npz?raw=true)

This is a NumPy save-file that contains two NumPy arrays, holding the initial values for `U` and `V`, respectively.  Once you have downloaded the file into your working directory, you can load the data using the following code snippet.  
"""
uvinitial = numpy.load('./data/uvinitial.npz')
U = uvinitial['U']
V = uvinitial['V']
####################

fig = pyplot.figure(figsize=(8,5))
pyplot.subplot(121)
pyplot.imshow(U, cmap = cm.RdBu)
pyplot.xticks([]), pyplot.yticks([]);
pyplot.subplot(122)
pyplot.imshow(V, cmap = cm.RdBu)
pyplot.xticks([]), pyplot.yticks([]);

####################

"""
## Sample Solution
"""
"""
Below is an animated gif showing the results of this solution for a different set of randomized initial block positions.  Each frame of the animation represents 100 timesteps.  

Just to get your juices flowing!
"""
"""
<img src="./figures/celldivision.gif">
"""
"""
## Exploring extra patterns
"""
"""
Once you have completed the assignment, you might want to explore a few more of the interesting patterns that can be obtained with the Gray-Scott model.  The conditions below will result in a variety of patterns and should work without any other changes to your existing code.

This pattern is called "Fingerprints."
"""
"""
<img src="./figures/fingerprint.gif">
"""
#Du, Dv, F, k = 0.00014, 0.00006, 0.035, 0.065 # Bacteria 2
#Du, Dv, F, k = 0.00016, 0.00008, 0.060, 0.062 # Coral
#Du, Dv, F, k = 0.00019, 0.00005, 0.060, 0.062 # Fingerprint
#Du, Dv, F, k = 0.00010, 0.00010, 0.018, 0.050 # Spirals
#Du, Dv, F, k = 0.00012, 0.00008, 0.020, 0.050 # Spirals Dense
#Du, Dv, F, k = 0.00010, 0.00016, 0.020, 0.050 # Spirals Fast
#Du, Dv, F, k = 0.00016, 0.00008, 0.020, 0.055 # Unstable
#Du, Dv, F, k = 0.00016, 0.00008, 0.050, 0.065 # Worms 1
#Du, Dv, F, k = 0.00016, 0.00008, 0.054, 0.063 # Worms 2
#Du, Dv, F, k = 0.00016, 0.00008, 0.035, 0.060 # Zebrafish
####################

"""
## References
"""
"""
*  Reaction-diffusion tutorial, by Karl Sims
http://www.karlsims.com/rd.html

*  Pearson, J. E. (1993). [Complex patterns in a simple system](http://www.sciencemag.org/content/261/5118/189), _Science_, Vol. 261(5118), 189-192 // [PDF](http://www3.nd.edu/~powers/pearson.pdf) from nd.edu.

*  Pattern Parameters from [http://www.aliensaint.com/uo/java/rd/](http://www.aliensaint.com/uo/java/rd/)
"""
"""
---
###### The cell below loads the style of the notebook
"""
from IPython.core.display import HTML
css_file = '../../styles/numericalmoocstyle.css'
HTML(open(css_file, "r").read())
####################

print (" The presented result might be overlapping. ".center(60, "*"))
show()
