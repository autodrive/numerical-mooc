# -*- coding: utf8 -*-
from pylab import *
"""
###### Content under Creative Commons Attribution license CC-BY 4.0, code under MIT license © 2015 L.A. Barba, G.F. Forsyth, B. Knaepen
"""
"""
# Relax and hold steady
"""
"""
This is the fourth and last notebook of **Module 5** (*"Relax and hold steady"*), dedicated to elliptic PDEs. In the [previous notebook](./05_03_Iterate.This.ipynb), we examined how different algebraic formulations can speed up the iterative solution of the Laplace equation, compared to the simplest (but slowest) Jacobi method.  The Gauss-Seidel and successive-over relaxation methods both provide faster algebraic convergence than Jacobi. But there is still room for improvement.  

In this lesson, we'll take a look at the very popular [conjugate gradient](https://en.wikipedia.org/wiki/Conjugate_gradient_method) (CG) method.  
The CG method solves linear systems with coefficient matrices that are symmetric and positive-definite. It is either used on its own, or in conjunction with multigrid—a technique that we'll explore later on its own (optional) course module.

For a real understanding of the CG method, there is no better option than studying the now-classic monograph by Jonathan Shewchuck: *"An introduction to the conjugate gradient method without the agonizing pain"* (1994). Here, we try to give you a brief summary to explain the implementation in Python.
"""
"""
### Test problem
"""
"""
Let's return to the Poisson equation example from [Lesson 2](./05_02_2D.Poisson.Equation.ipynb).

\begin{equation}
\nabla^2 p = -2\left(\frac{\pi}{2}\right)^2\sin\left( \frac{\pi x}{L} \right) \cos\left(\frac{\pi y}{L}\right)
\end{equation}

in the domain 

$$\left\lbrace \begin{align*}
0 &\leq x\leq 1  \\
-0.5 &\leq y \leq 0.5 
\end{align*} \right.$$

with boundary conditions 

$$p=0 \text{ at } \left\lbrace 
\begin{align*}
x&=0\\
y&=0\\
y&=-0.5\\
y&=0.5
\end{align*} \right.$$

We will solve this equation by assuming an initial state of $p=0$ everywhere, and applying boundary conditions to relax via the Laplacian operator.
"""
"""
## Head in the right direction!
"""
"""
Recall that in its discretized form, the Poisson equation reads,

$$\frac{p_{i+1,j}^{k}-2p_{i,j}^{k}+p_{i-1,j}^{k}}{\Delta x^2}+\frac{p_{i,j+1}^{k}-2 p_{i,j}^{k}+p_{i,j-1}^{k}}{\Delta y^2}=b_{i,j}^{k}$$

The left hand side represents a linear combination of the values of $p$ at several grid points and this linear combination has to be equal to the value of the source term, $b$, on the right hand side.

Now imagine you gather the values $p_{i,j}$ of $p$ at all grid points into a big vector ${\bf p}$ and you do the same for $b$ using the same ordering. Both vectors ${\bf p}$ and ${\bf b}$ contain $N=nx*ny$ values and thus belong to $\mathbb{R}^N$. The discretized Poisson equation corresponds to the following linear system:

\begin{equation}
A{\bf p}={\bf b},
\end{equation}

where $A$ is an $N\times N$ matrix. Although we will not directly use the matrix form of the system in the CG algorithm, it is useful to examine the problem this way to understand how the method works.

All iterative methods start with an initial guess, $\mathbf{p}^0$, and modify it in a way such that we approach the solution. This can be viewed as modifying the vector of discrete $p$ values on the grid by adding another vector, i.e., taking a step of magnitude $\alpha$ in a direction $\mathbf{d}$, as follows:

\begin{equation}
{\bf p}^{k+1}={\bf p}^k + \alpha {\bf d}^k
\end{equation}


The iterations march towards the solution by taking steps along the direction vectors ${\bf d}^k$, with the scalar $\alpha$ dictating how big a step to take at each iteration.  We *could* converge faster to the solution if we just knew how to carefully choose the direction vectors and the size of the steps. But how to do that?
"""
"""
## The residual
"""
"""
One of the tools we use to find the right direction to step to is called the *residual*.  What is the residual?  We're glad you asked!

We know that, as the iterations proceed, there will be some error between the calculated value, $p^k_i$, and the exact solution $p^{exact}_i$.  We may not know what the exact solution is, but we know it's out there.  The error is:

\begin{equation}
e^k_i = p^k_i - p^{exact}_i
\end{equation}

**Note:** We are talking about error at a specific point $i$, not a measure of error across the entire domain.  

What if we recast the Poisson equation in terms of a not-perfectly-relaxed $\bf p^k$?

\begin{equation}
A \bf p^k \approx b
\end{equation}

We write this as an approximation because $\bf p^k \neq p$.  To "fix" the equation, we need to add an extra term to account for the difference in the Poisson equation $-$ that extra term is called the residual.  We can write out the modified Poisson equation like this:

\begin{equation}
{\bf r^k} + A \bf p^k = b
\end{equation}
"""
"""
## The method of steepest descent
"""
"""
Before considering the more-complex CG algorithm, it is helpful to introduce a simpler approach called the *method of steepest descent*. At iteration $0$, we choose an initial guess. Unless we are immensely lucky, it will not satisfy the Poisson equation and we will have,

\begin{equation}
{\bf b}-A{\bf p}^0={\bf r}^0\ne {\bf 0}
\end{equation}

The vector ${\bf r}^0$ is the initial residual and measures how far we are from satisfying the linear system. We can monitor the residual vector at each iteration, as it gets (hopefully) smaller and smaller: 

\begin{equation}
{\bf r}^k={\bf b}-A{\bf p}^k
\end{equation}

We make two choices in the method of steepest descent:

1. the direction vectors are the residuals ${\bf d}^k = {\bf r}^k$, and
2. the length of the step makes the $k+1^{th}$ residual orthogonal to the $k^{th}$ residual.

There are good (not very complicated) reasons to justify these choices and you should read one of the references to understand them. But since we want you to converge to the end of the notebook in a shorter time, please accept them for now. 

Choice 2 requires that,

\begin{align}
{\bf r}^{k+1}\cdot {\bf r}^{k} = 0 \nonumber \\
\Leftrightarrow ({\bf b}-A{\bf p}^{k+1}) \cdot {\bf r}^{k} = 0 \nonumber \\
\Leftrightarrow ({\bf b}-A({\bf p}^{k}+\alpha {\bf r}^k)) \cdot {\bf r}^{k} = 0 \nonumber \\
\Leftrightarrow ({\bf r}^k+\alpha A{\bf r}^k) \cdot {\bf r}^{k} = 0 \nonumber \\
\alpha = \frac{{\bf r}^k \cdot {\bf r}^k}{A{\bf r}^k \cdot {\bf r}^k}.
\end{align}

We are now ready to test this algorithm.

To begin, let's import libraries, set up our mesh and import some helper functions:
"""
import numpy
from math import pi
#%matplotlib inline
####################

from laplace_helper import plot_3D, L2_rel_error
from cg_helper import poisson_2d, p_analytical
####################

# Parameters
nx = 101
ny = 101
xmin = 0
xmax = 1
ymin = -0.5
ymax = 0.5

l2_target = 1e-10

# Spacing
dx = (xmax-xmin)/(nx-1)
dy = (ymax-ymin)/(ny-1)

# Mesh
x  = numpy.linspace(xmin,xmax,nx)
y  = numpy.linspace(ymin,ymax,ny)
X,Y = numpy.meshgrid(x,y)

# Source
L = xmax-xmin
b = -2*(pi/L)**2*numpy.sin(pi*X/L)*numpy.cos(pi*Y/L)

# Initialization
p_i  = numpy.zeros((ny,nx))

# Analytical solution
pan = p_analytical(X,Y,L)

####################

"""
### Time to code steepest descent!  

Let's quickly review the solution process:

1. Calculate the residual, $\bf r^k$, which also serves as the direction vector, $\bf d^k$
2. Calculate the step size $\alpha$
3. Update ${\bf p}^{k+1}={\bf p}^k + \alpha {\bf d}^k$

"""
"""
##### How do we calculate the residual?  

We have an equation for the residual above: 
\begin{equation}
{\bf r}^k={\bf b}-A{\bf p}^k
\end{equation}

Remember that $A$ is just a stand-in for the discrete Laplacian, which taking $\Delta x=\Delta y$ is:

\begin{equation}
\nabla^2 p^k = 4p^k_{i,j} - \left(p^{k}_{i,j-1} + p^k_{i,j+1} + p^{k}_{i-1,j} + p^k_{i+1,j} \right)
\end{equation}
"""
"""
##### What about calculating $\alpha$?

The calculation of $\alpha$ is relatively straightforward, but does require evaluating the term $A{\bf r^k}$, but we just wrote the discrete $A$ operator above. You just need to apply that same formula to $\mathbf{r}^k$.
"""
def steepest_descent_2d(p, b, dx, dy, l2_target):
    '''
    Performs steepest descent relaxation
    Assumes Dirichlet boundary conditions p=0
    
    Parameters:
    ----------
    p : 2D array of floats
        Initial guess
    b : 2D array of floats
        Source term
    dx: float
        Mesh spacing in x direction
    dy: float
        Mesh spacing in y direction
    l2_target: float
        iteration exit criterion
        
    Returns:
    -------
    p: 2D array of float
        Distribution after relaxation
    '''
    ny, nx = p.shape
    r  = numpy.zeros((ny,nx)) # residual
    Ar  = numpy.zeros((ny,nx)) # to store result of matrix multiplication
    
    l2_norm = 1
    iterations = 0
    l2_conv = []
    
    # Iterations
    while l2_norm > l2_target:

        pd = p.copy()
        
        r[1:-1,1:-1] = b[1:-1,1:-1]*dx**2 + 4*pd[1:-1,1:-1] - \
            pd[1:-1,2:] - pd[1:-1,:-2] - pd[2:, 1:-1] - pd[:-2, 1:-1]
        
        Ar[1:-1,1:-1] = -4*r[1:-1,1:-1]+r[1:-1,2:]+r[1:-1,:-2]+\
            r[2:, 1:-1] + r[:-2, 1:-1]

        rho = numpy.sum(r*r)
        sigma = numpy.sum(r*Ar)
        alpha = rho/sigma

        p = pd + alpha*r
        
        # BCs are automatically enforced
        
        l2_norm = L2_rel_error(pd,p)
        iterations += 1
        l2_conv.append(l2_norm)
    
    print('Number of SD iterations: {0:d}'.format(iterations))
    return p, l2_conv     


####################

"""
Let's see how it performs on our example problem.
"""
p, l2_conv = steepest_descent_2d(p_i.copy(), b, dx, dy, l2_target)
L2_rel_error(p, pan)
####################

"""
Not bad! it took only *two* iterations to reach a solution that meets our exit criterion. Although this seems great, the steepest descent algorithm is not too good when used with large systems or more complicated right-hand sides in the Poisson equation (we'll examine this below!). We can get better performance if we take a little more care in selecting the direction vectors, $\bf d^k$.
"""
"""
## The method of conjugate gradients
"""
"""
With steepest descent, we know that two **successive** jumps are orthogonal, but that's about it.  There is nothing to prevent the algorithm from making several jumps in the same (or a similar) direction.  Imagine you wanted to go from the intersection of 5th Avenue and 23rd Street to the intersection of 9th Avenue and 30th Street. Knowing that each segment has the same computational cost (one iteration), would you follow the red path or the green path?
"""
"""
<img src="./figures/jumps.png" width=350>
"""
"""
#### Figure 1. Do you take the red path or the green path?
"""
"""
The method of conjugate gradients reduces the number of jumps by making sure the algorithm never selects the same direction twice. The size of the jumps is now given by:

\begin{equation}
\alpha = \frac{{\bf r}^k \cdot {\bf r}^k}{A{\bf d}^k \cdot {\bf d}^k}
\end{equation}

and the direction vectors by:

\begin{equation}
{\bf d}^{k+1}={\bf r}^{k+1}+\beta{\bf d}^{k}
\end{equation}

where $\beta = \frac{{\bf r}^{k+1} \cdot {\bf r}^{k+1}}{{\bf r}^k \cdot {\bf r}^k}$.

The search directions are no longer equal to the residuals but are instead a  linear combination of the residual and the previous search direction. It turns out that CG converges to the exact solution (up to machine accuracy) in a maximum of $N$ iterations! When one is satisfied with an approximate solution, many fewer steps are needed than with any other method. Again, the derivation of the algorithm is not immensely difficult and can be found in Shewchuk (1994).

"""
"""
### Implementing Conjugate Gradients
"""
"""
We will again update $\bf p$ according to 

\begin{equation}
{\bf p}^{k+1}={\bf p}^k + \alpha {\bf d}^k
\end{equation}

but use the modified equations above to calculate $\alpha$ and ${\bf d}^k$.  

You may have noticed that $\beta$ depends on both ${\bf r}^{k+1}$ and ${\bf r}^k$ and that makes the calculation of ${\bf d}^0$ a little bit tricky.  Or impossible (using the formula above).  Instead we set ${\bf d}^0 = {\bf r}^0$ for the first step and then switch for all subsequent iterations.  

Thus, the full set of steps for the method of conjugate gradients is:

Calculate ${\bf d}^0 = {\bf r}^0$ (just  once), then

1. Calculate $\alpha = \frac{{\bf r}^k \cdot {\bf r}^k}{A{\bf d}^k \cdot {\bf d}^k}$
2. Update ${\bf p}^{k+1}$
3. Calculate ${\bf r}^{k+1} = {\bf r}^k - \alpha A {\bf d}^k$ $\ \ \ \ $(see <a href='#references'>Shewchuk (1994)</a>)
4. Calculate $\beta = \frac{{\bf r}^{k+1} \cdot {\bf r}^{k+1}}{{\bf r}^k \cdot {\bf r}^k}$
5. Calculate ${\bf d}^{k+1}={\bf r}^{k+1}+\beta{\bf d}^{k}$
6. Repeat!
"""
def conjugate_gradient_2d(p, b, dx, dy, l2_target):
    '''Performs cg relaxation
    Assumes Dirichlet boundary conditions p=0
    
    Parameters:
    ----------
    p : 2D array of floats
        Initial guess
    b : 2D array of floats
        Source term
    dx: float
        Mesh spacing in x direction
    dy: float
        Mesh spacing in y direction
    l2_target: float
        exit criterion
        
    Returns:
    -------
    p: 2D array of float
        Distribution after relaxation
    '''
    ny, nx = p.shape
    r  = numpy.zeros((ny,nx)) # residual
    Ad  = numpy.zeros((ny,nx)) # to store result of matrix multiplication 
    
    l2_norm = 1
    iterations = 0
    l2_conv = []
    
    # Step-0 We compute the initial residual and 
    # the first search direction is just this residual
    
    r[1:-1,1:-1] = b[1:-1,1:-1]*dx**2 + 4*p[1:-1,1:-1] - \
        p[1:-1,2:] - p[1:-1,:-2] - p[2:, 1:-1] - p[:-2, 1:-1]
    d = r.copy()
    rho = numpy.sum(r*r)
    Ad[1:-1,1:-1] = -4*d[1:-1,1:-1]+d[1:-1,2:]+d[1:-1,:-2]+\
        d[2:, 1:-1] + d[:-2, 1:-1]
    sigma = numpy.sum(d*Ad)
    
    # Iterations
    while l2_norm > l2_target:

        pk = p.copy()
        rk = r.copy()
        dk = d.copy()
        
        alpha = rho/sigma

        p = pk + alpha*dk
        r = rk- alpha*Ad
        
        rhop1 = numpy.sum(r*r)
        beta = rhop1 / rho
        rho = rhop1
        
        d = r + beta*dk
        Ad[1:-1,1:-1] = -4*d[1:-1,1:-1] + d[1:-1,2:] + d[1:-1,:-2] + \
            d[2:, 1:-1] + d[:-2, 1:-1]
        sigma = numpy.sum(d*Ad)
        
        # BCs are automatically enforced
        
        l2_norm = L2_rel_error(pk,p)
        iterations += 1
        l2_conv.append(l2_norm)
    
    print('Number of CG iterations: {0:d}'.format(iterations))
    return p, l2_conv     

####################

p, l2_conv = conjugate_gradient_2d(p_i.copy(), b, dx, dy, l2_target)
L2_rel_error(p,pan)
####################

"""
The method of conjugate gradients also took two iterations to reach a solution that meets our exit criterion. But let's compare this to the number of iterations needed for the Jacobi iteration:
"""
p, l2_conv = poisson_2d(p_i.copy(), b, dx, dy, l2_target)
####################

"""
For our test problem, we get substantial gains in terms of computational cost using the method of steepest descent or the conjugate gradient method.
"""
"""
## More difficult Poisson problems
"""
"""
The conjugate gradient method really shines when one needs to solve more difficult Poisson problems. To get an insight into this, let's solve the Poisson problem using the same boundary conditions as the previous problem but with the following right-hand side,

\begin{equation}
b = \sin\left(\frac{\pi x}{L}\right) \cos\left(\frac{\pi y}{L}\right) + \sin\left(\frac{6\pi x}{L}\right) \cos\left(\frac{6\pi y}{L}\right)
\end{equation}

"""
b = (numpy.sin(pi*X/L)*numpy.cos(pi*Y/L)+
     numpy.sin(6*pi*X/L)*numpy.sin(6*pi*Y/L))
####################

p, l2_conv = poisson_2d(p_i.copy(), b, dx, dy, l2_target)

p, l2_conv = steepest_descent_2d(p_i.copy(), b, dx, dy, l2_target)

p, l2_conv = conjugate_gradient_2d(p_i.copy(), b, dx, dy, l2_target)
####################

"""
Now we can really appreciate the marvel of the CG method!
"""
"""
## References
"""
"""
<a id='references'></a>
Shewchuk, J. (1994). [An Introduction to the Conjugate Gradient Method Without the Agonizing Pain (PDF)](http://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf)

Ilya Kuzovkin, [The Concept of Conjugate Gradient Descent in Python](http://ikuz.eu/2015/04/15/the-concept-of-conjugate-gradient-descent-in-python/)
"""
from IPython.core.display import HTML
css_file = '../../styles/numericalmoocstyle.css'
HTML(open(css_file, "r").read())
####################

print (" The presented result might be overlapping. ".center(60, "*"))
show()
