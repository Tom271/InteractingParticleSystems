# Numerical Methods for an Interacting Particle System

 [![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
 [![Build Status](https://travis-ci.com/Tom271/InteractingParticleSystems.svg?branch=master)](https://travis-ci.com/Tom271/InteractingParticleSystems)

This package simulates the interacting particle system of Buttà et al. [[1]](#references) as well as the related model of Garnier et al. [[2]](#references). It is designed for particles positioned on the torus interacting through their velocities. It can be adjusted to other interactions.

## Installation
  To install the package, download the repository and open the terminal. Navigate to the directory containing the repo using `cd \PATH_TO_FOLDER\` then run `pip install .` – don't forget the period! The package can then be called using the standard  `import particle`.

## Basic Running
  All the functionality is contained within the `particle` folder. There are three main files:
  - `simulate.py`

    As the name suggests, all the simulating of the particles is done here. Any modifications to the interaction are done in this file, as well as adding new initial conditions through keyword strings. This file also calls `herdingfunctions.py` and `interactionfunctions.py`. These are kept separate for clarity and ease of adjustment.
  - `plotting.py`

    Contained in this file are various ways of plotting the data that `get_trajectories()` produces, and the analysis from `statistics.py`. Most interesting is `anim_torus()` for producing animations of the particles
  - `processing.py`

    If you're interested in running many simulations, this file contains functions for storing and loading the data in an efficient format. It also adds functionality for testing many different parameter sets in one go. Much of the functionality of `statistics.py` relies on the data being saved in this way, and not just passing the data in directly to the function.
  - `statistics.py`

    After simulating some data, use this file to do some post-processing and calculate some common statistics for the data. Currently includes:
      - `moving_average()` -- a rolling average of any input data
      - `Q_order_t()` -- an order parameter for quantifying how clustered the positions are
      - `CL2()` -- the centred $L^2$ discrepancy, a metric for assessing how far from uniform the positions are
      - `calculate_l1_convergence()` -- the $l^1$ discrepancy between the position data and a uniform distribution.
      - `calculate_stopping_time()` -- a measure for how long convergence in average velocity takes for a metastable state.
  Other simple statistics, such as the average velocity of the ensemble, are coded directly into the plotting functionality.


### Obtaining Trajectories
  To simulate the particle model the basic format is
  ```python
  import particle

  t,x,v = particle.simulate.get_trajectories(
      particle_count=100,
      D=1,
      initial_dist_x="two_clusters",
      initial_dist_v="pos_normal_dn",
      phi="Gamma",
      dt=0.1,
      T_end=100,
      G="Smooth",
      L=2 * np.pi ,
      scaling="Local",
      gamma=0.1,
  )
  ```
One can then produce an animation using the `anim_torus` function from `plotting.py`:
```python
  ani = plotting.anim_torus(t, x, v, mu_v=1, variance=1, L=2 * np.pi , framestep=1)
```
This gives an animation of the particles moving on the torus (green if moving clockwise, orange if anticlockwise) as well as four histograms. The top two histograms are of the particles positions at the current time, t (left) and positions up to the current time step [0,t] (right). The bottom two histograms are the same but for the velocities of the particles.



### Parameter Options
  For the full particle system, the following parameters are available.
  - `particle_count`: integer, number of particles to simulate

  - `dt`: float, time step to be use in Euler-Maruyama scheme. If not given, will be set as a tenth of the diffusion coefficient.

  - `T_end`: time at which to end simulation, float

  - `L`: the circumference of the circle on which the particles move, float

  - `D`: diffusion coefficient, denoted σ in the model, float

  - `initial_dist_x`: the initial positions of the particles. This can either be given as an array of length `particles` or alternatively can be one of:
    + `"uniform_dn"`: a [uniform distribution](https://en.wikipedia.org/wiki/Uniform_distribution_(continuous)) on [0,L]
    + `"one_cluster"`: a cluster of particles of width 2π /10 and centre π /2.
    + `"two_clusters"`: two clusters of particles both with width 2π /10 and centred at  π /2 and 3π /2

  - `initial_dist_v`:  the initial velocities of the particles. This can either be given as an array of length `particles` or alternatively can be one of:
    + `"pos_normal_dn"`: a [normal distribution](https://en.wikipedia.org/wiki/Normal_distribution) with mean 1 and variance 2.
    + `"neg_normal_dn"`: a [normal distribution](https://en.wikipedia.org/wiki/Normal_distribution) with mean -1 and variance 2.
    + `"uniform_dn"`: a [uniform distribution](https://en.wikipedia.org/wiki/Uniform_distribution_(continuous)) on [0,1]
    + `"pos_gamma_dn"`: a [Gamma distribution](https://en.wikipedia.org/wiki/Gamma_distribution) with shape 7.5 and scale 1.0

  - `interaction_function`: the interaction function φ, determined from a dictionary using one of the following strings:
    + `"Zero"`: no interaction between particles
    + `"Uniform"`: every particle interacts with every other particle equally, irrespective of separation
    + `"Indicator"`: a particle interacts with another if they are separated by less than L/10
    + `"Gamma"`: a particle interacts with another if they are separated by less than γ*L. Here γ is defined by the parameter `gamma`
    + `"Smoothed Indicator"`: An indicator function without the discontinuity
    + `"Garnier"`: the interaction function of [[2]](#references)

  - `gamma`: a float between 0 and 1 that is used if the `interaction_function` is ``"Gamma"``

  - `well_depth`: a float greater than 4 that is used if the `interaction_function` is ``"Garnier"``

  - `herding_function`: the herding function G, determined from a dictionary using one of the following strings:
    + `"zero"`: no herding takes place
    + `"step"`: a step function herding common in the literature
    + `"smooth"`: a smooth herding function more amenable to analysis
    + `"Garnier"`: the herding function of [[2]](#references).


  - `scaling`: the denominator of the fraction in the interaction term (the argument of G) can be either:
    + `"Local"`: scales the interaction by the number of particles the current particle is interacting with
    + `"Global"`: scales the interaction by the total number of particles in the system, as in [[2]](#references)


## Documentation
  A deeper dive into the model is available [here](https://tom271.github.io/InteractingParticleSystems/). For further documentation, look at the docstrings of the functions.

#### References:
[1] [P. Buttà, F. Flandoli, M. Ottobre, and B. Zegarlinski. A non-linear kinetic model of self-propelled particles with multiple equilibria. Kinetic & Related Models, 12(4):791–827, 2019.](https://arxiv.org/abs/1804.01247)

[2] [J. Garnier, G. Papanicolaou, T-W. Yang, Mean field model for collective motion bistability, Discrete & Continuous Dynamical Systems, 24(2): 851-879, 2019.](https://arxiv.org/abs/1611.02194)


**The authors were supported by The Maxwell Institute Graduate School in Analysis and its Applications, a Centre for Doctoral Training funded by the UK Engineering and Physical Sciences Research Council (grant EP/L016508/01), the Scottish Funding Council, Heriot-Watt University and the University of Edinburgh.**
