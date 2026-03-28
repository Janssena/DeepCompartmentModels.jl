# DeepCompartmentModels.jl

A package for fitting Deep Compartment Models in julia. 

Most of the basic functionality for fitting these models has been implemented. 
If you have any suggestions to improve the package, please do not hesitate to 
open an [issue](https://github.com/Janssena/DeepCompartmentModels.jl/issues/new) 
or submit a pull request. 

### Introduction

The deep compartment model (DCMs) framework is novel deep learning based 
modeling framework for fitting machine learning (ML) models to time-series data 
in the medical domain. The aim of these models is to provide insights for the 
personalization of treatment of patients. The package aims to combine techinques 
from the field of ML and pharmacometrics in order to produce more reliable models.

A main problem with using machine learning as-is for this purpose is that there 
is a large amount of heterogeneity in measurement timing or treatment 
interventions between patients. In standard ML-based algorithms, we have to 
supply information such as the time points of interest or the administered dose 
directly as inputs to the model, while we are uncertain how the algorithm will 
treat them. Generally, we thus see that such information is interpreted 
incorrectly, and thus raises questions regarding the reliablility of the 
predictions. Instead, by using a system of differential equations, we can 
explicitly handle time and other interventions. The standard DCM structure uses 
compartment models to constrain the solution to follow certain expectations 
about drug kinetics and dynamics. Aside from improving model reliability, this 
also reduces the need for large data sets as we can supply prior knowledge about 
drug dynamics to the model *a priori*.

We are current working on bringing NeuralODE capabilities to the framework.

### Installation

**Installing Julia.**

Most pharmacometricians are used to programming in R, and likely do not yet have 
julia installed. You can download the appropriate julia installer 
[here](https://julialang.org/downloads/).  


**Installing the DeepCompartmentModels package**

After installing julia, run the `julia` command in an open command line or bash 
window to launch a julia REPL. Enter the following commands:

```julia
julia> ]
pkg> add DeepCompartmentModels

# or 

julia> using Pkg
julia> Pkg.add("DeepCompartmentModels")
```

### Fitting a model

A DCM consists of a neural network and a system of differential 
equations. [Lux](https://lux.csail.mit.edu/stable/) is a machine learning 
library that aids in defining complex neural network architectures. It is 
automatically loaded in the REPL session after running 
`using DeepCompartmentModels`, so you can direclty make use of functions like 
`Lux.Chain` and `Lux.Dense`.

The DeepCompartmentModels package already exports some compartmental structures 
including one_comp! and two_comp!. Pull requests adding new compartmental 
structures are very welcome!

```julia
import CSV

using DataFrames
using DeepCompartmentModels # Also re-exports Lux and Optimisers

df = DataFrame(CSV.File("my_dataset.csv"))

population = load(df, [:WEIGHT, :AGE])

# Our data set contains two covariates, which we feed into a hidden layer with 16 neurons
ann = Lux.Chain(
    Lux.Dense(2, 16, Lux.relu), 
    Lux.Dense(16, 4, Lux.softplus), # Our differential equation has four parameters
)

obj_fn = SSE() # Minimize the sum of squared errors (equivalent to the mean squared error)
model = DCM(two_comp!, ann)
ps, st = setup(obj_fn, model)

# Select an optimizer and a learning rate.
opt = Optimisers.Adam(1e-2)

# Fit the model:
opt_state, ps, st = fit(obj_fn, model, population, opt, ps, st; epochs = 100);

# Making predictions.
predict(model, population[1], ps, st)
```

### Citing this work

Whenever you use contents from this package, please help us spread interest by 
citing the original work on which this package has been based:

> Janssen, A., Leebeek, F. W., Cnossen, M. H., Mathôt, R. A., OPTI‐CLOT study group and SYMPHONY consortium, (2022). Deep compartment models: a deep learning approach for the reliable prediction of time‐series data in pharmacokinetic modeling. CPT: Pharmacometrics & Systems Pharmacology, 11(7), 934-945.
