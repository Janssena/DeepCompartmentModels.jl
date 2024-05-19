# DeepCompartmentModels.jl

A package for fitting Deep Compartment Models in julia. 

Most of the basic functionality for fitting deep compartment models has been 
implemented. If you have any suggestions to improve the package, please do not 
hesitate to open an 
[issue](https://github.com/Janssena/DeepCompartmentModels.jl/issues/new) or 
submit a pull request. 

### Introduction

Deep Compartment Models are novel deep learning based modeling framework for 
fitting machine learning models to time-series data in the medical domain. The 
aim of these models is to provide insights for the personalization of treatment 
of patients. The package aims to combine techinques from the field of machine 
learning and pharmacometrics in order to produce more reliable models.

A main problem with using machine learning as-is for this purpose is that there 
is a large amount of heterogeneity in measurement timing or treatment 
intervations between patients. In standard machine learning based approaches, we 
have to supply information such as the time points of interest or the 
administered dose directly as inputs to the model, while we are uncertain how 
the algorithm will treat them. Generally, we thus see that such information is 
interpreted incorrectly, and thus raises questions regarding the reliablility of 
the predictions. Instead, by using a system of differential equations, we can 
explicitly handle such information. The classic deep compartment model structure 
uses compartment models to constrain the solution to follow certain expectations 
about drug kinetics and dynamics. Aside from improving model reliability, this 
also reduces the need for large data sets as we can supply prior knowledge about 
drug dynamics to the model *a priori*.

### Installation

**Installing Julia.**

Most pharmacometricians are used to programming in R, and likely do not yet have 
julia installed. You can download the appropriate julia installer 
[here](https://julialang.org/downloads/).  


**Installing the DeepCompartmentModels package**

After installing julia run the `julia` command in an open command line or bash 
window to launch a julia REPL. Enter the following commands:

```julia
julia> ]
pkg> add DeepCompartmentModels

# or 

julia> using Pkg
julia> Pkg.add("DeepCompartmentModels")
```

### Fitting a model

Deep compartment models consist of a neural network and a system of differential 
equations. [Lux](https://lux.csail.mit.edu/stable/) is a machine learning 
library that aids in definiting complex neural network architectures. It is 
automatically loaded in the REPL session after running 
`using DeepCompartmentModels`, so you can direclty make use of functions like 
`Lux.Chain` and `Lux.Dense`.


```julia
import Optimisers
import CSV

using DataFrames
using DeepCompartmentModels

df = DataFrame(CSV.File("my_dataset.csv"))

population = load(df, [:WEIGHT, :AGE])

ann = Chain(
    # Our data set contains two covariates, which we feed into a hidden layer with 16 neurons
    Dense(2, 16, relu), 
    Dense(16, 4, softplus), # Our differential equation has four parameters
)

# DeepCompartmentModels already exports some compartmental structures including two_comp!
model = DCM(two_comp!, ann, 2) 

fit!(model, population, Optimisers.Adam(), 500) # optimize neural network for 500 epochs

predict(model, population[1]) # predict the concentration for the first individual in the population.
```

### Citing this work

Whenever you use contents from this package, please help us spread interest by 
citing the original works on which this package has been based:

> Janssen, A., Leebeek, F. W., Cnossen, M. H., Mathôt, R. A., OPTI‐CLOT study group and SYMPHONY consortium, (2022). Deep compartment models: a deep learning approach for the reliable prediction of time‐series data in pharmacokinetic modeling. CPT: Pharmacometrics & Systems Pharmacology, 11(7), 934-945.