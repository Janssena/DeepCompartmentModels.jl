# DeepCompartmentModels.jl



A package for fitting Deep Compartment Models in julia. 

This package is still under development, although most of the basic functionality have been implemented. If you have any suggestions with respect to improving this package, please do not hesitate to open an [issue](https://github.com/Janssena/DeepCompartmentModels.jl/issues/new) or submit a pull request. 



#### Introduction

Deep Compartment Models are a combination of neural networks and a system of differential equations. Their goal is to provide a more reliable framework for using deep learning approaches for the prediction of time series data (specifically in the field of pharmacometrics). A main problem with using machine learning as-is for this purpose is that there is a large amount of heterogeneity in blood sampling time or treatment strategies between patients. In regular machine learning approaches, we have to supply information as the time point of interest or the administered dose directly as inputs in the model, while we are uncertain how the algorithm will treat them. Generally, we thus see that such information is interpreted incorrect, and thus raises questions regarding the accuracy of the predictions. Instead, by using a system of differential equations, we can explicitly handle such information, and by using compartment models constrain the solution to follow a certain structure. This might also reduce the need for large data sets as we can supply prior knowledge about drug dynamics to the model *a priori*.



#### Installation

**Installing Julia. **

Most pharmacometricians are used to programming in R, and likely do not yet have julia installed. You can download the appropriate julia installer [here](https://julialang.org/downloads/).  



**Installing the DeepCompartmentModels package**

After installing julia run the `julia` command in an open command line or bash window to launch a julia REPL. Enter the following commands:

```julia
julia> ]
pkg> add DeepCompartmentModels

# or 

julia> using Pkg
julia> Pkg.add("DeepCompartmentModels")
```



#### Fitting a model

Deep Compartment Models consist of a neural network and a system of differential equations. [Flux](https://fluxml.ai/Flux.jl/stable/) is a machine learning library that simplifies initializing neural networks models.  

```julia
import Flux

using DeepCompartmentModels

population = load("myDataset.csv", [:WEIGHT, :AGE])

ann = Flux.Chain(
    # Our data set contains two covariates, which we feed into a hidden layer with 16 neurons
	Flux.Dense(2, 16), 
    Flux.Dense(16, 4), # Our differential equation has four parameters
)

# DeepCompartmentModels already exports some compartmental structures including two_comp!
model = DCM(two_comp!, ann, 2) 

fit!(model, population) # optimize neural network

predict(model, population) # predict the concentration for the population.
```



For a more comprehensive walkthrough you can look browse through our [tutorial folder](https://github.com/Janssena/DeepCompartmentModels.jl/tree/main/tutorial) which contains jupyter notebooks with more examples.



