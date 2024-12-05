module DeepCompartmentModels

using Reexport

@reexport using Lux
@reexport using Distributions
@reexport using DistributionsAD
@reexport using OrdinaryDiffEq
@reexport using SciMLSensitivity
@reexport using ComponentArrays

import Zygote.ChainRules: @ignore_derivatives, ignore_derivatives
import Lux.Functors
import Optimisers
import SciMLBase
import FiniteDiff
import Zygote
import Random
import Optim
import Lux

using LinearAlgebra
using PartialFunctions

include("lib/compartment_models.jl");
export  unpack, one_comp!, one_comp_abs!, two_comp!, two_comp_abs!

include("lib/population.jl");
export  AbstractIndividual, BasicIndividual, Individual, Population, get_x, 
        get_y, load

include("lib/model.jl");
export  predict, forward_ode, forward_ode_with_dv, construct_p, AbstractModel,
        AbstractDEModel, Parameterization, MeanSqrt, MeanVar

include("lib/error.jl");
export  AdditiveError, ProportionalError, CombinedError, CustomError, 
        AbstractErrorModel, init_sigma, make_dist

include("lib/objectives.jl");
export  AbstractObjective, FixedObjective, MixedObjective, SSE, LogLikelihood, 
        FO, FOCE, VariationalELBO, MeanField, FullRank, objective, logprior, 
        logq, getq, elbo, optimize_etas, predict_de_parameters
        
include("lib/constrain.jl");
export  constrain

include("lib/mixed_effects.jl");
export  get_random_effects, make_etas

include("lib/natgrads.jl");
export  NaturalDescent, NaturalDescentMean, NaturalDescentVar, update_opt_state!

include("lib/nn.jl");
export  StandardNeuralNetwork, SNN, predict

include("lib/dcm.jl");
export  DeepCompartmentModel, DCM, forward, predict_typ_parameters

include("lib/node.jl");
export NeuralODE

include("lib/low_dim_node.jl");
export  LowDimensionalNeuralODE, LowDimNODE

include("lib/auto_encoding_node.jl");
export  AutoEncodingNeuralODE, VAENODE

include("lib/initializers.jl");
export  setup, init_theta, init_error, init_omega, init_phi

include("lib/optimization.jl")
export  fit, update_state

include("lib/callbacks.jl");
export  generate_dosing_callback

include("lib/lux.helpers.jl");
export  Normalize, AddGlobalParameters, Combine, SingleHeadedBranch, 
        MultiHeadedBranch, make_branch, interpret_branch
end
