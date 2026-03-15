module DeepCompartmentModels

import OrdinaryDiffEq.SciMLBase

abstract type AbstractModel{M,E} end
abstract type AbstractDEModel{P,M,E,S<:SciMLBase.AbstractSensitivityAlgorithm} <: AbstractModel{M,E} end

abstract type Parameterisation end
struct MeanVar <: Parameterisation end # Is used with Natural gradients / Bayesian Learning Rule
struct MeanSqrt <: Parameterisation end # needs to be constrained during optimization

export  AbstractModel, AbstractDEModel, Parameterisation, MeanVar, MeanSqrt

using Reexport

@reexport import Optimisers
@reexport import Accessors
@reexport import Lux

@reexport using Distributions
@reexport using OrdinaryDiffEq
@reexport using DistributionsAD
@reexport using SciMLSensitivity

import Zygote.ChainRules: @non_differentiable, @ignore_derivatives, ignore_derivatives
import Zygote
import Random

using Static
using Functors
using ThreadPools
using LinearAlgebra
using InvertedIndices
using ComponentArrays
using LogExpFunctions

include("lib/compartment_models.jl");
export  unpack, one_comp!, one_comp_abs!, two_comp!, two_comp_abs!

include("lib/individuals.jl"); include("lib/population.jl");
export  AbstractIndividual, BasicIndividual, TimeVariableIndividual, MOIndividual, 
        Individual, Population, get_x, get_t, get_tx, get_y, load

include("lib/error.jl");
export  AbstractErrorModel, AdditiveError, ProportionalError, CombinedError, 
        CustomError, ErrorModelSet, make_dist, var

include("lib/dcm.jl");
export  DeepCompartmentModel, DCM, predict_typ_parameters, predict_de_parameters, 
        predict

include("lib/nn.jl");
export  StandardNeuralNetwork, SNN, predict

include("lib/ude.jl");
export  UniversalDiffEq, AbstractUDEType, BasicUDE, TimeConcatUDE, build_problem

include("lib/solve.jl");
export  solve, solve_for_target, construct_p

include("lib/random_effects.jl");
export  get_random_effects, make_etas, sample_gaussian, update_epsilon!

include("lib/objectives.jl");
export  MSE, SSE, LogLikelihood, VariationalELBO, mse, sse, 
        loglikelihood, kldivergence, logprior, logq, getq, logjoint, 
        elbo

include("lib/setup.jl");
export  setup, setup_phi

include("lib/gradients.jl");
export  gradient, create_batches, take_batch, residual_error_value_and_gradient

include("lib/vem.jl");
export  m_step, optimise_omega, optimise_residual_error

include("lib/callbacks.jl");
export  generate_dosing_callback

include("lib/lux.helpers.jl");
export  Normalize, AddGlobalParameters, Combine, SingleHeadedBranch, 
        MultiHeadedBranch, make_branch, interpret_branch
end
