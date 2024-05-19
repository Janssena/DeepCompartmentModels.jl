module DeepCompartmentModels

using Reexport

@reexport using Lux
@reexport using Distributions
@reexport using DistributionsAD
@reexport using ComponentArrays
@reexport using SciMLSensitivity
@reexport using DifferentialEquations

import Zygote.ChainRules: @ignore_derivatives, ignore_derivatives
import LinearAlgebra: Diagonal, Symmetric, diag
import Optimisers
import Bijectors
import Zygote
import Random

using PartialFunctions

include("lib/population.jl");
export  AbstractIndividual, BasicIndividual, Individual, Population, get_x

include("lib/model.jl");
export  predict, update!, forward_ode

include("lib/error.jl");
export  Additive, Proportional, Combined, Custom, ErrorModel, init_sigma, 
        variance

include("lib/objectives.jl");
export  SSE, LogLikelihood, VariationalELBO, MeanField, FullRank, MonteCarlo, 
        SampleAverage, objective, init_params, init_params!, init_phi, 
        init_samples!, adapt!, fit!
        
include("lib/constrain.jl");
export  constrain, constrain_phi

include("lib/nn.jl");
export  StandardNeuralNetwork, SNN

include("lib/dcm.jl");
export  DeepCompartmentModel, DCM

include("lib/hybrid_dcm.jl");
export  HybridDCM

include("lib/basic_hybrid_model.jl");
export  BasicHybridModel

include("lib/low_dim_node.jl");
export  LowDimensionalNODE

include("lib/latent_encoder_decoder.jl");
export  LatentEncoderDecoder

include("lib/callbacks.jl");
export  generate_dosing_callback

include("lib/lux.helpers.jl");
export  Normalize, AddGlobalParameters, Combine, SingleHeadedBranch, 
        MultiHeadedBranch, make_branch, interpret_branch

include("lib/compartment_models.jl");
export  unpack, one_comp!, one_comp_abs!, two_comp!, two_comp_abs!

end