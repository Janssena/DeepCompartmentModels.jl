const PACKAGE_API_GROUPS = (
    core = (:AbstractModel, :AbstractDEModel, :Parameterisation, :MeanVar, :MeanSqrt),
    compartments = (:unpack, :one_comp!, :one_comp_abs!, :two_comp!, :two_comp_abs!),
    data = (:AbstractIndividual, :BasicIndividual, :TimeVariableIndividual,
            :MOIndividual, :Individual, :Population, :get_x, :get_t, :get_tx,
            :get_y, :load),
    errors = (:AbstractErrorModel, :AdditiveError, :ProportionalError,
              :CombinedError, :CustomError, :ErrorModelSet, :make_dist, :var),
    models = (:DeepCompartmentModel, :DCM, :StandardNeuralNetwork, :SNN,
              :UniversalDiffEq, :AbstractUDEType, :BasicUDE, :TimeConcatUDE,
              :build_problem, :predict_typ_parameters, :predict_de_parameters,
              :predict),
    solving = (:solve, :solve_for_target, :construct_p, :generate_dosing_callback),
    random_effects = (:get_random_effects, :sample_gaussian, :update_epsilon!),
    objectives = (:MSE, :SSE, :LogLikelihood, :VariationalELBO, :mse, :sse,
                  :loglikelihood, :kldivergence, :logprior, :logq, :getq,
                  :logjoint, :elbo),
    fitting_primitives = (:setup, :setup_phi, :gradient, :create_batches,
                          :take_batch, :residual_error_value_and_gradient,
                          :m_step, :optimise_omega, :optimise_residual_error),
    lux_helpers = (:Normalize, :AddGlobalParameters, :Combine,
                   :SingleHeadedBranch, :MultiHeadedBranch, :make_branch,
                   :interpret_branch),
)

@testset "package-owned exports resolve" begin
    public_names = Set(names(DeepCompartmentModels; all = false, imported = false))
    for (area, symbols) in pairs(PACKAGE_API_GROUPS)
        @testset "$area" begin
            for symbol in symbols
                @test symbol in public_names
                @test isdefined(DeepCompartmentModels, symbol)
            end
        end
    end
end

@testset "dormant APIs are not advertised" begin
    public_names = Set(names(DeepCompartmentModels; all = false, imported = false))
    @test :make_etas ∉ public_names
    @test :NeuralODE ∉ public_names
    @test :LowDimNODE ∉ public_names
    @test :AutoEncodingNODE ∉ public_names

    # `fit` leaks into the namespace through a broad dependency re-export. The
    # obsolete DCM implementation in lib/optimization.jl is not loaded.
    @test :fit in public_names
    @test !any(method -> method.module === DeepCompartmentModels,
               methods(DeepCompartmentModels.fit))
end
