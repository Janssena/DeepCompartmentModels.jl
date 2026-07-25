const PACKAGE_API_GROUPS = (
    core = (:AbstractModel, :AbstractDEModel, :Parameterisation, :MeanVar, :MeanSqrt),
    compartments = (:unpack, :one_comp!, :one_comp_abs!, :two_comp!, :two_comp_abs!),
    data = (:AbstractIndividual, :BasicIndividual, :TimeVariableIndividual,
            :MOIndividual, :Individual, :Population, :get_x, :get_t, :get_tx,
            :get_y, :load),
    errors = (:AbstractErrorModel, :AdditiveError, :ProportionalError,
              :CombinedError, :CustomError, :ErrorModelSet, :make_dist, :var),
    structural = (:AbstractParameterTransform, :IdentityTransform, :LogTransform,
                  :LogitTransform, :StructuralParameter, :StructuralParameters,
                  :parameter_names, :parameter_units, :parameter_transforms,
                  :parameter_bounds,
                  :natural_parameters, :unconstrained_parameters),
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
                          :m_step, :optimise_omega, :optimise_residual_error,
                          :fit),
    fit_results = (:FitResult, :isconverged, :niterations,
                   :objective_history, :fit_status, :coef, :coefnames,
                   :coefunits, :empirical_bayes),
    uncertainty = (:uncertainty, :FixedEffectUncertainty, :MixedEffectUncertainty,
                   :vcov, :stderror, :confint),
    lux_helpers = (:Normalize, :InitialScale, :AddGlobalParameters, :Combine,
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

    # `fit` is the StatsAPI generic re-exported through Distributions. The
    # package owns explicit fixed-effect and VariationalELBO methods.
    @test :fit in public_names
    @test any(method -> method.module === DeepCompartmentModels,
              methods(DeepCompartmentModels.fit))
    @test applicable(
        fit, VariationalELBO([1]), toy_dcm(; error = AdditiveError(0.1f0)),
        toy_population(; n = 1), Optimisers.Adam())
end
