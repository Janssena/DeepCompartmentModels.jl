import LinearAlgebra: I, diag
import Zygote
import Random
import Plots
import Optim

using ApproximateGPs
using ParameterHandling
using Distributions

Random.seed!(42);

k_true = [30.0, 1.5]
kernel_true = k_true[1] * (SqExponentialKernel() ∘ ScaleTransform(k_true[2]))

jitter = 1e-8  # for numeric stability
lgp = LatentGP(GP(kernel_true), BernoulliLikelihood(), jitter)
x_true = 0:0.02:6
f_true, y_true = rand(lgp(x_true))

Plots.plot(x_true, f_true; seriescolor="red", label="")  # Plot the sampled function

μ = mean.(lgp.lik.(f_true))
Plots.plot(x_true, μ; seriescolor="red", label="")

N = 30  # The number of training points
mask = sample(1:length(x_true), N; replace=false, ordered=true)  # Subsample some input locations
x, y = x_true[mask], y_true[mask]

Plots.scatter(x, y; label="Sampled outputs")
Plots.plot!(x_true, mean.(lgp.lik.(f_true)); seriescolor="red", label="True mean")

M = 15  # number of inducing points
raw_initial_params = (
    k=(var=positive(rand()), precision=positive(rand())),
    z=bounded.(range(0.1, 5.9; length=M), 0.0, 6.0),  # constrain z to simplify optimisation
    m=zeros(M),
    A=positive_definite(Matrix{Float64}(I, M, M)),
);

flat_init_params, unflatten = ParameterHandling.flatten(raw_initial_params)
unpack = ParameterHandling.value ∘ unflatten;

lik = BernoulliLikelihood()
jitter = 1e-3  # added to aid numerical stability

function build_SVGP(params::NamedTuple)
    kernel = params.k.var * (SqExponentialKernel() ∘ ScaleTransform(params.k.precision))
    f = LatentGP(GP(kernel), lik, jitter)
    q = MvNormal(params.m, params.A)
    fz = f(params.z).fx
    return SparseVariationalApproximation(fz, q), f
end

function loss(params::NamedTuple)
    svgp, f = build_SVGP(params)
    fx = f(x)
    return -elbo(svgp, fx, y; 
        # quadrature=ApproximateGPs.GPLikelihoods.MonteCarloExpectation(100)
        quadrature=ApproximateGPs.GPLikelihoods.DefaultExpectationMethod()
    )
end;


Zygote.gradient(loss ∘ unpack, flat_init_params)

opt = Optim.optimize(
    loss ∘ unpack,
    θ -> only(Zygote.gradient(loss ∘ unpack, θ)),
    flat_init_params,
    Optim.LBFGS(;
        alphaguess=Optim.LineSearches.InitialStatic(; scaled=true),
        linesearch=Optim.LineSearches.BackTracking(),
    ),
    Optim.Options(; iterations=4_000);
    inplace=false,
)

final_params = unpack(opt.minimizer)

svgp_opt, f_opt = build_SVGP(final_params)
post_opt = posterior(svgp_opt)
l_post_opt = LatentGP(post_opt, BernoulliLikelihood(), jitter)

post_f_samples = rand(l_post_opt.f(x_true, 1e-6), 100)
post_μ_samples = mean.(l_post_opt.lik.(post_f_samples))

Plots.plot(0:0.1:6, post_opt)
Plots.scatter!(final_params.z, final_params.m, yerr=sqrt.(diag(final_params.A)), label="Inducing points", ylim=(-22, 22))

plt = Plots.plot(x_true, post_μ_samples; seriescolor="red", linealpha=0.2, label="")
Plots.scatter!(plt, x, y; seriescolor="blue", label="Data points")
Plots.plot!(x_true, mean.(lgp.lik.(f_true)); seriescolor="green", linewidth=3, label="True function")



# MonteCarloExpectation is not as good in this setting.