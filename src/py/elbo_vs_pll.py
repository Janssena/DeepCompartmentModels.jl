import matplotlib.pyplot as plt
import gpytorch
import torch
import math

class ApproximateGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(-1))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def train_and_test_approximate_gp(objective_function_cls):
    model = ApproximateGPModel(torch.linspace(0, 1, 100))
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    objective_function = objective_function_cls(likelihood, model, num_data=train_y.numel())
    optimizer = torch.optim.Adam(list(model.parameters()) + list(likelihood.parameters()), lr=0.1)

    # Train
    model.train()
    likelihood.train()
    for _ in range(training_iterations):
        output = model(train_x)
        loss = -objective_function(output, train_y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Test
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        f_dist = model(train_x)
        mean = f_dist.mean
        f_lower, f_upper = f_dist.confidence_region()
        y_dist = likelihood(f_dist)
        y_lower, y_upper = y_dist.confidence_region()

    # Plot model
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    line, = ax.plot(train_x, mean, "blue")
    ax.fill_between(train_x, f_lower, f_upper, color=line.get_color(), alpha=0.3, label="q(f)")
    ax.fill_between(train_x, y_lower, y_upper, color=line.get_color(), alpha=0.1, label="p(y)")
    ax.scatter(train_x, train_y, c='k', marker='.', label="Data")
    ax.legend(loc="best")
    ax.set(xlabel="x", ylabel="y")
    plt.show()

# Define some training data
train_x = torch.linspace(0, 1, 100)
train_y = torch.cos(train_x * 2 * math.pi) + torch.randn(100).mul(train_x.pow(3) * 1.)

fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.scatter(train_x, train_y, c='k', marker='.', label="Data")
ax.set(xlabel="x", ylabel="y")
plt.show()

training_iterations = 50

train_and_test_approximate_gp(gpytorch.mlls.VariationalELBO)

train_and_test_approximate_gp(gpytorch.mlls.PredictiveLogLikelihood)

# Matches docs!
model = ApproximateGPModel(torch.linspace(0, 1, 100))
likelihood = gpytorch.likelihoods.GaussianLikelihood()
objective_function = gpytorch.mlls.PredictiveLogLikelihood(likelihood, model, num_data=train_y.numel())
optimizer = torch.optim.Adam(list(model.parameters()) + list(likelihood.parameters()), lr=0.1)

model.train()
likelihood.train()
output = model(train_x) # Gets posterior marginals?

gpytorch.mlls.PredictiveLogLikelihood(likelihood, model, num_data=train_y.numel())._log_likelihood_term(output, train_y)
mean, covar = output.mean, output.lazy_covariance_matrix
noise_covar = likelihood._shaped_noise_covar(mean.shape)
full_covar = covar + noise_covar
marginal = output.__class__(mean, full_covar)
indep_dist = torch.distributions.Normal(marginal.mean, marginal.variance.clamp_min(1e-8).sqrt())
res = indep_dist.log_prob(train_y)
sum(res)
# equals
res2 = ((train_y - indep_dist.mean).square()) / indep_dist.variance + indep_dist.variance.log() + math.log(2 * math.pi)
res2 = res2.mul(-0.5)
sum(res2)

gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.numel())._log_likelihood_term(output, train_y)
noise = likelihood._shaped_noise_covar(output.mean.shape).diagonal(dim1=-1, dim2=-2)
noise = noise.view(*noise.shape[:-1], *output.event_shape)
mean, variance = output.mean, output.variance
res = ((train_y - mean).square() + variance) / noise + noise.log() + math.log(2 * math.pi)
res = res.mul(-0.5)
sum(res)
# This takes the marginals (?) and runs them through a multivariate normal distribution

# So what is happening here is that for GaussianDistributions, the GP variance + noise estimate
# is used as the likelihood function variance. In the variational ELBO above
# the GP variance is added to the residual error (y - y_hat)^2 + GP_var and the 
# noise estimate is used instead.

# We should implement the way GPytorch does this for any likelihood function 
# when we want to directly predict y and GP parameters (instead of first 
# estimating PK parameters and then running the GP on PK parameter estimates):

def _expected_log_prob(y, f_dist):
    likelihood_samples = _draw_likelihood_samples(f_dist)
    res = likelihood_samples.log_prob(y).mean(dim=0)
    return res

def _draw_likelihood_samples(f_dist, num_monte_carlo=10):
        # if self.training:
        num_event_dims = len(f_dist.event_shape)
        f_dist = torch.distributions.Normal(f_dist.mean, f_dist.variance.sqrt())
        f_dist = torch.distributions.Independent(f_dist, num_event_dims - 1)
        function_samples = f_dist.rsample(torch.Size(([num_monte_carlo])))

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        conditional = likelihood(function_samples)
        return conditional

sum(_draw_likelihood_samples(output, num_monte_carlo=10_000).log_prob(train_y).mean(dim=0))
dist = torch.distributions.Normal(output.mean, output.variance.sqrt())

function_samples = torch.distributions.Independent(dist, 0).rsample(torch.Size(([10_000])))
noises = likelihood._shaped_noise_covar(function_samples.shape).diagonal(dim1=-1, dim2=-2)
dist_fs = torch.distributions.Normal(function_samples, noises.sqrt())
dist_fs.log_prob(train_y).sum(1).mean()

sum(dist.log_prob(train_y))
sum(torch.distributions.Independent(dist, 0).log_prob(train_y))



def _log_marginal(y, f_dist):
        likelihood_samples = _draw_likelihood_samples(f_dist)
        log_probs = likelihood_samples.log_prob(y)
        res = log_probs.sub(math.log(log_probs.size(0))).logsumexp(dim=0)

        # log(sum(exp.(log_prob - log.(N)))) # N = num monte carlo samples
        return res

sum(_expected_log_prob(train_y, output))
sum(_log_marginal(train_y, output))

# Then we should figure out what model(train_x) outputs. This is a full-rank 
# MvNormal distribution: How do we get this from AbstractGPs?

