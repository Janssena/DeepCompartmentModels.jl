# Tutorials

The tutorials are [Quarto](https://quarto.org) notebooks in the
[`tutorials/`](https://github.com/Janssena/DeepCompartmentModels.jl/tree/main/tutorials) folder of the
repository. When this site is built, `docs/make.jl` converts each notebook into a page, and Documenter executes every
Julia cell. The code you see here is therefore the code that was run to produce the output and figures next to it.
The same notebooks can also be rendered on their own with `quarto render`.

| tutorial | what it covers |
|:--|:--|
| [A first Deep Compartment Model](../generated/01-basic-dcm.md) | the whole loop on theophylline data: individuals, network, model, `SSE`, training, predictions |
| [Loading data and dosing events](../generated/05-loading-data.md) | dosing callbacks, bolus and infusion doses, building populations from warfarin and tobramycin tables |
| [Mixed effects with the variational ELBO](../generated/02-mixed-effects.md) | random effects, warm starts, the M-step, typical versus individual predictions |
| [Objectives, error models and parameter structure](../generated/07-objectives.md) | `SSE`, `MSE`, `LogLikelihood`, `VariationalELBO`, error models and the layout of `ps` and `st` |
| [Writing your own ODE model](../generated/06-custom-ode.md) | the conventions for custom ODE functions, with a transit-compartment model |
| [Interpretable covariate effects](../generated/04-covariate-effects.md) | branch networks and reading covariate effects |

## Running them yourself

The notebooks read the datasets in the repository's `data/` folder (theophylline, warfarin and tobramycin).
To run one in Julia without Quarto, start Julia in the `docs` environment and run the cells in order, or use Quarto:

```bash
quarto render tutorials/01-basic-dcm.qmd
```

Quarto needs a Jupyter kernel for Julia (`using IJulia; IJulia.installkernel("Julia")`).

!!! note
    Parts of the interface do not work yet, so several topics, such as universal differential equations,
    PK/PD with several outputs and occasions, have no tutorial. See [Status and limitations](../limitations.md).
