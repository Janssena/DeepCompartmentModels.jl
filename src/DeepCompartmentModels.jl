module DeepCompartmentModels

__precompile__()

import DifferentialEquations: DiscreteCallback

include("./lib/population.jl");
export Population, Individual

include("./lib/model.jl");
export DCM, predict, fit!, mse, monitor_loss

include("./lib/dataset.jl");
export load, normalize, normalize_inv, normalize⁻¹, create_split

include("./lib/compartment_models.jl");
export one_comp!, two_comp!

end
