abstract type AbstractParameterTransform end

"""Unconstrained identity transform with bounds `(-Inf, Inf)`."""
struct IdentityTransform <: AbstractParameterTransform end

"""Positive log transform with natural-scale bounds `(0, Inf)`."""
struct LogTransform <: AbstractParameterTransform end

"""Logit transform from the real line to the open interval `(lower, upper)`."""
struct LogitTransform{T<:AbstractFloat} <: AbstractParameterTransform
    lower::T
    upper::T

    function LogitTransform(lower::Real, upper::Real)
        lo, hi = promote(float(lower), float(upper))
        isfinite(lo) && isfinite(hi) || throw(ArgumentError(
            "LogitTransform bounds must be finite."))
        lo < hi || throw(ArgumentError(
            "LogitTransform lower bound must be smaller than its upper bound."))
        return new{typeof(lo)}(lo, hi)
    end
end

_parameter_bounds(::IdentityTransform, ::Type{T}) where {T} = (T(-Inf), T(Inf))
_parameter_bounds(::LogTransform, ::Type{T}) where {T} = (zero(T), T(Inf))
_parameter_bounds(t::LogitTransform, ::Type{T}) where {T} =
    (T(t.lower), T(t.upper))

_to_natural(::IdentityTransform, value) = value
_to_natural(::LogTransform, value) = exp(value)
_to_natural(t::LogitTransform, value) =
    t.lower + (t.upper - t.lower) * logistic(value)

_to_unconstrained(::IdentityTransform, value) = value
function _to_unconstrained(::LogTransform, value)
    value > zero(value) || throw(DomainError(
        value, "a LogTransform parameter must be strictly positive."))
    return log(value)
end
function _to_unconstrained(t::LogitTransform, value)
    t.lower < value < t.upper || throw(DomainError(
        value, "a LogitTransform parameter must lie strictly inside " *
               "($(t.lower), $(t.upper))."))
    proportion = (value - t.lower) / (t.upper - t.lower)
    return log(proportion) - log1p(-proportion)
end

"""
    StructuralParameter(name, initial; unit=nothing, transform=LogTransform())

Specification for one conventional structural-model parameter. `initial` is
always supplied on the natural/reporting scale. `transform` defines both the
optimizer-scale mapping and the open natural-scale bounds:

- `IdentityTransform()` for an unbounded parameter;
- `LogTransform()` for a strictly positive parameter;
- `LogitTransform(lower, upper)` for a parameter inside finite bounds.

`unit` is descriptive metadata and is never used to rescale a value silently.
"""
struct StructuralParameter{T<:AbstractFloat,U,TR<:AbstractParameterTransform}
    name::Symbol
    initial::T
    unit::U
    transform::TR
end

function StructuralParameter(
    name::Symbol,
    initial::Real;
    unit::Union{Nothing,AbstractString}=nothing,
    transform::AbstractParameterTransform=LogTransform(),
)
    isempty(string(name)) && throw(ArgumentError("parameter name cannot be empty."))
    value = float(initial)
    isfinite(value) || throw(ArgumentError(
        "initial value for $name must be finite."))
    # Validate the natural-scale domain at construction time.
    _to_unconstrained(transform, value)
    stored_unit = unit === nothing ? nothing : String(unit)
    return StructuralParameter(name, value, stored_unit, transform)
end

"""
    StructuralParameters(parameters...)

A no-covariate, non-neural Lux layer for conventional structural parameters.
It returns parameters in declared order on their natural scale and broadcasts
the same typical values across subjects. Optimizer parameters are stored in
`ps.unconstrained`; names, units, transformations and bounds remain in the layer.

# Example

```julia
layer = StructuralParameters(
    StructuralParameter(:ka, 1.5; unit="1/h"),
    StructuralParameter(:CL, 3.0; unit="L/h"),
    StructuralParameter(:V, 30.0; unit="L"),
)
model = DCM(one_comp_abs!, layer; target=2)
```

`StructuralParameters([1.5, 3.0, 30.0])` remains available as a concise
positive-parameter constructor, with generated names `θ1`, `θ2`, ... and
unspecified units. Explicit specifications are required for meaningful reports.
"""
struct StructuralParameters{P<:Tuple} <: Lux.AbstractLuxLayer
    parameters::P

    function StructuralParameters(parameters::P) where {P<:Tuple}
        isempty(parameters) && throw(ArgumentError(
            "StructuralParameters requires at least one parameter."))
        all(parameter -> parameter isa StructuralParameter, parameters) ||
            throw(ArgumentError(
                "every StructuralParameters entry must be a StructuralParameter."))
        names = map(parameter -> parameter.name, parameters)
        allunique(names) || throw(ArgumentError(
            "structural parameter names must be unique, got $names."))
        return new{P}(parameters)
    end
end

StructuralParameters(parameters::StructuralParameter...) =
    StructuralParameters(parameters)

function StructuralParameters(initial::AbstractVector{<:Real})
    parameters = map(eachindex(initial)) do index
        StructuralParameter(Symbol("θ", index), initial[index])
    end
    return StructuralParameters(Tuple(parameters))
end

function StructuralParameters(
    names::AbstractVector{Symbol},
    initial::AbstractVector{<:Real};
    units=fill(nothing, length(names)),
    transforms=fill(LogTransform(), length(names)),
)
    length(names) == length(initial) == length(units) == length(transforms) ||
        throw(DimensionMismatch(
            "names, initial values, units and transforms must have equal length."))
    parameters = map(eachindex(names)) do index
        StructuralParameter(
            names[index], initial[index]; unit=units[index],
            transform=transforms[index])
    end
    return StructuralParameters(Tuple(parameters))
end

parameter_names(layer::StructuralParameters) =
    Symbol[parameter.name for parameter in layer.parameters]
parameter_units(layer::StructuralParameters) =
    Union{Nothing,String}[parameter.unit for parameter in layer.parameters]
parameter_transforms(layer::StructuralParameters) =
    AbstractParameterTransform[parameter.transform for parameter in layer.parameters]
parameter_bounds(layer::StructuralParameters) = [
    _parameter_bounds(parameter.transform, typeof(parameter.initial))
    for parameter in layer.parameters
]

function unconstrained_parameters(
    layer::StructuralParameters,
    natural::AbstractVector{<:Real},
)
    length(natural) == length(layer.parameters) || throw(DimensionMismatch(
        "received $(length(natural)) values for $(length(layer.parameters)) " *
        "structural parameters."))
    return collect(map(layer.parameters, natural) do parameter, value
        _to_unconstrained(parameter.transform, value)
    end)
end

function natural_parameters(
    layer::StructuralParameters,
    unconstrained::AbstractVector{<:Real},
)
    length(unconstrained) == length(layer.parameters) || throw(DimensionMismatch(
        "received $(length(unconstrained)) optimizer values for " *
        "$(length(layer.parameters)) structural parameters."))
    return collect(map(layer.parameters, unconstrained) do parameter, value
        _to_natural(parameter.transform, value)
    end)
end

function natural_parameters(layer::StructuralParameters, ps::NamedTuple)
    haskey(ps, :unconstrained) || throw(ArgumentError(
        "StructuralParameters expects optimizer parameters named `unconstrained`."))
    return natural_parameters(layer, ps.unconstrained)
end

function Lux.initialparameters(::Random.AbstractRNG, layer::StructuralParameters)
    initial = [parameter.initial for parameter in layer.parameters]
    return (unconstrained=unconstrained_parameters(layer, initial),)
end
Lux.initialstates(::Random.AbstractRNG, ::StructuralParameters) = NamedTuple()
Lux.parameterlength(layer::StructuralParameters) = length(layer.parameters)
Lux.statelength(::StructuralParameters) = 0

function (layer::StructuralParameters)(x::AbstractMatrix, ps, st)
    values = natural_parameters(layer, ps)
    return repeat(values, 1, size(x, 2)), st
end
(layer::StructuralParameters)(::AbstractVector, ps, st) =
    (natural_parameters(layer, ps), st)

function Base.show(io::IO, layer::StructuralParameters)
    print(io, "StructuralParameters(", join(parameter_names(layer), ", "), ")")
end
