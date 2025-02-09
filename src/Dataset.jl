module DatasetModule

using DynamicQuantities:
    AbstractDimensions,
    Dimensions,
    SymbolicDimensions,
    Quantity,
    uparse,
    sym_uparse,
    DEFAULT_DIM_BASE_TYPE

using ..UtilsModule: subscriptify, get_base_type
using ..ProgramConstantsModule: BATCH_DIM, FEATURE_DIM, DATA_TYPE, LOSS_TYPE
using ...InterfaceDynamicQuantitiesModule: get_si_units, get_sym_units


mutable struct Dataset{
    T<:DATA_TYPE,
    L<:LOSS_TYPE,
    AX<:AbstractMatrix{T},
    AY<:Union{AbstractVector{T},Nothing},
    AW<:Union{AbstractVector{T},Nothing},
    NT<:NamedTuple,
    XU<:Union{AbstractVector{<:Quantity},Nothing},
    YU<:Union{Quantity,Nothing},
    XUS<:Union{AbstractVector{<:Quantity},Nothing},
    YUS<:Union{Quantity,Nothing},
}
    X::AX
    y::AY
    n::Int
    nfeatures::Int
    weighted::Bool
    weights::AW
    extra::NT
    avg_y::Union{T,Nothing}
    use_baseline::Bool
    baseline_loss::L
    variable_names::Array{String,1}
    display_variable_names::Array{String,1}
    y_variable_name::String
    X_units::XU
    y_units::YU
    X_sym_units::XUS
    y_sym_units::YUS
end

function Dataset(
    X::AbstractMatrix{T},
    y::Union{AbstractVector{T},Nothing}=nothing;
    loss_type::Type{L}=Nothing,
    weights::Union{AbstractVector{T},Nothing}=nothing,
    variable_names::Union{Array{String,1},Nothing}=nothing,
    display_variable_names=variable_names,
    y_variable_name::Union{String,Nothing}=nothing,
    extra::NamedTuple=NamedTuple(),
    X_units::Union{AbstractVector,Nothing}=nothing,
    y_units=nothing,
    # Deprecated:
    varMap=nothing,
    kws...,
) where {T<:DATA_TYPE,L}
    Base.require_one_based_indexing(X)
    y !== nothing && Base.require_one_based_indexing(y)
    # Deprecation warning:
    variable_names = variable_names

    n = size(X, BATCH_DIM)
    nfeatures = size(X, FEATURE_DIM)
    weighted = weights !== nothing
    variable_names = if variable_names === nothing
        ["x$(i)" for i in 1:nfeatures]
    else
        variable_names
    end
    display_variable_names = if display_variable_names === nothing
        ["x$(subscriptify(i))" for i in 1:nfeatures]
    else
        display_variable_names
    end

    y_variable_name = if y_variable_name === nothing
        ("y" ∉ variable_names) ? "y" : "target"
    else
        y_variable_name
    end
    avg_y = if y === nothing
        nothing
    else
        if weighted
            sum(y .* weights) / sum(weights)
        else
            sum(y) / n
        end
    end
    out_loss_type = if L === Nothing
        T <: Complex ? get_base_type(T) : T
    else
        L
    end

    use_baseline = true
    baseline = one(out_loss_type)
    y_si_units = get_si_units(T, y_units)
    y_sym_units = get_sym_units(T, y_units)

    # TODO: Refactor
    # This basically just ensures that if the `y` units are set,
    # then the `X` units are set as well.
    X_si_units = let (_X = get_si_units(T, X_units))
        if _X === nothing && y_si_units !== nothing
            get_si_units(T, [one(T) for _ in 1:nfeatures])
        else
            _X
        end
    end
    X_sym_units = let _X = get_sym_units(T, X_units)
        if _X === nothing && y_sym_units !== nothing
            get_sym_units(T, [one(T) for _ in 1:nfeatures])
        else
            _X
        end
    end

    error_on_mismatched_size(nfeatures, X_si_units)

    return Dataset{
        T,
        out_loss_type,
        typeof(X),
        typeof(y),
        typeof(weights),
        typeof(extra),
        typeof(X_si_units),
        typeof(y_si_units),
        typeof(X_sym_units),
        typeof(y_sym_units),
    }(
        X,
        y,
        n,
        nfeatures,
        weighted,
        weights,
        extra,
        avg_y,
        use_baseline,
        baseline,
        variable_names,
        display_variable_names,
        y_variable_name,
        X_si_units,
        y_si_units,
        X_sym_units,
        y_sym_units,
    )
end
function Dataset(
    X::AbstractMatrix,
    y::Union{<:AbstractVector,Nothing}=nothing;
    weights::Union{<:AbstractVector,Nothing}=nothing,
    kws...,
)
    T = promote_type(
        eltype(X),
        (y === nothing) ? eltype(X) : eltype(y),
        (weights === nothing) ? eltype(X) : eltype(weights),
    )
    X = Base.Fix1(convert, T).(X)
    if y !== nothing
        y = Base.Fix1(convert, T).(y)
    end
    if weights !== nothing
        weights = Base.Fix1(convert, T).(weights)
    end
    return Dataset(X, y; weights=weights, kws...)
end

function error_on_mismatched_size(_, ::Nothing)
    return nothing
end
function error_on_mismatched_size(nfeatures, X_units::AbstractVector)
    if nfeatures != length(X_units)
        error(
            "Number of features ($(nfeatures)) does not match number of units ($(length(X_units)))",
        )
    end
    return nothing
end

function has_units(dataset::Dataset)
    return dataset.X_units !== nothing || dataset.y_units !== nothing
end

end
