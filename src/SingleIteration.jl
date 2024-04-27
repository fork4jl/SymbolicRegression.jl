module SingleIterationModule

using DynamicExpressions: AbstractExpressionNode
using ..CoreModule: Options, Dataset, RecordType
using ..PopulationModule: Population
using ..HallOfFameModule: HallOfFame

using ..ConstantOptimizationModule: optimize_constants


function s_r_cycle(
    dataset::D,
    pop::P,
    ncycles,
    curmaxsize,
    running_search_statistics;
    verbosity,
    options::Options,
    record,
)::Tuple{
    P,HallOfFame{T,L,N},Float64
} where {T,L,D<:Dataset{T,L},N<:AbstractExpressionNode{T},P<:Population{T,L,N}}
    best_examples_seen = HallOfFame(options, T, L)

    return (pop, best_examples_seen, 0.0)
end

function optimize_and_simplify_population(
    dataset::D, pop::P, options::Options, curmaxsize, record
)::Tuple{P,Float64} where {T,L,D<:Dataset{T,L},P<:Population{T,L}}
    j = pop.n

    if options.should_optimize_constants
        pop.members[j], _ = optimize_constants(
            dataset, pop.members[j], options
        )
    end

    return (pop, 0.0)
end

end
