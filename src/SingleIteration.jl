module SingleIterationModule

using DynamicExpressions:
    AbstractExpressionNode,
    Node,
    constructorof,
    string_tree,
    simplify_tree!,
    combine_operators
using ..CoreModule: Options, Dataset, RecordType, DATA_TYPE, LOSS_TYPE
using ..ComplexityModule: compute_complexity
using ..PopMemberModule: PopMember, generate_reference
using ..PopulationModule: Population, finalize_scores, best_sub_pop
using ..HallOfFameModule: HallOfFame
using ..AdaptiveParsimonyModule: RunningSearchStatistics
using ..RegularizedEvolutionModule: reg_evol_cycle
using ..LossFunctionsModule: score_func_batched, batch_sample
using ..ConstantOptimizationModule: optimize_constants
using ..RecorderModule: @recorder

# Cycle through regularized evolution many times,
# printing the fittest equation every 10% through
function s_r_cycle(
    dataset::D,
    pop::P,
    ncycles::Int,
    curmaxsize::Int,
    running_search_statistics::RunningSearchStatistics;
    verbosity::Int=0,
    options::Options,
    record::RecordType,
)::Tuple{
    P,HallOfFame{T,L,N},Float64
} where {T,L,D<:Dataset{T,L},N<:AbstractExpressionNode{T},P<:Population{T,L,N}}
    best_examples_seen = HallOfFame(options, T, L)

    return (pop, best_examples_seen, 0.0)
end

function optimize_and_simplify_population(
    dataset::D, pop::P, options::Options, curmaxsize::Int, record::RecordType
)::Tuple{P,Float64} where {T,L,D<:Dataset{T,L},P<:Population{T,L}}
    j = pop.n

    if options.should_optimize_constants
        pop.members[j], array_num_evals[j] = optimize_constants(
            dataset, pop.members[j], options
        )
    end

    return (pop, 0.0)
end

end
