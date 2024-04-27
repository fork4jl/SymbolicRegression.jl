module ConstantOptimizationModule

using LineSearches: LineSearches
using Optim: Optim
using DynamicExpressions: Node, count_constants, get_constant_refs
using ..CoreModule: Options, Dataset, DATA_TYPE, LOSS_TYPE
using ..UtilsModule: get_birth_order
using ..LossFunctionsModule: eval_loss, loss_to_score, batch_sample
using ..PopMemberModule: PopMember

function optimize_constants(
    dataset::Dataset{T,L}, member::P, options::Options
)::Tuple{P,Float64} where {T<:DATA_TYPE,L<:LOSS_TYPE,P<:PopMember{T,L}}
    dispatch_optimize_constants(dataset, member, options, nothing)
end

function dispatch_optimize_constants(
    dataset::Dataset{T,L}, member::P, options::Options, idx
)::Tuple{P,Float64} where {T<:DATA_TYPE,L<:LOSS_TYPE,P<:PopMember{T,L}}
    nconst = count_constants(member.tree)
    nconst == 0 && return (member, 0.0)

    _optimize_constants(
        dataset,
        member,
        options,
        options.optimizer_algorithm,
        options.optimizer_options,
        idx,
    )
end

function _optimize_constants(
    dataset, member::P, options, algorithm, optimizer_options, idx
)::Tuple{P,Float64} where {T,L,P<:PopMember{T,L}}
    tree = member.tree
    f(t) = eval_loss(t, dataset, options; regularization=false, idx=idx)::L
    result = Optim.optimize(f, tree, algorithm, optimizer_options)

    return member, 0.0
end

end
