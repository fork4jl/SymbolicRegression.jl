module ConstantOptimizationModule


using Optim: Optim
using DynamicExpressions: count_constants
using ..CoreModule: Options, Dataset, DATA_TYPE, LOSS_TYPE
using ..PopMemberModule: PopMember

function optimize_constants(
    dataset::Dataset{T,L}, member::P, options::Options
)::Tuple{P,Float64} where {T<:DATA_TYPE,L<:LOSS_TYPE,P<:PopMember{T,L}}
    _dispatch_optimize_constants(dataset, member, options)
end

function _dispatch_optimize_constants(
    dataset::Dataset{T,L}, member::P, options::Options
)::Tuple{P,Float64} where {T<:DATA_TYPE,L<:LOSS_TYPE,P<:PopMember{T,L}}
    nconst = count_constants(member.tree)
    nconst == 0 && return (member, 0.0)

    tree = member.tree
    f(t) = 0.0

    Optim.optimize(f, tree, options.optimizer_algorithm, options.optimizer_options)

    return member, 0.0
end

end
