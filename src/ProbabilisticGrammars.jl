module ProbabilisticGrammars

import Base: show, length, getindex, iterate, eltype, insert!

export default, OneOrTwo, One, Two
export Tree, label, children, labels, innerlabels, leaflabels
export Rule, -->, ⋅, lhs, rhs
export derivation2tree, tree2derivation
export CNFG, chartparse, push_rules!
export CountScoring

#########
# Utils #
#########

default(::Type{T}) where {T <: Number} = zero(T)
default(::Type{Char}) = '0'

struct OneOrTwo{T}
    length :: Int
    fst    :: T
    snd    :: T

    OneOrTwo(fst::T) where T = new{T}(1, fst, default(T))

    function OneOrTwo(fst, snd) 
        fst_, snd_ = promote(fst, snd)
        new{typeof(fst_)}(2, fst_, snd_)
    end
end

One(x) = OneOrTwo(x)
Two(x1, x2) = OneOrTwo(x1, x2)

length(xs::OneOrTwo) = xs.length
eltype(::Type{<:OneOrTwo{T}}) where T = T 

function getindex(xs::OneOrTwo, i)
    if i == 1
        xs.fst
    elseif i == 2 && length(xs) == 2
        xs.snd
    else
        throw(BoundsError(xs, i))
    end
end

function iterate(xs::OneOrTwo, i=0)
    if i < length(xs)
        (xs[i+1], i+1)
    end
end

#########
# Trees #
#########

struct Tree{T}
    label :: T
    children :: Vector{Tree{T}}
end

Tree(label::T) where T = Tree(label, Tree{T}[])
Tree(label, children::Tree...) = Tree(label, collect(children))
label(tree::Tree) = tree.label
children(tree::Tree) = tree.children
isleaf(tree::Tree) = isempty(tree.children)
eltype(::Type{Tree{T}}) where T = T
labels(tree::Tree) = label.(subtrees(tree))
innerlabels(tree::Tree) = [label(t) for t in subtrees(tree) if !isleaf(t)]
leaflabels(tree::Tree) = [label(t) for t in subtrees(tree) if isleaf(t)]

function subtrees(tree::Tree{T}, out=Tree{T}[]) where T
    push!(out, tree)
    if !isleaf(tree)
        for child in children(tree)
            subtrees(child, out)
        end
    end
    return out
end

function map(f, tree::Tree)
    if isleaf(tree)
        Tree(f(tree.label))
    else
        Tree(f(tree.label), [map(f, c) for c in tree.children])
    end
end

function dict2tree(f, dict; label_key="label", children_key="children")
    label = f(dict[label_key])
    T = typeof(label)
    children = Tree{T}[dict2tree(f, child) for child in dict[children_key]]
    Tree(label, children)
end

dict2tree(dict; args...) = dict2tree(identity, dict; args...)

function zip_trees(t1, t2)
    @assert length(t1.children) == length(t2.children)
    if isleaf(t1)
        Tree((t1.label, t2.label))
    else
        zipped_children = map(zip_trees, t1.children, t2.children)
        Tree((t1.label, t2.label), zipped_children)
    end
end

#########
# Rules #
#########

const Rhs{T} = OneOrTwo{T}

struct Rule{T}
    lhs :: T
    rhs :: Rhs{T}
end

Rule(lhs, rhs...) = Rule(lhs, OneOrTwo(rhs...))
-->(lhs::T, rhs::T) where T = Rule(lhs, rhs)
-->(lhs, rhs) = Rule(lhs, rhs...)
⋅(x1, x2) = (x1, x2)
eltype(::Type{<:Rule{T}}) where T = T
lhs(r::Rule) = r.lhs
rhs(r::Rule) = r.rhs

function show(io::IO, r::Rule)
    if length(r.rhs) == 1
        print(io, "$(r.lhs) --> $(r.rhs.fst)")
    else
        print(io, "$(r.lhs) --> $(r.rhs.fst) ⋅ $(r.rhs.snd)")
    end
end

function derivation2tree(derivation)
    i = 0
    next_rule()  = (i += 1; derivation[i])
    backtrack()  = (i -= 1)
    more_rules() = (i < length(derivation))

    function rewrite(label)
        r = next_rule()
        if lhs(r) == label
            children = [more_rules() ? rewrite(c) : Tree(c) for c in rhs(r)]
            Tree(label, children)
        else
            backtrack()
            Tree(label)
        end
    end

    rewrite(lhs(first(derivation)))
end

function tree2derivation(tree::Tree{T}) where T
    [label(t) --> label.(children(t)) for t in subtrees(tree) if !isleaf(t)]
end

derivation = ['A' --> 'A'⋅'A', 'A' --> 'a', 'A' --> 'a']
tree = derivation2tree(derivation)


############
# Grammars #
############

"""
    CNFG(start_symbols, rules)

Grammar in Chomsky-normal form.
"""
struct CNFG{T}
    start_symbols :: Set{T}
    lhss          :: Dict{Rhs{T}, Vector{T}} # left-hand sides

    function CNFG(start_symbols, rules)
        T = eltype(eltype(typeof(rules)))
        lhss = Dict{Rhs{T}, Vector{T}}()
        for r in rules
            lhss_r = get!(() -> T[], lhss, rhs(r))
            push!(lhss_r, lhs(r))
        end
        return new{T}(Set(collect(start_symbols)), lhss)
    end
end

"""
    push_rules!(grammar, stack, rhs...)

Push all rules with given right-hand side to stack.
"""
function push_rules!(grammar::CNFG, stack, xs...)
    rhs = OneOrTwo(xs...)
    if haskey(grammar.lhss, rhs)
        for lhs in grammar.lhss[rhs]
            push!(stack, lhs --> rhs)
        end
    end
end

#######################
# Scores and scorings #
#######################

mulscores(scoring, s1, s2, s3) = mulscores(scoring, s1, mulscores(scoring, s2, s3))

struct CountScoring end
scoretype(::CountScoring, grammar) = Int
rulescore(::CountScoring, rule) = 1
addscores(::CountScoring, left, right) = left + right
mulscores(::CountScoring, left, right) = left * right

###########
# Parsing #
###########

const ChartCell{T, S} = Dict{T, S} # map left-hand sides to scores
const Chart{T, S} = Matrix{ChartCell{T, S}}

function empty_chart(::Type{T}, ::Type{S}, n)::Chart{T, S} where {T, S}
    [ Dict{T, S}() for _ in 1:n, _ in 1:n ]
end

function insert!(cell::ChartCell, scoring, lhs, s::S) where S    
    if haskey(cell, lhs)
        cell[lhs] = addscores(scoring, cell[lhs], s)
    else
        cell[lhs] = s
    end
end

"""
    chartparse(grammar, scoring, sequence)
"""
function chartparse(grammar, scoring, sequence)
    chartparse(grammar, scoring, [[x] for x in sequence])
end

function chartparse(grammar, scoring, sequence::Vector{Vector{T}}) where T
    n = length(sequence)
    S = scoretype(scoring, grammar)
    chart = empty_chart(T, S, n)
    stack = Vector{Rule{T}}() # channel for communicating completions
    # using a single stack is much more efficient than constructing multiple arrays
    stack_unary = Vector{Tuple{T, S}}()

    score(rule) = rulescore(scoring, rule)

    # terminal completions
    for (i, terminals) in enumerate(sequence)
        for terminal in terminals
            push_rules!(grammar, stack, terminal)
        end
        while !isempty(stack)
            rule = pop!(stack)
            insert!(chart[i, i], scoring, lhs(rule), score(rule))
        end
    end

    for l in 1:n - 1         # length
        for i in 1:n - l     # start index
            j = i + l        # end index

            # binary completions
            for k in i:j - 1 # split index
                for (rhs1, s1) in chart[i, k]
                    for (rhs2, s2) in chart[k + 1, j]
                        push_rules!(grammar, stack, rhs1, rhs2)
                        while !isempty(stack)
                            rule = pop!(stack)
                            s = mulscores(scoring, score(rule), s1, s2)
                            insert!(chart[i, j], scoring, lhs(rule), s)
                        end
                    end
                end
            end

            # unary completions
            for (rhs, s) in chart[i, j]
                push_rules!(grammar, stack, rhs)
                while !isempty(stack)
                    rule = pop!(stack)
                    push!(stack_unary, (lhs(rule), mulscores(scoring, score(rule), s)))
                end
            end
            while !isempty(stack_unary)
                (lhs, s) = pop!(stack_unary)
                insert!(chart[i, j], scoring, lhs, s)
            end
        end
    end

    return chart
end




end # module
