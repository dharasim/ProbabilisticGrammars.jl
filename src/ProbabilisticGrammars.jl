module ProbabilisticGrammars

import Base: show, length, getindex, iterate, eltype, insert!, zero, iszero

using LogProbs: LogProb
using SimpleProbabilisticPrograms: logpdf, DirCat, symdircat

export default, OneOrTwo, One, Two, ⊣
export Tree, label, children, labels, innerlabels, leaflabels
export Rule, -->, ⋅, lhs, rhs
export derivation2tree, tree2derivation
export StdCategory, T, NT, isterminal, isnonterminal
export CNFG, chartparse, push_rules!
export symdircat_ruledist
export CountScoring, InsideScoring, BooleanScoring
export AllDerivationScoring, getallderivations
export WeightedDerivationScoring, sample_derivations

#########
# Utils #
#########

⊣(tag, x) = x.tag == tag

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

######################
# Rule distributions #
######################

struct DirCatRuleDist{T}
    dists :: Dict{T, DirCat{Rule{T}}}
end

(d::DirCatRuleDist)(x) = d.dists[x]

function symdircat_ruledist(xs, rules, concentration=1.0)
    applicable_rules(x) = filter(r -> lhs(r) == x, rules)
    dists = Dict(
        x => symdircat(applicable_rules(x), concentration) 
        for x in xs
    )
    DirCatRuleDist(dists)
end

# struct ConstDirCatRuleDist{T}
#     dist :: DirCat{Rule{T}}
# end
# 
# (d::ConstDirCatRuleDist)(x) = d.dist

####################################
# Standard category implementation #
####################################

struct StdCategory{T}
    isterminal :: Bool
    val        :: T
end

T(val) = StdCategory(true, val)
NT(val) = StdCategory(false, val)

T(c::StdCategory)  = StdCategory(true, c.val)
NT(c::StdCategory) = StdCategory(false, c.val)

isterminal(c::StdCategory) = c.isterminal
isnonterminal(c::StdCategory) = !c.isterminal
default(::Type{StdCategory{T}}) where T = StdCategory(default(Bool), default(T))

function show(io::IO, c::StdCategory)
    if isterminal(c)
        print(io, "T($(c.val))")
    else
        print(io, "NT($(c.val))")
    end
end

############
# Grammars #
############

"""
    CNFG(start_symbols, rules)

Grammar in Chomsky-normal form (CNF).

The user is responsible for all rules being in CNF.
"""
struct CNFG{T}
    lhss :: Dict{Rhs{T}, Vector{T}} # left-hand sides

    function CNFG(rules)
        T = eltype(eltype(typeof(rules)))
        lhss = Dict{Rhs{T}, Vector{T}}()
        for r in rules
            lhss_r = get!(() -> T[], lhss, rhs(r))
            push!(lhss_r, lhs(r))
        end
        return new{T}(lhss)
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

eltype(::CNFG{T}) where T = T
ruletype(grammar) = Rule{eltype(grammar)}

#######################
# Scores and scorings #
#######################

mulscores(scoring, s1, s2, s3) = mulscores(scoring, s1, mulscores(scoring, s2, s3))

struct CountScoring end
scoretype(::CountScoring, grammar) = Int
rulescore(::CountScoring, rule) = 1
addscores(::CountScoring, left, right) = left + right
mulscores(::CountScoring, left, right) = left * right

struct InsideScoring{D} ruledist::D end
scoretype(::InsideScoring, grammar) = LogProb
addscores(::InsideScoring, left, right) = left + right
mulscores(::InsideScoring, left, right) = left * right
function rulescore(sc::InsideScoring, rule)
    d = sc.ruledist(lhs(rule))
    LogProb(logpdf(d, rule), islog=true)
end

struct BooleanScoring end
scoretype(::BooleanScoring, grammar) = Bool
rulescore(::BooleanScoring, rule) = true
addscores(::BooleanScoring, left, right) = left || right
mulscores(::BooleanScoring, left, right) = left && right

struct Derivations{T}
    all :: Vector{Vector{Rule{T}}}
end
iszero(ds::Derivations) = isempty(ds.all)
zero(::Type{Derivations{T}}) where T = Derivations(Vector{Rule{T}}[])
getallderivations(ds::Derivations) = ds.all

struct AllDerivationScoring end
const ADS = AllDerivationScoring
scoretype(::ADS, grammar) = Derivations{eltype(grammar)}
rulescore(::ADS, rule) = Derivations([[rule]])
addscores(::ADS, left, right) = Derivations([left.all; right.all])
mulscores(::ADS, left, right) = Derivations([[l; r] for l in left.all for r in right.all])

######################################################################
### Free-semiring scorings with manually managed pointer structure ###
######################################################################

# Implementation idea: break rec. structure with indices into a vector (store).
# Ihe store contains unboxed values, which reduces GC times.
# Additionally, it allows to update probabilities without parsing again 
# (not yet implemented).

@enum ScoreTag ADD MUL VAL ZERO

struct ScoredFreeEntry{S, V}
    tag        :: ScoreTag
    score      :: S
    value      :: V
    index      :: Int
    leftIndex  :: Int
    rightIndex :: Int

    # addition and multiplication
    function ScoredFreeEntry(
        store :: Vector{ScoredFreeEntry{S, V}},
        op    :: Union{typeof(+), typeof(*)},
        left  :: ScoredFreeEntry{S, V}, 
        right :: ScoredFreeEntry{S, V}
    ) where {S, V}
        tag(::typeof(+)) = ADD
        tag(::typeof(*)) = MUL
        score = op(left.score, right.score)
        value = left.value # dummy value
        index = length(store) + 1
        x = new{S, V}(tag(op), score, value, index, left.index, right.index)
        push!(store, x)
        return x
    end

    # scored values
    function ScoredFreeEntry(
        store :: Vector{ScoredFreeEntry{S, V}},
        score :: S,
        value :: V
    ) where {S, V}
        index = length(store) + 1
        x = new{S, V}(VAL, score, value, index)
        push!(store, x)
        return x
    end

    # constant zero
    function ScoredFreeEntry(::Type{S}, ::Type{V}) where {S, V}
        new{S, V}(ZERO, zero(S))
    end
end

zero(::Type{ScoredFreeEntry{S, V}}) where {S, V} = ScoredFreeEntry(S, V)
iszero(x::ScoredFreeEntry) = x.tag == ZERO

# Weighted Derivation Scoring (WDS)
struct WeightedDerivationScoring{D, T, L}
    ruledist :: D
    store    :: Vector{ScoredFreeEntry{LogProb, Rule{T}}}
    logpdf   :: L
end

const WDS{D, T, L} = WeightedDerivationScoring{D, T, L}

function WeightedDerivationScoring(ruledist, grammar, logpdf=logpdf)
    WDS(ruledist, ScoredFreeEntry{LogProb, ruletype(grammar)}[], logpdf)
end

scoretype(::WDS, grammar) = ScoredFreeEntry{LogProb, ruletype(grammar)}

function rulescore(sc::WDS, rule)
    logp = LogProb(sc.logpdf(sc.ruledist(lhs(rule)), rule), islog=true)
    ScoredFreeEntry(sc.store, logp, rule)
end

function addscores(sc::WDS, x, y)
  ZERO == x.tag && return y
  ZERO == y.tag && return x
  return ScoredFreeEntry(sc.store, +, x, y)
end

function mulscores(sc::WDS, x, y)
  ZERO == x.tag && return x
  ZERO == y.tag && return y
  return ScoredFreeEntry(sc.store, *, x, y)
end

function sample_derivations(
    sc::WDS, x::ScoredFreeEntry{S, V}, n::Int
  ) where {S, V}
  vals = Vector{V}()
  for _ in 1:n
    sample_derivation!(vals, sc, x)
  end
  vals
end

function sample_derivation!(vals, sc::WDS, x::ScoredFreeEntry{S, V}) where {S, V}
  if VAL ⊣ x 
    push!(vals, x.value)
  elseif ADD ⊣ x
    goleft = rand(S) < sc.store[x.leftIndex].score / x.score
    index = goleft ? x.leftIndex : x.rightIndex
    sample_derivation!(vals, sc, sc.store[index])
  elseif MUL ⊣ x
    sample_derivation!(vals, sc, sc.store[x.leftIndex])
    sample_derivation!(vals, sc, sc.store[x.rightIndex])
  else # ZERO ⊣ x
    error("cannot sample from zero")
  end
end

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
