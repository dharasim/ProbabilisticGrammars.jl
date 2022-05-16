module ProbabilisticGrammars

import Base: *, +, show, map, length, getindex, iterate, eltype, insert!, zero, iszero

using LogProbs: LogProb
using Setfield: @set
using DataStructures: counter, Accumulator
using ProgressMeter: Progress, progress_map
using SimpleProbabilisticPrograms: SimpleProbabilisticPrograms
using SimpleProbabilisticPrograms: logpdf, logvarpdf, insupport, add_obs!, DirCat, symdircat

export default, OneOrTwo, One, Two, ⊣, normalize
export Tree, label, children, labels, innerlabels, leaflabels, tree_similarity
export Rule, -->, ⋅, lhs, rhs
export derivation2tree, tree2derivation
export symdircat_ruledist, observe_trees!
export StdCategory, T, NT, isterminal, isnonterminal
export CNFP, chartparse, push_rules!, ruletype
export CountScoring, InsideScoring, BooleanScoring
export AllDerivationScoring, getallderivations
export WeightedDerivationScoring, sample_derivations
export BestDerivationScoring, getbestderivation
export ProbabilisticGrammar, train_on_trees!, use_map_params, predict_tree
export estimate_rule_counts, runvi

#########
# Utils #
#########

normalize(xs) = xs ./ sum(xs)

⊣(tag, x) = x.tag == tag

default(::Type{T}) where {T <: Number} = zero(T)
default(::Type{Char}) = '0'
default(::Type{Tuple{T1, T2}}) where {T1, T2} = (default(T1), default(T2))

*(a::Accumulator, n::Number) = Accumulator(Dict(k => v*n for (k,v) in a.map))
*(n::Number, a::Accumulator) = Accumulator(Dict(k => n*v for (k,v) in a.map))
+(a::Accumulator, n::Number) = Accumulator(Dict(k => v+n for (k,v) in a.map))
+(n::Number, a::Accumulator) = Accumulator(Dict(k => n+v for (k,v) in a.map))

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

function relabel_with_spans(tree)
    k = 0 # leaf index
    next_leafindex() = (k += 1; k)
    span(i, j) = (from=i, to=j)
    combine(span1, span2) = span(span1.from, span2.to)
  
    function relabel(tree) 
        if isleaf(tree)
            i = next_leafindex()
            Tree(span(i,i))
        elseif length(tree.children) == 1
            child = relabel(tree.children[1])
            Tree(child.label, child)
        elseif length(tree.children) == 2
            left  = relabel(tree.children[1])
            right = relabel(tree.children[2])
            Tree(combine(left.label, right.label), left, right)
        else
            error("tree is not binary")
        end
    end
  
    return relabel(tree)
end
  
function collapse_unaries(tree)
    if isleaf(tree)
        tree
    elseif length(tree.children) == 1
        collapse_unaries(tree.children[1])
    else
        Tree(tree.label, map(collapse_unaries, tree.children))
    end
end
  
function constituent_spans(tree)
    tree |> collapse_unaries |> relabel_with_spans |> innerlabels
end
  
function tree_similarity(tree1, tree2)
    spans1 = constituent_spans(tree1)
    spans2 = constituent_spans(tree2)
    @assert length(spans1) == length(spans2)
    length(intersect(spans1, spans2)) / length(spans1)
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

    label = lhs(first(derivation))
    rewrite(label)::Tree{typeof(label)}
end

function tree2derivation(tree::Tree{T}) where T
    [label(t) --> label.(children(t)) for t in subtrees(tree) if !isleaf(t)]
end

derivation = ['A' --> 'A'⋅'A', 'A' --> 'a', 'A' --> 'a']
tree = derivation2tree(derivation)

######################
# Rule distributions #
######################

# struct DirCatRuleDist{T}
#     dists :: Dict{T, DirCat{Rule{T}}}
# end
# 
# (d::DirCatRuleDist)(x) = d.dists[x]
# 
# function symdircat_ruledist(xs, rules, concentration=1.0)
#     applicable_rules(x) = filter(r -> lhs(r) == x, rules)
#     dists = Dict(
#         x => symdircat(applicable_rules(x), concentration) 
#         for x in xs
#     )
#     DirCatRuleDist(dists)
# end

import Base: rand
import Distributions: logpdf
using Random: AbstractRNG

struct GenericCategorical{T}
    probs :: Dict{T, Float64}
end

function rand(rng::AbstractRNG, gc::GenericCategorical)
    r = rand(rng)
    q = 0
    for (x, p) in gc.probs
        q += p
        if q > r; return x; end
    end
end

logpdf(gc::GenericCategorical, x) = log(gc.probs[x])

function observe_rule!(ruledist, rule, count=1)
    if insupport(ruledist(lhs(rule)), rule)
        add_obs!(ruledist(lhs(rule)), rule, count)
    else
        @info "Rule $rule not observed. It's not in the rule distribution."
    end
end
  
function observe_trees!(ruledist, trees)
    for tree in trees, rule in tree2derivation(tree)
        observe_rule!(ruledist, rule)
    end
end

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

##########
# Parser #
##########

"""
    CNFP(rules)

Parser for a grammar in Chomsky-normal form (CNF).

The user is responsible for all rules being in CNF.
"""
struct CNFP{T}
    rules :: Set{Rule{T}}
    lhss  :: Dict{Rhs{T}, Vector{T}} # left-hand sides

    function CNFP(rules)
        T = eltype(eltype(typeof(rules)))
        lhss = Dict{Rhs{T}, Vector{T}}()
        ruleset = Set(collect(rules))
        for r in ruleset
            lhss_r = get!(() -> T[], lhss, rhs(r))
            push!(lhss_r, lhs(r))
        end
        return new{T}(ruleset, lhss)
    end
end

"""
    push_rules!(parser, stack, rhs...)

Push all rules with given right-hand side to stack.
"""
function push_rules!(parser::CNFP, stack, xs...)
    rhs = OneOrTwo(xs...)
    if haskey(parser.lhss, rhs)
        for lhs in parser.lhss[rhs]
            push!(stack, lhs --> rhs)
        end
    end
end

eltype(::CNFP{T}) where T = T
ruletype(parser) = Rule{eltype(parser)}

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
    chartparse(parser, scoring, sequence)
"""
function chartparse(parser, scoring, sequence)
    chartparse(parser, scoring, [[x] for x in sequence])
end

function chartparse(parser, scoring, sequence::Vector{Vector{T}}) where T
    n = length(sequence)
    S = scoretype(scoring, parser)
    chart = empty_chart(T, S, n)
    stack = Vector{Rule{T}}() # channel for communicating completions
    # using a single stack is much more efficient than constructing multiple arrays
    stack_unary = Vector{Tuple{T, S}}()

    score(rule) = rulescore(scoring, rule)

    # terminal completions
    for (i, terminals) in enumerate(sequence)
        for terminal in terminals
            push_rules!(parser, stack, terminal)
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
                        push_rules!(parser, stack, rhs1, rhs2)
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
                push_rules!(parser, stack, rhs)
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

#######################
# Scores and scorings #
#######################

mulscores(scoring, s1, s2, s3) = mulscores(scoring, s1, mulscores(scoring, s2, s3))

struct CountScoring end
scoretype(::CountScoring, parser) = Int
rulescore(::CountScoring, rule) = 1
addscores(::CountScoring, left, right) = left + right
mulscores(::CountScoring, left, right) = left * right

struct InsideScoring{D} ruledist::D end
scoretype(::InsideScoring, parser) = LogProb
addscores(::InsideScoring, left, right) = left + right
mulscores(::InsideScoring, left, right) = left * right
function rulescore(sc::InsideScoring, rule)
    d = sc.ruledist(lhs(rule))
    LogProb(logpdf(d, rule), islog=true)
end

struct BooleanScoring end
scoretype(::BooleanScoring, parser) = Bool
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
scoretype(::ADS, parser) = Derivations{eltype(parser)}
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

function WeightedDerivationScoring(ruledist, parser, logpdf=logpdf)
    WDS(ruledist, ScoredFreeEntry{LogProb, ruletype(parser)}[], logpdf)
end

scoretype(::WDS, parser) = ScoredFreeEntry{LogProb, ruletype(parser)}

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

###########################
# Best derivation scoring #
###########################

@enum DerivationTag CONCAT RULE

# Each best derivation value is either an inner tree node with tag CONCAT
# or a leaf node with tag RULE.
# The rules of the derivation are the rules of the tree's leafs.
# The usage of indices essentially implements a tiny garbage collector that is
# much more efficient then Julia's default GC in this case, because the default
# GC traverses reference graphs recursively.
struct BestDerivation{T}
    tag        :: DerivationTag
    prob       :: LogProb
    rule       :: Rule{T}
    index      :: Int
    leftIndex  :: Int
    rightIndex :: Int
  
    # concatenation
    function BestDerivation(
            store :: Vector{BestDerivation{T}}, 
            left  :: BestDerivation{T}, 
            right :: BestDerivation{T},
        ) where T
        prob = left.prob * right.prob
        rule = left.rule # dummy value
        index = length(store) + 1
        x = new{T}(CONCAT, prob, rule, index, left.index, right.index)
        push!(store, x)
        return x
    end
  
    # single-rule derivations
    function BestDerivation(store, prob::LogProb, rule::Rule{T}) where T
        index = length(store) + 1
        x = new{T}(RULE, prob, rule, index)
        push!(store, x)
        return x
    end
end

struct BestDerivationScoring{D, T}
    ruledist :: D
    store    :: Vector{BestDerivation{T}}
end

const BDS{D, T} = BestDerivationScoring{D, T}
BDS(ruledist, parser) = BDS(ruledist, BestDerivation{eltype(parser)}[])
scoretype(::BDS, parser) = BestDerivation{eltype(parser)}
addscores(::BDS, left, right) = left.prob >= right.prob ? left : right
mulscores(sc::BDS, left, right) = BestDerivation(sc.store, left, right)

function rulescore(sc::BDS, rule)
  logp = LogProb(logpdf(sc.ruledist(lhs(rule)), rule), islog=true)
  BestDerivation(sc.store, logp, rule)
end

function getbestderivation(sc::BDS, bd::BestDerivation{T}) where T
    function getbestderivation!(out, sc, bd)
        if RULE ⊣ bd
            push!(out, bd.rule)
        else # CONCAT ⊣ bd
            getbestderivation!(out, sc, sc.store[bd.leftIndex])
            getbestderivation!(out, sc, sc.store[bd.rightIndex])
        end
    end

    derivation = Rule{T}[]
    getbestderivation!(derivation,sc, bd)
    bd.prob.log, derivation
end

#########################
# Probabilistic grammar #
#########################

struct ProbabilisticGrammar{P, D, R, S2S}
    parser    :: P
    dists     :: D
    ruledist  :: R
    seq2start :: S2S
end

function train_on_trees!(grammar::ProbabilisticGrammar, trees)
    observe_trees!(grammar.ruledist, trees)
    return grammar
end

# use maximum a priori parameters
function use_map_params(grammar::ProbabilisticGrammar)
    map_dists = map(use_map_params, grammar.dists)
    @set grammar.dists = map_dists
end

function use_map_params(dict::Dict)
    Dict(k => use_map_params(v) for (k, v) in dict)
end

function use_map_params(dc::DirCat)
    if !dc.logpdfs_uptodate
        SimpleProbabilisticPrograms.update_logpdfs!(dc)
    end
    s = sum(values(dc.pscounts))
    map_params = Dict(x => c/s for (x, c) in dc.pscounts)
    GenericCategorical(map_params)
end

function predict_tree(grammar::ProbabilisticGrammar, seq)
    scoring = BestDerivationScoring(grammar.ruledist, grammar.parser)
    chart   = chartparse(grammar.parser, scoring, seq)
    forest  = chart[1, end][grammar.seq2start(seq)]
    logprob, derivation = getbestderivation(scoring, forest)
    derivation2tree(derivation)
end

#########################
# Variational inference #
#########################

function estimate_rule_counts(
      grammar::ProbabilisticGrammar, sequences; 
      seq2numtrees=seq->length(seq)^2, showprogress=true
    )
    p = Progress(length(sequences); desc="estimating rule counts: ", enabled=showprogress)
    estimates_per_sequence = progress_map(sequences; progress=p) do sequence
        scoring = WDS(grammar.ruledist, grammar.parser, logvarpdf)
        chart = chartparse(grammar.parser, scoring, sequence)
        forest = chart[1, end][grammar.seq2start(sequence)]
        n = seq2numtrees(sequence)
        return 1/n * counter(sample_derivations(scoring, forest, n))
    end
    reduce(merge!, estimates_per_sequence)
end

# run variational inference
function runvi(
      mk_grammar, sequences; 
      epochs, seq2numtrees=seq->length(seq)^2, showprogress=true
    )
    grammar = mk_grammar()
    for e in 1:epochs
        showprogress ? println("epoch $e of $epochs") : nothing
        rule_counts = estimate_rule_counts(grammar, sequences; seq2numtrees, showprogress)
        grammar = mk_grammar()
        for (rule, pscount) in rule_counts
            add_obs!(grammar.ruledist(lhs(rule)), rule, pscount)
        end
    end
    return grammar
end
# 
# # run variational inference with automatic prior initialization
# function runvi(
#     epochs, mk_prior, grammar::Grammar, other_estimation_args...;
#     showprogress=true
#   )
#   runvi(epochs, mk_prior, mk_prior(), grammar, other_estimation_args...; showprogress)
# end

end # module