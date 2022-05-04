include("JazzTreebank.jl")
import .JazzTreebank as JHT # Jazz Harmony Treebank

using ProbabilisticGrammars
using Pitches: Pitches
using Statistics: mean

tunes, treebank = JHT.load_tunes_and_treebank();

const all_forms = instances(JHT.ChordForm)
const all_chords = collect(
  JHT.Chord(Pitches.parsespelledpitch(letter * acc), form) 
  for letter in 'A':'G'
  for acc in ("b", "#", "")
  for form in all_forms
)

##########################
# Probabilistic grammars #
##########################

mutable struct ProbabilisticGrammar{P, D, F}
    parser    :: P
    ruledist  :: D
    seq2start :: F
end

function prior_grammar(model)
    p = parser(model)
    ProbabilisticGrammar(p, prior(model, p), seq -> seq2start(model, seq))
end

function treebank_grammar(model)
    grammar = prior_grammar(model)
    observe_trees!(grammar.ruledist, trees(model))
    return grammar
end

function predict_tree(grammar::ProbabilisticGrammar, seq)
    scoring = BestDerivationScoring(grammar.ruledist, grammar.parser)
    chart = chartparse(grammar.parser, scoring, seq)
    forest = chart[1, end][grammar.seq2start(seq)]
    logprob, derivation = getbestderivation(scoring, forest)
    derivation2tree(derivation)
end

#######################
# Model specification #
#######################

struct SimpleHarmonyModel
    rulekinds :: Vector{Symbol} # [:leftheaded, :rightheaded]
end

function parser(model::SimpleHarmonyModel)
    ts  = T.(all_chords)  # terminals
    nts = NT.(all_chords) # nonterminals

    # termination and duplication rules are always included
    rules = [nt --> t for (nt, t) in zip(nts, ts)]
    append!(rules, [nt --> nt⋅nt for nt in nts])

    # include headed rules
    if :leftheaded in model.rulekinds
        append!(rules, [nt1 --> nt1⋅nt2 for nt1 in nts for nt2 in nts if nt1 != nt2])
    end
    if :rightheaded in model.rulekinds
        append!(rules, [nt2 --> nt1⋅nt2 for nt1 in nts for nt2 in nts if nt1 != nt2])
    end

    CNFP(rules)
end

function prior(model::SimpleHarmonyModel, parser=parser(model))
    symdircat_ruledist(NT.(all_chords), parser.rules, 0.1)
end

seq2start(::SimpleHarmonyModel, seq) = NT(seq[end])
trees(::SimpleHarmonyModel) = [tune["harmony_tree"] for tune in treebank]

###################
# Model execution #
###################

model = SimpleHarmonyModel([:rightheaded])
grammar = treebank_grammar(model);
@time accs = map(trees(model)) do tree
    prediction = predict_tree(grammar, leaflabels(tree))
    tree_similarity(tree, prediction)
end
mean(accs)
