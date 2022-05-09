include("JazzTreebank.jl")

begin # imports and constants
    import .JazzTreebank as JHT # Jazz Harmony Treebank
    import ProbabilisticGrammars: push_rules!
    import SimpleProbabilisticPrograms: recover_trace
    import Base: eltype

    using ProbabilisticGrammars
    using SimpleProbabilisticPrograms: SimpleProbabilisticPrograms, logpdf, symdircat, DirCat, @probprog, probprogtype
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
end 

begin # ProbabilisticGrammar
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
end

begin # SimpleHarmonyModel
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
end

begin # RhythmParser
    struct RhythmParser
        splitratios :: Set{Rational{Int}}
    end

    const RhythmCategory = StdCategory{Rational{Int}}
    eltype(::RhythmParser) = RhythmCategory

    function push_rules!(::RhythmParser, stack, c::RhythmCategory)
        if isterminal(c)
            push!(stack, NT(c) --> c)
        end
    end

    function push_rules!(parser::RhythmParser, stack, c1::RhythmCategory, c2::RhythmCategory)
        if isnonterminal(c1) && isnonterminal(c2)
            s = sum(c1.val + c2.val)
            ratio = c1.val / s
            if ratio in parser.splitratios
                push!(stack, NT(s) --> c1⋅c2)
            end
        end
    end
end

begin # SimpleRhythmModel
    struct SimpleRhythmModel 
        max_denom :: Int
    end

    @probprog function simple_rhythm_model(lhs, dists)
        terminate ~ dists.terminate
        if terminate
            return lhs --> T(lhs)
        else
            ratio ~ dists.ratio
            return lhs --> NT(lhs.val*ratio) ⋅ NT(lhs.val*(1-ratio))
        end
        return
    end

    function recover_trace(::probprogtype(simple_rhythm_model), rule)
        if length(rhs(rule)) == 1
            (; terminate=true, ratio=default(Rational{Int}))
        else
            ratio = rhs(rule)[1].val / (rhs(rule)[1].val + rhs(rule)[2].val)
            (; terminate=false, ratio)
        end
    end

    function prior(::SimpleRhythmModel, parser::RhythmParser)
        dists = (
            terminate = DirCat(Dict(true => 1, false => 1)),
            ratio     = DirCat(Dict(ratio => 0.1 for ratio in parser.splitratios)),
        )
        lhs -> simple_rhythm_model(lhs, dists)
    end

    function parser(model::SimpleRhythmModel)
        m = model.max_denom
        splitratios = Set(n//d for n in 1:m for d in n+1:m)
        RhythmParser(splitratios)
    end

    seq2start(::SimpleRhythmModel, seq) = NT(1//1)
    trees(::SimpleRhythmModel) = [tune["rhythm_tree"] for tune in treebank]
end

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

model = SimpleRhythmModel(100)
grammar = treebank_grammar(model);
@time accs = map(trees(model)) do tree
    prediction = predict_tree(grammar, leaflabels(tree))
    tree_similarity(tree, prediction)
end
mean(accs)