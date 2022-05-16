include("JazzTreebank.jl")

begin # imports and constants
    import .JazzTreebank as JHT # Jazz Harmony Treebank
    import ProbabilisticGrammars: push_rules!, default
    import SimpleProbabilisticPrograms: recover_trace
    import Base: eltype

    using ProbabilisticGrammars
    using SimpleProbabilisticPrograms: SimpleProbabilisticPrograms, logpdf, symdircat, DirCat, @probprog, probprogtype, Dirac, ProbProg
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

    function use_map_params!(grammar::ProbabilisticGrammar)

    end

    function predict_tree(grammar::ProbabilisticGrammar, seq)
        scoring = BestDerivationScoring(grammar.ruledist, grammar.parser)
        chart   = chartparse(grammar.parser, scoring, seq)
        forest  = chart[1, end][grammar.seq2start(seq)]
        logprob, derivation = getbestderivation(scoring, forest)
        derivation2tree(derivation)
    end
end

begin # simple harmony grammar
    function harmony_rules(rulekinds)
        ts  = T.(all_chords)  # terminals
        nts = NT.(all_chords) # nonterminals

        # termination and duplication rules are always included
        rules = [[nt --> t for (nt, t) in zip(nts, ts)]; [nt --> nt⋅nt for nt in nts]]

        # include headed rules
        if :leftheaded in rulekinds
            append!(rules, [nt1 --> nt1⋅nt2 for nt1 in nts for nt2 in nts if nt1 != nt2])
        end
        if :rightheaded in rulekinds
            append!(rules, [nt2 --> nt1⋅nt2 for nt1 in nts for nt2 in nts if nt1 != nt2])
        end

        return rules
    end

    function simple_harmony_grammar(;
            rulekinds = [:leftheaded, :rightheaded],
            concentration = 0.1,
        )
        rules = harmony_rules(rulekinds)
        parser = CNFP(rules)
        dircat(nt) = symdircat([r for r in rules if lhs(r)==nt], concentration)
        dists = (
            rule = Dict(nt => dircat(nt) for nt in NT.(all_chords)),
        )
        ruledist(lhs) = simple_harmony_model(lhs, dists)
        seq2start(seq) = NT(seq[end])
        ProbabilisticGrammar(parser, dists, ruledist, seq2start)
    end

    @probprog function simple_harmony_model(lhs, dists)
        rule ~ dists.rule[lhs]
        return rule
    end

    recover_trace(::ProbProg{typeof(simple_harmony_model)}, rule) = (rule = rule, )
    istermination(harmony_rule) = isone(length(rhs(harmony_rule)))
end

begin # TranspInvModel
    all_intervals(from, to) = Pitches.SpelledIC.(from:to)
    dists = (
        terminate = symdircat([true, false], 0.1),
        rightheaded = symdircat([true, false], 0.1),
        interval = symdircat(all_intervals(-12, 12), 0.1),
        form = Dict(f => symdircat(all_forms, 0.1) for f in all_forms),
    )

    @probprog function transp_inv_model(lhs, dists)
        terminate ~ dists.terminate
        if terminate 
            return lhs --> T(lhs)
        end

        interval ~ dists.interval
        form     ~ dists.form[lhs.val.form]
        chord    = NT(JHT.Chord(lhs.val.root + interval, form))
        if lhs == chord
            return lhs --> lhs⋅lhs
        else
            rightheaded ~ dists.rightheaded
            return rightheaded ? (lhs --> chord⋅lhs) : (lhs --> lhs⋅chord)
        end
    end

    function recover_trace(::probprogtype(transp_inv_model), rule)
        
    end
end

# nt = NT(first(all_chords))
# d = transp_inv_model(nt, dists)
# rule = rand(d)
# logpdf(d, rule)

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

begin # simple rhythm grammar
    ratios_max_denom(m) = Set(n//d for n in 1:m for d in n+1:m)
    splitrule(lhs, ratio) = lhs --> NT(lhs.val*ratio) ⋅ NT(lhs.val*(1-ratio))

    function simple_rhythm_grammar(; splitratios=ratios_max_denom(100))
        parser = RhythmParser(splitratios)
        dists = (
            terminate = symdircat([true, false], 1),
            ratio     = symdircat(splitratios, 0.1),
        )
        ruledist(lhs) = simple_rhythm_model(lhs, dists)
        seq2start(seq) = NT(1//1)
        ProbabilisticGrammar(parser, dists, ruledist, seq2start)
    end

    @probprog function simple_rhythm_model(lhs, dists)
        terminate ~ dists.terminate
        if terminate
            return lhs --> T(lhs)
        else
            ratio ~ dists.ratio
            return splitrule(lhs, ratio)
        end
        return
    end

    function recover_trace(::probprogtype(simple_rhythm_model), rule)
        if length(rhs(rule)) == 1
            (terminate=true, ratio=default(Rational{Int}))
        else
            ratio = rhs(rule)[1].val / (rhs(rule)[1].val + rhs(rule)[2].val)
            (terminate=false, ratio=ratio)
        end
    end
end

begin # ProductParser
    struct ProductParser{P1, P2, R1, R2}
        components :: Tuple{P1, P2}
        stacks     :: Tuple{Vector{R1}, Vector{R2}}

        function ProductParser(component1::P1, component2::P2) where {P1, P2}
            R1, R2 = ruletype(component1), ruletype(component2)
            new{P1, P2, R1, R2}((component1, component2), (R1[], R2[]))
        end
    end

    function eltype(parser::ProductParser) 
        p1, p2 = parser.components
        Tuple{eltype(p1), eltype(p2)}
    end

    function unzip(xs::OneOrTwo)
        if isone(length(xs))
            (One(xs[1][1]), One(xs[1][2]))
        else
            (Two(xs[1][1], xs[2][1]), Two(xs[1][2], xs[2][2]))
        end
    end
    unzip(xs) = (zip(xs...)...,)
    product_rule(r1, r2) = (lhs(r1), lhs(r2)) --> zip(rhs(r1), rhs(r2))

    # inverse to product_rule
    # needs to be written-out explicitely to ensure type inference of result
    function product_rule_components(product_rule)
        component1 = if isone(length(rhs(product_rule)))
            lhs(product_rule)[1] --> rhs(product_rule)[1][1]
        else
            lhs(product_rule)[1] --> rhs(product_rule)[1][1] ⋅ rhs(product_rule)[2][1]
        end
        component2 = if isone(length(rhs(product_rule)))
            lhs(product_rule)[2] --> rhs(product_rule)[1][2]
        else
            lhs(product_rule)[2] --> rhs(product_rule)[1][2] ⋅ rhs(product_rule)[2][2]
        end

        (component1, component2)
    end

    function push_rules!(parser::ProductParser, stack, cs...)
        p1, p2 = parser.components
        s1, s2 = parser.stacks
        cs1, cs2 = unzip(cs)

        push_rules!(p1, s1, cs1...)
        push_rules!(p2, s2, cs2...)
        for r1 in s1, r2 in s2
            push!(stack, product_rule(r1, r2))
        end
        empty!(s1)
        empty!(s2)
        return nothing
    end
end

begin # simple product grammar
    function simple_product_grammar(;
            rulekinds = [:leftheaded, :rightheaded],
            concentration = 0.1,
            splitratios = ratios_max_denom(100),
        )
        h_rules = harmony_rules(rulekinds)
        parser = ProductParser(
            CNFP(h_rules),
            RhythmParser(splitratios),
        )
        dircat(h_nt) = symdircat([r for r in h_rules if lhs(r)==h_nt], concentration)
        dists = (
            harmony_rule = Dict(h_nt => dircat(h_nt) for h_nt in NT.(all_chords)),
            ratio = symdircat(splitratios, 0.1),
        )
        ruledist(lhs) = simple_product_model(lhs, dists)
        seq2start(seq) = (NT(seq[end][1]), NT(1//1))
        ProbabilisticGrammar(parser, dists, ruledist, seq2start)
    end

    @probprog function simple_product_model(lhs, dists)
        harmony_lhs, rhythm_lhs = lhs
        harmony_rule ~ dists.harmony_rule[harmony_lhs]
        if istermination(harmony_rule)
            rhythm_rule = rhythm_lhs --> T(rhythm_lhs)
        else
            ratio ~ dists.ratio
            rhythm_rule = splitrule(rhythm_lhs, ratio)
        end
        return product_rule(harmony_rule, rhythm_rule)
    end

    function recover_trace(::probprogtype(simple_product_model), product_rule)
        harmony_rule, rhythm_rule = product_rule_components(product_rule)
        ratio = rhs(rhythm_rule)[1].val / lhs(rhythm_rule).val
        (; harmony_rule, ratio)
    end
end

###################
# Model execution #
###################

grammar = simple_harmony_grammar(rulekinds=[:rightheaded]);
trees = [tune["harmony_tree"] for tune in treebank];
train_on_trees!(grammar, trees);
@time accs = map(trees) do tree
    prediction = predict_tree(grammar, leaflabels(tree))
    tree_similarity(tree, prediction)
end; mean(accs)

grammar = simple_rhythm_grammar();
trees = [tune["rhythm_tree"] for tune in treebank];
train_on_trees!(grammar, trees);
@time accs = map(trees) do tree
    prediction = predict_tree(grammar, leaflabels(tree))
    tree_similarity(tree, prediction)
end; mean(accs)

grammar = simple_product_grammar(rulekinds=[:rightheaded]);
trees = [tune["product_tree"] for tune in treebank];
train_on_trees!(grammar, trees);
@time accs = map(trees) do tree
    prediction = predict_tree(grammar, leaflabels(tree))
    tree_similarity(tree, prediction)
end; mean(accs)