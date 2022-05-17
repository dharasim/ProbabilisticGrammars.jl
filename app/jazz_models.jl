include("JazzTreebank.jl")
include("calkin_wilf.jl")

begin # imports and constants
    import .JazzTreebank as JHT # Jazz Harmony Treebank
    import ProbabilisticGrammars: push_rules!, default
    import SimpleProbabilisticPrograms: recover_trace
    import Base: eltype

    using ProbabilisticGrammars
    using SimpleProbabilisticPrograms: SimpleProbabilisticPrograms, logpdf, symdircat, DirCat, @probprog, probprogtype, Dirac, ProbProg
    using Pitches: Pitches
    using Statistics: mean
    using Distributions: Geometric
    using Setfield: @set

    tunes, treebank = JHT.load_tunes_and_treebank();

    const all_forms = instances(JHT.ChordForm)
    const all_chords = collect(
        JHT.Chord(Pitches.parsespelledpitch(letter * acc), form) 
        for letter in 'A':'G'
        for acc in ("b", "#", "")
        for form in all_forms
    )
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

begin # transpositionally invariant grammar
    all_intervals(from, to) = Pitches.SpelledIC.(from:to)

    function transpinv_harmony_grammar(; rulekinds=[:rightheaded, :leftheaded])
        rules = harmony_rules(rulekinds)
        parser = CNFP(rules)
        ints = all_intervals(-12, 12)
        dists = (
            terminate   = symdircat([true, false], 1),
            interval    = Dict(f => symdircat(ints, 0.1) for f in all_forms),
            form        = Dict((i, f) => symdircat(all_forms, 0.1) for i in ints for f in all_forms),
            rightheaded = symdircat([true, false], 0.1),
        )
        ruledist(lhs) = transpinv_harmony_model(lhs, dists, rulekinds)
        seq2start(seq) = NT(seq[end])
        ProbabilisticGrammar(parser, dists, ruledist, seq2start)
    end

    @probprog function transpinv_harmony_model(lhs, dists, rulekinds)
        terminate ~ dists.terminate
        if terminate 
            return lhs --> T(lhs)
        end

        interval ~ dists.interval[lhs.val.form]
        form     ~ dists.form[interval, lhs.val.form]
        chord    = NT(JHT.Chord(lhs.val.root + interval, form))
        if lhs == chord
            return lhs --> lhs⋅lhs
        elseif :rightheaded in rulekinds && :leftheaded in rulekinds
            rightheaded ~ dists.rightheaded
            return rightheaded ? (lhs --> chord⋅lhs) : (lhs --> lhs⋅chord)
        elseif :rightheaded in rulekinds
            return lhs --> chord⋅lhs
        elseif :leftheaded in rulekinds
            return lhs --> lhs⋅chord
        else
            error("rulekinds = $rulekinds must include either leftheaded or rightheaded")
        end
    end

    function recover_trace(::probprogtype(transpinv_harmony_model), rule)
        if istermination(rule)
            terminate   = true
            rightheaded = default(Bool)
            interval    = default(Pitches.SpelledIC)
            form        = default(JHT.ChordForm)
        else
            terminate   = false
            rightheaded = lhs(rule) == rhs(rule)[2]
            chord       = rightheaded ? rhs(rule)[1] : rhs(rule)[2]
            interval    = chord.val.root - lhs(rule).val.root
            form        = chord.val.form
        end
        (; terminate, rightheaded, interval, form)
    end
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

begin # transpinv product grammar
    function transpinv_product_grammar(; 
            rulekinds=[:leftheaded, :rightheaded], 
            splitratios = ratios_max_denom(100),
        )
        h_rules = harmony_rules(rulekinds)
        parser = ProductParser(
            CNFP(h_rules),
            RhythmParser(splitratios),
        )
        ints = all_intervals(-12, 12)
        dists = (
            terminate   = symdircat([true, false], 1),
            interval    = Dict(f => symdircat(ints, 0.1) for f in all_forms),
            form        = Dict((i, f) => symdircat(all_forms, 0.1) for i in ints for f in all_forms),
            rightheaded = symdircat([true, false], 0.1),
            ratio = symdircat(splitratios, 0.1),
        )
        ruledist(lhs) = transpinv_product_model(lhs, dists, rulekinds)
        seq2start(seq) = (NT(seq[end][1]), NT(1//1))
        ProbabilisticGrammar(parser, dists, ruledist, seq2start)
    end

    @probprog function transpinv_product_model(lhs, dists, rulekinds)
        harmony_lhs, rhythm_lhs = lhs
        harmony_rule ~ transpinv_harmony_model(harmony_lhs, dists, rulekinds)
        if istermination(harmony_rule)
            rhythm_rule = rhythm_lhs --> T(rhythm_lhs)
        else
            ratio ~ dists.ratio
            rhythm_rule = splitrule(rhythm_lhs, ratio)
        end
        return product_rule(harmony_rule, rhythm_rule)
    end

    function recover_trace(::probprogtype(transpinv_product_model), product_rule)
        harmony_rule, rhythm_rule = product_rule_components(product_rule)
        ratio = rhs(rhythm_rule)[1].val / lhs(rhythm_rule).val
        (; harmony_rule, ratio)
    end
end

begin # regularized rhythm grammar
    function regularized_rhythm_grammar(; maxlvl=8, lvlaccept=0.75)
        splitratios = mapreduce(proper_ratios_of_calkin_wilf_level, union, 1:maxlvl)
        parser = RhythmParser(splitratios)
        dists = (
            terminate = symdircat([true, false], 1),
            levelm1   = Geometric(1 - lvlaccept),
            ratio     = symdircat.(proper_ratios_of_calkin_wilf_level.(1:maxlvl), 0.1),
        )
        ruledist(lhs) = regularized_rhythm_model(lhs, dists)
        seq2start(seq) = NT(1//1)
        ProbabilisticGrammar(parser, dists, ruledist, seq2start)
    end

    @probprog function regularized_rhythm_model(lhs, dists)
        terminate ~ dists.terminate
        if terminate; return lhs --> T(lhs); end
        levelm1 ~ dists.levelm1 # level minus one
        level = min(levelm1 + 1, maxlvl) # hacky solution to reduce complexity
        ratio ~ dists.ratio[level]
        return splitrule(lhs, ratio)
    end

    function recover_trace(::probprogtype(regularized_rhythm_model), rule)
        if istermination(rule)
            terminate = true
            levelm1   = default(Int)
            ratio     = default(Rational{Int})
        else
            terminate = false
            ratio     = rhs(rule)[1].val / (rhs(rule)[1].val + rhs(rule)[2].val)
            levelm1   = calkin_wilf_level(ratio) - 1
        end
        return (; terminate, levelm1, ratio)
    end
end
