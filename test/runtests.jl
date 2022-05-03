using ProbabilisticGrammars
using Test

@testset "utils" begin
    @test length(One(42)) == 1
    @test length(Two(42, π)) == 2
    @test One(42)[1] == 42
    @test Two(42, π)[2] ≈ π
    @test eltype(One(42)) == typeof(42)
    @test collect(Two(3, 4)) == [3, 4]
end

@testset "rules" begin
    r = 1 --> 2 ⋅ 3
    @test lhs(r) == 1
    @test rhs(r) == Two(2, 3)
    @test string(r) == "1 --> 2 ⋅ 3"
end

@testset "derivations and trees" begin
    derivation = ['A' --> 'A'⋅'A', 'A' --> 'a', 'A' --> 'a']
    tree = derivation2tree(derivation)
    @test tree2derivation(tree) == derivation

    derivation = [
        'A' --> 'B', 
        'B' --> 'C'⋅'D',
        'C' --> 'c',
        'D' --> 'B'⋅'A',
        'B' --> 'b',
        'A' --> 'a'
    ]
    tree = derivation2tree(derivation)
    @test tree2derivation(tree) == derivation
    @test tree isa Tree{Char}
    @test Char == eltype(tree)
    @test labels(tree) == ['A', 'B', 'C', 'c', 'D', 'B', 'b', 'A', 'a']
    @test leaflabels(tree) == ['c', 'b', 'a']
    @test innerlabels(tree) == ['A', 'B', 'C', 'D', 'B', 'A']
end

@testset "binary tree grammar" begin
    a, A = 'a', 'A'
    rules = [A --> A⋅A, A --> a]
    grammar = CNFG(rules)

    chart = chartparse(grammar, CountScoring(), fill(a, 10))
    @test [chart[1, n][A] for n in 1:10] == [1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862]

    chart = chartparse(grammar, AllDerivationScoring(), fill(a, 10))
    @test length(getallderivations(chart[1, 10][A])) == 4862

    scoring = InsideScoring(symdircat_ruledist([A], rules))
    chart = chartparse(grammar, scoring, fill(a, 10))
    @test exp(chart[1, 10][A].log) > 0

    scoring = WeightedDerivationScoring(symdircat_ruledist([A], rules), grammar)
    chart = chartparse(grammar, scoring, fill(a, 10))
    derivation = sample_derivations(scoring, chart[1, 10][A], 1)
    @test leaflabels(derivation2tree(derivation)) == fill(a, 10)
end
