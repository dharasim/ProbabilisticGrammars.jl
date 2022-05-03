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
    @test string(1 --> 2 ⋅ 3) == "1 --> 2 ⋅ 3"
end