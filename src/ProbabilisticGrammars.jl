module ProbabilisticGrammars

import Base: show, length, getindex, iterate, eltype

export default, OneOrTwo, One, Two
export Rule, -->, ⋅

#########
# Utils #
#########

default(::Type{T}) where {T <: Number} = zero(T)

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
# Rules #
#########

struct Rule{T}
    lhs :: T
    rhs :: OneOrTwo{T}
end

Rule(lhs, rhs...) = Rule(lhs, OneOrTwo(rhs...))
-->(lhs::T, rhs::T) where T = Rule(lhs, rhs)
-->(lhs, rhs) = Rule(lhs, rhs...)
⋅(x1, x2) = (x1, x2)

function show(io::IO, r::Rule)
    if length(r.rhs) == 1
        print(io, "$(r.lhs) --> $(r.rhs.fst)")
    else
        print(io, "$(r.lhs) --> $(r.rhs.fst) ⋅ $(r.rhs.snd)")
    end
end


end # module
