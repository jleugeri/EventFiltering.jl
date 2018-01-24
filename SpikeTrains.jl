using InverseLaplace

struct SpikeTrain{T}
    times::Vector{T}
end

struct FilteredSpikeTrain{T}
    spikes::SpikeTrain{T}
    F::Vector{Function}
end

(Base.conv(s::SpikeTrain{T}, F::Function)::FilteredSpikeTrain{T}) where T = FilteredSpikeTrain{T}(s, [F])
(Base.conv(s::FilteredSpikeTrain{T}, F::Function)::FilteredSpikeTrain{T}) where T = FilteredSpikeTrain{T}(s.spikes, [s.F; F])
Base.length(s::SpikeTrain) = length(s.times)
Base.length(s::FilteredSpikeTrain) = length(s.spikes)

"""Merge two spike trains (times must be SORTED!)"""
function Base.:+(s::SpikeTrain{S}, t::SpikeTrain{S})::SpikeTrain{S} where S
    new_times = S[]
    (i1,i2) = (1,1)
    (n1,n2) = length.((s, t))
    while (i1 <= n1) & (i2 <= n2)
        if(s.times[i1] < t.times[i2])
            push!(new_times, s.times[i1])
            i1 += 1
        else
            push!(new_times, t.times[i2])
            i2 += 1
        end
    end

    while i1 <= n1
        push!(new_times, s.times[i1])
        i1 += 1
    end

    while i2 <= n2
        push!(new_times, t.times[i2])
        i2 += 1
    end

    return SpikeTrain{S}(new_times)
end

(Base.getindex(s::SpikeTrain{T}, r)::SpikeTrain{T}) where T = SpikeTrain{T}(s[r])

function slice(s::SpikeTrain{T}, from, to) where T
    r = searchsortedfirst(s.times, from):searchsortedlast(s.times, to)
    return SpikeTrain{T}(s.times[r])
end

function (s::FilteredSpikeTrain{T})(t::T) where T
    spikes = slice(s.spikes, 0, t)

    F = x -> prod([F(x) for F in s.F])
    f = ILt(F, talbot)
    print(t-spikes.times)
    return sum(f(t-spikes.times))
end
