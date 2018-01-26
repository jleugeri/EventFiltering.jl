"""
Module to simplify working with events and filters.
--jleugeri, 2017


Usage example:
==============

julia```
using Plots
pyplot()
e1 = EventTrain([1.0, 2.0, 3.0])
e2 = 2*EventTrain([0.5, 1.5, 2.5, 3.5])
e3 = 1.5 * e1 + e2

F = LaplaceFilter(s->1/(s+1))
X = F∘F∘e3
M = x - F∘x
plot(X, 0,10, color=:blue, event_color=:blue, show_events=false)
plot!(F∘x, 0, 10, color=:red, show_events=false)
plot!(M, 0, 10, color=:black)
'''
"""
module EventFiltering
using InverseLaplace, Plots

export Filter, LaplaceFilter, GammaFilter, ExponentialFilter, EventTrain, FilteredEventTrain, freeze, fromto, ℒ, delay
    # Base.:∘,
    # Base.:+,
    # Base.:-,
    # Base.:*,
    # Base.length,
    # Base.getindex,

# Helper function to evaluate a list of functions on given arguments
call_all(fs, args...) = [f(args...) for f in fs]

###########
# Filters #
###########

abstract type Filter{T, E} end


### LaplaceFilter
struct LaplaceFilter{T} <: Filter{T, Float64}
    terms::Vector{Function}
end
LaplaceFilter(t::Vector{Function}) = LaplaceFilter{Float64}(t)
LaplaceFilter(t::Function) = LaplaceFilter{Float64}(Function[t])

# Convolve LaplaceFilters by convolution theorem
Base.:∘(F1::LaplaceFilter{T}, F2::LaplaceFilter{T}) where T = LaplaceFilter{T}([F1.terms; F2.terms])

# Evaluate LaplaceFilter
ℒ(F::LaplaceFilter) = x->convert(Complex, prod(call_all(F.terms,x)))
freeze(F::LaplaceFilter; method=talbot, accuracy=32) = ILt(ℒ(F), method, accuracy)
(F::LaplaceFilter{T})(t::T) where T = convert(Float64,freeze(F)(t))

# Add LaplaceFilters
Base.:+(F1::LaplaceFilter{T}, F2::LaplaceFilter{T}) where T = LaplaceFilter{T}([s->convert(Float64, ℒ(F1)(s)+ℒ(F2)(s))])
# Scale LaplaceFilters
Base.:*(a, F::LaplaceFilter{T}) where T = LaplaceFilter{T}([x->a; F.terms])
# Sign-Invert LaplaceFilters
Base.:-(F1::LaplaceFilter) = (-1) * F1
# Subtract LaplaceFilters
Base.:-(F1::LaplaceFilter, F2::LaplaceFilter) = F1 + (-F2)

### GammaFilter
struct GammaFilter{T,E} <: Filter{T, E}
    α::T
    k::T
end

# Convolve GammaFilters (convert to LaplaceFilter if necessary)
function Base.:∘(F1::GammaFilter{T,E}, F2::GammaFilter{T,E}) where {T,E}
    if F1.α == F2.α
        GammaFilter{T,E}(F1.α, F1.k+F2.k)
    else
        LaplaceFilter(F1)∘LaplaceFilter(F2)
    end
end


# Evaluate GammaFilter
ℒ(F::GammaFilter) = s->(F.α/(s+F.α))^F.k
freeze(F::GammaFilter) = F
(F::GammaFilter)(t) = (F.α^F.k)/gamma(F.k)*t^(F.k-1)*exp(-F.α*t)
(F::GammaFilter)(t::Vector) = F.(t)

# Convert GammaFilter to LaplaceFilter
LaplaceFilter(F::GammaFilter{T,E}) where {T,E} = LaplaceFilter{T}([ℒ(F)])

# ExponentialFilter
ExponentialFilter(α=1.0) = GammaFilter{Float64, Float64}(α, 1)


#####################################
# EventTrain and FilteredEventTrain #
#####################################

struct EventTrain{T,E}
    times::Vector{T}
    events::Vector{E}
end
function EventTrain(times::Vector{T}) where T
    if ~issorted(times)
        sort!(times)
    end
    EventTrain{T, Float64}(times, ones(Float64, length(times)))
end

struct FilteredEventTrain{T,E,F<:Filter{T,E}}
    event_train::EventTrain{T,E}
    F::F
end

Base.length(s::EventTrain) = length(s.times)
Base.length(s::FilteredEventTrain) = length(s.event_train)
Base.endof(s::EventTrain) = length(s)
Base.endof(s::FilteredEventTrain) = length(s)

"""Merge two event trains (times must be SORTED!)"""
function (Base.:+(s::EventTrain{S,E}, t::EventTrain{T, F})::EventTrain{promote_type(S,T),promote_type(E,F)}) where {S,T,E,F}
    ST = promote_type(S,T)
    EF = promote_type(E,F)
    new_times = ST[]
    new_events = EF[]
    (i1,i2) = (1,1)
    (n1,n2) = length.((s, t))
    while (i1 <= n1) & (i2 <= n2)
        if(s.times[i1] < t.times[i2])
            push!(new_times, s.times[i1])
            push!(new_events, s.events[i1])
            i1 += 1
        else
            push!(new_times, t.times[i2])
            push!(new_events, t.events[i2])
            i2 += 1
        end
    end

    while i1 <= n1
        push!(new_times, s.times[i1])
        push!(new_events, s.events[i1])
        i1 += 1
    end

    while i2 <= n2
        push!(new_times, t.times[i2])
        push!(new_events, t.events[i2])
        i2 += 1
    end

    return EventTrain{ST, EF}(new_times, new_events)
end
# Add FilteredEventTrains
function Base.:+(s1::FilteredEventTrain{T,E,F},s2::FilteredEventTrain{T,G,H}) where {T,E,F,G,H}
    return if s1.F == s2.F
        FilteredEventTrain{T,E,F}(s1.event_train+s2.event_train, s1.F)
    elseif s1.event_train == s2.event_train
        FilteredEventTrain{T,E,F}(s1.event_train, s1.F+s2.F)
    else
        x->(s1(x)+s2(x))
    end
end

# Sign-Invert EventTrains
Base.:-(s1::EventTrain) = (-1) * s1
# Sign-Invert FilteredEventTrains
Base.:-(s1::FilteredEventTrain) = (-1) * s1

# Subtract EventTrains
Base.:-(s1::EventTrain, s2::EventTrain) = s1 + (-s2)
# Subtract FilteredEventTrains
Base.:-(s1::FilteredEventTrain, s2::FilteredEventTrain) = s1 + (-s2)

# Scale EventTrain
Base.:*(a, s::EventTrain{S,E}) where {S,E} = EventTrain{S,E}(s.times, a.*s.events)
# Scale FilteredEventTrain
Base.:*(a, s::FilteredEventTrain{S,E,F}) where {S,E,F} = FilteredEventTrain{S,E,F}(a.*s.event_train, s.F)

# Subsample events of a (Filtered)EventTrain
Base.getindex(s::EventTrain{T, E}, r) where {T,E} = EventTrain{T, E}(s.times[r], s.events[r])
Base.getindex(s::FilteredEventTrain{T, E, F}, r) where {T,E,F} = FilteredEventTrain{T, E, F}(s.event_train[r], s.F)

# Slice a time interval out of a (Filtered)EventTrain
function fromto(s::EventTrain{T, E}, from, to) where {T,E}
    if from == :start
        from = minimum(s.times)-1.0
    end
    if to == :end
        to = maximum(s.times)+1.0
    end

    r = searchsortedfirst(s.times, from):searchsortedlast(s.times, to)
    return s[r]
end
fromto(s::FilteredEventTrain{T, E, F}, from, to) where {T,E,F} = FilteredEventTrain{T, E, F}(fromto(s.event_train, from, to), s.F)

# Shift a (Filtered)EventTrain in time
delay(s::EventTrain{T, E}, dt) where {T,E} = EventTrain{T, E}(sort(s.times + dt), s.events)
delay(s::FilteredEventTrain{T, E, F}, dt) where {T,E,F} = FilteredEventTrain{T, E, F}(delay(s.event_train, dt), s.F)

# Convolve two EventTrains
Base.:∘(e1::EventTrain{T, E}, e2::EventTrain{T, E}) where {T,E} = mapreduce(dtw->delay(dtw[2]*e1, dtw[1]), +, zip(e2.times,e2.events))
# Convolve two FilteredEventTrains
Base.:∘(e1::FilteredEventTrain{T, E, F}, e2::FilteredEventTrain{T, E, F}) where {T,E,F} = FilteredEventTrain(e1.event_train∘e2.event_train, e1.F∘e2.F)
# Convolve one FilteredEventTrain and one EventTrain
Base.:∘(e1::FilteredEventTrain{T, E, F}, e2::EventTrain{T, E}) where {T,E,F} = FilteredEventTrain(e1.event_train∘e2, e1.F)
Base.:∘(e1::EventTrain{T, E}, e2::FilteredEventTrain{T, E, F}) where {T,E,F} = e2∘e1

################################################
# Interaction between filters and event trains #
################################################


# Use linearity to chain filters
Base.:∘(F::FF, e::FilteredEventTrain{T, E, <:Filter{T,E}}) where {T,E,FF<:Filter{T,E}} = FilteredEventTrain(e.event_train, F∘e.F)

# Apply Filter to event-train
Base.:∘(F::FT, e::EventTrain{T, E}) where {T,E, FT<:Filter{T,E}} = FilteredEventTrain{T, E, FT}(e, F)

# Evaluate FilteredEventTrain
function ((s::FilteredEventTrain{T,E,F})(t::T, args...)::promote_type(T,E)) where {T,E,F}
    s_frozen = freeze(s, args...)
    return s_frozen(t)
end

function freeze(s::FilteredEventTrain{T,E,F}, max_range=nothing) where {T,E,F}
    TE = promote_type(T,E)
    F_frozen = freeze(s.F)

    function frozen(t::T)::TE
        sub_train = fromto(s.event_train, max_range==nothing ? :start : t-max_range, t)
        f= convert(Vector{TE}, F_frozen(t-sub_train.times))
        return f ⋅ sub_train.events
    end
end


####################
# Plotting recipes #
####################



@recipe f(e::EventTrain{T,E}, y=0.0) where {T,E<:Real} = ([e.times'; e.times'; e.times'][:], [fill(y, length(e.times))'; y+e.events'; fill(NaN, length(e.times))'][:])
@recipe f(e::EventTrain, y=0.0, height=1.0) = ([e.times'; e.times'; e.times'][:], [fill(y, length(e.times))'; fill(y+height, length(e.times))'; fill(NaN, length(e.times))'][:])

@recipe function f(e::FilteredEventTrain, xmin=:start, xmax=:end; show_events=false, show_kernels=false, stack_kernels=true, event_color=:black)
    if xmax==:end
        xmax = maximum(e.event_train.times)
    end
    if xmin==:start
        xmin = minimum(e.event_train.times)
    end

    xlims := (xmin, xmax)

    if show_events
        @series begin
            label := ""
            color := event_color
            e.event_train
        end
    end

    if show_kernels
        f = freeze(e.F)
        sub = fromto(e, :start, xmax)
        for (t,w) ∈ zip(sub.event_train.times, sub.event_train.events)
            if t >= xmax
                continue
            end

            if stack_kernels
                @series begin
                    linewidth := 1
                    linestyle := :dot
                    color := event_color
                    label := ""
                    (x->convert(Float64, fromto(sub, :start, t)(x))), t, xmax
                end
            else
                @series begin
                    linewidth := 1
                    linestyle := :dot
                    color := event_color
                    label := ""
                    (x->convert(Float64, x>=t ? (w*f(x-t)) : 0.0)), t, xmax
                end
            end
        end
    end

    @series begin
        label --> ""
        x->e(x), xmin, xmax
    end
end
end
