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
    terms::Vector{<:Function}
end
LaplaceFilter(t::Vector{<:Function}) = LaplaceFilter{Float64}(t)
LaplaceFilter(t::Function) = LaplaceFilter{Float64}(Function[t])

# Convolve LaplaceFilters by convolution theorem
Base.:∘(F1::LaplaceFilter, F2::LaplaceFilter) = LaplaceFilter([F1.terms; F2.terms])

# Evaluate LaplaceFilter
function ℒ(F::LaplaceFilter)
    function LLaplace(s::V)::V where V
        prod(call_all(F.terms,s))
    end
end

function freeze(F::LaplaceFilter{T}; method=talbot, accuracy=32) where T
    ilt = ILt(ℒ(F), method, accuracy)
    function frozen(x::T)::T
        ilt(x)
    end
end

((F::LaplaceFilter{T})(t::T)::T) where T = freeze(F)(t)

# Add LaplaceFilters
Base.:+(F1::LaplaceFilter, F2::LaplaceFilter) = LaplaceFilter([s->convert(Float64, ℒ(F1)(s)+ℒ(F2)(s))])
# Scale LaplaceFilters
Base.:*(a, F::LaplaceFilter) = LaplaceFilter([x->a; F.terms])
# Sign-Invert LaplaceFilters
Base.:-(F1::LaplaceFilter) = (-1) * F1
# Subtract LaplaceFilters
Base.:-(F1::LaplaceFilter, F2::LaplaceFilter) = F1 + (-F2)

### GammaFilter
struct GammaFilter{T,A,K} <: Filter{T, Float64}
    α::A
    k::K
end
GammaFilter(α::A, k::K) where {A,K} = GammaFilter{Float64,A,K}(α, k)

# Convolve GammaFilters (convert to LaplaceFilter if necessary)
function Base.:∘(F1::GammaFilter, F2::GammaFilter)
    if F1.α == F2.α
        GammaFilter(F1.α, F1.k+F2.k)
    else
        LaplaceFilter(F1)∘LaplaceFilter(F2)
    end
end


# Evaluate GammaFilter
function ℒ(F::GammaFilter)
    function Lgamma(s::V)::V where V
        (F.α/(s+F.α))^F.k
    end
end

freeze(F::GammaFilter) = F
((F::GammaFilter{T,A,K})(t::T)::promote_type(T,A,K)) where {T,A,K} = (F.α^F.k)/gamma(F.k)*t^(F.k-1)*exp(-F.α*t)

# Convert GammaFilter to LaplaceFilter
LaplaceFilter(F::GammaFilter) = LaplaceFilter([ℒ(F)])

# ExponentialFilter
ExponentialFilter(α=1.0) = GammaFilter(α, 1)


#####################################
# EventTrain and FilteredEventTrain #
#####################################

struct EventTrain{T,E}
    times::Vector{T}
    events::E
    function EventTrain(times::Vector{T}, events::E = 1) where {T,E}
        if ~issorted(times)
            ord = sortperm(times)
            times = times[ord]

            if isa(events, Vector)
                events = events[ord]
            end
        end
        new{T,E}(times, events)
    end
end


struct FilteredEventTrain{T,E,F<:Filter}
    event_train::EventTrain{T,E}
    F::F
end

#FilteredEventTrain(e::EventTrain{T,E,F}, F::F) where {T,E,F<:Filter} = FilteredEventTrain{T,E,F}(e, F)

Base.length(s::EventTrain) = length(s.times)
Base.length(s::FilteredEventTrain) = length(s.event_train)
Base.endof(s::EventTrain) = length(s)
Base.endof(s::FilteredEventTrain) = length(s)

"""Merge two event trains (times must be SORTED!)"""
function Base.:+(s::EventTrain{S,E}, t::EventTrain{T, F}) where {S,T,E,F}
    ST = promote_type(S,T)
    new_times = ST[]

    EF = if (E<:Vector) | (F<:Vector)
        # We need a vector to hold the events
        eltype_E = E<:Vector ? eltype(E) : E
        eltype_F = F<:Vector ? eltype(F) : F
        Vector{promote_type(eltype_E,eltype_F)}
    elseif s.events != t.events
        # the events of s and t are different and must thus be stored in a vector
        Vector{promote_type(E,F)}
    else
        # We don't need a vector to hold the events
        promote_type(E,F)
    end

    new_events = (EF<:Vector) ? EF() : EF(s.events)

    (i1,i2) = (1,1)
    (n1,n2) = length.((s, t))
    while (i1 <= n1) & (i2 <= n2)
        if(s.times[i1] < t.times[i2])
            push!(new_times, s.times[i1])

            if EF<:Vector
                push!(new_events, (E<:Vector) ? s.events[i1] : s.events)
            end
            i1 += 1
        else
            push!(new_times, t.times[i2])
            if EF<:Vector
                push!(new_events, (F<:Vector) ? t.events[i2] : t.events)
            end
            i2 += 1
        end
    end

    while i1 <= n1
        push!(new_times, s.times[i1])
        if EF<:Vector
            push!(new_events,(E<:Vector) ? s.events[i1] : s.events)
        end
        i1 += 1
    end

    while i2 <= n2
        push!(new_times, t.times[i2])
        if EF<:Vector
            push!(new_events,(F<:Vector) ? t.events[i2] : t.events)
        end
        i2 += 1
    end

    return EventTrain(new_times, new_events)
end
# Add FilteredEventTrains
function Base.:+(s1::FilteredEventTrain,s2::FilteredEventTrain)
    return if s1.F == s2.F
        FilteredEventTrain(s1.event_train+s2.event_train, s1.F)
    elseif s1.event_train == s2.event_train
        FilteredEventTrain(s1.event_train, s1.F+s2.F)
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
Base.:*(a, s::EventTrain) = EventTrain(s.times, a.*s.events)
# Scale FilteredEventTrain
Base.:*(a, s::FilteredEventTrain) = FilteredEventTrain(a.*s.event_train, s.F)

# Subsample events of a (Filtered)EventTrain
Base.getindex(s::EventTrain{T,<:Vector}, r) where {T} = EventTrain(s.times[r], s.events[r])
Base.getindex(s::EventTrain, r) = EventTrain(s.times[r], s.events)
Base.getindex(s::FilteredEventTrain, r) = FilteredEventTrain(s.event_train[r], s.F)

# Slice a time interval out of a (Filtered)EventTrain
function fromto(s::EventTrain, from, to)
    if from == :start
        from = minimum(s.times)-1.0
    end
    if to == :end
        to = maximum(s.times)+1.0
    end

    r = searchsortedfirst(s.times, from):searchsortedlast(s.times, to)
    return s[r]
end
fromto(s::FilteredEventTrain, from, to) = FilteredEventTrain(fromto(s.event_train, from, to), s.F)

# Shift a (Filtered)EventTrain in time
function delay(s::EventTrain{T, E}, dt) where {T,E}
    new_times = s.times+dt
    ord = sortperm(new_times)
    EventTrain(new_times[ord], (E<:Vector) ? s.events[ord] : s.events)
end

delay(s::FilteredEventTrain, dt) = FilteredEventTrain(delay(s.event_train, dt), s.F)

# Convolve two EventTrains
Base.:∘(e1::EventTrain{S, E}, e2::EventTrain{T, F}) where {S,T,E,F} = ((E<:Vector) | (F<:Vector)) ? mapreduce(dtw->delay(dtw[2]*e1, dtw[1]), +, zip(e2.times,e2.events)) : mapreduce(dt->delay(e2.events*e1, dt), +, e2.times)

# Convolve two FilteredEventTrains
Base.:∘(e1::FilteredEventTrain, e2::FilteredEventTrain) = FilteredEventTrain(e1.event_train∘e2.event_train, e1.F∘e2.F)
# Convolve one FilteredEventTrain and one EventTrain
Base.:∘(e1::FilteredEventTrain, e2::EventTrain) = FilteredEventTrain(e1.event_train∘e2, e1.F)
Base.:∘(e1::EventTrain, e2::FilteredEventTrain) = e2∘e1

################################################
# Interaction between filters and event trains #
################################################


# Use linearity to chain filters
Base.:∘(F::Filter, e::FilteredEventTrain) = FilteredEventTrain(e.event_train, F∘e.F)

# Apply Filter to event-train
Base.:∘(F::Filter, e::EventTrain) = FilteredEventTrain(e, F)

# Evaluate FilteredEventTrain
function ((s::FilteredEventTrain{T,E,F})(t::T, args...)) where {T,E,F}
    s_frozen = freeze(s, args...)
    return s_frozen(t)
end

function freeze(s::FilteredEventTrain{T,E,F}, max_range=nothing) where {T,E,F}
    F_frozen = freeze(s.F)

    function frozen(t::T)
        sub_train = fromto(s.event_train, max_range==nothing ? :start : t-max_range, t)
        return sum(F_frozen.(t.-sub_train.times).*sub_train.events)
    end
end


####################
# Plotting recipes #
####################


@recipe function f(f::Filter, xmin, xmax)
    x->f(x), xmin, xmax
end

@recipe function f(e::EventTrain, y=0.0)
    y = (length(y) == length(e.times)) ? y : fill(y, length(e.times))
    ([e.times'; e.times'; e.times'][:], [y'; (y+e.events)'; fill(NaN, length(e.times))'][:])
end

@recipe function f(e::FilteredEventTrain{T,E,F}, xmin=:start, xmax=:end; show_events=false, show_kernels=false, stack_kernels=true, event_color=:black) where {T,E,F}
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

        if stack_kernels
            for t ∈ sub.event_train.times
                if t<xmax
                    @series begin
                        linewidth := 1
                        linestyle := :dot
                        color := event_color
                        label := ""
                        (x->convert(Float64, fromto(sub, :start, t)(x))), t, xmax
                    end
                end
            end
        elseif (E<:Vector)
            for (t,w) ∈ zip(sub.event_train.times, sub.event_train.events)
                if t<xmax
                    @series begin
                        linewidth := 1
                        linestyle := :dot
                        color := event_color
                        label := ""
                        (x->convert(Float64, x>=t ? (w*f(x-t)) : 0.0)), t, xmax
                    end
                end
            end
        else
            for t ∈ sub.event_train.times
                if t<xmax
                    @series begin
                        linewidth := 1
                        linestyle := :dot
                        color := event_color
                        label := ""
                        (x->convert(Float64, x>=t ? (sub.event_train.events*f(x-t)) : 0.0)), t, xmax
                    end
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
