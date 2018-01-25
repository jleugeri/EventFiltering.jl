# EventFiltering.jl
## Events and Filtering.

Defines Filters and Events.
Events are assumed to be pairs of times and values.
Filters are operators that map sequences of events onto functions of time.
This design is for example suitable for working with neural spike trains and neural filter responses.

## Filters
Thus far, `LaplaceFilter`s and `GammaFilter`s are implemented.

### LaplaceFilters
`LaplaceFilter`s are defined by their Laplace Transform.
They are closed under linear operations (scaling, addition, sign-inversion, subtraction) as well as convolution.

### GammaFilters
`GammaFilter`s are specific filters with a shape equal to the PDF of a Gamma-Distribution.
For equal constants `α`, they are closed under convolution; for different `\alpha`s,
they are automatically converted to `LaplaceFilter`s before convolution.
`GammaFilter`s are very efficient, yet less flexible than `LaplaceFilter`s.
For operations like addition, subtraction, negation etc. they must be manually converted to `LaplaceFilter`s.


## Events
Events are represented in either an `EventTrain`, or a `FilteredEventTrain`.
The former represents pure sequences of events, whereas the latter represents a pair of event sequence and a filter.
Both support subsampling of spikes via indexing or slicing of a time-interval using `fromto`.

`FilteredEventTrain`s can be evaluated as functions. If a lot of function evaluations are required,
`freeze`ing the `FilteredEventTrain` first can speed up the process a lot -- in particular if the filter is truncated to a small `max_range`.

Basic plotting of `(Filtered)EventTrain`s is provided via a `Plots` user recipe.

### Convolutions of event trains
`(Filtered)EventTrain`s can be convolved as well.
This can be useful for composing complex, structured event sequences.

## Usage examples:

```julia
# Working with event trains:
e1 = EventTrain([1.0, 2.0, 3.0])
e2 = 2*EventTrain([0.5, 1.5, 2.5, 3.5])
e3 = 1.5 * e1 + e2

F = ExponentialFilter(1.0)
X = F∘F∘e3
M = X - F∘X

# Plotting:
using Plots
pyplot()

plot(X, 0,10, color=:blue, event_color=:blue, show_events=false, show_kernels=true)
plot!(F∘X, 0, 10, color=:red, show_events=true, show_kernels=true)
plot!(M, 0, 10, color=:black)
```

For more examples, see the the [docs](./docs/) folder.
