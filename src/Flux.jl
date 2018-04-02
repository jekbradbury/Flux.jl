__precompile__()

module Flux

# Zero Flux Given

using Requires, Reexport
using MacroTools: @forward

export Chain, Dense, RNN, LSTM, GRU, Conv2D,
  Dropout, LayerNorm, BatchNorm,
  SGD, ADAM, Momentum, Nesterov, AMSGrad,
  param, params, mapleaves

@reexport using NNlib

# include("tracker/Tracker.jl")
# using .Tracker
# export Tracker
# import .Tracker: data

module Tracker
param(x) = x
data(x) = x
function back! end
abstract type TrackedArray{T,N} end
TrackedVector{T} = TrackedArray{T,1}
TrackedMatrix{T} = TrackedArray{T,2}
end

import .Tracker: param, TrackedArray, TrackedMatrix, TrackedVector

include("optimise/Optimise.jl")
using .Optimise

include("oset.jl")
include("utils.jl")
include("onehot.jl")
include("treelike.jl")

include("layers/stateless.jl")
include("layers/basic.jl")
include("layers/conv.jl")
include("layers/recurrent.jl")
include("layers/normalisation.jl")

# include("data/Data.jl")

# include("jit/JIT.jl")

@require CuArrays include("cuda/cuda.jl")

end # module
