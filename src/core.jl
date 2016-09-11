
using PiecewiseAffineTransforms
using MultivariateStats
using ActiveAppearanceModels
using Images
using ImageView
using Colors
using FaceDatasets
using Boltzmann
using HDF5
using JLD
using Clustering

const DATA_DIR = expanduser("~/data")
const DATA_DIR_PUT = joinpath(DATA_DIR, "PUT")
const DATA_DIR_CK = joinpath(DATA_DIR, "CK")
const MODEL_DIR = expanduser("~/data/models")
const RESIZERATIO = .25

include("view.jl")
include("utils.jl")
