
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

const DATA_DIR_CK = expanduser("~/data/CK")
const MODEL_DIR = expanduser("~/data/models")
const RESIZERATIO = .5

include("view.jl")
incllude("utils.jl")





