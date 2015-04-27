
using PiecewiseAffineTransforms
using ActiveAppearanceModels
using MultivariateStats
using Images
using ImageView

include("data.jl")

# 0. read images
# 1. train AAM
# 2. collect masked
# 3. map masked
# 4. apply PCA
# 5. apply RBM



function main()
    imgs = read_images_ck(imgdir, count=1000)
    shapes = read_shapes_ck(shapedir, count=1000)
    aam = AAModel()
    @time train(aam, imgs, shapes)
end
