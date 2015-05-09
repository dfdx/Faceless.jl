
function viewshape(img::Image, lms::Matrix{Float64})
    imgc, img2 = view(img)
    for i=1:size(lms, 1)
        annotate!(imgc, img2, AnnotationPoint(lms[i, 2], lms[i, 1], shape='.',
                                              size=4))
    end
    imgc, img2
end
viewshape(mat::Matrix{Float64}, lms::Shape) = viewshape(convert(Image, mat), lms)


function nviewall(imgs, padding=10)
    @assert(length(imgs) > 0, "Need at least one image")
    h, w = size(imgs[1])
    n = length(imgs)
    rows = int(floor(sqrt(n)))
    cols = int(ceil(n / rows))
    halfpad = div(padding, 2)
    dat = zeros(rows * (h + padding), cols * (w + padding))
    for i=1:n
        ## wt = W[i, :]
        ## wt = reshape(wt, length(wt))
        ## wim = map_nonzeros(IMSIZE, wt, nzs)
        img = imgs[i]
        img = img ./ (maximum(img) - minimum(img))
        r = div(i - 1, cols) + 1
        c = rem(i - 1, cols) + 1
        dat[(r-1)*(h+padding)+halfpad+1 : r*(h+padding)-halfpad,
            (c-1)*(w+padding)+halfpad+1 : c*(w+padding)-halfpad] = imgs[i]
    end
    nview(dat)
    return dat
end
