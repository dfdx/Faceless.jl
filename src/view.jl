
function viewshape(img::Image, lms::Matrix{Float64})
    imgc, img2 = view(img)
    for i=1:size(lms, 1)
        annotate!(imgc, img2, AnnotationPoint(lms[i, 2], lms[i, 1], shape='.',
                                              size=4))
    end
    imgc, img2
end
viewshape(mat::Matrix{Float64}, lms::Shape) = viewshape(convert(Image, mat), lms)
