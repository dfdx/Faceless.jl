
const DATA_DIR_CK = expanduser("~/work/ck-data")


# TODO: remove if everything is ok with cogn-kanade dataset
function checklength()
    imgdir = joinpath(DATA_DIR_CK, "cohn-kanade-images")
    imgfiles = sort(walkdir(imgdir, pred=(p -> endswith(p, ".png"))))
    shapedir = joinpath(DATA_DIR_CK, "Landmarks")
    shapefiles = sort(walkdir(shapedir, pred=(p -> endswith(p, ".txt"))))
    i = 1
    j = 1
    while i < length(imgfiles) && j < length(shapefiles)
        expectedshapefile = replace(imgfiles[i], "cohn-kanade-images", "Landmarks")
        expectedshapefile = replace(expectedshapefile, ".png", "_landmarks.txt")
        if basename(shapefiles[j]) == "S109_002_00000008_landmarks.txt"
            j += 1
            continue  # note: i is not increased on this iteration
        end
        if expectedshapefile != shapefiles[j]
            println("problem: i=$i;j=$j")
            break
        end
        i += 1
        j += 1
    end
end
