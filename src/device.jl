function select_device! end
function get_warpsize end


function devices end
function device end
function deviceid end

function name end

device(backend::Backend, i::Integer) = devices(backend)[i]
device(x::SubArray) = device(parent(x))

function deviceid(x::AbstractArray)
    p = parent(x)
    p === x ? deviceid(device(x)) : deviceid(p)
end

get_warpsize(src::AbstractArray) = get_warpsize(device(src))