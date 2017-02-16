
import OpenCL: cl
import OpenCL.cl: CLArray, to_host
using CLBLAS
import CLBLAS: gemm 

const single_kernel = """
__kernel void single(
                     const __global float* x,
                     __global float* y)
 {
   const int i = get_global_id(0);
   y[i] = 1.0 / (1.0 + exp(-x[i]));
 }
"""

const exp_kernel = """
__kernel void exp_k(
                     const __global float* x,
                     __global float* y)
 {
   const int i = get_global_id(0);
   y[i] = exp(-x[i]);
 }
"""


const log_kernel = """
__kernel void log_k(
                     const __global float* x,
                     __global float* y)
 {
   const int i = get_global_id(0);
   y[i] = 1.0 + x[i];
 }
"""

const div_kernel = """
__kernel void div_k(
                     const __global float* x,
                     __global float* y)
 {
   const int i = get_global_id(0);
   y[i] = 1.0 / x[i];
 }
"""


function main2()
    x = rand(Float32, 50_000)
    y = similar(x)
    
    device, ctx, queue = cl.create_compute_context()    
    x_buff = cl.Buffer(Float32, ctx, (:r, :copy), hostbuf=x)
    y_buff = cl.Buffer(Float32, ctx, :w, length(x))
    
    p = cl.Program(ctx, source=single_kernel) |> cl.build!
    k = cl.Kernel(p, "single")
    ev = queue(k, size(x), nothing, x_buff, y_buff)
    @time begin
        for i=1:1_000
            ev = queue(k, size(x), nothing, x_buff, y_buff; wait_on=[ev])
        end
        cl.wait(ev)
    end

    exp_p = cl.Program(ctx, source=exp_kernel) |> cl.build!
    exp_k = cl.Kernel(exp_p, "exp_k")
    log_p = cl.Program(ctx, source=log_kernel) |> cl.build!
    log_k = cl.Kernel(log_p, "log_k")
    div_p = cl.Program(ctx, source=div_kernel) |> cl.build!
    div_k = cl.Kernel(div_p, "div_k")
    @time begin
        for i=1:1_000
            ev = queue(exp_k, size(x), nothing, x_buff, y_buff; wait_on=[ev])
            ev = queue(log_k, size(x), nothing, y_buff, y_buff; wait_on=[ev])
            ev = queue(div_k, size(x), nothing, y_buff, y_buff; wait_on=[ev])
        end
        cl.wait(ev)
    end

    
end


function main()
    device, ctx, queue = cl.create_compute_context()
    p = cl.Program(ctx, source=template_kernel) |> cl.build!
    W = rand(Float32, 128*16, 64*16)
    x = rand(Float32, 64*16, 1)
    # b = rand(Float32, 128, 1)
    y = zeros(Float32, 128*16, 1)
    d_W = CLArray(queue, W)
    d_x = CLArray(queue, x)
    # d_b = CLArray(queue, b)
    d_y = CLArray(queue, y)
    
    @time begin
        ev = nothing
        for i=1:1000
            ev = gemm!('N', 'N', Float32(1.0), d_W, d_x, Float32(0.0), d_y)
        end
        cl.wait(ev)
    end

    @time begin
        for i=1:1000
            gemm!('N', 'N', Float32(1.0), W, x, Float32(0.0), y)
        end
    end
end
