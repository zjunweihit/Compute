const sampler_t sampler = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

/*
 *  3 x 3 kernel to convolve with original image to compute derivatives.
 *
 *       [ -1   0  +1 ]
 *  Gx = [ -2   0  +2 ]
 *       [ -1   0  +1 ]

 *       [ -1  -2  -1 ]
 *  Gy = [  0   0   0 ]
 *       [ +1  +2  +1 ]
 *
 *       [ p00  p01  p20 ]
 *   p = [ p10  p11  p12 ]
 *       [ p20  p21  p22 ]
 */

__kernel void sobel(__read_only image2d_t src, __write_only image2d_t dst)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);

    if (x >= get_image_width(src) || y >= get_image_height(src))
        return;

    float4 p00 = read_imagef(src, sampler, (int2)(x - 1, y - 1));
    float4 p01 = read_imagef(src, sampler, (int2)(x - 1, y));
    float4 p02 = read_imagef(src, sampler, (int2)(x - 1, y + 1));

    float4 p10 = read_imagef(src, sampler, (int2)(x    , y - 1));
    float4 p12 = read_imagef(src, sampler, (int2)(x    , y + 1));

    float4 p20 = read_imagef(src, sampler, (int2)(x + 1, y - 1));
    float4 p21 = read_imagef(src, sampler, (int2)(x + 1, y));
    float4 p22 = read_imagef(src, sampler, (int2)(x + 1, y + 1));

    float3 gx = -p00.xyz + p20.xyz
                - 2.0f * (p01.xyz - p12.xyz)
                -p20.xyz + p22.xyz;
    float3 gy = -p00.xyz + p20.xyz
                + 2.0f * (p21.xyz - p01.xyz)
                -p02.xyz + p22.xyz;
    float3 g = native_sqrt(gx * gx + gy * gy);

    write_imagef(dst, (int2)(x, y), (float4)(g.x, g.y, g.z, 1.0f));
}
