const sampler_t sampler = CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

/*
 */

__kernel void gray(__read_only image2d_t src, __write_only image2d_t dst)
{
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);

    if (x >= get_image_width(src) || y >= get_image_height(src))
        return;

    float3 pixel = read_imagef(src, sampler, (int2)(x, y)).xyz;
    float3 g = 0.11 * pixel.x + 0.59 * pixel.y + 0.30 * pixel.z;

    write_imagef(dst, (int2)(x, y), (float4)(g.x, g.y, g.z, 1.0f));
}
