__kernel void gaussian_filter(__read_only image2d_t src,
                              __write_only image2d_t dst,
                              sampler_t sampler,
                              int width,
                              int height)
{
    float weights[9] = {
        1.0f, 2.0f, 1.0f,
        2.0f, 4.0f, 2.0f,
        1.0f, 2.0f, 1.0f,
    };
    int2 startPos = (int2)(get_global_id(0) - 1, get_global_id(1) - 1);
    int2 endPos   = (int2)(get_global_id(0) + 1, get_global_id(1) + 1);
    int2 outPos   = (int2)(get_global_id(0),     get_global_id(1));

    if (outPos.x > width || outPos.y > height)
        return;


    int w = 0;
    float4 color = (float4) {0.0f, 0.0f, 0.0f, 0.0f};
    for(int x = startPos.x; x <= endPos.x; ++x) {
        for(int y = startPos.y; y <= endPos.y; ++y) {
            color += weights[w++] * read_imagef(src, sampler, (int2)(x, y));
        }
    }
    color /= 16.0f;
    write_imagef(dst, outPos, color);
}
