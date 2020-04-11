#pragma once
template<typename T, typename SeedT = std::default_random_engine::result_type>
auto generate_uniformly_distributed_vec(std::size_t count, T min, T max, SeedT seed = std::default_random_engine::default_seed)
{
    std::default_random_engine generator{ seed };
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    std::vector<T> vals(count);

    double scale = max - min;


    assert(scale > T{});
    assert(scale > T{});

    for (int i = 0; i < count; ++i)
    {
        vals[i] = min + static_cast<T>(scale * distribution(generator));
    }

    return vals;
}
