#include "quad_integration/integration_method.hpp"

namespace {

template <typename Scalar>
void load_integration_methods()
{
    [[maybe_unused]] quad_integration::QuadParameters<Scalar>
        quad_parameters{};

    using QuanticsParameters =
        quad_integration::QuanticsFitParameters<Scalar>;
    using RunnerParameters =
        typename QuanticsParameters::Quantics::RunnerParam;

    [[maybe_unused]] QuanticsParameters quantics_parameters{
        RunnerParameters{1, 1, 1}, {}};
}

} // namespace

int main()
{
    load_integration_methods<double>();
    load_integration_methods<dd_128>();
    return 0;
}
