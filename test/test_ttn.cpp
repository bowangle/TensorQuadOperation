#include "type_double_double.h"
#include "type_float128_boost.h"
#include "type_int128.h"
#include "ttn_base.hpp"

#include <QTgrid-quad/grid.h>
#include <xfac_quad/tree/tensor_tree_ci.hpp>

#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <type_traits>

namespace {

using Clock = std::chrono::steady_clock;
constexpr int dimensions = 3;
constexpr int bits_per_variable = 3;
constexpr int physical_count = dimensions * bits_per_variable;
constexpr int junction = physical_count;
const std::string output_dir = "test/output/ttn/";
std::size_t checks = 0;

void check(bool condition, const std::string& message)
{
    ++checks;
    if (!condition)
        throw std::runtime_error(message);
}

template<typename Exception, typename Function>
void check_throws(Function&& function, const std::string& message)
{
    bool caught = false;
    try {
        function();
    }
    catch (const Exception&) {
        caught = true;
    }
    check(caught, message);
}

template<typename Function>
void timed(const std::string& label, Function&& function)
{
    const auto start = Clock::now();
    function();
    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    std::cout << "[" << label << "] OK (" << ms << " ms)\n";
}

template<typename Real>
double to_double(const Real& value)
{
    if constexpr (std::is_same_v<Real, dd_128>)
        return value.x[0];
    else
        return static_cast<double>(value);
}

template<typename T>
using RealOf = typename Eigen::NumTraits<T>::Real;

template<typename T>
RealOf<T> abs2(const T& value)
{
    return value.real() * value.real() + value.imag() * value.imag();
}

template<typename T>
RealOf<T> numeric_tolerance()
{
    return RealOf<T>(1000000) * Eigen::NumTraits<RealOf<T>>::epsilon();
}

// Both functions genuinely depend on all three coordinates. They have low
// tree ranks, allowing a small exhaustive grid and inexpensive precision tests.
template<typename T>
T function_1(const std::vector<RealOf<T>>& point)
{
    using std::cos;
    const auto& x = point.at(0);
    const auto& y = point.at(1);
    const auto& z = point.at(2);
    return T(cos(x) * cos(RealOf<T>(3) * y) * cos(RealOf<T>(3) * z),
             cos(RealOf<T>(2) * x + y - z));
}

template<typename T>
T function_2(const std::vector<RealOf<T>>& point)
{
    const auto& x = point.at(0);
    const auto& y = point.at(1);
    const auto& z = point.at(2);
    return T(RealOf<T>(1) + x*y + y*z + z*x,
             -(x + RealOf<T>(2)*y + RealOf<T>(3)*z));
}

template<typename Real, typename Sint>
qtgrid::MultQTGrid<Real, Sint> make_grid()
{
    return {{Real(-1), Real(-1), Real(-1)}, {Real(1), Real(1), Real(1)},
            {bits_per_variable, bits_per_variable, bits_per_variable}};
}

// makeTuckerTree uses axis + bit*dimensions, whereas MultQTGrid uses
// axis*bits_per_variable + bit. Preserve the topology and neighbor/axis order.
xfac_quad::TopologyTree make_grouped_tucker_tree()
{
    const auto source = xfac_quad::makeTuckerTree(dimensions, bits_per_variable);
    const auto grouped_id = [](int node) {
        return node >= 0 && node < physical_count
            ? (node % dimensions) * bits_per_variable + node / dimensions : node;
    };
    xfac_quad::TopologyTree tree(grouped_id(source.root));
    for (int node : source.nodes)
        tree.nodes.insert(grouped_id(node));
    for (const auto& [node, neighbors] : source.neigh)
        for (int neighbor : neighbors.from_int())
            tree.neigh[grouped_id(node)].push_back(grouped_id(neighbor));
    check(xfac_quad::isValid(tree), "grouped Tucker topology is valid");
    return tree;
}

std::vector<std::vector<int>> all_points()
{
    // Exhaust all 8^3 coordinate combinations, including every leaf's bits.
    std::vector<std::vector<int>> points(std::size_t(1) << physical_count,
                                         std::vector<int>(physical_count));
    for (std::size_t code = 0; code < points.size(); ++code)
        for (int bit = 0; bit < physical_count; ++bit)
            points[code][bit] = static_cast<int>((code >> bit) & 1);
    return points;
}

template<typename T>
void check_values(const std::string& label, const std::vector<T>& values,
                  const std::vector<T>& reference, RealOf<T> tolerance)
{
    check(values.size() == reference.size(), label + ": result count");
    RealOf<T> max_error2 = 0;
    RealOf<T> scale2 = 1;
    for (std::size_t i = 0; i < values.size(); ++i) {
        max_error2 = std::max(max_error2, abs2(T(values[i] - reference[i])));
        scale2 = std::max(scale2, abs2(reference[i]));
    }
    std::cout << "  " << label << ": scaled max error = "
              << std::sqrt(to_double(max_error2 / scale2)) << "\n";
    check(max_error2 <= tolerance*tolerance*scale2, label + ": values preserved");
}

template<typename T>
RealOf<T> relative_error2(const std::vector<T>& values, const std::vector<T>& reference)
{
    check(values.size() == reference.size(), "dense error: result count");
    RealOf<T> error2 = 0;
    RealOf<T> norm2 = 0;
    for (std::size_t i = 0; i < values.size(); ++i) {
        error2 += abs2(T(values[i] - reference[i]));
        norm2 += abs2(reference[i]);
    }
    check(norm2 > RealOf<T>(0), "nonzero reference norm");
    return error2 / norm2;
}

template<typename T>
void check_same_body(const xfac_quad::TensorTree<T>& a, const xfac_quad::TensorTree<T>& b)
{
    std::ostringstream topology_a, topology_b;
    a.tree.save(topology_a);
    b.tree.save(topology_b);
    check(topology_a.str() == topology_b.str(), "topology and axis ordering preserved");
    check(a.M.size() == b.M.size(), "tensor count preserved");
    for (std::size_t node = 0; node < a.M.size(); ++node) {
        const auto& x = a.M[node];
        const auto& y = b.M[node];
        check(x.n_rows == y.n_rows && x.n_cols == y.n_cols && x.n_slices == y.n_slices,
              "tensor shape preserved");
        check(x.data == y.data, "tensor data preserved exactly");
    }
}

template<typename T>
void check_metadata(const TTN<T>& ttn)
{
    const auto& body = ttn.get_TTN_body();
    const auto shapes = ttn.get_shape();
    const auto counts = ttn.compute_nb_value_core();
    check(shapes.size() == body.M.size(), "get_shape tensor count");
    check(counts.size() == body.M.size(), "per-core value count size");
    Eigen::Index total = 0;
    for (std::size_t node = 0; node < body.M.size(); ++node) {
        const auto& tensor = body.M[node];
        check(shapes[node] == std::make_tuple(tensor.n_rows, tensor.n_cols, tensor.n_slices),
              "get_shape uses stored axis order");
        const auto size = static_cast<Eigen::Index>(tensor.data.size());
        check(counts[node] == size, "per-core count matches actual storage");
        total += size;
    }
    check(ttn.compute_tot_nb_value() == total, "total value count matches actual storage");
    std::vector<int> expected_chi;
    for (const auto& [from, neighbors] : body.tree.neigh)
        for (int to : neighbors.from_int()) {
            if (to == -1 || from >= to) continue;
            const Eigen::Index rank = xfac_quad::cubeToMat_R(body.M.at(from), neighbors.pos(to)).cols();
            const Eigen::Index other_rank = xfac_quad::cubeToMat_R(
                body.M.at(to), body.tree.neigh.at(to).pos(from)).cols();
            check(rank == other_rank, "neighboring bond dimensions agree");
            expected_chi.push_back(static_cast<int>(rank));
        }
    check(ttn.compute_list_chi() == expected_chi, "bond list ordering and dimensions");
    check(expected_chi.size() + 1 == body.M.size(), "one chi per undirected tree edge");
    const Eigen::Index max_chi = expected_chi.empty() ? 0
        : *std::max_element(expected_chi.begin(), expected_chi.end());
    check(ttn.get_chi() == max_chi, "get_chi matches maximum virtual dimension");
}

template<typename T>
void check_canonical(const TTN<T>& ttn)
{
    check(ttn.get_w() != -1, "canonical centre is defined");
    const auto& body = ttn.get_TTN_body();
    using Matrix = typename xfac_quad::Tensor3D<T>::MatrixX;
    for (const auto& [from, to] : body.tree.leavesToRoot(ttn.get_w())) {
        const Matrix matrix = xfac_quad::cubeToMat_R(body.M.at(from), body.tree.neigh.at(from).pos(to));
        const Matrix gram = matrix.adjoint() * matrix;
        const RealOf<T> error = (gram - Matrix::Identity(gram.rows(), gram.cols())).norm();
        const RealOf<T> tolerance = numeric_tolerance<T>() * RealOf<T>(gram.rows());
        check(error <= tolerance, "isometry toward centre at node " + std::to_string(from));
    }
    check_metadata(ttn);
}

void prepare_fixture(int function_id, const std::string& filename)
{
    using T = std::complex<float128>;
    const auto grid = make_grid<float128, long long>();
    const auto tree = make_grouped_tucker_tree();
    const auto function = function_id == 1 ? function_1<T> : function_2<T>;
    xfac_quad::TensorTreeCIParam<T> param;
    param.bondDim = 16;
    param.reltol = float128(100) * Eigen::NumTraits<float128>::epsilon();
    param.fullPiv = true;
    param.pivot1 = grid.coord_to_id({float128(0), float128(0), float128(0)});
    xfac_quad::TensorTreeCI<T> ci(
        [&grid, function](std::vector<int> id) { return function(grid.id_to_coord(id)); },
        tree, std::vector<int>(physical_count, 2), param);
    ci.iterate(6);
    const auto points = all_points();
    std::vector<T> values, reference;
    for (const auto& point : points) {
        values.push_back(ci.tt.eval(point));
        reference.push_back(function(grid.id_to_coord(point)));
    }
    check_values("TCI analytic reference", values, reference, numeric_tolerance<T>());
    ci.tt.save(filename);
}

template<typename T, typename Sint>
void test_ttn(const std::string& type_name, int function_id, const std::string& filename)
{
    using Real = RealOf<T>;
    const auto start = Clock::now();
    std::cout << "\n========================================\nTesting " << type_name
              << ", f" << function_id << "\n========================================\n";
    const auto grid = make_grid<Real, Sint>();
    const auto points = all_points();
    const TTN<T> raw(filename, 0, Real(-1), -1);
    const auto& original = raw.get_TTN_body();
    const auto reference = raw.eval_list(points);
    std::vector<T> analytic;
    for (const auto& point : points) {
        const auto coordinate = grid.id_to_coord(point);
        check(grid.coord_to_id(coordinate) == point, "mqtgrid coordinate roundtrip");
        analytic.push_back(function_id == 1 ? function_1<T>(coordinate) : function_2<T>(coordinate));
    }

    timed("load, eval, eval_list, accessors", [&] {
        check(raw.get_w() == -1, "constructor supports undefined centre");
        check(raw.get_max_bond_dim() == 0 && raw.get_reltol() == Real(-1), "default parameters");
        check_metadata(raw);
        check_values("analytic three-variable function", reference, analytic, numeric_tolerance<T>());
        for (std::size_t i = 0; i < points.size(); ++i)
            check(reference[i] == raw.eval(points[i]), "eval_list agrees with serial eval");
        check(raw.eval_list({}).empty(), "empty eval_list");
        TTN<T> from_body(original, 7, Real(1)/Real(1000), -1);
        check_same_body(original, from_body.get_TTN_body());
        check(from_body.get_max_bond_dim() == 7 && from_body.get_reltol() == Real(1)/Real(1000),
              "explicit constructor parameters");
        TTN<T> loaded(filename);
        check(loaded.get_w() == 0, "filename constructor initialises centre");
        check_canonical(loaded);
        for (const auto& [a, b, c] : loaded.get_shape())
            std::cout << "(" << a << "," << b << "," << c << ") ";
        std::cout << "chi=" << loaded.get_chi() << "\n";
    });

    timed("initialise and canonify at every physical and artificial node", [&] {
        for (const auto& [node, neighbors] : original.tree.neigh) {
            (void)neighbors;
            TTN<T> ttn(original, 0, Real(-1), -1);
            ttn.canonify_to_center(node);
            check(ttn.get_w() == node, "canonify sets requested centre");
            check_canonical(ttn);
            check_values("centre " + std::to_string(node), ttn.eval_list(points), reference, numeric_tolerance<T>());
            TTN<T> constructed(original, 0, Real(-1), node);
            check(constructed.get_w() == node, "constructor initialises requested centre");
            check_canonical(constructed);
        }
    });

    timed("move centre across branches and back", [&] {
        TTN<T> ttn(original, 0, Real(-1), -1);
        // Also exercises the fallback from an undefined centre.
        for (const auto& [node, neighbors] : original.tree.neigh) {
            (void)neighbors;
            ttn.move_canonical_center(node);
            check(ttn.get_w() == node, "move reaches requested centre");
            check_canonical(ttn);
        }
        for (auto it = original.tree.neigh.rbegin(); it != original.tree.neigh.rend(); ++it) {
            ttn.move_canonical_center(it->first);
            check_canonical(ttn);
        }
        check_values("all centre moves", ttn.eval_list(points), reference, numeric_tolerance<T>());
        const auto snapshot = ttn.get_TTN_body();
        ttn.move_canonical_center(ttn.get_w());
        ttn.canonify_to_center(-1);
        check_same_body(snapshot, ttn.get_TTN_body());
    });

    timed("centre movement leaves off-path tensors untouched", [&] {
        TTN<T> ttn(original, 0, Real(-1), 0);
        const auto snapshot = ttn.get_TTN_body();
        const int target = bits_per_variable;
        std::map<int, int> parent;
        for (const auto& [from, to] : original.tree.leavesToRoot(target)) parent.emplace(from, to);
        std::set<int> path{target};
        for (int node = ttn.get_w(); node != target; node = parent.at(node)) path.insert(node);
        ttn.move_canonical_center(target);
        for (const auto& [node, neighbors] : original.tree.neigh) {
            (void)neighbors;
            if (!path.contains(node))
                check(snapshot.M.at(node).data == ttn.get_TTN_body().M.at(node).data,
                      "off-path tensor unchanged");
        }
        check_canonical(ttn);
        check_values("cross-branch move", ttn.eval_list(points), reference, numeric_tolerance<T>());
    });

    timed("compress_svd without truncation", [&] {
        for (int center : {-1, 0, bits_per_variable, junction}) {
            TTN<T> ttn(original, 2, Real(1)/Real(10), center);
            const auto chi_before = ttn.get_chi();
            // Defaults must not silently use the stored cap/tolerance.
            ttn.compress_svd();
            check(ttn.get_w() == (center == -1 ? original.tree.root : center), "compression restores centre");
            check(ttn.get_chi() <= chi_before, "compression does not grow bonds");
            check(ttn.get_max_bond_dim() == 2 && ttn.get_reltol() == Real(1)/Real(10),
                  "compression leaves stored parameters unchanged");
            check_canonical(ttn);
            check_values("untruncated SVD", ttn.eval_list(points), reference, numeric_tolerance<T>());
        }
    });

    timed("compress_svd tolerance and rank-cap sweeps", [&] {
        TTN<T> baseline(original, 0, Real(-1), junction);
        for (Real tolerance : {Real(100)*Eigen::NumTraits<Real>::epsilon(), Real(1)/Real(100)}) {
            auto ttn = baseline;
            ttn.compress_svd(tolerance, 0);
            check(ttn.get_w() == junction, "tolerance compression preserves centre");
            check_canonical(ttn);
            const Real error2 = relative_error2(ttn.eval_list(points), reference);
            // reltol is local, not global: use the conservative sum of per-edge bounds.
            const Real bound = Real(original.M.size() - 1)*tolerance + numeric_tolerance<T>();
            check(error2 <= bound*bound, "dense error within accumulated local tolerance bound");
            std::cout << "  reltol=" << to_double(tolerance) << " relative L2 error="
                      << std::sqrt(to_double(error2)) << " chi=" << ttn.get_chi() << "\n";
        }
        check(baseline.get_chi() > 1, "fixture exercises nontrivial ranks");
        for (int cap = std::min<int>(4, static_cast<int>(baseline.get_chi())); cap >= 1; --cap) {
            auto ttn = baseline;
            ttn.compress_svd(Real(-1), cap);
            check(ttn.get_chi() <= cap, "rank cap enforced on every bond");
            check(ttn.get_w() == junction, "rank-cap compression preserves centre");
            check_canonical(ttn);
            std::cout << "  cap=" << cap << " chi=" << ttn.get_chi() << " relative L2 error="
                      << std::sqrt(to_double(relative_error2(ttn.eval_list(points), reference))) << "\n";
        }
        check_same_body(original, raw.get_TTN_body());
        check_values("copy independence", baseline.eval_list(points), reference, numeric_tolerance<T>());
    });

    timed("optional QR+SVD uses stored rank cap", [&] {
        TTN<T> ttn(original, 1, Real(-1), -1);
        ttn.canonify_to_center(junction, true);
        check(ttn.get_chi() == 1, "canonify compresses using stored cap");
        check_canonical(ttn);
        TTN<T> moving(original, 1, Real(-1), 0);
        const int target = bits_per_variable;
        moving.move_canonical_center(target, true);
        check_canonical(moving);
        std::map<int, int> parent;
        for (const auto& [from, to] : original.tree.leavesToRoot(target)) parent.emplace(from, to);
        for (int node = 0; node != target; node = parent.at(node)) {
            const auto& body = moving.get_TTN_body();
            check(xfac_quad::cubeToMat_R(body.M.at(node), body.tree.neigh.at(node).pos(parent.at(node))).cols() == 1,
                  "centre movement compresses each traversed bond using stored cap");
        }
    });

    timed("save/reload filename and stream roundtrips", [&] {
        const std::string saved = output_dir + "roundtrip_" + type_name + "_f" + std::to_string(function_id) + ".tt";
        raw.save(saved);
        TTN<T> reloaded(saved, 0, Real(-1), -1);
        check_same_body(original, reloaded.get_TTN_body());
        check_values("save/reload", reloaded.eval_list(points), reference, numeric_tolerance<T>());
        {
            std::ofstream out(saved);
            raw.save(out);
            check(static_cast<bool>(out), "stream save succeeds");
        }
        TTN<T> streamed(saved, 0, Real(-1), -1);
        check_same_body(original, streamed.get_TTN_body());
    });

    timed("invalid inputs", [&] {
        auto ttn = raw;
        const int missing_node = static_cast<int>(original.M.size()) + 1;
        check_throws<std::invalid_argument>([&] { ttn.canonify_to_center(missing_node); }, "invalid canonical centre rejected");
        check_throws<std::invalid_argument>([&] { ttn.move_canonical_center(missing_node); }, "invalid movement target rejected");
        check_throws<std::invalid_argument>([&] { (void)ttn.eval({}); }, "wrong evaluation dimension rejected");
        check_throws<std::runtime_error>([&] { TTN<T> missing(output_dir + "missing/input.tt"); }, "missing file rejected");
        check_same_body(original, ttn.get_TTN_body());
    });

    std::cout << "Total time: " << std::chrono::duration<double, std::milli>(Clock::now() - start).count() << " ms\n";
}

// A known Schmidt spectrum independently checks that reltol (not just the rank
// cap) is respected. The only nonzero values are F(0,0,0)=1 and F(1,1,1)=i/10000.
template<typename T>
void test_known_spectrum()
{
    using Real = RealOf<T>;
    xfac_quad::TensorTree<T> body(xfac_quad::makeTuckerTree(3, 1));
    for (int node : body.tree.nodes) {
        body.M.at(node) = xfac_quad::Tensor3D<T>(2, 1, 2);
        body.M.at(node)(0, 0, 0) = T(1);
        body.M.at(node)(1, 0, 1) = T(1);
    }
    body.M.at(3) = xfac_quad::Tensor3D<T>(2, 2, 2);
    body.M.at(3)(0, 0, 0) = T(1);
    body.M.at(3)(1, 1, 1) = T(Real(0), Real(1)/Real(10000));
    for (int center : {0, 3}) {
        TTN<T> loose(body, 0, Real(-1), center);
        loose.compress_svd(Real(1)/Real(1000), 0);
        check(loose.get_chi() == 1, "loose reltol removes small Schmidt value");
        check(loose.get_w() == center, "known-spectrum compression restores centre");
        check_canonical(loose);
        check(abs2(T(loose.eval({0, 0, 0}) - T(1))) <= numeric_tolerance<T>()*numeric_tolerance<T>(),
              "large Schmidt component preserved");
        check(abs2(loose.eval({1, 1, 1})) <= numeric_tolerance<T>()*numeric_tolerance<T>(),
              "small Schmidt component removed");
        TTN<T> tight(body, 0, Real(-1), center);
        tight.compress_svd(Real(1)/Real(1000000), 0);
        check(tight.get_chi() == 2, "tight reltol retains small Schmidt value");
        check_canonical(tight);
        check(abs2(T(tight.eval({1, 1, 1}) - body.eval({1, 1, 1})))
                  <= numeric_tolerance<T>()*numeric_tolerance<T>(), "retained small component preserved");
    }
    for (bool truncate : {false, true}) {
        const Real tolerance = truncate ? Real(1)/Real(1000) : Real(1)/Real(1000000);
        TTN<T> moving(body, 0, tolerance, 0);
        moving.move_canonical_center(3, true);
        check(moving.get_w() == 3, "stored-tolerance move reaches junction");
        check_canonical(moving);
        const auto& result = moving.get_TTN_body();
        const Eigen::Index rank = xfac_quad::cubeToMat_R(
            result.M.at(0), result.tree.neigh.at(0).pos(3)).cols();
        check(rank == (truncate ? 1 : 2), "centre movement respects stored reltol");
        const T expected = truncate ? T(0) : body.eval({1, 1, 1});
        check(abs2(T(moving.eval({1, 1, 1}) - expected))
                  <= numeric_tolerance<T>()*numeric_tolerance<T>(), "stored-tolerance Schmidt component");
    }
}

template<typename T>
void test_scalar(const std::string& name)
{
    for (int function_id : {1, 2}) {
        const std::string filename = output_dir + "function_" + std::to_string(function_id) + ".tt";
        test_ttn<T, long long>(name + "_int64", function_id, filename);
        test_ttn<T, util::i128>(name + "_int128", function_id, filename);
    }
    timed(name + " known Schmidt spectrum", [] { test_known_spectrum<T>(); });
}

} // namespace

int main()
{
    try {
        std::cout << std::scientific << std::setprecision(6);
        std::filesystem::create_directories(output_dir);
        std::srand(7);
        for (int function_id : {1, 2})
            timed("float128 TCI fixture f" + std::to_string(function_id), [&] {
                prepare_fixture(function_id, output_dir + "function_" + std::to_string(function_id) + ".tt");
            });
        test_scalar<std::complex<double>>("complex_double");
        test_scalar<std::complex<float128>>("complex_float128");
        test_scalar<std::complex<dd_128>>("complex_dd128");
        std::cout << "\nAll TTN tests passed (" << checks << " checks).\n";
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "\nFAILED after " << checks << " checks: " << error.what() << "\n";
        return 1;
    }
}
