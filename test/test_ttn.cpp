#include "type_double_double.h"
#include "type_float128_boost.h"
#include "type_int128.h"
#include "ttn_base.hpp"

#include <QTgrid-quad/grid.h>
#include <xfac_quad/tree/tensor_tree_ci.hpp>

#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
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
void test_branch_operations()
{
    using Real = RealOf<T>;
    using Tensor = xfac_quad::Tensor3D<T>;
    const Real tolerance = numeric_tolerance<T>();
    const T identity_weight(Real(3)/Real(10), Real(2)/Real(10));

    // A site-dependent product operator catches reversed branch ordering.
    const auto local_operator = [](int site, int length, int row, int column) -> T {
        if (site == 0) return T(row != column ? 1 : 0); // X
        if (site == length - 1) return T(row == column ? (row == 0 ? 1 : -1) : 0); // Z
        return T(row == column ? row + 1 : 0);
    };

    for (int length : {1, 3}) {
        for (bool reverse_axes : {false, true}) {
            auto tree = xfac_quad::makeTuckerTree(3, length);
            xfac_quad::TensorTree<T> body(tree);
            for (const auto& [node, neighbors] : tree.neigh) {
                const int degree = static_cast<int>(neighbors.size());
                Tensor tensor(2, degree > 1 ? 2 : 1, 2);
                for (Eigen::Index k = 0; k < tensor.n_slices; ++k)
                    for (Eigen::Index j = 0; j < tensor.n_cols; ++j)
                        for (Eigen::Index i = 0; i < tensor.n_rows; ++i) {
                            const int seed = 1 + node * 17 + int(i) * 7 + int(j) * 11 + int(k) * 13;
                            tensor(i, j, k) = T(Real(seed % 19 - 9)/Real(10),
                                                 Real(seed % 13 - 6)/Real(10));
                        }
                body.M.at(node) = std::move(tensor);
            }
            if (reverse_axes) {
                // Swap both topology slots and tensor axes, preserving the dense tensor.
                for (int node : tree.nodes) {
                    const auto neighbors = body.tree.neigh.at(node).from_int();
                    body.tree.neigh.at(node) = {};
                    body.tree.neigh.at(node).push_back(neighbors.size() > 1 ? neighbors[1] : -1);
                    body.tree.neigh.at(node).push_back(neighbors[0]);
                    const auto source = body.M.at(node);
                    Tensor swapped(source.n_cols, source.n_rows, source.n_slices);
                    for (Eigen::Index p = 0; p < source.n_slices; ++p)
                        swapped.slice(p) = source.slice_const(p).transpose();
                    body.M.at(node) = std::move(swapped);
                }
            }
            const int count = 3 * length;
            std::vector<std::vector<int>> points(std::size_t(1) << count, std::vector<int>(count));
            for (std::size_t code = 0; code < points.size(); ++code)
                for (int bit = 0; bit < count; ++bit)
                    points[code][bit] = int((code >> bit) & 1);
            const int center = count; // Junction outside every branch.
            const TTN<T> baseline(body, 0, Real(-1), center);
            const auto original_values = baseline.eval_list(points);

            for (int branch_id = 0; branch_id < 3; ++branch_id) {
                const auto& branch = body.tree.branches->at(branch_id);
                auto roundtrip = baseline;
                auto extracted = roundtrip.extract_branch(branch_id);
                check(extracted.get_w() == length - 1, "extracted MPS center at attachment endpoint");
                check(roundtrip.get_w() == branch.nodes.back(), "extraction moves TTN center");
                check(std::as_const(extracted).get_core().back().n_slices == 2,
                      "branch fixture exercises nontrivial dangling leg");
                check_values("extraction preserves TTN", roundtrip.eval_list(points), original_values, tolerance);
                roundtrip.insert_branch(branch_id, extracted);
                check(roundtrip.get_w() == branch.nodes.back(), "default insertion center");
                check_canonical(roundtrip);
                roundtrip.insert_branch(branch_id, extracted, center);
                check(roundtrip.get_w() == center, "insertion restores requested center");
                check_canonical(roundtrip);
                check_values("branch roundtrip", roundtrip.eval_list(points), original_values, tolerance);

                // W = product(local_operator) + identity_weight * I.
                // For multiple sites, the two independent channels give MPO bond dimension two.
                std::vector<Tensor> operator_cores;
                for (int site = 0; site < length; ++site) {
                    Tensor core(site == 0 ? 1 : 2, 4, site == length - 1 ? 1 : 2);
                    for (int row = 0; row < 2; ++row)
                        for (int column = 0; column < 2; ++column) {
                            const int physical = 2 * row + column;
                            core(0, physical, 0) = local_operator(site, length, row, column);
                            const T identity = row == column ? T(1) : T(0);
                            core(site == 0 ? 0 : 1, physical, site == length - 1 ? 0 : 1)
                                += (site == 0 ? identity_weight : T(1)) * identity;
                        }
                    operator_cores.push_back(std::move(core));
                }
                MPO<T> mpo(std::move(operator_cores), 0, Real(-1));
                std::vector<T> expected(points.size(), T(0));
                for (std::size_t output = 0; output < points.size(); ++output) {
                    expected[output] = identity_weight * original_values[output];
                    for (int input = 0; input < (1 << length); ++input) {
                        std::size_t input_code = output;
                        T coefficient(1);
                        for (int site = 0; site < length; ++site) {
                            const int node = branch.nodes[site];
                            const int bit = (input >> site) & 1;
                            coefficient *= local_operator(site, length, points[output][node], bit);
                            input_code &= ~(std::size_t(1) << node);
                            input_code |= std::size_t(bit) << node;
                        }
                        expected[output] += coefficient * original_values[input_code];
                    }
                }
                for (const std::string method : {"qrsvd", "zip-up", "optimize"}) {
                    auto result = baseline;
                    result.mul_branch(branch_id, mpo, false, method, 2, 6);
                    check(result.get_w() == center, method + ": original TTN center restored");
                    check_canonical(result);
                    const auto& updated = result.get_TTN_body();
                    const int endpoint = branch.nodes.back();
                    const int axis = updated.tree.neigh.at(endpoint).pos(branch.attachment);
                    check(xfac_quad::cubeToMat_R(updated.M.at(endpoint), axis).cols() == 2,
                          method + ": attachment dimension preserved");
                    const Real error2 = relative_error2(result.eval_list(points), expected);
                    std::cout << "  branch=" << branch_id << " length=" << length
                              << " reversed=" << reverse_axes << " method=" << method
                              << " relative L2 error=" << std::sqrt(to_double(error2)) << '\n';
                    check(error2 <= tolerance * tolerance, method + ": dense branch multiplication agrees");
                }
            }
        }
    }
}

void test_branch_dense_stress()
{
    using T = std::complex<double>;
    using Tensor = xfac_quad::Tensor3D<T>;
    constexpr int length = 3;
    constexpr int count = 9;
    const auto points = all_points();
    for (unsigned seed : {17u, 83u, 211u}) {
        std::mt19937 generator(seed);
        std::uniform_real_distribution<double> distribution(-0.5, 0.5);
        const auto random_value = [&]() { return T(distribution(generator), distribution(generator)); };
        xfac_quad::TensorTree<T> body(xfac_quad::makeTuckerTree(3, length));
        for (const auto& [node, neighbors] : body.tree.neigh) {
            Tensor core(4, neighbors.size() > 1 ? 4 : 1, node < count ? 2 : 4);
            for (auto& value : core.data) value = random_value();
            body.M.at(node) = std::move(core);
        }
        std::vector<Tensor> operator_cores;
        for (int site = 0; site < length; ++site) {
            Tensor core(site == 0 ? 1 : 3, 4, site == length - 1 ? 1 : 3);
            for (auto& value : core.data) value = random_value();
            operator_cores.push_back(std::move(core));
        }
        // Independently enumerate MPO bond indices to obtain all 8 x 8 matrix entries.
        std::array<std::array<T, 8>, 8> dense_operator{};
        for (int row = 0; row < 8; ++row)
            for (int column = 0; column < 8; ++column)
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b) {
                        const int p0 = 2 * (row & 1) + (column & 1);
                        const int p1 = 2 * ((row >> 1) & 1) + ((column >> 1) & 1);
                        const int p2 = 2 * ((row >> 2) & 1) + ((column >> 2) & 1);
                        dense_operator[row][column] += operator_cores[0](0, p0, a)
                            * operator_cores[1](a, p1, b) * operator_cores[2](b, p2, 0);
                    }
        const auto dense_apply = [&](const std::vector<T>& input, int branch_id) {
            std::vector<T> output(input.size());
            const auto& nodes = body.tree.branches->at(branch_id).nodes;
            for (std::size_t code = 0; code < input.size(); ++code) {
                int row = 0;
                for (int site = 0; site < length; ++site)
                    row |= points[code][nodes[site]] << site;
                for (int column = 0; column < 8; ++column) {
                    std::size_t input_code = code;
                    for (int site = 0; site < length; ++site) {
                        input_code &= ~(std::size_t(1) << nodes[site]);
                        input_code |= std::size_t((column >> site) & 1) << nodes[site];
                    }
                    output[code] += dense_operator[row][column] * input[input_code];
                }
            }
            return output;
        };
        MPO<T> mpo(operator_cores, 0, -1);
        for (const std::string method : {"qrsvd", "zip-up", "optimize"}) {
            for (int initial_center : {-1, 0, 7, count}) {
                TTN<T> result(body, 0, -1, initial_center);
                auto expected = result.eval_list(points);
                // Repeat a branch, then act on another branch to exercise composition and gauges.
                for (int branch_id : {0, 0, 2, 1}) {
                    const auto& branch = body.tree.branches->at(branch_id);
                    expected = dense_apply(expected, branch_id);
                    const int old_center = result.get_w();
                    result.mul_branch(branch_id, mpo, false, method, 2, 8);
                    check(result.get_w() == (old_center == -1 ? branch.nodes.back() : old_center),
                          "stress multiplication restores center or establishes branch center");
                    check_canonical(result);
                    const auto& updated = result.get_TTN_body();
                    const int axis = updated.tree.neigh.at(branch.nodes.back()).pos(branch.attachment);
                    check(xfac_quad::cubeToMat_R(updated.M.at(branch.nodes.back()), axis).cols() == 4,
                          "stress attachment dimension remains four");
                    const double error = std::sqrt(relative_error2(result.eval_list(points), expected));
                    std::cout << "  stress seed=" << seed << " method=" << method
                              << " initial_center=" << initial_center << " branch=" << branch_id
                              << " relative L2 error=" << error << '\n';
                    check(error < 1e-11, "repeated dense complex MPO application agrees");
                }
            }
        }
        // Check an actual truncation bound for the exact-QR/SVD method.
        const auto expected = dense_apply(TTN<T>(body).eval_list(points), 0);
        for (double tolerance : {1e-8, 1e-3}) {
            TTN<T> truncated(body, 0, tolerance, count);
            truncated.mul_branch(0, mpo, false, "qrsvd");
            check_canonical(truncated);
            const double error = std::sqrt(relative_error2(truncated.eval_list(points), expected));
            check(error <= (length - 1) * tolerance + 1e-11,
                  "branch QR/SVD error obeys accumulated local truncation bound");
        }
        TTN<T> capped(body, 1, -1, count);
        capped.mul_branch(0, mpo, true, "qrsvd");
        check_canonical(capped);
        check(capped.get_w() == count, "compressed return restores center");
        const auto& capped_body = capped.get_TTN_body();
        const auto& nodes = body.tree.branches->at(0).nodes;
        for (int site = 0; site < length; ++site) {
            const int neighbor = site + 1 < length ? nodes[site + 1] : count;
            const int axis = capped_body.tree.neigh.at(nodes[site]).pos(neighbor);
            check(xfac_quad::cubeToMat_R(capped_body.M.at(nodes[site]), axis).cols() == 1,
                  "rank cap applies to branch bonds and compressed return edge");
        }
    }
}

template<typename T>
void test_large_tci_branch(RealOf<T> truncation_tolerance)
{
    using Real = RealOf<T>;
    using Tensor = xfac_quad::Tensor3D<T>;
    constexpr int bits = 30;
    constexpr int count = 3 * bits;
    const Real theta = Real(7)/Real(10);
    const T coefficient(Real(3)/Real(10), Real(2)/Real(10));
    const Real comparison_tolerance = Real(1000) * truncation_tolerance;
    const auto phase_factor = [](Real angle) {
        using std::cos;
        using std::sin;
        return T(cos(angle), sin(angle));
    };
    const auto tree = xfac_quad::makeTuckerTree(3, bits);
    // Preserve makeTuckerTree's interleaved node IDs. The leaf bit is most significant.
    const auto coordinates = [](const std::vector<int>& id) {
        std::vector<Real> point(3, Real(-1));
        for (int bit = 0; bit < bits; ++bit)
            for (int variable = 0; variable < 3; ++variable)
                point[variable] += Real(id[variable + 3 * bit]) / Real(std::uint64_t(1) << bit);
        return point;
    };
    const auto function = [&](const std::vector<int>& id) { return function_1<T>(coordinates(id)); };
    xfac_quad::TensorTreeCIParam<T> parameters;
    parameters.bondDim = 16;
    parameters.reltol = truncation_tolerance;
    parameters.fullPiv = true;
    parameters.pivot1.assign(count, 0);
    for (int variable = 0; variable < 3; ++variable) parameters.pivot1[variable] = 1;
    xfac_quad::TensorTreeCI<T> ci(function, tree, std::vector<int>(count, 2), parameters);
    ci.iterate(8);
    const TTN<T> baseline(ci.tt, 64, truncation_tolerance, count);
    check(baseline.get_TTN_body().tree.nodes.size() == count, "large TCI has 90 physical sites");
    std::cout << "  large TCI chi=" << baseline.get_chi() << '\n';

    // Include corners, single-bit perturbations and reproducible held-out random points.
    std::vector<std::vector<int>> points;
    for (int corner = 0; corner < 8; ++corner) {
        std::vector<int> id(count);
        for (int node = 0; node < count; ++node) id[node] = (corner >> (node % 3)) & 1;
        points.push_back(std::move(id));
    }
    for (int node = 0; node < count; ++node) {
        std::vector<int> id(count, 0);
        id[node] = 1;
        points.push_back(std::move(id));
    }
    std::mt19937 generator(3090);
    for (int sample = 0; sample < 512; ++sample) {
        std::vector<int> id(count);
        for (auto& bit : id) bit = int(generator() & 1);
        points.push_back(std::move(id));
    }
    std::vector<T> analytic;
    for (const auto& id : points) analytic.push_back(function(id));
    const auto original = baseline.eval_list(points);
    check_values("90-site TCI analytic samples", original, analytic, comparison_tolerance);

    // W = I + c D X^{tensor bits}, with D(s,s) = exp(i theta x(s)).
    // Each channel is a product operator, yielding internal MPO dimension two.
    std::vector<Tensor> cores;
    for (int site = 0; site < bits; ++site) {
        Tensor core(site == 0 ? 1 : 2, 4, site == bits - 1 ? 1 : 2);
        for (int row = 0; row < 2; ++row) {
            core(0, 3 * row, 0) = T(1);
            T phase = phase_factor(theta * Real(row) / Real(std::uint64_t(1) << site));
            if (site == 0) phase *= coefficient * phase_factor(-theta);
            core(site == 0 ? 0 : 1, 2 * row + (1 - row), site == bits - 1 ? 0 : 1) = phase;
        }
        cores.push_back(std::move(core));
    }
    MPO<T> mpo(std::move(cores), 64, truncation_tolerance);
    for (int branch_id = 0; branch_id < 3; ++branch_id) {
        const auto& branch = tree.branches->at(branch_id);
        std::vector<std::vector<int>> reflected = points;
        for (auto& id : reflected)
            for (int node : branch.nodes) id[node] = 1 - id[node];
        const auto reflected_values = baseline.eval_list(reflected);
        std::vector<T> expected, analytic_expected;
        for (std::size_t sample = 0; sample < points.size(); ++sample) {
            const T factor = coefficient * phase_factor(theta * coordinates(points[sample])[branch_id]);
            expected.push_back(original[sample] + factor * reflected_values[sample]);
            analytic_expected.push_back(analytic[sample] + factor * function(reflected[sample]));
        }
        const int endpoint = branch.nodes.back();
        const int axis = tree.neigh.at(endpoint).pos(branch.attachment);
        const auto boundary = xfac_quad::cubeToMat_R(baseline.get_TTN_body().M.at(endpoint), axis).cols();
        check(boundary > 1, "large TCI exercises nontrivial attachment");
        for (const std::string method : {"qrsvd", "zip-up", "optimize"}) {
            auto result = baseline;
            result.mul_branch(branch_id, mpo, false, method, 2, 6);
            check(result.get_w() == count, "large branch restores junction center");
            check_canonical(result);
            check(xfac_quad::cubeToMat_R(result.get_TTN_body().M.at(endpoint), axis).cols() == boundary,
                  "large branch preserves attachment dimension");
            const auto values = result.eval_list(points);
            const Real multiplication_error2 = relative_error2(values, expected);
            const Real analytic_error2 = relative_error2(values, analytic_expected);
            std::cout << "  90-site branch=" << branch_id << " method=" << method
                      << " samples=" << points.size() << " multiplication relative L2="
                      << std::sqrt(to_double(multiplication_error2))
                      << " analytic relative L2=" << std::sqrt(to_double(analytic_error2))
                      << " chi=" << result.get_chi() << '\n';
            check(multiplication_error2 < comparison_tolerance * comparison_tolerance,
                  "large branch sampled multiplication error");
            check_values("90-site analytic operator samples", values, analytic_expected,
                         Real(10) * comparison_tolerance);
        }
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
    timed(name + " branch extraction, insertion and MPO multiplication", [] { test_branch_operations<T>(); });
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
        timed("dense complex branch stress and truncation", test_branch_dense_stress);
        timed("complex_double three-variable 30-bit TCI branch multiplication", [] {
            test_large_tci_branch<std::complex<double>>(1e-12);
        });
        timed("complex_dd128 three-variable 30-bit TCI branch multiplication", [] {
            test_large_tci_branch<std::complex<dd_128>>(dd_128(1)/dd_128(1e27));
        });
        std::cout << "\nAll TTN tests passed (" << checks << " checks).\n";
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "\nFAILED after " << checks << " checks: " << error.what() << "\n";
        return 1;
    }
}
