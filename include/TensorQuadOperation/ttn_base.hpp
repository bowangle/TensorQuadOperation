#pragma once

#include "TensorQuadOperation/mpo_base.h"
#include "TensorQuadOperation/magic_tensor.hpp"
#include <xfac_quad/cubemat_helper.h>
#include <xfac_quad/tree/tensor_tree.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <map>
#include <ostream>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace TensorQuadOperation {

template<typename T>
class TTN {
public:
    using VecTT = std::vector<xfac_quad::Tensor3D<T>>;
    using RealScalar = typename Eigen::NumTraits<T>::Real;

protected:
    xfac_quad::TensorTree<T> ttn_body;
    int w = -1;                         // No defined working tensor.
    int max_bond_dim = 0;                // No bond-dimension cap.
    RealScalar reltol = RealScalar(-1);  // No relative-tolerance truncation.

public:
    TTN(
        xfac_quad::TensorTree<T> TensorTree_body_,
        int max_bond_dim_=0,
        RealScalar reltol_=-1,
        int w_=0)
        :
        ttn_body(TensorTree_body_),
        w(-1),
        max_bond_dim(max_bond_dim_),
        reltol(reltol_)
        {
            _initialize_w(w_);
        }

    TTN(
        const std::string& filename,
        int max_bond_dim_=0,
        RealScalar reltol_=-1,
        int w_=0)
        : TTN(xfac_quad::TensorTree<T>::load(filename),
              max_bond_dim_, reltol_, w_)
    {}

    const xfac_quad::TensorTree<T>& get_TTN_body() const { return ttn_body; }
    int get_w() const { return w; }
    int get_max_bond_dim() const { return max_bond_dim; }
    RealScalar get_reltol() const { return reltol; }

    // Tensor shapes in node-index order, using the stored axis ordering.
    std::vector<std::tuple<Eigen::Index, Eigen::Index, Eigen::Index>> get_shape() const
    {
        std::vector<std::tuple<Eigen::Index, Eigen::Index, Eigen::Index>> shapes;
        shapes.reserve(ttn_body.M.size());
        for (const auto& tensor : ttn_body.M)
            shapes.emplace_back(tensor.n_rows, tensor.n_cols, tensor.n_slices);
        return shapes;
    }

    // Maximum dimension over tree bonds, excluding physical and dummy legs.
    Eigen::Index get_chi() const
    {
        Eigen::Index chi = 0;
        for (const auto& [from, neighbors] : ttn_body.tree.neigh) {
            const auto& tensor = ttn_body.M.at(from);
            const std::array<Eigen::Index, 3> shape{
                tensor.n_rows, tensor.n_cols, tensor.n_slices};
            for (int to : neighbors.from_int()) {
                // Inspect each undirected edge once; -1 denotes a dummy leg.
                if (to == -1 || from >= to)
                    continue;
                chi = std::max(chi, shape.at(neighbors.pos(to)));
            }
        }
        return chi;
    }

    T eval(const std::vector<int>& id) const
    {
        return ttn_body.eval(id);
    }

    void save(const std::string& filename) const
    {
        ttn_body.save(filename);
    }

    void save(std::ostream& out) const
    {
        ttn_body.save(out);
    }

    // SVD-compress all bonds while preserving the canonical centre.
    void compress_svd(RealScalar reltol_ = RealScalar(-1), int max_bond_dim_ = -1)
    {
        const auto& tree = ttn_body.tree;
        const int center = (w == -1) ? tree.root : w;
        move_canonical_center(center);

        // Track the DFS path: QR moves outward without truncation, while
        // SVD compresses each bond once on the return to its parent.
        // Every decomposition starts at the current canonical centre.
        std::vector<int> path{center};
        xfac_quad::MatSVDFixedTol<T> decompose{reltol_, max_bond_dim_};
        for (const auto& [from, to] : tree.rootToLeaves(center)) {
            if (path.size() > 1 && to == path[path.size() - 2]) {
                xfac_quad::compress_bond(
                    ttn_body.M, from, to, tree, decompose);
                path.pop_back();
            }
            else {
                _canonify_bond(from, to);
                path.push_back(to);
            }
            w = to;
        }
    }

    // =============================================================
    //                      TTN Canonicalisation:
    // =============================================================

    // Canonicalise toward new_w using QR only, without truncation.
    void canonify_to_center(int new_w)
    {
        if (new_w == -1)
            return;

        const auto& tree = ttn_body.tree;
        if (!tree.neigh.contains(new_w))
            throw std::invalid_argument("TTN::canonify_to_center: invalid node");

        for (const auto& [from, to] : tree.leavesToRoot(new_w)) {
            _canonify_bond(from, to, false);
        }
        w = new_w;
    }

    // Move an existing centre along the unique path to new_w.
    void move_canonical_center(int new_w, bool compress=false)
    {
        const auto& tree = ttn_body.tree;
        if (!tree.neigh.contains(new_w))
            throw std::invalid_argument("TTN::move_canonical_center: invalid node");

        if (w == -1) {
            canonify_to_center(new_w);
            if (compress){
                compress_svd(reltol, max_bond_dim);
            }
            return;
        }
        if (w == new_w)
            return;

        // Root the topology at the target to find each node's next step.
        // Only tensors on the path from w to new_w are factorised.
        std::map<int, int> toward_center;
        for (const auto& [from, to] : tree.leavesToRoot(new_w))
            toward_center.emplace(from, to);

        while (w != new_w) {
            const int next = toward_center.at(w);
            _canonify_bond(w, next, compress);
            w = next;
        }
    }

    void _initialize_w(int new_w)
    {
        canonify_to_center(new_w);
    }

    // Make from isometric toward to and absorb the remainder into to.
    void _canonify_bond(int from, int to, bool compress=false)
    {
        using MatrixX = typename xfac_quad::Tensor3D<T>::MatrixX;
        // compress_bond always requests left-orthogonal factors: A = Q * R.
        auto decompose = [this, compress](const MatrixX& matrix, bool leftOrthogonal) {
            auto ab = xfac_quad::MatQR<T>{}(matrix, leftOrthogonal);
            if (compress) {
                xfac_quad::SVDDecomp<T> svd(
                    ab[1], /*leftOrthogonal=*/true, reltol, max_bond_dim);
                // Keep Q * U at the source and absorb S * V^H into its neighbour.
                ab[0] = (ab[0] * svd.left()).eval();
                ab[1] = svd.right();
            }
            return ab;
        };

        xfac_quad::compress_bond(
            ttn_body.M, from, to, ttn_body.tree, decompose);
    }

    // =============================================================
    //                      TTN operator:
    // =============================================================

    /// Return a scaled copy (ttn * scalar), leaving this TTN unchanged.
    /// Preserve the centre, bond dimensions, topology, and truncation settings.
    TTN operator*(T scalar) const
    {
        TTN result(*this);
        result *= scalar;
        return result;
    }

    /// Return a scaled copy with the scalar on the left (scalar * ttn).
    friend TTN operator*(T scalar, const TTN& tree)
    {
        return tree * scalar;
    }

    /// Scale in place (ttn *= scalar) and return this TTN by reference.
    /// Scale only the canonical centre, or the root if w == -1; no sweep or truncation.
    /// Keep w and all bond dimensions unchanged, including when scalar is zero.
    TTN& operator*=(T scalar)
    {
        // Scaling only the centre preserves all other isometries, even for zero.
        // With no defined centre, scale the root without canonicalising the tree.
        const int node = w == -1 ? ttn_body.tree.root : w;
        for (auto& value : ttn_body.M.at(node).data)
            value *= scalar;
        return *this;
    }

    /// Causal convolution with one independent 1D MPS kernel per branch.
    /// For three variables, the discrete result is:
    ///   C[p,q,r] = sum(i=0..p) sum(j=0..q) sum(k=0..r)
    ///                f[i,j,k] * Kx[p-i] * Ky[q-j] * Kz[r-k].
    /// Its continuum counterpart, after applying the grid-cell volume, is:
    ///   dx * dy * dz * C[p,q,r] approximately equals
    ///     integral(u=ax..x[p]) integral(v=ay..y[q]) integral(w=az..z[r])
    ///       f(u,v,w) * Kx(x[p]-u) * Ky(y[q]-v) * Kz(z[r]-w) dw dv du.
    /// kernels[i] acts on branch i; cores are binary and least-significant first.
    /// Return a new TTN, retaining the bit counts and leaving both inputs unchanged.
    /// Discard overflow without periodic wraparound; apply no grid-volume scaling.
    /// Each kernel must match its branch length and have trivial boundary bonds.
    /// Requested truncation settings override the inputs' settings for every contraction.
    /// final_compress adds a full-tree SVD sweep with the same settings.
    TTN conv(
        const std::vector<MPS<T>>& kernels,
        int max_bond_dim_ = 0,
        RealScalar reltol_ = RealScalar(-1),
        bool final_compress = true) const
    {
        if (!ttn_body.tree.branches || ttn_body.tree.branches->empty() ||
            kernels.size() != ttn_body.tree.branches->size())
            throw std::invalid_argument("TTN::conv: expected one kernel per branch and nonempty branch metadata");

        // Validate every branch and kernel before beginning any contraction.
        std::set<int> covered;
        for (std::size_t i = 0; i < kernels.size(); ++i) {
            _branch_axes(static_cast<int>(i));
            const auto& branch = ttn_body.tree.branches->at(i);
            for (int node : branch.nodes) {
                if (!covered.insert(node).second)
                    throw std::invalid_argument("TTN::conv: branches must not overlap");
                if (ttn_body.M.at(node).n_slices != 2)
                    throw std::invalid_argument("TTN::conv: expected signal physical dimension two");
            }
            const auto& cores = kernels[i].get_core();
            if (cores.size() != branch.nodes.size())
                throw std::invalid_argument("TTN::conv: kernel length must match its branch");
            if (cores.front().n_rows != 1 || cores.back().n_slices != 1)
                throw std::invalid_argument("TTN::conv: kernel boundary bonds must have dimension one");
            for (std::size_t site = 0; site < cores.size(); ++site) {
                const auto& core = cores[site];
                if (core.n_cols != 2 || core.n_rows <= 0 || core.n_slices <= 0)
                    throw std::invalid_argument("TTN::conv: invalid binary kernel core dimensions");
                if (site > 0 && cores[site - 1].n_slices != core.n_rows)
                    throw std::invalid_argument("TTN::conv: kernel internal bond mismatch");
            }
        }
        if (covered != ttn_body.tree.nodes)
            throw std::invalid_argument("TTN::conv: branches must cover all physical nodes");

        // Use the requested settings throughout, without modifying either input.
        TTN configured_signal(*this);
        configured_signal.max_bond_dim = max_bond_dim_;
        configured_signal.reltol = reltol_;
        for (std::size_t i = 0; i < kernels.size(); ++i) {
            MPS<T> configured_kernel(kernels[i].get_core(), max_bond_dim_, reltol_,
                                     kernels[i].get_w());

            // Steps 1-2: build the magic tensor and contract the kernel's m' leg.
            const auto magic_result = magic_tensor_product(configured_kernel);
            // Step 3: interpret the remaining (n'*m) cores as an MPO.
            auto convolution_mpo = reshape_magic_res(magic_result);
            // Step 4: apply it to this branch. Centre moves are QR-only; the MPO
            // multiplication uses the requested settings on both operands.
            configured_signal.mul_branch(static_cast<int>(i), std::move(convolution_mpo),
                                         /*compress=*/false, "zip-up");
        }
        if (final_compress)
            configured_signal.compress_svd(reltol_, max_bond_dim_);
        return configured_signal;
    }

    /// Apply an MPO to one branch by extracting, multiplying, and reinserting its MPS.
    /// Restore the original centre when defined; otherwise keep the branch endpoint.
    /// compress controls centre preparation/restoration; multiplication independently
    /// uses the operands' truncation settings. The caller's MPO is unchanged.
    void mul_branch(
        int branch_id,
        MPO<T> mpo,
        bool compress=false,
        std::string const& method = "optimize",
        int order = 2,
        int n_sweeps = 2)
    {
        _branch_axes(branch_id);
        const auto& branch = ttn_body.tree.branches->at(branch_id);
        const auto& operator_cores = std::as_const(mpo).get_core();
        if (operator_cores.size() != branch.nodes.size())
            throw std::invalid_argument("TTN::mul_branch: MPO length mismatch");
        if (operator_cores.front().n_rows != 1 || operator_cores.back().n_slices != 1)
            throw std::invalid_argument("TTN::mul_branch: MPO boundaries must have dimension one");
        for (int node : branch.nodes)
            if (ttn_body.M.at(node).n_slices != 2)
                throw std::invalid_argument("TTN::mul_branch: expected physical dimension two");
        if (method != "zip-up" && method != "qrsvd" && method != "optimize")
            throw std::invalid_argument("TTN::mul_branch: unknown multiplication method");
        const int original_w = w;
        // 1: move canonical center of the tree in the branch root
        // _extract_branch optionally compresses during this move.
        // 2: extract the branch mps from it and set it's canonical center 
        //    (have an additional dangling leg on the side)
        auto branch_mps = _extract_branch(branch_id, compress);
        // 3 : multiply the mpo with the branch mps 
        auto result = mpo._mul(branch_mps, method, order, n_sweeps);
        // 4: put the result back in the tree and move the canonical center back to the original position
        _insert_branch(branch_id, std::move(result), original_w, compress);
    }

    /// Prepare the centre at the branch endpoint, optionally compressing, then
    /// return a leaf-to-endpoint MPS with an open right bond to the attachment.
    MPS<T> _extract_branch(int branch_id, bool compress)
    {
        const auto axes = _branch_axes(branch_id);
        const auto& branch = ttn_body.tree.branches->at(branch_id);
        // 1: move canonical center of the tree in the branch root (updates TTN::w)
        move_canonical_center(branch.nodes.back(), compress);
        // 2: extract the branch mps from it and set it's canonical center 
        //    (have an additional dangling leg on the side)
        VecTT cores;
        cores.reserve(branch.nodes.size());
        for (std::size_t i = 0; i < branch.nodes.size(); ++i) {
            const auto& source = ttn_body.M.at(branch.nodes[i]);
            const std::array<Eigen::Index, 2> shape{source.n_rows, source.n_cols};
            xfac_quad::Tensor3D<T> core(shape[axes[i][0]], source.n_slices,
                                       shape[axes[i][1]]);
            for (Eigen::Index right = 0; right < core.n_slices; ++right)
                for (Eigen::Index physical = 0; physical < core.n_cols; ++physical)
                    for (Eigen::Index left = 0; left < core.n_rows; ++left) {
                        std::array<Eigen::Index, 2> index;
                        index[axes[i][0]] = left;
                        index[axes[i][1]] = right;
                        core(left, physical, right) = source(index[0], index[1], physical);
                    }
            cores.push_back(std::move(core));
        }
        // 3 : return the branch mps
        return MPS<T>(std::move(cores), max_bond_dim, reltol,
                      static_cast<int>(branch.nodes.size()) - 1);
    }

    /// Replace a branch with an MPS, then restore original_w using compress.
    /// If original_w == -1, keep the centre at the branch endpoint.
    /// The incoming MPS must use the same attachment gauge as the tree.
    void _insert_branch(int branch_id, MPS<T> branch_mps, int original_w, bool compress)
    {
        // Prepare the replacement before moving the TTN center.
        // Validate the branch topology and recover each node's stored bond axes.
        const auto axes = _branch_axes(branch_id);
        const auto& branch = ttn_body.tree.branches->at(branch_id);
        const auto& cores = std::as_const(branch_mps).get_core();
        // Check the requested center and compatibility with the existing branch.
        // The leaf boundary is trivial; the right boundary connects to the tree.
        if (original_w != -1 && !ttn_body.tree.neigh.contains(original_w))
            throw std::invalid_argument("TTN::_insert_branch: invalid target center");
        if (cores.size() != branch.nodes.size())
            throw std::invalid_argument("TTN::_insert_branch: branch length mismatch");
        const auto& endpoint = ttn_body.M.at(branch.nodes.back());
        const std::array<Eigen::Index, 2> endpoint_shape{endpoint.n_rows, endpoint.n_cols};
        if (cores.front().n_rows != 1 ||
            cores.back().n_slices != endpoint_shape[axes.back()[1]])
            throw std::invalid_argument("TTN::_insert_branch: boundary dimension mismatch");
        for (std::size_t i = 0; i < cores.size(); ++i) {
            if (cores[i].n_cols != ttn_body.M.at(branch.nodes[i]).n_slices ||
                cores[i].n_rows <= 0 || cores[i].n_slices <= 0)
                throw std::invalid_argument("TTN::_insert_branch: invalid core dimensions");
            if (i > 0 && cores[i - 1].n_slices != cores[i].n_rows)
                throw std::invalid_argument("TTN::_insert_branch: internal bond mismatch");
        }
        // Canonicalize the incoming MPS at its last site, the attachment endpoint.
        // branch_mps is a local copy; cores references its updated tensors.
        branch_mps.shift_w(static_cast<int>(cores.size()) - 1);
        // Prepare step 2 by converting (left, physical, right) MPS axes back
        // to the stored tree axes, with the physical index on axis two.
        // Finish allocation and conversion before changing the TTN.
        VecTT replacements;
        replacements.reserve(cores.size());
        for (std::size_t i = 0; i < cores.size(); ++i) {
            const auto& core = cores[i];
            std::array<Eigen::Index, 2> shape;
            shape[axes[i][0]] = core.n_rows;
            shape[axes[i][1]] = core.n_slices;
            xfac_quad::Tensor3D<T> tensor(shape[0], shape[1], core.n_cols);
            for (Eigen::Index right = 0; right < core.n_slices; ++right)
                for (Eigen::Index physical = 0; physical < core.n_cols; ++physical)
                    for (Eigen::Index left = 0; left < core.n_rows; ++left) {
                        std::array<Eigen::Index, 2> index;
                        index[axes[i][0]] = left;
                        index[axes[i][1]] = right;
                        tensor(index[0], index[1], physical) = core(left, physical, right);
                    }
            replacements.push_back(std::move(tensor));
        }
        // 1: move the TTN canonical center to the branch attachment node before replacing the branch
        // The branch-side endpoint is nodes.back(); attachment itself is outside the branch.
        move_canonical_center(branch.nodes.back());
        // A QR center move can change bond dimensions, so recheck the attachment.
        const auto& centered_endpoint = ttn_body.M.at(branch.nodes.back());
        const std::array<Eigen::Index, 2> centered_shape{
            centered_endpoint.n_rows, centered_endpoint.n_cols};
        if (cores.back().n_slices != centered_shape[axes.back()[1]])
            throw std::invalid_argument("TTN::_insert_branch: attachment dimension changed during center movement");
        // 2: insert the branch mps into the tree and set its canonical center
        //    (have an additional dangling leg on the side)
        for (std::size_t i = 0; i < replacements.size(); ++i)
            ttn_body.M.at(branch.nodes[i]) = std::move(replacements[i]);
        w = branch.nodes.back();
        // 3: restore the original center, optionally compressing the return path.
        // If original_w == -1, keep the center at the branch endpoint.
        if (original_w != -1)
            move_canonical_center(original_w, compress);
    }

private:
    // Validate the chain and map its left/right bonds to stored tensor axes.
    std::vector<std::array<int, 2>> _branch_axes(int branch_id) const
    {
        const auto& tree = ttn_body.tree;
        if (!tree.branches || branch_id < 0 ||
            static_cast<std::size_t>(branch_id) >= tree.branches->size())
            throw std::invalid_argument("TTN: invalid branch id or missing branch metadata");
        const auto& branch = tree.branches->at(branch_id);
        if (branch.nodes.empty())
            throw std::invalid_argument("TTN: empty branch");
        std::set<int> visited;
        std::vector<std::array<int, 2>> axes;
        for (std::size_t i = 0; i < branch.nodes.size(); ++i) {
            const int node = branch.nodes[i];
            if (!tree.nodes.contains(node) || !visited.insert(node).second)
                throw std::invalid_argument("TTN: branch must contain distinct physical nodes");
            const int previous = i == 0 ? -1 : branch.nodes[i - 1];
            const int next = i + 1 == branch.nodes.size() ? branch.attachment : branch.nodes[i + 1];
            const auto& neighbors = tree.neigh.at(node);
            for (int neighbor : neighbors.from_int())
                if (neighbor != -1 && neighbor != previous && neighbor != next)
                    throw std::invalid_argument("TTN: branch has an unexpected external connection");
            int left = previous == -1 ? -1 : neighbors.pos(previous);
            int right = next == -1 ? -1 : neighbors.pos(next);
            if ((previous != -1 && (left < 0 || left > 1)) ||
                (next != -1 && (right < 0 || right > 1)))
                throw std::invalid_argument("TTN: branch bond must occupy axis zero or one");
            if (left == -1) left = right == -1 ? 0 : 1 - right;
            if (right == -1) right = 1 - left;
            if (left == right)
                throw std::invalid_argument("TTN: invalid branch ordering");
            const auto& tensor = ttn_body.M.at(node);
            const std::array<Eigen::Index, 2> shape{tensor.n_rows, tensor.n_cols};
            if (tensor.n_slices <= 0 || shape[0] <= 0 || shape[1] <= 0 ||
                (previous == -1 && shape[left] != 1) || (next == -1 && shape[right] != 1))
                throw std::invalid_argument("TTN: invalid branch tensor dimensions");
            if (next != -1) {
                const int reciprocal = tree.neigh.at(next).pos(node);
                const auto& neighbor = ttn_body.M.at(next);
                const std::array<Eigen::Index, 3> neighbor_shape{
                    neighbor.n_rows, neighbor.n_cols, neighbor.n_slices};
                if (reciprocal < 0 || reciprocal > 2 || shape[right] != neighbor_shape[reciprocal])
                    throw std::invalid_argument("TTN: branch bond dimension mismatch");
            }
            axes.push_back({left, right});
        }
        if (visited.contains(branch.attachment))
            throw std::invalid_argument("TTN: attachment must be outside the branch");
        return axes;
    }

public:

    // =============================================================
    //                      TTN misc:
    // =============================================================

    // Evaluate the TTN at a list of points.
    std::vector<T> eval_list(const std::vector<std::vector<int>>& points) const
    {
        std::vector<T> results(points.size());
        #pragma omp parallel for schedule(static)
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(points.size()); ++i)
            results[i] = eval(points[i]);
        return results;
    }

    // One dimension per tree bond, in node/neighbor order, without boundary padding.
    std::vector<int> compute_list_chi() const
    {
        std::vector<int> chi;
        chi.reserve(ttn_body.M.size());
        for (const auto& [from, neighbors] : ttn_body.tree.neigh) {
            const auto& tensor = ttn_body.M.at(from);
            const std::array<Eigen::Index, 3> shape{
                tensor.n_rows, tensor.n_cols, tensor.n_slices};
            for (int to : neighbors.from_int()) {
                if (to == -1 || from >= to)
                    continue;
                chi.push_back(static_cast<int>(shape.at(neighbors.pos(to))));
            }
        }
        return chi;
    }

    // Number of stored values per tensor, in node-index order.
    std::vector<Eigen::Index> compute_nb_value_core() const
    {
        std::vector<Eigen::Index> counts;
        counts.reserve(ttn_body.M.size());
        for (const auto& tensor : ttn_body.M)
            counts.push_back(tensor.n_rows * tensor.n_cols * tensor.n_slices);
        return counts;
    }

    // Total number of stored values across all tensors.
    Eigen::Index compute_tot_nb_value() const
    {
        Eigen::Index total = 0;
        for (const auto& tensor : ttn_body.M)
            total += tensor.n_rows * tensor.n_cols * tensor.n_slices;
        return total;
    }

};

} // namespace TensorQuadOperation
