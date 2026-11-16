#pragma once

#include <xfac_quad/cubemat_helper.h>
#include <xfac_quad/tree/tensor_tree.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <map>
#include <ostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

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
    void compress_svd(RealScalar reltol = RealScalar(-1), int max_bond_dim = -1)
    {
        const auto& tree = ttn_body.tree;
        const int center = (w == -1) ? tree.root : w;
        move_canonical_center(center);

        // Track the DFS path: QR moves outward without truncation, while
        // SVD compresses each bond once on the return to its parent.
        // Every decomposition starts at the current canonical centre.
        std::vector<int> path{center};
        xfac_quad::MatSVDFixedTol<T> decompose{reltol, max_bond_dim};
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

    // Canonicalise toward new_w using QR, optionally truncating R with SVD.
    void canonify_to_center(int new_w, bool compress=false)
    {
        if (new_w == -1)
            return;

        const auto& tree = ttn_body.tree;
        if (!tree.neigh.contains(new_w))
            throw std::invalid_argument("TTN::canonify_to_center: invalid node");

        for (const auto& [from, to] : tree.leavesToRoot(new_w)) {
            _canonify_bond(from, to, compress);
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
            canonify_to_center(new_w, compress);
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

    void _initialize_w(int new_w, bool compress=false)
    {
        canonify_to_center(new_w, compress);
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
