#pragma once

#include <string>
#include <algorithm>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <tuple>
#include <type_traits>
#include <vector>

#include <mps_base.h>
#include <mpo_base.h>
#include <grid.h>
#include <xf-qd-runner.hpp>
#include "fft_param.hpp"
#include "magic_tensor_qft.hpp"

template <typename Cscalar, typename Sint>
class FFTmps{
public:
    using RealScalar = typename Eigen::NumTraits<Cscalar>::Real;

    // constructor where we provide everything (mps, grid and mpo)
    FFTmps(
        MPS<Cscalar> mps_,
        QTGrid<RealScalar, Sint> grid_,
        MPO<Cscalar> qft_mpo_,
        int padding_bit_ = 0,
        int max_chi_ = 200,
        RealScalar reltol_ = Eigen::NumTraits<RealScalar>::epsilon() * 100)
    :
    mps(std::move(mps_)),
    grid(std::move(grid_)),
    QFT_E_T(std::move(qft_mpo_)),
    padding_bit(padding_bit_),
    reltol(reltol_),
    max_chi(max_chi_)
    {
        _add_n_padding(mps, grid, padding_bit);
    }
    
    

    // constructor where we provide path to file:
    // assume mps armadillo file format with .tt extension for tt and grid file format in _grid.tt
    FFTmps(
        const std::string& prefix_mps,
        const std::string& folder_qft_data = "mpo_data/data/",
        int padding_bit_ = 0,
        int max_chi_ = 200,
        RealScalar reltol_ = Eigen::NumTraits<RealScalar>::epsilon() * 100)
    :
    mps(MPS<Cscalar>(prefix_mps + ".tt", max_chi_, reltol_, /*w_=*/-1)),
    grid(QTGrid<RealScalar, Sint>(prefix_mps + "_grid_E.json")),
    QFT_E_T(magic_tensor_qft::load_compressed_qft_mpo<Cscalar>(mps.get_size() + padding_bit_, folder_qft_data, reltol_, max_chi_)),
    padding_bit(padding_bit_),
    reltol(reltol_),
    max_chi(max_chi_)
    {
        _add_n_padding(mps, grid, padding_bit);
    }

    // Convenience constructor configured from persisted FFT parameters.
    FFTmps(
        const std::string& prefix_mps,
        const std::string& folder_qft_data,
        const fft_param<Cscalar>& param,
        const std::string& prefix_output = "")
        : FFTmps(prefix_mps,
                 folder_qft_data,
                 param.padding_bit,
                 param.max_chi,
                 param.reltol)
    {
        Cscalar f_b;
        std::vector<RealScalar> l_disc;
        std::vector<Cscalar> l_f_disc;

        std::tie(f_b, l_disc, l_f_disc) =
            load_fvalues_from_json<Cscalar>(prefix_mps + "_f_values.json");

        fft_discontinuous_loopless(
            l_disc, l_f_disc, f_b, param.do_shift, param.method);

        if (!prefix_output.empty()) {
            mps.save(prefix_output);
            grid.save_json(prefix_output);
            param.save(prefix_output + "_fft_param.json");
        }
    }

    // Convenience constructor loading FFT parameters from JSON.
    FFTmps(
        const std::string& prefix_mps,
        const std::string& folder_qft_data,
        const std::string& fft_param_path,
        const std::string& prefix_output = "")
        : FFTmps(prefix_mps,
                 folder_qft_data,
                 fft_param<Cscalar>(fft_param_path),
                 prefix_output)
    {}

    // --- reverse cores: swap left↔right on each core, then reverse order ---
    // TODO we need to move this to TensorQuadOperation
    static std::vector<Tensor3D<Cscalar>> reverse_cores(const std::vector<Tensor3D<Cscalar>>& cores)
    {
        std::vector<Tensor3D<Cscalar>> out;
        out.reserve(cores.size());
        for (int i = static_cast<int>(cores.size()) - 1; i >= 0; --i) {
            auto const& src = cores[i];
            Tensor3D<Cscalar> dst(src.n_right, src.n_phys, src.n_left);
            for (Eigen::Index l = 0; l < src.n_left; ++l)
                for (Eigen::Index p = 0; p < src.n_phys; ++p)
                    for (Eigen::Index r = 0; r < src.n_right; ++r)
                        dst(r, p, l) = src(l, p, r);
            out.push_back(std::move(dst));
        }
        return out;
    }

    // --- vanilla FFT (no trapezoid rule) ---
    // WARNING TODO later
    void fft_vanilla(bool do_shift = true,
                     const std::string& method = "zip-up")
    {
        // 1. Reverse cores for endian consistency with the QFT MPO
        mps = MPS<Cscalar>(reverse_cores(mps.get_core()), mps.get_max_bond_dim(), mps.get_reltol(), mps.get_w());

        // 2. Scale before the QFT to keep the contracted MPS well-conditioned.
        mps *= magic_tensor_qft::real_sqrt(RealScalar(grid.get_N()))
             * grid.get_dx() / (RealScalar(2) * pi<RealScalar>());

        // 3. Apply QFT MPO
        mps = QFT_E_T._mul(mps, method);

        // 4. Time-dependent phase: exp(-1j * t * a)
        RealScalar dt = RealScalar(2) * pi<RealScalar>() / (RealScalar(grid.get_N()) * grid.get_dx());
        auto mps_phase = _phase_mps_exp_1j_t_a(grid.get_nBits(), grid.get_a(), dt);
        mps_phase = MPS<Cscalar>(mps_phase.get_core(), max_chi, reltol, mps_phase.get_w());
        auto phase_mpo = MPO<Cscalar>::from_mps(mps_phase);
        mps = phase_mpo._mul(mps, method);

        // 5. Optional fftshift
        if (do_shift)
            mps = _fft_shift(mps);

        // 6. Update grid to dual grid
        grid = grid.build_dual_grid(do_shift);
    }

    // --- trapezoid FFT (with trapezoid rule) ---
    // WARNING TODO to test, both case padding_bit=0 and padding_bit>0, with the three method.
    void fft_trapez(Cscalar f_b, bool do_shift = true, const std::string &method = "zip-up")
    {
        // 1. Reverse cores for endian consistency with the QFT MPO
        mps = MPS<Cscalar>(reverse_cores(mps.get_core()), mps.get_max_bond_dim(), mps.get_reltol(), mps.get_w());

        // 2. Putting the correction on each end of the MPS
        {
            // left correction
            std::vector<int> zeros_bits(grid.get_nBits(), 0);
            Cscalar value_left = mps.eval(zeros_bits);

            if (padding_bit == 0) {
                // b aliases onto index 0: one delta covers both endpoints
                mps = mps + _mps_single_point(zeros_bits, (f_b - value_left) / Cscalar(2, 0));
            } else {
                // b is a real grid point on the padded grid
                std::vector<int> b_orig_bits(grid.get_nBits(), 0);
                b_orig_bits[padding_bit - 1] = 1;   // MSB-first
                mps = mps - _mps_single_point(zeros_bits,  value_left / Cscalar(2, 0));
                mps = mps + _mps_single_point(b_orig_bits, f_b        / Cscalar(2, 0));
            }
        }

        // 3. Scale before the QFT to keep the contracted MPS well-conditioned.
        mps *= magic_tensor_qft::real_sqrt(RealScalar(grid.get_N()))
             * grid.get_dx() / (RealScalar(2) * pi<RealScalar>());

        // 4. Apply QFT MPO
        mps = QFT_E_T._mul(mps, method);

        // 5. Time-dependent phase: exp(-1j * t * a)
        RealScalar dt = RealScalar(2) * pi<RealScalar>() / (RealScalar(grid.get_N()) * grid.get_dx());
        auto mps_phase = _phase_mps_exp_1j_t_a(grid.get_nBits(), grid.get_a(), dt);
        mps_phase = MPS<Cscalar>(mps_phase.get_core(), max_chi, reltol, mps_phase.get_w());
        auto phase_mpo = MPO<Cscalar>::from_mps(mps_phase);
        mps = phase_mpo._mul(mps, method);

        // 6. Optional fftshift
        if (do_shift)
            mps = _fft_shift(mps);

        // 7. Update grid to dual grid
        grid = grid.build_dual_grid(do_shift);
    }

    void fft_discontinuous_with_loop(const std::vector<RealScalar> &l_disc,
                        const std::vector<Cscalar>    &l_f_disc,
                        Cscalar f_b,
                        bool do_shift = true,
                        const std::string& method = "zip-up")
    {
        if (l_disc.size() != l_f_disc.size())
            throw std::invalid_argument("l_disc and l_f_disc must have equal length");

        // 1. Reverse cores -> MSB-first
        mps = MPS<Cscalar>(reverse_cores(mps.get_core()), mps.get_max_bond_dim(),
                        mps.get_reltol(), mps.get_w());

        const int nB = int(mps.get_core().size());
        const Sint N_orig = Sint(1) << (nB - padding_bit);
        const RealScalar a = grid.get_a(), dx = grid.get_dx();

        // --- validate and index the discontinuities (conditions 1 and 6) ---
        std::vector<Sint> idx;
        for (size_t j = 0; j < l_disc.size(); ++j) {
            RealScalar q = (l_disc[j] - a) / dx;
            const Sint i = _round_to_sint(q);
            _warn_if_discontinuity_is_off_grid(
                "fft_discontinuous_with_loop", j, l_disc[j], a, dx, q, i);
            if (i <= Sint(0) || i >= N_orig)
                throw std::invalid_argument("discontinuity outside (a, b)");
            if (j > 0 && i <= idx.back())
                throw std::invalid_argument("l_disc must be strictly increasing");
            idx.push_back(i);
        }

        auto bits_of = [nB](Sint n) {
            std::vector<int> v(nB, 0);
            for (int k = 0; k < nB; ++k)
                v[k] = int((n >> (nB - 1 - k)) & Sint(1));
            return v;
        };

        const size_t K = idx.size();               // K discontinuities -> K+1 intervals
        MPS<Cscalar> mps_out = _zero_mps(nB);
        const MPS<Cscalar> g = mps;                // untouched copy for evaluations

        for (size_t i = 0; i <= K; ++i) {
            const bool first = (i == 0), last = (i == K);
            const Sint L = first ? Sint(0) : idx[i - 1];
            const Sint U = last  ? N_orig : idx[i];

            // --- restrict to [L, U) ---
            auto ind = _mps_indicator_interval(nB, L, U, !first, !last, max_chi, reltol);
            MPS<Cscalar> part = MPO<Cscalar>::from_mps(ind)._mul(g, method);

            // --- left endpoint: halve the value at L ---
            std::vector<int> L_bits = bits_of(L);
            Cscalar g_L = g.eval(L_bits);
            part = part - _mps_single_point(L_bits, g_L / Cscalar(2, 0));

            // --- right endpoint: half the LEFT LIMIT, at a slot outside [L,U) ---
            if (!last) {
                // grid point U holds the right limit; we add half the left limit
                part = part + _mps_single_point(bits_of(U), l_f_disc[i] / Cscalar(2, 0));
            } else if (padding_bit > 0) {
                part = part + _mps_single_point(bits_of(N_orig), f_b / Cscalar(2, 0));
            } else {
                // b aliases onto index 0, which is outside this interval when K >= 1
                std::vector<int> zeros(nB, 0);
                part = part + _mps_single_point(zeros, f_b / Cscalar(2, 0));
            }

            mps_out = mps_out + part;
        }

        mps = mps_out;

        // Scale before the QFT to keep the contracted MPS well-conditioned.
        mps *= magic_tensor_qft::real_sqrt(RealScalar(grid.get_N()))
             * grid.get_dx() / (RealScalar(2) * pi<RealScalar>());
        mps = QFT_E_T._mul(mps, method);
        RealScalar dt = RealScalar(2) * pi<RealScalar>()
                    / (RealScalar(grid.get_N()) * grid.get_dx());
        auto mps_phase = _phase_mps_exp_1j_t_a(grid.get_nBits(), grid.get_a(), dt);
        mps_phase = MPS<Cscalar>(mps_phase.get_core(), max_chi, reltol, mps_phase.get_w());
        mps = MPO<Cscalar>::from_mps(mps_phase)._mul(mps, method);
        if (do_shift) mps = _fft_shift(mps);
        grid = grid.build_dual_grid(do_shift);
    }

    // --- FFT of a piecewise-continuous function, trapezoid-corrected ---
    //
    // l_disc      : discontinuity positions, strictly increasing, strictly in (a,b)
    // l_f_disc[j] : LEFT limit  g(d_j - eps).  The grid already stores the RIGHT
    //               limit g(d_j), so the left one cannot be recovered from the MPS.
    // f_b         : g(b), the right endpoint, which is not on the grid.
    //
    // Conditions for O(1/N^2):
    //   1. every d_j falls exactly on a grid point
    //   2. d_j is the FIRST point of the interval to its right (half-open [d_j, d_{j+1}))
    //   3. l_f_disc holds left limits
    // Conditions 4 (one global grid) and 5 (global a in the phase) are automatic
    // here, since there is only ever one transform.
    //
    // Each correction sets its slot to the average of the two one-sided limits;
    // index 0 additionally absorbs b, which aliases onto it when unpadded.
    void fft_discontinuous_loopless(const std::vector<RealScalar> &l_disc,
                        const std::vector<Cscalar>    &l_f_disc,
                        Cscalar f_b,
                        bool do_shift = true,
                        const std::string& method = "zip-up")
    {
        if (l_disc.size() != l_f_disc.size())
            throw std::invalid_argument("l_disc and l_f_disc must have equal length");

        // 1. Reverse cores -> MSB-first, for endian consistency with the QFT MPO
        mps = MPS<Cscalar>(reverse_cores(mps.get_core()), mps.get_max_bond_dim(),
                        mps.get_reltol(), mps.get_w());

        // 2. Endpoint and discontinuity corrections, all before the QFT
        {
            const int  nB     = int(mps.get_core().size());
            const Sint N_orig = Sint(1) << (nB - padding_bit);
            const RealScalar a  = grid.get_a();
            const RealScalar dx = grid.get_dx();

            auto bits_of = [nB](Sint n) {
                std::vector<int> v(nB, 0);
                for (int k = 0; k < nB; ++k)
                    v[k] = int((n >> (nB - 1 - k)) & Sint(1));
                return v;
            };

            // --- interior discontinuities: slot d_j -> (g(d_j^+) + g(d_j^-)) / 2 ---
            // Evaluate every value against the ORIGINAL mps before modifying it.
            std::vector<std::vector<int>> d_bits;
            std::vector<Cscalar>          d_delta;
            Sint prev = Sint(0);
            for (size_t j = 0; j < l_disc.size(); ++j) {
                RealScalar q = (l_disc[j] - a) / dx;
                const Sint i = _round_to_sint(q);
                _warn_if_discontinuity_is_off_grid(
                    "fft_discontinuous_loopless", j, l_disc[j], a, dx, q, i);
                if (i <= Sint(0) || i >= N_orig)
                    throw std::invalid_argument("discontinuity outside (a, b)");
                if (j > 0 && i <= prev)
                    throw std::invalid_argument("l_disc must be strictly increasing");
                prev = i;

                auto b_j = bits_of(i);
                Cscalar g_j = mps.eval(b_j);                       // right limit, on grid
                d_bits.push_back(b_j);
                d_delta.push_back((l_f_disc[j] - g_j) / Cscalar(2, 0));
            }

            // --- endpoints ---
            std::vector<int> zeros(nB, 0);
            Cscalar g_a = mps.eval(zeros);

            if (padding_bit == 0) {
                // b = a + N*dE aliases exactly onto index 0, since
                // exp(-i*b*tau_m) == exp(-i*a*tau_m) for every m on the grid.
                mps = mps + _mps_single_point(zeros, (f_b - g_a) / Cscalar(2, 0));
            } else {
                // b is a real grid point at index N_orig
                mps = mps - _mps_single_point(zeros, g_a / Cscalar(2, 0));
                mps = mps + _mps_single_point(bits_of(N_orig), f_b / Cscalar(2, 0));
            }

            // --- apply the discontinuity corrections ---
            for (size_t j = 0; j < d_bits.size(); ++j)
                mps = mps + _mps_single_point(d_bits[j], d_delta[j]);
        }

        // 3. Scale before the QFT to keep the contracted MPS well-conditioned.
        mps *= magic_tensor_qft::real_sqrt(RealScalar(grid.get_N()))
             * grid.get_dx() / (RealScalar(2) * pi<RealScalar>());

        // 4. Apply QFT MPO
        mps = QFT_E_T._mul(mps, method);

        // 5. Time-dependent phase: exp(-1j * t * a)
        RealScalar dt = RealScalar(2) * pi<RealScalar>()
                    / (RealScalar(grid.get_N()) * grid.get_dx());
        auto mps_phase = _phase_mps_exp_1j_t_a(grid.get_nBits(), grid.get_a(), dt);
        mps_phase = MPS<Cscalar>(mps_phase.get_core(), max_chi, reltol, mps_phase.get_w());
        mps = MPO<Cscalar>::from_mps(mps_phase)._mul(mps, method);

        // 6. Optional fftshift
        if (do_shift)
            mps = _fft_shift(mps);

        // 7. Update grid to dual grid
        grid = grid.build_dual_grid(do_shift);
    }

    // --- vanilla inverse FFT (no trapezoid rule) ---
    // `a_primal` is the left bound of the ORIGINAL energy interval (E_0).
    // It cannot be recovered from the dual grid, so it must be supplied.
    // `do_shift` must match the value used in the forward transform.
    // WARNING TODO to test, cycle with fft -> ifft ->fft
    void ifft_vanilla(RealScalar a_primal, bool do_shift = true)
    {
        // 1. Undo the fftshift (swapping the MSB physical index is an involution)
        if (do_shift)
            mps = _fft_shift(mps);

        // 2. Undo the time-dependent phase: multiply by exp(+1j * t * a)
        //    dt is the dual-grid spacing, i.e. the current grid.get_dx()
        RealScalar dt = grid.get_dx();
        auto mps_phase = _phase_mps_exp_1j_t_a_conj(grid.get_nBits(), a_primal, dt);
        mps_phase = MPS<Cscalar>(mps_phase.get_core(), max_chi, reltol, mps_phase.get_w());
        auto phase_mpo = MPO<Cscalar>::from_mps(mps_phase);
        mps = phase_mpo._mul(mps);

        // 3. Undo the two scalar factors sqrt(N) and dE/(2*pi) in one step.
        //    1 / (sqrt(N) * dE/(2*pi)) = sqrt(N) * dt,  since dE = 2*pi/(N*dt).
        mps *= magic_tensor_qft::real_sqrt(RealScalar(grid.get_N())) * dt;

        // 4. Apply the inverse QFT MPO
        mps = dagger_mpo(QFT_E_T)._mul(mps);

        // 5. Reverse cores back to the original endianness
        mps = MPS<Cscalar>(reverse_cores(mps.get_core()), mps.get_max_bond_dim(),
                        mps.get_reltol(), mps.get_w());

        // 6. Recover the primal grid
        grid = grid.build_dual_grid(do_shift);
    }

    Eigen::Index get_chi() const { return mps.get_chi(); }
    MPS<Cscalar> const& get_mps() const { return mps; }
    QTGrid<RealScalar, Sint> const& get_grid() const { return grid; }
    Cscalar eval(std::vector<int> const& bits) const { return mps.eval(bits); }
    RealScalar get_a() const { return grid.get_a(); }

private:
    MPS<Cscalar> mps;
    QTGrid<RealScalar, Sint> grid;
    MPO<Cscalar> QFT_E_T;
    int padding_bit;
    RealScalar reltol = Eigen::NumTraits<RealScalar>::epsilon() * 100;
    int max_chi;

    // --- padding -----------------------------------------------------------
    static void _add_one_slow_bit(MPS<Cscalar>& mps, QTGrid<RealScalar, Sint>& grid)
    {
        auto cores = mps.get_core();
        Eigen::Index chi_right = cores.back().n_right;
        Tensor3D<Cscalar> new_core(chi_right, 2, 1);
        for (Eigen::Index r = 0; r < chi_right; ++r)
            new_core(r, 0, 0) = Cscalar(1, 0);
        cores.push_back(std::move(new_core));
        mps = MPS<Cscalar>(std::move(cores), mps.get_max_bond_dim(), mps.get_reltol(), mps.get_w());
        grid.update_padding_1h_bit();
    }

    static void _add_n_padding(MPS<Cscalar>& mps, QTGrid<RealScalar, Sint>& grid, int n)
    {
        for (int i = 0; i < n; ++i)
            _add_one_slow_bit(mps, grid);
    }

    // --- fftshift: swap |0⟩ ↔ |1⟩ on the last (MSB) core ---
    static MPS<Cscalar> _fft_shift(MPS<Cscalar> const& mps)
    {
        auto cores = mps.get_core();
        auto& last = cores.back();
        Tensor3D<Cscalar> swapped(last.n_left, last.n_phys, last.n_right);
        for (Eigen::Index l = 0; l < last.n_left; ++l)
            for (Eigen::Index r = 0; r < last.n_right; ++r) {
                swapped(l, 0, r) = last(l, 1, r);
                swapped(l, 1, r) = last(l, 0, r);
            }
        cores.back() = std::move(swapped);
        return MPS<Cscalar>(std::move(cores), mps.get_max_bond_dim(), mps.get_reltol(), mps.get_w());
    }

    // --- MPS for half-trapezoid rule: 1/2 at position 0 (all bits 0),
    //     1 everywhere else. Uses bond dimension 2 automaton:
    //     state 0 = "still all zeros", state 1 = "seen a non-zero bit".
    static MPS<Cscalar> _mps_left_trap_correction(int nBits, int max_chi_=0, RealScalar reltol_=-1, int w_=0)
    {
        std::vector<Tensor3D<Cscalar>> cores;
        if (nBits == 1) {
            Tensor3D<Cscalar> core(1, 2, 1);
            core(0, 0, 0) = Cscalar(0.5, 0);
            core(0, 1, 0) = Cscalar(1, 0);
            cores.push_back(std::move(core));
            return MPS<Cscalar>(std::move(cores));
        }

        // first core (MSB in post-reverse_cores convention)
        Tensor3D<Cscalar> c0(1, 2, 2);
        c0(0, 0, 0) = Cscalar(1, 0);  // bit=0 → stay in state 0
        c0(0, 1, 1) = Cscalar(1, 0);  // bit=1 → go to state 1
        cores.push_back(std::move(c0));

        // middle cores: pass through unless already in state 1
        for (int i = 0; i < nBits - 2; ++i) {
            Tensor3D<Cscalar> cm(2, 2, 2);
            cm(0, 0, 0) = Cscalar(1, 0);  // st0,bit0 → st0
            cm(0, 1, 1) = Cscalar(1, 0);  // st0,bit1 → st1
            cm(1, 0, 1) = Cscalar(1, 0);  // st1,bit0 → st1
            cm(1, 1, 1) = Cscalar(1, 0);  // st1,bit1 → st1
            cores.push_back(std::move(cm));
        }

        // last core (LSB): output the weight
        Tensor3D<Cscalar> cl(2, 2, 1);
        cl(0, 0, 0) = Cscalar(0.5, 0);  // st0,bit0 → 0.5 (all zeros!)
        cl(0, 1, 0) = Cscalar(1, 0);    // st0,bit1 → 1
        cl(1, 0, 0) = Cscalar(1, 0);    // st1,bit0 → 1
        cl(1, 1, 0) = Cscalar(1, 0);    // st1,bit1 → 1
        cores.push_back(std::move(cl));

        return MPS<Cscalar>(std::move(cores), max_chi_, reltol_, w_);
    }

    // --- MPS that is non-zero only at a single bit pattern ---
    static MPS<Cscalar> _mps_single_point(const std::vector<int>& bits, Cscalar value)
    {
        int nBits = static_cast<int>(bits.size());
        std::vector<Tensor3D<Cscalar>> cores;
        cores.reserve(nBits);
        for (int k = 0; k < nBits; ++k) {
            Tensor3D<Cscalar> core(1, 2, 1);
            core(0, bits[k], 0) = (k == 0) ? value : Cscalar(1, 0);
            cores.push_back(std::move(core));
        }
        return MPS<Cscalar>(std::move(cores));
    }

    // --- MPS that evaluates to a constant value everywhere ---
    static MPS<Cscalar> _constant_mps(int nBits, Cscalar value)
    {
        std::vector<Tensor3D<Cscalar>> cores;
        cores.reserve(nBits);
        for (int k = 0; k < nBits; ++k) {
            Tensor3D<Cscalar> core(1, 2, 1);
            Cscalar v = (k == 0) ? value : Cscalar(1, 0);
            core(0, 0, 0) = v;
            core(0, 1, 0) = v;
            cores.push_back(std::move(core));
        }
        return MPS<Cscalar>(std::move(cores));
    }

    // --- phase MPS: exp(-1j * t_m * a) ------------------------------------
    // Generate all bit phases from one base-angle exponential.  Independent
    // exp() calls at a*dt*2^k require increasingly inaccurate large-argument
    // reduction (notably for dd_real) and destroy the cancellation used by
    // small negative two's-complement indices.
    static MPS<Cscalar> _phase_mps_exp_1j_t_a(int nBits, RealScalar a, RealScalar dt)
    {
        std::vector<Tensor3D<Cscalar>> cores;
        cores.reserve(nBits);

        auto normalize_phase = [](Cscalar value) {
            using std::abs;
            const RealScalar magnitude = abs(value);
            return value / Cscalar(magnitude, 0);
        };

        // Lower bits have weights +2^k in the signed time index.
        Cscalar phase = normalize_phase(std::exp(Cscalar(0, -a * dt)));
        for (int k = 0; k < nBits - 1; ++k) {
            Tensor3D<Cscalar> core(1, 2, 1);
            core(0, 0, 0) = Cscalar(1, 0);
            core(0, 1, 0) = phase;
            cores.push_back(std::move(core));

            // Keep the bit phases mutually coherent and on the unit circle.
            // Normalization prevents a radial roundoff error from being
            // doubled by every subsequent squaring.
            phase = normalize_phase(phase * phase);
        }

        // The MSB has two's-complement weight -2^(nBits-1), hence the
        // conjugate of the phase produced by the same doubling recurrence.
        Tensor3D<Cscalar> msb_core(1, 2, 1);
        msb_core(0, 0, 0) = Cscalar(1, 0);
        msb_core(0, 1, 0) = std::conj(phase);
        cores.push_back(std::move(msb_core));

        return MPS<Cscalar>(std::move(cores));
    }

    static void _warn_if_discontinuity_is_off_grid(
        const char* implementation,
        size_t discontinuity_number,
        RealScalar discontinuity,
        RealScalar a,
        RealScalar dx,
        RealScalar q,
        Sint nearest_index)
    {
        using std::abs;
        const RealScalar grid_error = abs(q - RealScalar(nearest_index));
        const RealScalar warning_threshold = RealScalar(1e-9);
        if (grid_error <= warning_threshold)
            return;

        const RealScalar ulp_scale =
            Eigen::NumTraits<RealScalar>::epsilon()
            * std::max(RealScalar(1), abs(q));

        std::ostringstream warning;
        warning << std::setprecision(
                       std::numeric_limits<RealScalar>::max_digits10)
                << "[FFT warning] " << implementation
                << ": discontinuity[" << discontinuity_number << "]"
                << "; x=" << discontinuity
                << ", a=" << a
                << ", dx=" << dx
                << ", q=" << q
                << ", nearest_index=" << RealScalar(nearest_index)
                << ", |q-index|=" << grid_error
                << ", approximate_ulps=" << grid_error / ulp_scale
                << ", warning_threshold=" << warning_threshold
                << '\n';
        std::cerr << warning.str();
    }

    // Round a grid coordinate directly to the configured index type. In
    // particular, preserve both components of dd_128 instead of narrowing
    // through long long.
    static Sint _round_to_sint(RealScalar value)
    {
        using std::ceil;
        using std::floor;
        const RealScalar rounded = value < RealScalar(0)
            ? ceil(value - RealScalar(0.5))
            : floor(value + RealScalar(0.5));

        if constexpr (std::is_same_v<RealScalar, dd_128>) {
            return static_cast<Sint>(rounded._hi())
                 + static_cast<Sint>(rounded._lo());
        } else if constexpr (std::is_arithmetic_v<Sint>
                             || std::is_same_v<Sint, __int128>) {
            return static_cast<Sint>(rounded);
        } else {
            return rounded.template convert_to<Sint>();
        }
    }

    // Diagonal indicator for L <= n < U, as an MPS over MSB-first bits.
    // Bond dim 4: state = (sL, sU), sL in {equal-so-far, already-greater},
    //                               sU in {equal-so-far, already-less}.
    // has_L / has_U = false disables that bound (interval open on that side).
    // Requires Tensor3D to zero-initialise.
    static MPS<Cscalar> _mps_indicator_interval(int nBits, Sint L, Sint U,
                                                bool has_L, bool has_U,
                                                int max_chi_, RealScalar reltol_)
    {
        auto bit_of = [nBits](Sint x, int k) -> int {
            return int((x >> (nBits - 1 - k)) & Sint(1));
        };

        const int S = 4;                                   // 0=(E,E) 1=(E,L) 2=(G,E) 3=(G,L)
        const int s_init = (has_L ? 0 : 2) + (has_U ? 0 : 1);

        std::vector<Tensor3D<Cscalar>> cores;
        for (int k = 0; k < nBits; ++k) {
            const int l = has_L ? bit_of(L, k) : 0;
            const int u = has_U ? bit_of(U, k) : 0;
            const int dl = (k == 0)          ? 1 : S;
            const int dr = (k == nBits - 1)  ? 1 : S;
            Tensor3D<Cscalar> c(dl, 2, dr);

            for (int s = 0; s < S; ++s) {
                if (k == 0 && s != s_init) continue;
                const int sL = s / 2, sU = s % 2;
                for (int v = 0; v < 2; ++v) {
                    int nL, nU;
                    if      (sL == 1 || !has_L) nL = 1;    // already greater
                    else if (v == l)            nL = 0;
                    else if (v >  l)            nL = 1;
                    else                        continue;  // n < L, dead
                    if      (sU == 1 || !has_U) nU = 1;    // already less
                    else if (v == u)            nU = 0;
                    else if (v <  u)            nU = 1;
                    else                        continue;  // n > U, dead

                    const int row = (k == 0) ? 0 : s;
                    if (k == nBits - 1) {
                        if (nU != 1) continue;             // equality with U excluded
                        c(row, v, 0) += Cscalar(1, 0);
                    } else {
                        c(row, v, nL * 2 + nU) += Cscalar(1, 0);
                    }
                }
            }
            cores.push_back(std::move(c));
        }
        return MPS<Cscalar>(std::move(cores), max_chi_, reltol_);
    }

    // Zero MPS: every entry is 0. Bond dimension 1 throughout.
    static MPS<Cscalar> _zero_mps(int nBits, int max_chi_ = 0, RealScalar reltol_ = -1)
    {
        std::vector<Tensor3D<Cscalar>> cores;
        cores.reserve(nBits);
        for (int k = 0; k < nBits; ++k) {
            Tensor3D<Cscalar> c(1, 2, 1);
            c(0, 0, 0) = Cscalar(0, 0);
            c(0, 1, 0) = Cscalar(0, 0);
            cores.push_back(std::move(c));
        }
        return MPS<Cscalar>(std::move(cores), max_chi_, reltol_);
    }

};
