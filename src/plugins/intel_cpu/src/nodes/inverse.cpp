// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "inverse.hpp"

#include "nodes/common/cpu_memcpy.h"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/inverse.hpp"
#include "utils/bfloat16.hpp"

// Parallel LU decomposition algorithm with partial pivoting
// Based on the lectures by Prof. Dr. Thomas Huckle, Parallel Numerics

namespace ov {
namespace intel_cpu {
namespace node {

Inverse::Inverse(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op, EMPTY_PORT_MASK)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        THROW_CPU_NODE_ERR(errorMessage);
    }

    auto inverse_op = as_type_ptr<op::v14::Inverse>(op);
    m_adjoint = inverse_op->get_adjoint();

    constant = ConstantType::StrictNoConst;

    m_const_input = is_type<op::v0::Constant>(op->get_input_node_ptr(INPUT_PORT));
}

bool Inverse::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (op->get_type_info() != op::v14::Inverse::get_type_info_static()) {
            errorMessage = "Only Inverse operation from the opset14 is supported by the CPU plugin.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

void Inverse::getSupportedDescriptors() {
    if (getParentEdges().size() != 1) {
        THROW_CPU_NODE_ERR("has incorrect number of input edges.");
    }
    if (getChildEdges().empty()) {
        THROW_CPU_NODE_ERR("has incorrect number of output edges.");
    }
}

void Inverse::initSupportedPrimitiveDescriptors() {
    m_input_precision = getOriginalInputPrecisionAtPort(INPUT_PORT);
    if (!one_of(m_input_precision, ov::element::f32, ov::element::f16, ov::element::bf16)) {
        m_input_precision = ov::element::f32;
    }

    addSupportedPrimDesc({{LayoutType::ncsp, m_input_precision, m_const_input}},
                         {{LayoutType::ncsp, m_input_precision}},
                         ref_any);
}

void Inverse::prepareParams() {
    const auto& input_shape = getParentEdgeAt(INPUT_PORT)->getMemory().getStaticDims();

    if (input_shape.size() < 2) {
        THROW_CPU_NODE_ERR("has incompatible 'data' shape ",
                           PartialShape(input_shape),
                           ". Only tensors of rank at least 2 are allowed.");
    }

    m_side = input_shape.back();
    m_side_squared = m_side * m_side;
    m_batches_count = 1;

    for (size_t i = 0; i < input_shape.size() - 2; ++i) {
        m_batches_count = m_batches_count * input_shape[i];
    }
}

bool Inverse::created() const {
    return getType() == Type::Inverse;
}

void Inverse::execute(dnnl::stream strm) {
    OV_SWITCH(intel_cpu,
              InverseExecute,
              this,
              m_input_precision,
              OV_CASE(ov::element::bf16, bfloat16_t),
              OV_CASE(ov::element::f16, ov::float16),
              OV_CASE(ov::element::f32, float))
}

void Inverse::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

template <typename T>
void Inverse::inverse() {
    const auto* data = getSrcDataAtPortAs<const T>(INPUT_PORT);
    auto* output = getDstDataAtPortAs<T>(OUTPUT_PORT);

    std::vector<T> L(m_side_squared, T{0});
    std::vector<T> U(m_side_squared, T{0});
    std::vector<T> P(m_side, T{0});

    for (size_t b = 0; b < m_batches_count; ++b) {
        bool sign = true;
        lu_decomposition(data, L, U, P, sign, b);

        for (size_t column = 0; column < m_side; ++column) {
            lu_solve(output, L, U, P, b, column);
        }

        if (m_adjoint) {
            // Multiply by det(A) = det(U)
            to_adjoint(output, U, sign, b);
        }
    }
}

template <typename T>
void Inverse::lu_decomposition(const T* data,
                               std::vector<T>& L,
                               std::vector<T>& U,
                               std::vector<T>& P,
                               bool& sign,
                               size_t b) {
    // Make L identity, U a copy of data and P a range(0, side)
    size_t batch_idx = b * m_side_squared;

    std::fill(L.begin(), L.end(), T{0});
    cpu_parallel_memcpy(&U[0], &data[batch_idx], sizeof(T) * m_side_squared);

    parallel_for(m_side, [&](size_t i) {
        L[i * m_side + i] = T{1};
        P[i] = static_cast<T>(i);
    });

    for (size_t k = 0; k < m_side; ++k) {
        size_t pivot_row = k;
        size_t k_idx = k * m_side;
        size_t pivot_idx = pivot_row * m_side;

        size_t remaining_columns = m_side - k;
        size_t remaining_rows = remaining_columns - 1;

        // Find maximum value pivot - non-parallel
        for (size_t i = (k + 1) * m_side, j = k + 1; i < m_side_squared; i += m_side, ++j) {
            if (abs(U[i + k]) > abs(U[pivot_idx + k])) {
                pivot_row = j;
                pivot_idx = pivot_row * m_side;
            }
        }

        if (pivot_row != k) {
            // Swap rows in L, U and P
            sign = !sign;
            std::swap(P[k], P[pivot_row]);
            parallel_for(m_side, [&](size_t i) {
                std::swap(L[k_idx + i], L[pivot_idx + i]);
                std::swap(U[k_idx + i], U[pivot_idx + i]);
            });
        }

        parallel_for(remaining_rows, [&](size_t i) {
            size_t i_idx = (i + k + 1) * m_side;
            L[i_idx + k] = U[i_idx + k] / U[k_idx + k];
        });

        parallel_for(remaining_rows * remaining_columns, [&](size_t i) {
            size_t i_idx = (i / remaining_columns + k + 1) * m_side;
            size_t j_idx = i % remaining_columns + k;

            U[i_idx + j_idx] = U[i_idx + j_idx] - L[i_idx + k] * U[k_idx + j_idx];
        });
    }
}

template <typename T>
void Inverse::lu_solve(T* output, std::vector<T>& L, std::vector<T>& U, std::vector<T>& P, size_t b, size_t column) {
    std::vector<T> B(m_side, T{0});
    std::vector<T> X(m_side, T{0});
    std::vector<T> Y(m_side, T{0});
    B[column] = T{1};

    // Forward substitution: Ly = Pb - not possible to be parallel
    for (size_t i = 0; i < m_side; ++i) {
        Y[i] = B[P[i]];
        size_t i_idx = i * m_side;
        for (size_t j = 0; j < i; ++j) {
            Y[i] = Y[i] - L[i_idx + j] * Y[j];
        }
    }

    // Backward substitution: Ux = y - not possible to be parallel
    for (size_t i = 0; i < m_side; ++i) {
        size_t i_adj = m_side - i - 1;
        size_t i_idx = i_adj * m_side;
        X[i_adj] = Y[i_adj];
        for (size_t j = i_adj + 1; j < m_side; ++j) {
            X[i_adj] = X[i_adj] - U[i_idx + j] * X[j];
        }
        X[i_adj] = X[i_adj] / U[i_idx + i_adj];
    }

    // Substitute back to get result
    size_t batch_idx = b * m_side_squared;
    parallel_for(m_side, [&](size_t row) {
        output[batch_idx + row * m_side + column] = X[row];
    });
}

template <typename T>
void Inverse::to_adjoint(T* output, std::vector<T>& U, bool sign, size_t b) {
    T determinant = sign ? T{1} : T{-1};

    for (size_t i = 0; i < m_side; ++i) {
        determinant = determinant * U[i * m_side + i];
    }

    size_t batch_idx = b * m_side_squared;
    parallel_for(m_side_squared, [&](size_t i) {
        output[batch_idx + i] = output[batch_idx + i] * determinant;
    });
}

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
