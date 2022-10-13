// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

// clang-format off
#include "openvino/openvino.hpp"

#include "samples/args_helper.hpp"
#include "samples/common.hpp"
#include "samples/latency_metrics.hpp"
#include "samples/slog.hpp"
// clang-format on

using Ms = std::chrono::duration<double, std::ratio<1, 1000>>;

int main(int argc, char* argv[]) {
    try {
        slog::info << ov::get_openvino_version() << slog::endl;
        if (argc != 2) {
            slog::info << "Usage : " << argv[0] << " <path_to_model>" << slog::endl;
            return EXIT_FAILURE;
        }
        // Optimize for latency. Most of the devices are configured for latency by default,
        // but there are exceptions like MYRIAD
        ov::AnyMap latency{{ov::hint::performance_mode.name(), ov::hint::PerformanceMode::LATENCY}};
        // Create ov::Core and use it to compile a model
        // Pick device by replacing CPU, for example AUTO:GPU,CPU.
        // Using MULTI device is pointless in sync scenario
        // because only one instance of ov::InferRequest is used
        ov::CompiledModel compiled_model = ov::Core{}.compile_model(argv[1], "CPU", latency);
        ov::InferRequest ireq = compiled_model.create_infer_request();
        // Fill input data for the ireq
        for (const ov::Output<const ov::Node>& model_input : compiled_model.inputs()) {
            fill_tensor_random(ireq.get_tensor(model_input));
        }
        // Warm up
        ireq.infer();
        // Run benchmarking for seconds_to_run seconds
        std::chrono::seconds seconds_to_run{15};
        std::vector<double> latencies;
        auto start = std::chrono::steady_clock::now();
        auto time_point = start;
        auto time_point_to_finish = start + seconds_to_run;
        while (time_point < time_point_to_finish) {
            ireq.infer();
            auto iter_end = std::chrono::steady_clock::now();
            latencies.push_back(std::chrono::duration_cast<Ms>(iter_end - time_point).count());
            time_point = iter_end;
        }
        auto end = time_point;
        double duration = std::chrono::duration_cast<Ms>(end - start).count();
        // Report results
        slog::info << "Count:      " << latencies.size() << " iterations" << slog::endl;
        slog::info << "Duration:   " << duration << " ms" << slog::endl;
        slog::info << "Latency:" << slog::endl;
        size_t percent = 50;
        LatencyMetrics latency_metrics{latencies, "", percent};
        latency_metrics.write_to_slog();
        slog::info << "Throughput: " << double_to_string(latencies.size() * 1000 / duration) << " FPS" << slog::endl;
    } catch (const std::exception& ex) {
        slog::err << ex.what() << slog::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
