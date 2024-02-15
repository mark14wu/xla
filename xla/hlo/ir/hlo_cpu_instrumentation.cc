/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/hlo/ir/hlo_cpu_instrumentation.h"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/dynamic_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "xla/service/custom_call_status.h"
#include "xla/service/custom_call_target_registry.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/status.h"
#include "xla/status_macros.h"
#include "xla/statusor.h"
#include "xla/types.h"
#include "xla/util.h"
// #include "tensorflow/core/lib/core/errors.h"
// #include "tensorflow/core/platform/errors.h"
// #include "tensorflow/core/platform/logging.h"
namespace xla {

namespace jpr {

enum JPrStatus { JPrOK, JPrFAIL };
struct JPrEventArgs {
  std::string device_type;
  std::string call_stack;
  JPrEventArgs(const std::string _device_type, const std::string _call_stack)
      : device_type(_device_type), call_stack(_call_stack) {}
};

struct JPrEvent {
  std::string name;
  long long ts;
  std::shared_ptr<JPrEventArgs> args_ptr;
  JPrEvent(const std::string _name, const long long _ts,
           const std::string _device_type, const std::string _call_stack)
      : name(_name), ts(_ts) {
    args_ptr = std::make_shared<JPrEventArgs>(_device_type, _call_stack);
  }
};

class JPrTracer {
 public:
  JPrStatus addEvent(const std::string _name, const std::string _device_type,
                     const std::string _call_stack) {
    auto ts = std::chrono::duration_cast<std::chrono::microseconds>(
                  std::chrono::system_clock::now().time_since_epoch())
                  .count();
    _events.emplace_back(_name, ts, _device_type, _call_stack);

    const auto& event = _events[_events.size() - 1];
    std::cout << std::setw(30) << event.name << "|";
    std::cout << std::setw(20) << std::fixed << std::setprecision(0)
              << (double)event.ts << "|";
    std::cout << std::setw(10) << event.args_ptr->device_type << "|";
    std::cout << event.args_ptr->call_stack << std::endl;

    return JPrOK;
  }
  JPrStatus Print() {
    std::cout << std::setw(30) << "HLO Instruction Call"
              << "|";
    std::cout << std::setw(20) << "Time Stamp"
              << "|";
    std::cout << std::setw(10) << "Device Type"
              << "|";
    std::cout << "Call Stack" << std::endl;
    for (const auto& event : _events) {
      std::cout << std::setw(30) << event.name << "|";
      std::cout << std::setw(20) << std::fixed << std::setprecision(0)
                << (double)event.ts << "|";
      std::cout << std::setw(10) << event.args_ptr->device_type << "|";
      std::cout << event.args_ptr->call_stack << std::endl;
    }
    return JPrOK;
  }

 private:
  std::vector<JPrEvent> _events;
};

JPrTracer jpr_tracer;

}  // namespace jpr

void CpuInstrCallback(void* output, void** inputs, const char* opaque,
                      size_t opaque_len, XlaCustomCallStatus* status) {
  auto call_name = std::string(opaque);
  jpr::jpr_tracer.addEvent(call_name, "CPU", " - ");
}

// XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(symbol, address, "Host")
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("cpu_instrumentation_callback",
                                         &CpuInstrCallback, "Host");
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("cpu_instrumentation_callback",
                                         &CpuInstrCallback, "CUDA");

/*static*/ StatusOr<bool> HloCpuInstr::RunOnComputation(
    HloComputation* computation) {
  bool changed = true;

  // Left empty
  return changed;
}
StatusOr<bool> HloCpuInstr::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = true;
  auto main_computation = module->entry_computation();

  std::vector<xla::HloInstruction*> instructions;
  for (xla::HloInstruction* instruction : main_computation->instructions()) {
    instructions.push_back(instruction);
  }
  int instr_cnt = 0;
  for (xla::HloInstruction* instruction : instructions) {
    auto opaque = std::string("custom_call");
    if (instr_cnt != 0) {
      opaque = opaque + std::string(".") + std::to_string(instr_cnt);
    }
    instr_cnt++;
    // instruction->name();
    auto instr = Cast<HloCustomCallInstruction>(
        main_computation->AddInstruction(HloInstruction::CreateCustomCall(
            ShapeUtil::MakeShape(F32, {}),
            /*operands=*/{},
            /*custom_call_target=*/"cpu_instrumentation_callback", opaque,
            CustomCallApiVersion::API_VERSION_STATUS_RETURNING_UNIFIED)));
    instr->set_custom_call_has_side_effect(true);
    TF_RETURN_IF_ERROR(instruction->AddControlDependencyTo(instr));
    for (xla::HloInstruction* user : instruction->users()) {
      TF_RETURN_IF_ERROR(instr->AddControlDependencyTo(user));
    }
  }

  return changed;
}

}  // namespace xla
