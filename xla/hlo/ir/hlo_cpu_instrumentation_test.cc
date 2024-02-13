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

#include "hlo_cpu_instrumentation.h"

#include <memory>
#include <iostream>

#include "xla/literal_util.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/client/xla_computation.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/tfrt_cpu_pjrt_client.h"

int main() {
  std::cout << "start building hlomodule!" << std::endl;
  auto builder = xla::HloComputation::Builder("123");
  auto constant1 = builder.AddInstruction(
      xla::HloInstruction::CreateConstant(xla::LiteralUtil::CreateR0<float>(42.0f)));
  auto constant2 = builder.AddInstruction(
      xla::HloInstruction::CreateConstant(xla::LiteralUtil::CreateR0<float>(123.0f)));
  builder.AddInstruction(xla::HloInstruction::CreateBinary(
      constant1->shape(), xla::HloOpcode::kAdd, constant1, constant2));

  std::unique_ptr<xla::HloModule> module = std::make_unique<xla::HloModule>("module", xla::HloModuleConfig());

  auto computation = module->AddEntryComputation(builder.Build());
  std::cout << "add entry computation success." << std::endl;

  // convert hlomodule to xla computation
  xla::XlaComputation xla_computation(module->ToProto());
  std::cout << "convert hlomodule to xla computation success." << std::endl;

  // get a client
  std::unique_ptr<xla::PjRtClient> client = xla::GetTfrtCpuClient(/*asynchronous=*/true).value();
  std::cout << "get client success." << std::endl;

  // compile to executable
  xla::CompileOptions compile_options;
  std::unique_ptr<xla::PjRtLoadedExecutable> executable = client->Compile(xla_computation, compile_options).value();
  std::cout << "compile success." << std::endl;

  auto compiled_modules = executable->GetHloModules().value();
  for (auto compiled_module: compiled_modules) {
    std::cout << "hlomodule before modify: " << std::endl << compiled_module->ToString() << std::endl;
    std::cout << "=========================" << std::endl;
    xla::HloCpuInstr cpu_instr;
    if (!cpu_instr.Run(compiled_module.get()).value()) {
      std::cout << "cpu instr run fail!" << std::endl;
    } else {
      std::cout << "cpu instr run success." << std::endl;
    }
    std::cout << "after modify hlomodule: " << std::endl << compiled_module->ToString() << std::endl;
    std::cout << "=========================" << std::endl;
  }

  auto modified_modules = executable->GetHloModules().value();
  for (auto modified_module: modified_modules) {
    std::cout << "modified module: " << std::endl;
    std::cout << modified_module->ToString() << "=========================" << std::endl;
  }

  // execute the module
  xla::ExecuteOptions execute_options;
  auto results = executable->Execute({{}}, execute_options).value();
  std::cout << "execute success." << std::endl;

  return 0;
}
