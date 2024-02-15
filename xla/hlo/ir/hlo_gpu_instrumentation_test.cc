#include <iostream>

#include "xla/literal_util.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/client/xla_computation.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"

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
  auto client = xla::GetStreamExecutorGpuClient(xla::GpuClientOptions()).value();
  std::cout << "get a gpu client success." << std::endl;

  // compile to executable
  xla::CompileOptions compile_options;
  std::unique_ptr<xla::PjRtLoadedExecutable> executable = client->Compile(xla_computation, compile_options).value();
  std::cout << "compile success." << std::endl;

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
