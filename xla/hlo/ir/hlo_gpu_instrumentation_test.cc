#include <iostream>

#include "xla/literal_util.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/client/xla_computation.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"

int main() {
  std::cout << "start building hlomodule!" << std::endl;
  auto builder = xla::HloComputation::Builder("a_plus_b_times_c");

  auto param_shape = xla::ShapeUtil::MakeShape(xla::F32, {}); // Input shape is a scalar.
  auto a = builder.AddInstruction(
      xla::HloInstruction::CreateParameter(0, param_shape, "a"));
  auto b = builder.AddInstruction(
      xla::HloInstruction::CreateParameter(1, param_shape, "b"));
  auto c = builder.AddInstruction(
      xla::HloInstruction::CreateParameter(2, param_shape, "c"));

  auto a_plus_b = builder.AddInstruction(
      xla::HloInstruction::CreateBinary(param_shape, xla::HloOpcode::kAdd, a, b));

  // (a+b)*c
  auto result = builder.AddInstruction(
      xla::HloInstruction::CreateBinary(param_shape, xla::HloOpcode::kMultiply, a_plus_b, c));

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
    std::cout << "================================== modified module: ==================================" << std::endl;
    std::cout << modified_module->ToString();
    std::cout << "======================================================================================" << std::endl;
  }

  // create input data
  auto a_literal = xla::LiteralUtil::CreateR0<float>(2.0f);
  auto b_literal = xla::LiteralUtil::CreateR0<float>(3.0f);
  auto c_literal = xla::LiteralUtil::CreateR0<float>(5.0f);

  std::unique_ptr<xla::PjRtBuffer> param_a =
      client->BufferFromHostLiteral(a_literal, client->addressable_devices()[0])
          .value();
  std::unique_ptr<xla::PjRtBuffer> param_b =
      client->BufferFromHostLiteral(b_literal, client->addressable_devices()[0])
          .value();
  std::unique_ptr<xla::PjRtBuffer> param_c =
      client->BufferFromHostLiteral(c_literal, client->addressable_devices()[0])
          .value();

  // execute the module
  xla::ExecuteOptions execute_options;
  auto results = executable->Execute({{param_a.get(), param_b.get(), param_c.get()}}, execute_options).value();
  std::cout << "execute success." << std::endl;

  return 0;
}
