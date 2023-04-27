// RUN: rocmlir-opt -migraphx-to-tosa %s | rocmlir-driver -host-pipeline highlevel | rocmlir-gen -ph -print-results -rand none - | rocmlir-driver -arch %arch -c  | mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2
module {
  // CHECK: [1 1 1]
  // CHECK-NEXT: Unranked Memref base
  func.func @mlir_dot(%arg0: tensor<1x384x3072xf32>, %arg1: tensor<1x384x768xf32>, %arg2: tensor<1x768x3072xf32>) -> tensor<1x384x3072xf32> attributes {arch = "gfx90a:sramecc+:xnack-", kernel = "mixr"} {
    %0 = migraphx.multibroadcast(%arg2) {out_dyn_dims = [], out_lens = [1, 768, 3072]} : (tensor<1x768x3072xf32>) -> tensor<1x768x3072xf32>
    %1 = "tosa.const"() {value = dense<1.000000e+00> : tensor<1xf32>} : () -> tensor<1xf32>
    %2 = "tosa.const"() {value = dense<0.707106769> : tensor<1xf32>} : () -> tensor<1xf32>
    %3 = "tosa.const"() {value = dense<5.000000e-01> : tensor<1xf32>} : () -> tensor<1xf32>
    %4 = migraphx.multibroadcast(%3) {out_dyn_dims = [], out_lens = [1, 384, 3072]} : (tensor<1xf32>) -> tensor<1x384x3072xf32>
    %5 = migraphx.multibroadcast(%2) {out_dyn_dims = [], out_lens = [1, 384, 3072]} : (tensor<1xf32>) -> tensor<1x384x3072xf32>
    %6 = migraphx.multibroadcast(%1) {out_dyn_dims = [], out_lens = [1, 384, 3072]} : (tensor<1xf32>) -> tensor<1x384x3072xf32>
    %7 = migraphx.dot(%arg1, %0) {xdlopsV2 = true} : tensor<1x384x768xf32>, tensor<1x768x3072xf32> -> tensor<1x384x3072xf32>
    %8 = migraphx.add(%7, %arg0) : (tensor<1x384x3072xf32>, tensor<1x384x3072xf32>) -> tensor<1x384x3072xf32>
    %9 = migraphx.mul(%8, %5) : (tensor<1x384x3072xf32>, tensor<1x384x3072xf32>) -> tensor<1x384x3072xf32>
    %10 = migraphx.erf(%9) : (tensor<1x384x3072xf32>) -> tensor<1x384x3072xf32>
    %11 = migraphx.add(%10, %6) : (tensor<1x384x3072xf32>, tensor<1x384x3072xf32>) -> tensor<1x384x3072xf32>
    %12 = migraphx.mul(%8, %11) : (tensor<1x384x3072xf32>, tensor<1x384x3072xf32>) -> tensor<1x384x3072xf32>
    %13 = migraphx.mul(%12, %4) : (tensor<1x384x3072xf32>, tensor<1x384x3072xf32>) -> tensor<1x384x3072xf32>
    return %13 : tensor<1x384x3072xf32>
  }
}
