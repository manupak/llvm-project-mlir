// RUN: rocmlir-opt -migraphx-to-tosa --tosa-partition="anchor-ops=tosa.reduce_sum,tosa.matmul" --xmodel-async-graph --xmodel-target-kernels="targets=%arch" %s | rocmlir-driver -host-pipeline highlevel | rocmlir-gen -ph -print-results -p_verify=always -rand 1 -rand_type float -fut test_basic --verifier clone - | rocmlir-driver -host-pipeline xmodel -kernel-pipeline full -targets %arch | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// CLONE: 1e-05 < relDiff <= 1e-04: {{.*}} (0.000000%)
// CLONE-NEXT: 1e-04 < relDiff <= 1e-03: {{.*}} (0.000000%)
// CLONE-NEXT: 1e-03 < relDiff <= 1e-02: {{.*}} (0.000000%)
// CLONE-NEXT: 1e-02 < relDiff <= 1e-01: {{.*}} (0.000000%)
// CLONE-NEXT: 1e-01 < relDiff <= 1e+00: {{.*}} (0.000000%)
// CLONE-NEXT: 1e+00 < relDiff < inf : {{.*}} (0.000000%)
// CLONE-NEXT: relDiff = inf : {{.*}} (0.000000%)

func.func @test_basic(%arg0: tensor<1x1x768xf32>, %arg1: tensor<1x256x768xf32>, %arg2: tensor<1x256x3072xf32>, %arg3: tensor<1x3072x768xf32>) -> tensor<1x256x1xf32> {
    %0 = "migraphx.multibroadcast"(%arg0) {out_dyn_dims = [], out_lens = [1, 256, 768]} : (tensor<1x1x768xf32>) -> tensor<1x256x768xf32>
    %1 = "migraphx.dot"(%arg2, %arg3) : (tensor<1x256x3072xf32>, tensor<1x3072x768xf32>) -> tensor<1x256x768xf32>
    %2 = "migraphx.add"(%1, %0) : (tensor<1x256x768xf32>, tensor<1x256x768xf32>) -> tensor<1x256x768xf32>
    %3 = "migraphx.add"(%2, %arg1) : (tensor<1x256x768xf32>, tensor<1x256x768xf32>) -> tensor<1x256x768xf32>
    %4 = "migraphx.reduce_sum"(%3) {axes = [2]} : (tensor<1x256x768xf32>) -> tensor<1x256x1xf32>
    return %4 : tensor<1x256x1xf32>
}
