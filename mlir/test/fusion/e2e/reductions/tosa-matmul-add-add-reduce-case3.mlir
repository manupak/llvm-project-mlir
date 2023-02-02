// RUN: rocmlir-opt --tosa-partition="anchor-ops=tosa.reduce_sum,tosa.matmul" --xmodel-async-graph --xmodel-target-kernels="targets=%arch" %s | rocmlir-driver -host-pipeline highlevel | rocmlir-gen -ph -print-results -p_verify=always -rand 1 -rand_type float -fut test_basic --verifier clone - | rocmlir-driver -host-pipeline xmodel -kernel-pipeline full -targets %arch | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// CLONE: 1e-05 < relDiff <= 1e-04: {{.*}} (0.000000%)
// CLONE-NEXT: 1e-04 < relDiff <= 1e-03: {{.*}} (0.000000%)
// CLONE-NEXT: 1e-03 < relDiff <= 1e-02: {{.*}} (0.000000%)
// CLONE-NEXT: 1e-02 < relDiff <= 1e-01: {{.*}} (0.000000%)
// CLONE-NEXT: 1e-01 < relDiff <= 1e+00: {{.*}} (0.000000%)
// CLONE-NEXT: 1e+00 < relDiff < inf : {{.*}} (0.000000%)
// CLONE-NEXT: relDiff = inf : {{.*}} (0.000000%)

func.func @test_basic(%arg0: tensor<1x256x3072xf32>, %arg1: tensor<1x3072x768xf32>, %arg2: tensor<1x1x768xf32>, %arg3: tensor<1x256x768xf32>) -> tensor<1x256x1xf32> {
    %188 = "tosa.matmul"(%arg0, %arg1) {perf_config = "32,32,8,16,16,4,1,1"} : (tensor<1x256x3072xf32>, tensor<1x3072x768xf32>) -> tensor<1x256x768xf32>
    %189 = "tosa.add"(%188, %arg2) : (tensor<1x256x768xf32>, tensor<1x1x768xf32>) -> tensor<1x256x768xf32>
    %190 = "tosa.add"(%189, %arg3) : (tensor<1x256x768xf32>, tensor<1x256x768xf32>) -> tensor<1x256x768xf32>
    %191 = "tosa.reduce_sum"(%190) {axis = 2 : i64} : (tensor<1x256x768xf32>) -> tensor<1x256x1xf32>
    return %191 : tensor<1x256x1xf32>
}
