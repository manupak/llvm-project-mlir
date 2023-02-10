// RUN: rocmlir-gen -ph -print-results -rand none %s | rocmlir-driver -arch %arch -c  | /opt/rocm/bin/rocprof --stats mlir-cpu-runner -O2 --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext --entry-point-result=void | FileCheck %s

// CHECK-COUNT-256:  [0]
module {
  func.func @test_zero_init(%arg0: memref<1x256x1xf32>) attributes {kernel, arch = "", grid_size = 1, block_size = 256} {
    rock.zero_init_kernel %arg0 {arch = "", blockSize = 256 : i32, elemsPerThread = 1 : index, gridSize = 1 : i32} : memref<1x256x1xf32>
    return
  }
}
