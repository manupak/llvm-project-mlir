// This test is checking for larger reductions with larger block and grid sizes

// RUN: cat %s | rocmlir-gen -ph -print-results -fut test_basic -verifier clone - | rocmlir-driver -host-pipeline xmodel -kernel-pipeline full | /opt/rocm/bin/rocprof --stats xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// CLONE: [1 1 1]
// CLONE-NEXT: Unranked Memref base

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
module {
  func.func private @test_reduce__part_0(%arg0: memref<1x12x1xf32> {func.write_access}) {
    return
  }
  func.func private @test_basic__part_0(%arg0: memref<1x12x384xf32> {func.read_access}, %arg1: memref<1x384x384xf32> {func.read_access}, %arg2: memref<1x1x384xf32> {func.read_access}, %arg3: memref<1x12x384xf32> {func.read_access}, %arg4: memref<1x12x1xf32> {func.read_access, func.write_access}) {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = memref.alloc() {alignment = 128 : i64} : memref<1x12x384xf32>
    linalg.fill ins(%cst : f32) outs(%0 : memref<1x12x384xf32>)
    linalg.batch_matmul ins(%arg0, %arg1 : memref<1x12x384xf32>, memref<1x384x384xf32>) outs(%0 : memref<1x12x384xf32>)
    %1 = memref.collapse_shape %0 [[0, 1], [2]] : memref<1x12x384xf32> into memref<12x384xf32>
    %2 = memref.collapse_shape %arg2 [[0, 1, 2]] : memref<1x1x384xf32> into memref<384xf32>
    %3 = memref.collapse_shape %arg3 [[0, 1], [2]] : memref<1x12x384xf32> into memref<12x384xf32>
    %4 = memref.alloc() {alignment = 128 : i64} : memref<12x384xf32>
    linalg.generic {indexing_maps = [#map0, #map1, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%1, %2, %3 : memref<12x384xf32>, memref<384xf32>, memref<12x384xf32>) outs(%4 : memref<12x384xf32>) {
    ^bb0(%arg5: f32, %arg6: f32, %arg7: f32, %arg8: f32):
      %7 = arith.addf %arg5, %arg6 : f32
      %8 = arith.addf %7, %arg7 : f32
      linalg.yield %8 : f32
    }
    %5 = memref.alloc() {alignment = 128 : i64} : memref<12xf32>
    linalg.fill ins(%cst : f32) outs(%5 : memref<12xf32>)
    linalg.generic {indexing_maps = [#map0, #map2], iterator_types = ["parallel", "reduction"]} ins(%4 : memref<12x384xf32>) outs(%5 : memref<12xf32>) {
    ^bb0(%arg5: f32, %arg6: f32):
      %7 = arith.addf %arg5, %arg6 : f32
      linalg.yield %7 : f32
    }
    %6 = memref.expand_shape %5 [[0, 1, 2]] : memref<12xf32> into memref<1x12x1xf32>
    memref.dealloc %0 : memref<1x12x384xf32>
    memref.dealloc %4 : memref<12x384xf32>
    memref.copy %6, %arg4 : memref<1x12x1xf32> to memref<1x12x1xf32>
    return
  }
  func.func @test_basic(%arg0: memref<1x1x384xf32>, %arg1: memref<1x12x384xf32>, %arg2: memref<1x12x384xf32>, %arg3: memref<1x384x384xf32>, %arg4: memref<1x12x1xf32>) {
    %token0 = async.launch @test_reduce__part_0 (%arg4) : (memref<1x12x1xf32>) -> ()
    %token = async.launch @test_basic__part_0 [%token0] (%arg2, %arg3, %arg0, %arg1, %arg4) : (memref<1x12x384xf32>, memref<1x384x384xf32>, memref<1x1x384xf32>, memref<1x12x384xf32>, memref<1x12x1xf32>) -> ()
    async.await %token : !async.token
    return
  }
  module @__xmodule_gfx90a_srameccY_xnack_ attributes {xmodel.arch = "gfx90a:sramecc+:xnack-", xmodel.module} {
    func.func private @test_reduce__part_0(%arg0: memref<1x12x1xf32> {func.write_access}) attributes {kernel, original_func = @test_reduce__part_0, grid_size = 16, block_size = 512} {
      rock.zero_init_kernel %arg0 {arch = "", blockSize = 64 : i32, elemsPerThread = 1 : index, gridSize = 1 : i32} : memref<1x12x1xf32>
      return
    }
    func.func private @test_basic__part_0(%arg0: memref<1x12x384xf32> {func.read_access}, %arg1: memref<1x384x384xf32> {func.read_access}, %arg2: memref<1x1x384xf32> {func.read_access}, %arg3: memref<1x12x384xf32> {func.read_access}, %arg4: memref<1x12x1xf32> {func.read_access, func.write_access}) attributes {kernel, original_func = @test_basic__part_0} {
      %0 = memref.alloc() {alignment = 128 : i64} : memref<1x12x384xf32>
      rock.gemm %0 = %arg0 * %arg1 features =  mfma|dot|atomic_add storeMethod =  set {arch = "gfx90a:sramecc+:xnack-", perf_config = "64,16,8,16,16,4,1,1"} : memref<1x12x384xf32> = memref<1x12x384xf32> * memref<1x384x384xf32>
      %1 = memref.collapse_shape %0 [[0, 1], [2]] : memref<1x12x384xf32> into memref<12x384xf32>
      %2 = memref.collapse_shape %arg2 [[0, 1, 2]] : memref<1x1x384xf32> into memref<384xf32>
      %3 = memref.collapse_shape %arg3 [[0, 1], [2]] : memref<1x12x384xf32> into memref<12x384xf32>
      %4 = memref.alloc() {alignment = 128 : i64} : memref<12x384xf32>
      linalg.generic {indexing_maps = [#map0, #map1, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%1, %2, %3 : memref<12x384xf32>, memref<384xf32>, memref<12x384xf32>) outs(%4 : memref<12x384xf32>) {
      ^bb0(%arg5: f32, %arg6: f32, %arg7: f32, %arg8: f32):
        %6 = arith.addf %arg5, %arg6 : f32
        %7 = arith.addf %6, %arg7 : f32
        linalg.yield %7 : f32
      }
      %5 = memref.expand_shape %4 [[0, 1], [2]] : memref<12x384xf32> into memref<1x12x384xf32>
      rock.reduce  sum %5 into %arg4 {axis = 2 : index, blockSize = 256 : i32, gridSize = 768 : i32} : memref<1x12x384xf32> into memref<1x12x1xf32>
      return
    }
  }
}
