// RUN: rocmlir-opt --rock-fold-transpose -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-gridwise-gemm-to-blockwise -rock-linalg-align %s | FileCheck %s

// CHECK-LABEL: test_gemm_reduce_last_axis_fusion
func.func @test_gemm_reduce_last_axis_fusion(%arg0: memref<1x128x64xf32>, %arg1: memref<1x64x256xf32>, %arg2: memref<1x128x1xf32>) attributes {arch = "", kernel} {
  %0 = memref.alloc() : memref<1x128x256xf32>
  rock.gemm %0 = %arg0 * %arg1 features =  none storeMethod =  set {arch = ""} : memref<1x128x256xf32> = memref<1x128x64xf32> * memref<1x64x256xf32>
  // CHECK: %[[vecLoad:.*]] = vector.load
  // CHECK: %[[vecRed:.*]] = vector.reduction <add>, %[[vecLoad]]
  // CHECK: %[[vecRedReg:.*]] = rock.alloc() : memref<1xf32, 5>
  // CHECK: rock.in_bounds_store %[[vecRed]] -> %[[vecRedReg]][%c0]
  // CHECK: rock.global_store %[[vecRedReg]][%c0] -> %arg2[{{.*}}, {{.*}}, %c0] storeMethod( atomic_add) {{.*}} length = 1
  rock.reduce sum %0 into %arg2 {axis = 2 : index, blockSize = 256 : i32, gridSize = 1 : i32} : memref<1x128x256xf32> into memref<1x128x1xf32>
  return
}


// CHECK-LABEL: test_gemm_reduce_middle_axis_fusion
func.func @test_gemm_reduce_middle_axis_fusion(%arg0: memref<1x128x64xf32>, %arg1: memref<1x64x256xf32>, %arg2: memref<1x1x256xf32>) attributes {arch = "", kernel} {
  %0 = memref.alloc() : memref<1x128x256xf32>
  rock.gemm %0 = %arg0 * %arg1 features =  none storeMethod =  set {arch = ""} : memref<1x128x256xf32> = memref<1x128x64xf32> * memref<1x64x256xf32>
  // CHECK-NOT: vector.reduction
  // CHECK: rock.global_store {{.*}} -> %arg2[{{.*}}, %c0, {{.*}}] storeMethod( atomic_add)
  rock.reduce sum %0 into %arg2 {axis = 1 : index, blockSize = 256 : i32, gridSize = 1 : i32} : memref<1x128x256xf32> into memref<1x1x256xf32>
  return
}

// CHECK-LABEL: test_gemm_add_reduce_fusion
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @test_gemm_add_reduce_fusion(%arg0: memref<1x128x64xf32>, %arg1: memref<1x64x256xf32>, %arg2: memref<1x128x256xf32>, %arg3: memref<1x128x1xf32>) attributes {arch = "", kernel} {
  %0 = memref.alloc() : memref<1x128x256xf32>
  rock.gemm %0 = %arg0 * %arg1 features =  none storeMethod =  set {arch = ""} : memref<1x128x256xf32> = memref<1x128x64xf32> * memref<1x64x256xf32>
  %1 = memref.alloc() : memref<1x128x256xf32>
  //CHECK: linalg.generic {{.*}} outs(%[[addReg:.*]] : memref<4xf32, 5>)
  linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0, %arg2 : memref<1x128x256xf32>, memref<1x128x256xf32>) outs(%1 : memref<1x128x256xf32>) {
  ^bb0(%arg4: f32, %arg5: f32, %arg6: f32):
    %4 = arith.addf %arg4, %arg5 : f32
    linalg.yield %4 : f32
  }
  // CHECK: %[[vecLoad:.*]] = vector.load %[[addReg]][%c0]
  // CHECK: %[[vecRed:.*]] = vector.reduction <add>, %[[vecLoad]]
  // CHECK: %[[vecRedReg:.*]] = rock.alloc() : memref<1xf32, 5>
  // CHECK: rock.in_bounds_store %[[vecRed]] -> %[[vecRedReg]][%c0]
  // CHECK: rock.global_store %[[vecRedReg]][%c0] -> %arg3[{{.*}}, {{.*}}, %c0] storeMethod( atomic_add) {{.*}} length = 1
  rock.reduce sum %1 into %arg3 {axis = 2 : index, blockSize = 256 : i32, gridSize = 1 : i32} : memref<1x128x256xf32> into memref<1x128x1xf32>
  return
}
