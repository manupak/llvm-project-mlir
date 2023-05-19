//===- BlockwiseGemmToThreadwise - MLIR Rock ops lowering passes ---===//
//
// Copyright 2020 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================
//
// This pass converts rock.blockwise_* ops to rock.threadwise_*
// and lowers other higher-level ops like transform and fill in preparation for
// the threadwise lowering
//
//===-----------------------------------------------------===//
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/Tuning/GeneralGemmBlockStructure.h"
#include "mlir/Dialect/Rock/utility/AmdArchDb.h"
#include "mlir/Dialect/Rock/utility/builderUtils.h"
#include "mlir/Dialect/Rock/utility/loweringUtils.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

#include "AccelEmitter.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKBLOCKWISEGEMMTOTHREADWISEPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-blockwise-to-threadwise"

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::rock;

namespace {
struct RockLowerBlockwiseGemmToThreadwisePass
    : public rock::impl::RockBlockwiseGemmToThreadwisePassBase<
          RockLowerBlockwiseGemmToThreadwisePass> {
  void runOnOperation() override;
};

//===----------------------------------------------------------------------===//
// Fill lowering.
//===----------------------------------------------------------------------===//

struct FillRewritePattern : public OpConversionPattern<FillOp> {
  using OpConversionPattern<FillOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(FillOp op, FillOpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    Location loc = op.getLoc();
    MemRefType inputType = op.getInput().getType();
    ArrayRef<int64_t> inputShape = inputType.getShape();
    llvm::SmallVector<int64_t> lbs(inputShape.size(), 0);
    llvm::SmallVector<int64_t> strides(inputShape.size(), 1);

    buildAffineLoopNest(b, loc, lbs, inputShape, strides,
                        [value = adaptor.getValue(),
                         input = adaptor.getInput()](OpBuilder &b, Location loc,
                                                     ValueRange ivs) {
                          b.create<memref::StoreOp>(loc, value, input, ivs);
                        });

    b.replaceOp(op, {});
    return success();
  }
};

//===----------------------------------------------------------------------===//
// BlockwiseGemm lowering.
//===----------------------------------------------------------------------===//

// The structure of this lowing is documented at
// https://github.com/ROCmSoftwarePlatform/rocMLIR/issues/719
struct BlockwiseGemmRewritePattern
    : public OpConversionPattern<BlockwiseGemmOp> {
  using OpConversionPattern<BlockwiseGemmOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(BlockwiseGemmOp op,
                                BlockwiseGemmOpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    Location loc = op.getLoc();

    // Prepare some useful constants.
    Value zeroConstantOp = b.createOrFold<ConstantIndexOp>(loc, 0);

    MemRefType blockAType = op.getMatrixA().getType(),
               blockBType = op.getMatrixB().getType(),
               bufferCType = op.getMatrixC().getType();

    auto elementType = bufferCType.getElementType();

    int64_t k = blockAType.getShape()[0];
    int64_t m = blockAType.getShape()[1];
    int64_t n = blockBType.getShape()[1];
    int64_t kPack = blockAType.getShape()[2];

    // Non-accelerator path.

    // Obtain critical attributes.
    int64_t mC = bufferCType.getShape()[0];
    int64_t nC = bufferCType.getShape()[1];

    GeneralGemmParamsAttr params = op.getParams();
    uint32_t blockSize = params.getBlockSize();
    int64_t kPerThread = params.getKPerThread();
    int64_t mPerThread = params.getMPerThread();
    int64_t nPerThread = params.getNPerThread();

    GeneralGemmBlockStructure blockStructure =
        *deriveGeneralGemmBlockStructure(blockSize);

    int64_t mThreadsPerCuwave = blockStructure.mThreadsPerCuwave;
    int64_t nThreadsPerCuwave = blockStructure.nThreadsPerCuwave;
    int64_t cuwaveLen = mThreadsPerCuwave * nThreadsPerCuwave;

    int64_t mCuwavesPerBlock = blockStructure.mCuwavesPerBlock;
    int64_t nCuwavesPerBlock = blockStructure.nCuwavesPerBlock;
    int64_t numCuwaves = mCuwavesPerBlock * nCuwavesPerBlock;
    int64_t derivedBlockSize = numCuwaves * cuwaveLen;
    assert(blockSize == derivedBlockSize &&
           "block structure parameters must multiply to block size");

    int64_t mRepeat = mC / mPerThread;
    int64_t nRepeat = nC / nPerThread;

    if (mRepeat * mCuwavesPerBlock * mThreadsPerCuwave * mPerThread != m)
      return op.emitOpError("The m turing attributes don't multiply to M_LDS");
    if (nRepeat * nCuwavesPerBlock * nThreadsPerCuwave * nPerThread != n)
      return op.emitOpError("The n turing parameters don't multiply to N_LDS");

    LLVM_DEBUG(llvm::dbgs()
               << "M: " << m << "\n"
               << "mRepeat: " << mRepeat << "\n"
               << "mCuwavesPerBlock: " << mCuwavesPerBlock << "\n"
               << "mThreadsPerCuwave: " << mThreadsPerCuwave << "\n"
               << "mPerThread: " << mPerThread << "\n"
               << "n: " << n << "\n"
               << "nRepeat: " << nRepeat << "\n"
               << "nCuwavesPerBlock: " << nCuwavesPerBlock << "\n"
               << "nThreadsPerCuwave: " << nThreadsPerCuwave << "\n"
               << "nPerThread: " << nPerThread << "\n");

    auto ldsTidSplitter = [&](StringRef repeatName, int64_t repeatLen,
                              StringRef perThreadName,
                              int64_t perThreadLen) -> TopDownTMBuilder {
      TopDownTMBuilder splitTidForLDS(
          b, {"k", repeatName, "tid", perThreadName, "kpack"},
          {k, repeatLen, blockSize, perThreadLen, kPack}, loc);
      splitTidForLDS.passThrough({"k", repeatName});
      splitTidForLDS.merge({"m_cuwaves", "n_cuwaves", "m_cuwave", "n_cuwave"},
                           {2, 3, 4, 5}, "tid",
                           {mCuwavesPerBlock, nCuwavesPerBlock,
                            mThreadsPerCuwave, nThreadsPerCuwave});
      splitTidForLDS.passThrough({perThreadName, "kpack"}, {6, 7},
                                 {perThreadName, "kpack"});
      return splitTidForLDS;
    };

    TopDownTMBuilder splitTidA =
        ldsTidSplitter("m_repeat", mRepeat, "m_thread", mPerThread);
    TransformMapAttr splitTidAAttr = splitTidA.get();
    auto toLdsIndexA = TopDownTMBuilder::below(splitTidA, splitTidAAttr);
    toLdsIndexA.passThrough("k");
    toLdsIndexA.unmerge(
        "m", 1, {"m_repeat", "m_cuwaves", "m_cuwave", "m_thread"},
        {mRepeat, mCuwavesPerBlock, mThreadsPerCuwave, mPerThread});
    toLdsIndexA.ignore("n_cuwaves");
    toLdsIndexA.ignore("n_cuwave");
    toLdsIndexA.passThrough({"kpack"}, {2}, {"kpack"});
    TransformMapAttr toLdsIndexAAttr = toLdsIndexA.get();

    TopDownTMBuilder splitTidB =
        ldsTidSplitter("n_repeat", nRepeat, "n_thread", nPerThread);
    TransformMapAttr splitTidBAttr = splitTidB.get();
    auto toLdsIndexB = TopDownTMBuilder::below(splitTidB, splitTidBAttr);
    toLdsIndexB.passThrough("k");
    toLdsIndexB.unmerge(
        "n", 1, {"n_repeat", "n_cuwaves", "n_cuwave", "n_thread"},
        {nRepeat, nCuwavesPerBlock, nThreadsPerCuwave, nPerThread});
    toLdsIndexB.ignore("m_cuwaves");
    toLdsIndexB.ignore("m_cuwave");
    toLdsIndexB.passThrough({"kpack"}, {2}, {"kpack"});
    TransformMapAttr toLdsIndexBAttr = toLdsIndexB.get();

    Value matrixA, matrixB;
    ArrayAttr transformsA, transformsB;
    std::tie(matrixA, transformsA) =
        untransform(b, adaptor.getMatrixA(),
                    b.getArrayAttr({splitTidAAttr, toLdsIndexAAttr}));
    std::tie(matrixB, transformsB) =
        untransform(b, adaptor.getMatrixB(),
                    b.getArrayAttr({splitTidBAttr, toLdsIndexBAttr}));

    int64_t threadANumRegisters = kPerThread * mC * kPack;
    int64_t threadBNumRegisters = kPerThread * nC * kPack;

    // Alloc register for thread_a and thread_b.
    auto privateMemoryAddressSpace = b.getAttr<gpu::AddressSpaceAttr>(
        gpu::GPUDialect::getPrivateAddressSpace());
    auto threadARegisterMemRefType =
        MemRefType::get(threadANumRegisters, elementType, AffineMap{},
                        privateMemoryAddressSpace);
    auto threadAAllocOp = b.create<GpuAllocOp>(loc, threadARegisterMemRefType);

    auto threadBRegisterMemRefType =
        MemRefType::get(threadBNumRegisters, elementType, AffineMap{},
                        privateMemoryAddressSpace);
    auto threadBAllocOp = b.create<GpuAllocOp>(loc, threadBRegisterMemRefType);

    // Define views of register tiles for copies
    BottomUpTMBuilder viewA(b, {"raw"}, {threadANumRegisters}, loc);
    viewA.unmerge({"k", "m_repeat", "tid", "m_thread", "kpack"},
                  {0, 1, 2, 3, 4}, "raw",
                  {kPerThread, mRepeat, 1, mPerThread, kPack});
    TransformMapAttr threadACopyViewAttr = viewA.get();

    BottomUpTMBuilder viewB(b, {"raw"}, {threadBNumRegisters}, loc);
    viewB.unmerge({"k", "n_repeat", "tid", "n_thread", "kpack"},
                  {0, 1, 2, 3, 4}, "raw",
                  {kPerThread, nRepeat, 1, nPerThread, kPack});
    TransformMapAttr threadBCopyViewAttr = viewB.get();

    // Main loop.
    Value workitem = b.createOrFold<rock::WorkitemIdOp>(loc, b.getIndexType());
    LLVM_DEBUG(llvm::dbgs() << "Outer loop:\n "
                            << "k =  " << k << "\n"
                            << " kPerThread = " << kPerThread << "\n");
    auto loopOp = b.replaceOpWithNewOp<AffineForOp>(op, 0, k, kPerThread);
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(loopOp.getBody());
    Value kOffset = loopOp.getInductionVar();

    SmallVector<Value, 5> registerStartCoords(5, zeroConstantOp);
    SmallVector<Value, 5> ldsBufferAStartCoords = {
        kOffset, zeroConstantOp, workitem, zeroConstantOp, zeroConstantOp};
    auto copyALoop = b.create<TransformingForOp>(
        loc, ArrayRef<ValueRange>{ldsBufferAStartCoords, registerStartCoords},
        ArrayRef<Attribute>{transformsA, b.getArrayAttr(threadACopyViewAttr)},
        ArrayRef<int64_t>{kPerThread, mRepeat, 1, mPerThread, kPack},
        /*strides=*/std::nullopt, /*forceUnroll=*/true, /*indexDiffs=*/true);
    {
      OpBuilder::InsertionGuard copyAGuard(b);
      b.setInsertionPointToStart(copyALoop.getBody());
      Value aCopy = b.create<memref::LoadOp>(
          loc, matrixA, copyALoop.getLowerCoords(/*domain=*/0));
      Value aCast = createTypeConversionOp(b, loc, aCopy, elementType);
      b.create<memref::StoreOp>(loc, aCast, threadAAllocOp,
                                copyALoop.getLowerCoords(/*domain=*/1));
    }

    SmallVector<Value, 5> ldsBufferBStartCoords = {
        kOffset, zeroConstantOp, workitem, zeroConstantOp, zeroConstantOp};
    auto copyBLoop = b.create<TransformingForOp>(
        loc, ArrayRef<ValueRange>{ldsBufferBStartCoords, registerStartCoords},
        ArrayRef<Attribute>{transformsB, b.getArrayAttr(threadBCopyViewAttr)},
        ArrayRef<int64_t>{kPerThread, nRepeat, 1, nPerThread, kPack},
        /*strides=*/std::nullopt, /*forceUnroll=*/true, /*indexDiffs=*/true);
    {
      OpBuilder::InsertionGuard copyBGuard(b);
      b.setInsertionPointToStart(copyBLoop.getBody());
      Value bCopy = b.create<memref::LoadOp>(
          loc, matrixB, copyBLoop.getLowerCoords(/*domain=*/0));
      Value bCast = createTypeConversionOp(b, loc, bCopy, elementType);
      b.create<memref::StoreOp>(loc, bCast, threadBAllocOp,
                                copyBLoop.getLowerCoords(/*domain=*/1));
    }

    Value reshapedARegisters = reshapeBuffer(
        b, loc, threadAAllocOp, {"k", "m", "kpack"}, {kPerThread, mC, kPack});
    Value reshapedBRegisters = reshapeBuffer(
        b, loc, threadBAllocOp, {"k", "n", "kpack"}, {kPerThread, nC, kPack});
    // Actually do the gemm - this goes inside the look over kOffset
    b.create<ThreadwiseGemmOp>(loc, reshapedARegisters, reshapedBRegisters,
                               op.getMatrixC());

    return success();
  }
};

//===----------------------------------------------------------------------===//
// BlockwiseReduce lowering.
//===----------------------------------------------------------------------===//

struct BlockwiseReduceRewritePattern
    : public OpConversionPattern<BlockwiseReduceOp> {
  using OpConversionPattern<BlockwiseReduceOp>::OpConversionPattern;

  int64_t calculateNonReductionDimProduct(ArrayRef<int64_t> toReduceShape, int64_t axis) const {
    int64_t dimProduct = 1;
    for(size_t i=0; i < toReduceShape.size(); i++){
      if(i!=axis){
        dimProduct *= toReduceShape[i];
      }
    }
    return dimProduct;
  }

  // This should only be used if product non-reduction dims is
  // equal or larger than number threads in a block.
  //
  // Given a input tensor : D0, ... , Dr , ... , DN to reduce,
  // This function creates a view that maps the space of
  // [D0, ... , Dr , ... , DN] --> [tid, nrIter, rIter] where
  // tid is threads within the block, nrIter is non-reducing 
  // iterations within a thread and rIter is reducing iterations
  // within a thread. 
  //
  // NOTE : at this stage reducing iterations are calculated as if 
  //
  ArrayAttr createThreadViewForNRLargerThanThreads(Location loc, 
                             ArrayRef<int64_t> toReduceShape, 
                             int64_t blockSize, 
                             int64_t reduceAxis, PatternRewriter &rewriter) const {
    BottomUpTMBuilder threadsToTensor(rewriter, toReduceShape, loc);
    SmallVector<StringRef, 4> lowerNameRefs;
    threadsToTensor.getStartNames(lowerNameRefs);

    int64_t nonReduceMergeDimSize = 1;
    SmallVector<StringRef, 4> nonReduceNameRefs;
    for (auto dimAndSize : llvm::enumerate(toReduceShape)){
      size_t dim = dimAndSize.index();
      int64_t dimSize = dimAndSize.value();
      if(dim != reduceAxis){
        nonReduceMergeDimSize *= dimSize;
        nonReduceNameRefs.push_back(lowerNameRefs[dim]);
      }
    }
    threadsToTensor.merge("nrDim", 0, nonReduceNameRefs);
    threadsToTensor.passThrough("rIter", nonReduceNameRefs[reduceAxis]);
    TransformMapAttr mergeTrMap = threadsToTensor.get();

    threadsToTensor = BottomUpTMBuilder::above(threadsToTensor, mergeTrMap);
    int64_t nrThreads = (nonReduceMergeDimSize + (blockSize - 1)) / blockSize;
    threadsToTensor.pad({"nrDimPad"}, {0, blockSize * nrThreads - nonReduceMergeDimSize});
    threadsToTensor.passThrough("rIter", "rIter");
    TransformMapAttr padTrMap = threadsToTensor.get();

    threadsToTensor = BottomUpTMBuilder::above(threadsToTensor, padTrMap);
    threadsToTensor.unmerge({"tid", "nrIter"}, {0, 1}, "nrDimPad", {blockSize, nrThreads});
    threadsToTensor.passThrough("rIter", "rIter");
    TransformMapAttr unmergeTrMap = threadsToTensor.get();

    return rewriter.getArrayAttr({unmergeTrMap, padTrMap, mergeTrMap});
  }

  // This should only be used if product non-reduction dims is
  // less than number threads in a block.
  //
  // Given a input tensor : D0, ... , Dr , ... , DN to reduce,
  // This function creates a view that maps the space of
  // [D0, ... , Dr , ... , DN] --> [nrtid, rtid, rIter] where
  // nrtid = tid / product(non-reduction dims) is a reduction subgroup leader.
  // rtid = tid % product(non-reduction dims) is thread idx within a reduction subgroup.
  // | rtid | is the number of threads that'd participate in a reduction

  ArrayAttr createThreadViewforNRSmallerThanThreads(Location loc,
                                                    ArrayRef<int64_t> toReduceShape,
                                                    int64_t blockSize,
                                                    int64_t reduceAxis,
                                                    PatternRewriter &rewriter
                                                    ){
    BottomUpTMBuilder threadsToTensor(rewriter, toReduceShape, loc);
    SmallVector<StringRef, 4> lowerNameRefs;
    threadsToTensor.getStartNames(lowerNameRefs);

    int64_t nonReduceMergeDimSize = 1;
    SmallVector<StringRef, 4> nonReduceNameRefs;
    for (auto dimAndSize : llvm::enumerate(toReduceShape)){
      size_t dim = dimAndSize.index();
      int64_t dimSize = dimAndSize.value();
      if(dim != reduceAxis){
        nonReduceMergeDimSize *= dimSize;
        nonReduceNameRefs.push_back(lowerNameRefs[dim]);
      }
    }
    threadsToTensor.merge("nrDim", 0, nonReduceNameRefs);
    threadsToTensor.passThrough("rDim", nonReduceNameRefs[reduceAxis]);
    TransformMapAttr mergeTrMap = threadsToTensor.get();

    threadsToTensor = BottomUpTMBuilder::above(threadsToTensor, mergeTrMap);
    // Distribute threads that is extra than nonReduceMergeDimSize and use them
    // for reductions. This is a floor because max threads per non-reduction axes 
    // and we use remaining for reductions.
    int64_t rthreads = blockSize / nonReduceMergeDimSize;
    int64_t rDimPerRThread = toReduceShape[reduceAxis] + (rthreads - 1) / rthreads;
    threadsToTensor.pad({"rDimPad"}, {0, rthreads * rDimPerRThread - toReduceShape[reduceAxis]});
    threadsToTensor.passThrough("nrDim", "nrDim");
    TransformMapAttr padTrMap = threadsToTensor.get();

    threadsToTensor = BottomUpTMBuilder::above(threadsToTensor, padTrMap);
    threadsToTensor.unmerge({"rtid", "rIter"}, {0, 1}, "rDimPad", {rthreads, rDimPerRThread});
    threadsToTensor.passThrough("nrtid", "nrDim");
    TransformMapAttr unmergeTrMap = threadsToTensor.get();

    return rewriter.getArrayAttr({unmergeTrMap, padTrMap, mergeTrMap}); 
  }

  LogicalResult matchAndRewrite(BlockwiseReduceOp op,
                                BlockwiseReduceOpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    TransformMapAttr inputView = op.getInputRegViewAttr();
    TypedValue<MemRefType> inputReg = op.getInput();
    ArrayRef<int64_t> regShape = inputReg.getType().getShape();
    Type elemType = inputReg.getType().getElementType();
    TypedValue<MemRefType> workspaceLDSBuffer = op.getWorkspaceBuffer();
    int64_t vectorLength = inputReg.getType().getDimSize(0);
    Value zeroConstantOp = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    int64_t axis = op.getAxis().getSExtValue();
    int64_t blockSize = op.getBlockSize();
    auto privateMemoryAddressSpace = rewriter.getAttr<gpu::AddressSpaceAttr>(
        gpu::GPUDialect::getPrivateAddressSpace());
    // Get current workitem ID.
    WorkitemIdOp tid =
        rewriter.create<WorkitemIdOp>(loc, rewriter.getIndexType());

    // Create strides and bounds to iterate the virtual tensor
    TransformAttr upperTr = inputView.getOps()[0];
    SmallVector<int64_t, 4> regTensorShape;
    std::transform(upperTr.getUpperDims().begin(), 
                   upperTr.getUpperDims().end(), 
                   regTensorShape.begin(),
                   [](unsigned int val){
      return static_cast<int64_t>(val);
    });

    // The following RAII scope will create register to LDS store loop
    {
      SmallVector<int64_t> bounds(regTensorShape.size(), 1LL);
      SmallVector<int64_t> strides(regTensorShape.size(), 1LL);

      // Create thread-based view of the tensor

      //Create iteration inits.
      auto [buffer, regTransforms] = untransform(rewriter, inputReg);
      ArrayRef<int64_t> bufferShape =
        buffer.getType().cast<ShapedType>().getShape();
      Type loadTypeInputReg = elemType;
      for(int64_t dim=0; dim < regShape.size(); dim++){
        // if it is the reduction axis, we dont use the vectorization.
        if(axis == dim){
          strides[dim] = 1;
        }
        else{
          // Check the vectorLen w.r.t registers
          int64_t vectorLenReg = getMaxVectorizationForDatatype(regTransforms, dim, regShape[dim], bufferShape, elemType);
          // Check the vectorLen w.r.t virtual input tensor as we would need to store them to LDS
          SmallVector<Attribute> virtualInputViewTransforms {inputView.getOps()};
          int64_t vectorLenVirtual = getMaxVectorizationForDatatype(rewriter.getArrayAttr(virtualInputViewTransforms), dim, regShape[dim], inputView.getLowerBounds().asArrayRef(), elemType);
          int64_t vectorLen = std::min(vectorLenVirtual, vectorLenReg);
          Type loadTypeInputReg = vectorTypeOrSelf(elemType, vectorLen);
          strides[dim] = vectorLen;
          // There will only one dimension that will be vectorized.
          if(vectorLen > 1){
            break;
          }
        }
      }
      // This is viewing registers as the tensor as opposed to the flat array it is.
      // i.e. tensor of {d0, ... , dn} where d are dimensions of original tensor {D0, ... , Dn}.
      // where dx is a subset of Dx.
      SmallVector<Value, 4> registerTensorCoords (regTensorShape.size(), zeroConstantOp);
      SmallVector<Value, 4> registerInputViewCoords = registerTensorCoords;
      registerInputViewCoords.push_back(tid);

      //Store loop
      TransformingForOp LDSStoreLoop = rewriter.create<TransformingForOp>(
        loc, ArrayRef<ValueRange>{registerTensorCoords, registerInputViewCoords},
        ArrayRef<Attribute>{inputView, regTransforms}, ArrayRef<int64_t>(bounds),
        ArrayRef<int64_t>(strides),
        /*forceUnroll=*/true, /*useIndexDiffs=*/true);
      {
        PatternRewriter::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(LDSStoreLoop.getBody());
        Block::BlockArgListType registerLoadCoords = LDSStoreLoop.getLowerCoords(/*domain=*/0);
        Block::BlockArgListType ldsStoreCoords = LDSStoreLoop.getLowerCoords(/*domain=*/1);
        Value loadVal = rewriter.create<InBoundsLoadOp>(loc, loadTypeInputReg, inputReg, registerLoadCoords);
        rewriter.create<InBoundsStoreOp>(loc, loadVal, workspaceLDSBuffer, ldsStoreCoords);
      }
    }


    //Following RAII scope will create reduction loops.
    {
      int64_t nonReductionDimSizeProduct = calculateNonReductionDimProduct(regTensorShape, axis);
      if(blockSize < nonReductionDimSizeProduct){
        // This means there aren't enough threads to do a parallel reduction
        // each individual thread could do its own reduction.
        ArrayAttr threadViewTrs = createThreadViewForNRLargerThanThreads(loc, regTensorShape, blockSize, axis, rewriter);

        // inputView is viewing the input reg from the virtual preReduce tensor.
        // threadViewTrs is viewing thread-level from the virtual preReduce tensor.
        // So when we invert inputView and plug into threadViewTrs, we get input reg view from thread-level view.
        SmallVector<TransformMapAttr, 4> regToPreReduceTensorToThreadView = llvm::to_vector<4>(threadViewTrs.getAsRange<TransformMapAttr>());
        TransformMapAttr invertInputView = invertTransformMap(rewriter, inputView, loc);
        regToPreReduceTensorToThreadView.push_back(invertInputView);
        // TODO : create array attr to be used with down TransformingForOp
        ArrayAttr regToPreReduceTensorToThreadViewAttr = rewriter.getArrayAttr(regToPreReduceTensorToThreadView);

        ArrayRef<int64_t> threadViewShape = threadViewTrs[0].cast<TransformMapAttr>().getUpperBounds();
        constexpr size_t nrIterDim = 1;
        constexpr size_t rIterDim = 2;

        int64_t nrIterVectorLen = getMaxVectorizationForDatatype(threadViewTrs, nrIterDim, threadViewShape[nrIterDim], regTensorShape, elemType);
        AffineForOp nrIterLoop = rewriter.create<AffineForOp>(loc, 0, threadViewShape[nrIterDim] - 1, nrIterVectorLen);
        {
          // inside the loop.
          PatternRewriter::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(nrIterLoop.getBody());
          Value nrIter = nrIterLoop.getInductionVar();

          int64_t rIterVectorLen = getMaxVectorizationForDatatype(threadViewTrs, rIterDim, threadViewShape[rIterDim], regTensorShape, elemType);
          SmallVector<Value, 4> inits{tid, nrIter, zeroConstantOp};
          SmallVector<int64_t> bounds{1, 1, threadViewShape[rIterDim]};
          SmallVector<int64_t> strides{1, 1, rIterVectorLen};

          TransformingForOp reductionLoop = rewriter.create<TransformingForOp>(
          loc, inits, ArrayRef<Attribute>{threadViewTrs, regToPreReduceTensorToThreadView}, ArrayRef<int64_t>(bounds),
          ArrayRef<int64_t>(strides), /*forceUnroll=*/true, /*useIndexDiffs=*/true);
          {
            PatternRewriter::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(reductionLoop.getBody());
            Block::BlockArgListType LDSLoadCoords = reductionLoop.getLowerCoords(/*domain=*/0);

            if(nrIterVectorLen > 1){
              // This means non-reduction dimension is contigous in memory.
              Type loadTypeInputReg = vectorTypeOrSelf(elemType, nrIterVectorLen);
              // This is expected to be a vector of values that are not being reduced with each other.
              Value loadVal = rewriter.create<InBoundsLoadOp>(loc, loadTypeInputReg, workspaceLDSBuffer, LDSLoadCoords);
            }
            else if(rIterVectorLen > 1){
              // This means reduction dimension is contigous in memory.

            }
            else{
              // non of them are contigous in memory
            }

          }


        }


      }
      else{

      }
    }








    return success();
  }
};

//===----------------------------------------------------------------------===//
// BlockwiseGemmAccel lowering.
//===----------------------------------------------------------------------===//

struct BlockwiseGemmAccelRewritePattern
    : public OpConversionPattern<BlockwiseGemmAccelOp> {
  using OpConversionPattern<BlockwiseGemmAccelOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(BlockwiseGemmAccelOp op,
                                BlockwiseGemmAccelOpAdaptor adaptor,
                                ConversionPatternRewriter &b) const override {
    Location loc = op.getLoc();

    StringAttr arch = op.getArchAttr();
    RockAccelTuningParamAttrInterface tuningParams = op.getParams();
    int64_t M = tuningParams.getMPerBlock();
    int64_t N = tuningParams.getNPerBlock();
    int64_t K = tuningParams.getKpackPerBlock();
    int64_t mPerWave = tuningParams.getMPerWave();
    int64_t nPerWave = tuningParams.getNPerWave();
    int64_t KPack = tuningParams.getKpack();

    Type bufferElemTypeA =
        adaptor.getMatrixA().getType().cast<MemRefType>().getElementType();
    Type bufferElemTypeB =
        adaptor.getMatrixB().getType().cast<MemRefType>().getElementType();
    Type dataTypeA = bufferElemTypeA, dataTypeB = bufferElemTypeB;
    if (auto bufferVecTypeA = bufferElemTypeA.dyn_cast<VectorType>())
      dataTypeA = bufferVecTypeA.getElementType();
    if (auto bufferVecTypeB = bufferElemTypeB.dyn_cast<VectorType>())
      dataTypeB = bufferVecTypeB.getElementType();

    Value sourceOffsetA = adaptor.getWaveOffsetA();
    Value sourceOffsetB = adaptor.getWaveOffsetB();

    auto accelEmitterPtr = rock::accel::AccelEmitter::select(
        op.getFeatures(), dataTypeA, dataTypeB, arch, tuningParams);

    if (!accelEmitterPtr)
      return op.emitOpError("Unable to emit accelerator code.");

    // Extract relevant accelerator parameters
    rock::accel::AccelEmitterParams params = accelEmitterPtr->getParams();
    Type argTypeA = params.argTypeA;
    Type argTypeB = params.argTypeB;
    int64_t mRepeats = params.mRepeats;
    int64_t nRepeats = params.nRepeats;
    int64_t mPerAccel = params.mPerAccel;
    int64_t nPerAccel = params.nPerAccel;
    int64_t kBase = params.kBase;
    int64_t kpackPerThread = params.kpackPerThread;

    auto tid = b.create<WorkitemIdOp>(loc, b.getIndexType());
    const int64_t waveSize = rock::lookupArchInfo(arch).waveSize;
    auto laneId =
        b.create<RemUIOp>(loc, tid, b.create<ConstantIndexOp>(loc, waveSize));

    LLVM_DEBUG(llvm::dbgs()
               << "argVectorType A: " << argTypeA << "\n"
               << "argVectorType B: " << argTypeB << "\n"
               << "k_base: " << kBase << "\n"
               << "mPerWave: " << mPerWave << "\n"
               << "nPerWave: " << nPerWave << "\n"
               << "mRepeat: " << mRepeats << "\n"
               << "nRepeat: " << nRepeats << "\n"
               << "K: " << K << "\n"
               << "bufferA type: " << adaptor.getBufferA().getType() << "\n"
               << "bufferB type: " << adaptor.getBufferB().getType() << "\n");

    Value MConstantOp = b.create<ConstantIndexOp>(loc, M);
    Value NConstantOp = b.create<ConstantIndexOp>(loc, N);

    Value mPerAccelConstantOp = b.create<ConstantIndexOp>(loc, mPerAccel);
    Value nPerAccelConstantOp = b.create<ConstantIndexOp>(loc, nPerAccel);

    Value bufferA = adaptor.getBufferA();
    Value bufferB = adaptor.getBufferB();

    Value KPerThreadConstantOp = b.create<ConstantIndexOp>(loc, kpackPerThread);

    auto ldsToRegisterCopy = [&](Location loc, OpBuilder mnb, OpBuilder kb,
                                 Value sourceBase, Value mn_i, Value MN,
                                 Value k_i, Value K, Value mnPerMfmaGroup,
                                 Type ldsBufferElemType, Type dataType,
                                 Value ldsOrig, Value regDest) {
      // Compute source offset
      Value sourceOffset = accelEmitterPtr->computeLdsSourceOffset(
          kb, k_i, mnb, mn_i, b, MN, loc, sourceBase, laneId);

      Value value = kb.create<memref::LoadOp>(loc, ldsBufferElemType, ldsOrig,
                                              sourceOffset);

      auto bufferType = regDest.getType().cast<MemRefType>();
      Type bufferElementType = bufferType.getElementType();

      // We're loading in units of kPack, but storing in units of k_base.
      if (KPack == kBase) {
        Value destOffset = k_i;
        kb.create<memref::StoreOp>(loc, value, regDest, ValueRange{destOffset});
      } else if (KPack > kBase) {
        int64_t numStores = KPack / kBase;
        Value baseDestOffset = kb.createOrFold<arith::MulIOp>(
            loc, k_i, kb.createOrFold<arith::ConstantIndexOp>(loc, numStores));
        for (int64_t i = 0; i < numStores; ++i) {
          Value sliceStart =
              kb.createOrFold<arith::ConstantIndexOp>(loc, kBase * i);
          Value slice = kb.create<ExtractSliceOp>(loc, bufferElementType, value,
                                                  sliceStart);
          Value destOffset = kb.createOrFold<arith::AddIOp>(
              loc, baseDestOffset,
              kb.createOrFold<arith::ConstantIndexOp>(loc, i));
          kb.create<memref::StoreOp>(loc, slice, regDest,
                                     ValueRange{destOffset});
        }
      } else if (KPack < kBase) {
        // Here we are gathering loaded values into vectors for passing into
        // MFMAs.
        Value destValsPerKpack =
            kb.createOrFold<arith::ConstantIndexOp>(loc, kBase / KPack);
        // This is fine, since the inputs to MFMAs are contiguous in the k
        // dimension.
        Value destOffset =
            kb.createOrFold<arith::DivUIOp>(loc, k_i, destValsPerKpack);
        Value destVecPart =
            kb.createOrFold<arith::RemUIOp>(loc, k_i, destValsPerKpack);
        Value destSlicePos = kb.createOrFold<arith::MulIOp>(
            loc, destVecPart,
            b.createOrFold<arith::ConstantIndexOp>(loc, KPack));
        Value destVec = kb.create<memref::LoadOp>(
            loc, bufferElementType, regDest, ValueRange{destOffset});
        Value newDestVec = kb.create<InsertSliceOp>(
            loc, bufferElementType, value, destVec, destSlicePos);
        kb.create<memref::StoreOp>(loc, newDestVec, regDest,
                                   ValueRange{destOffset});
      }
    };

    auto ldsToRegisterCopyKdim =
        [&](OpBuilder outerLoopB, AffineForOp outerLoopBodyOp, Value sourceBase,
            Value MN, Value mnPerMfmaGroup, Type ldsBufferElemType,
            Type dataType, Value ldsOrig, Value regDest) {
          auto innerLoopK =
              outerLoopB.create<AffineForOp>(loc, 0, kpackPerThread);
          auto ilkb = ConversionPatternRewriter::atBlockBegin(
              innerLoopK.getBody(), outerLoopB.getListener());
          {
            OpBuilder::InsertionGuard guard(b);
            b.setInsertionPoint(outerLoopBodyOp);
            OpBuilder::InsertionGuard guardBody(outerLoopB);
            outerLoopB.setInsertionPointToStart(outerLoopBodyOp.getBody());
            ldsToRegisterCopy(loc, outerLoopB, ilkb, sourceBase,
                              outerLoopBodyOp.getInductionVar(), MN,
                              innerLoopK.getInductionVar(),
                              KPerThreadConstantOp, mnPerMfmaGroup,
                              ldsBufferElemType, dataType, ldsOrig, regDest);
          }
        };

    // load A from LDS into registers
    // for(index_t m_i = 0; m_i < mRepeats; ++m_i)
    //   for(index_t k_i = 0; k_i < KPerThread; ++k_i)
    //       ldsToRegisterCopy[m_i, k_i]
    auto outerLoopM = b.create<AffineForOp>(loc, 0, mRepeats);
    auto olmb = ConversionPatternRewriter::atBlockBegin(outerLoopM.getBody(),
                                                        b.getListener());
    ldsToRegisterCopyKdim(olmb, outerLoopM, sourceOffsetA, MConstantOp,
                          mPerAccelConstantOp, bufferElemTypeA, dataTypeA,
                          op.getMatrixA(), bufferA);

    // load B from LDS into registers
    // for(index_t n_i = 0; n_i < mRepeats; ++n_i)
    //   for(index_t k_i = 0; k_i < KPerThread; ++k_i)
    //       ldsToRegisterCopy[n_i, k_i]
    auto outerLoopN = olmb.create<AffineForOp>(loc, 0, nRepeats);
    auto olnb = ConversionPatternRewriter::atBlockBegin(outerLoopN.getBody(),
                                                        olmb.getListener());
    ldsToRegisterCopyKdim(olnb, outerLoopN, sourceOffsetB, NConstantOp,
                          nPerAccelConstantOp, bufferElemTypeB, dataTypeB,
                          op.getMatrixB(), bufferB);

    b.eraseOp(op);
    olnb.create<AccelGemmOp>(loc, outerLoopM.getInductionVar(),
                             outerLoopN.getInductionVar(), adaptor.getBufferA(),
                             adaptor.getBufferB(), adaptor.getMatrixC(), arch,
                             op.getFeaturesAttr(), tuningParams);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// GlobalLoadOp lowering.
//===----------------------------------------------------------------------===//
struct GlobalLoadRewritePattern : public OpRewritePattern<GlobalLoadOp> {
  using OpRewritePattern<GlobalLoadOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(GlobalLoadOp op,
                                PatternRewriter &b) const override {
    Location loc = op.getLoc();

    MemRefType sourceType = op.getSource().getType();
    Type sourceElemType = sourceType.getElementType();
    int64_t elemsPerWord = (32 / sourceElemType.getIntOrFloatBitWidth());
    int64_t maxLoadLen = 4 * elemsPerWord;

    Type resType = op.getResult().getType();
    int64_t totalLength = 1;
    if (auto vecType = resType.dyn_cast<VectorType>()) {
      totalLength = vecType.getNumElements();
    }
    // Don't use any vector magic if we don't need to
    if ((totalLength <= maxLoadLen) && (maxLoadLen % totalLength == 0)) {
      Type typeToLoad = sourceElemType;
      if (totalLength > 1)
        typeToLoad = VectorType::get({totalLength}, typeToLoad);
      BufferLoadOp load =
          b.create<BufferLoadOp>(loc, typeToLoad, op.getSource(), op.getValid(),
                                 op.getSourceCoord(), IntegerAttr(),
                                 /*oobIsOverload=*/nullptr);
      b.replaceOp(op, {load});
      return success();
    }
    int64_t remainingLength = totalLength;
    int64_t offset = 0;

    Value result = createZeroConstantOp(b, loc, resType);

    while (remainingLength > 0) {
      int64_t copyLength = std::min(remainingLength, maxLoadLen);

      // Clean up bad copy lengths
      if (copyLength != maxLoadLen && copyLength > (2 * elemsPerWord))
        copyLength = 2 * elemsPerWord;
      if (copyLength > elemsPerWord && copyLength % elemsPerWord != 0)
        copyLength = elemsPerWord;
      if (copyLength > 1 && copyLength < elemsPerWord)
        // TODO: revisit this to handle things like (2xi8) -> load short
        copyLength = 1;

      Type typeToLoad = sourceElemType;
      if (copyLength > 1)
        typeToLoad = VectorType::get({copyLength}, typeToLoad);

      IntegerAttr offsetAttr =
          (offset > 0) ? b.getIndexAttr(offset) : IntegerAttr();

      Value loaded =
          b.create<BufferLoadOp>(loc, typeToLoad, op.getSource(), op.getValid(),
                                 op.getSourceCoord(), offsetAttr,
                                 /*oobIsOverload=*/nullptr);
      if (totalLength == 1) {
        result = loaded;
      } else {
        Value offsetIdx = b.createOrFold<ConstantIndexOp>(loc, offset);
        result =
            b.create<InsertSliceOp>(loc, resType, loaded, result, offsetIdx);
      }

      remainingLength -= copyLength;
      offset += copyLength;
    }
    b.replaceOp(op, {result});
    return success();
  }
};

//===----------------------------------------------------------------------===//
// GlobalStore lowering.
//===----------------------------------------------------------------------===//
struct GlobalStoreRewritePattern : public OpRewritePattern<GlobalStoreOp> {
  using OpRewritePattern<GlobalStoreOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(GlobalStoreOp op,
                                PatternRewriter &b) const override {
    Location loc = op.getLoc();

    Value source = op.getSource();
    MemRefType sourceType = op.getSource().getType();
    Value sourceCoord = op.getSourceCoord();

    Type sourceElemType = sourceType.getElementType();
    Type destElemType = op.getDest().getType().getElementType();
    int64_t elemsPerWord = (32 / destElemType.getIntOrFloatBitWidth());
    int64_t maxWriteLen = 4 * elemsPerWord;
    int64_t remainingLength = op.getLength().getSExtValue();
    int64_t offset = 0;
    // Don't use any vector magic if we don't need to
    if ((remainingLength <= maxWriteLen) &&
        (maxWriteLen % remainingLength == 0)) {
      Type typeToLoad = sourceElemType;
      if (remainingLength > 1)
        typeToLoad = VectorType::get({remainingLength}, typeToLoad);
      Value loaded =
          b.create<InBoundsLoadOp>(loc, typeToLoad, source, sourceCoord);
      b.create<BufferStoreOp>(loc, loaded, op.getDest(), op.getValid(),
                              op.getDestCoord(), op.getFeaturesAttr(),
                              op.getStoreMethodAttr(), IntegerAttr(),
                              /*oobIsOverload=*/nullptr);
      b.eraseOp(op);
      return success();
    }
    while (remainingLength > 0) {
      int64_t copyLength = std::min(remainingLength, maxWriteLen);

      // Clean up bad copy lengths
      if (copyLength != maxWriteLen && copyLength > (2 * elemsPerWord))
        copyLength = 2 * elemsPerWord;
      if (copyLength > elemsPerWord && copyLength % elemsPerWord != 0)
        copyLength = elemsPerWord;
      if (copyLength > 1 && copyLength < elemsPerWord)
        copyLength = 1;

      Type typeToLoad = sourceElemType;
      if (copyLength > 1)
        typeToLoad = VectorType::get({copyLength}, typeToLoad);
      Type typeToStore = destElemType;
      if (copyLength > 1)
        typeToStore = VectorType::get({copyLength}, typeToStore);

      Value loadCoord = sourceCoord;
      if (offset > 0)
        loadCoord = b.createOrFold<AddIOp>(
            loc, sourceCoord, b.create<ConstantIndexOp>(loc, offset));
      Value loaded =
          b.create<InBoundsLoadOp>(loc, typeToLoad, source, loadCoord);
      IntegerAttr offsetAttr =
          (offset > 0) ? b.getIndexAttr(offset) : IntegerAttr();
      b.create<BufferStoreOp>(loc, loaded, op.getDest(), op.getValid(),
                              op.getDestCoord(), op.getFeaturesAttr(),
                              op.getStoreMethodAttr(), offsetAttr,
                              /*oobIsOverflow=*/nullptr);
      remainingLength -= copyLength;
      offset += copyLength;
    }
    b.eraseOp(op);
    return success();
  }
};

namespace {
struct ThreadwiseMemCpyRewritePattern
    : public OpConversionPattern<ThreadwiseMemCpyOp> {
  using OpConversionPattern<ThreadwiseMemCpyOp>::OpConversionPattern;

  // Depending on the memory space, the views might accept different
  // views (e.g. global memory might require all bid, tid and iter whilst
  // workgroup memory only requires tid and iter.). Equivalize the views
  // by adding a broadcast.
  std::tuple<ArrayAttr, ArrayAttr>
  equivalizeSrcDestViews(ThreadwiseMemCpyOp op, ArrayAttr srcViews,
                         ArrayAttr destViews,
                         ConversionPatternRewriter &b) const {
    Location loc = op.getLoc();
    SmallVector<Attribute> transformList;
    ArrayRef<int64_t> srcLowerShape;
    if (srcViews.empty()) {
      srcLowerShape = op.getSource().getType().getShape();
    } else {
      srcLowerShape =
          srcViews[0].cast<TransformMapAttr>().getUpperBounds().asArrayRef();
    }
    ArrayRef<int64_t> destLowerShape;
    if (destViews.empty()) {
      destLowerShape = op.getDest().getType().getShape();
    } else {
      destLowerShape =
          destViews[0].cast<TransformMapAttr>().getUpperBounds().asArrayRef();
    }

    if (srcLowerShape.size() == destLowerShape.size()) {
      return {srcViews, destViews};
    }

    if (srcLowerShape.size() < destLowerShape.size()) {
      BottomUpTMBuilder bcastAdder(b, srcLowerShape, loc);
      for (size_t i = 0; i < destLowerShape.size(); i++) {
        unsigned int dim = i;
        if (destLowerShape.size() - i > srcLowerShape.size()) {
          bcastAdder.broadcast({dim}, {destLowerShape[i]});
        } else {
          bcastAdder.passThrough({dim}, {dim});
        }
      }
      transformList.push_back(bcastAdder.get());
      transformList.append(srcViews.begin(), srcViews.end());
      return {b.getArrayAttr(transformList), destViews};
    } else {
      BottomUpTMBuilder bcastAdder(b, srcLowerShape, loc);
      for (size_t i = 0; i < srcLowerShape.size(); i++) {
        unsigned int dim = i;
        if (srcLowerShape.size() - i > destLowerShape.size()) {
          bcastAdder.broadcast({dim}, {srcLowerShape[i]});
        } else {
          bcastAdder.passThrough({dim}, {dim});
        }
      }
      transformList.push_back(bcastAdder.get());
      transformList.append(destViews.begin(), destViews.end());
      return {srcViews, b.getArrayAttr(transformList)};
    }
  }

  LogicalResult matchAndRewrite(ThreadwiseMemCpyOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &b) const final {
    Location loc = op.getLoc();
    TypedValue<MemRefType> sourceView = adaptor.getSource();
    TypedValue<MemRefType> destView = adaptor.getDest();

    auto [srcBuffer, srcTransforms] =
        untransform(b, sourceView, op.getExtraSrcViews().value_or(nullptr));
    MemRefType srcBufferType = srcBuffer.getType().cast<MemRefType>();
    auto [dstBuffer, dstTransforms] =
        untransform(b, destView, op.getExtraDstViews().value_or(nullptr));
    MemRefType dstBufferType = dstBuffer.getType().cast<MemRefType>();

    ArrayRef<int64_t> srcBufferShape =
        srcBuffer.getType().cast<ShapedType>().getShape();
    ArrayRef<int64_t> dstBufferShape =
        dstBuffer.getType().cast<ShapedType>().getShape();

    // Unless specified it is assumed to be global
    gpu::AddressSpace srcAddrSpace = gpu::AddressSpace::Global;
    if (srcBufferType.getMemorySpace()) {
      srcAddrSpace = srcBufferType.getMemorySpace()
                         .cast<gpu::AddressSpaceAttr>()
                         .getValue();
    }
    // Unless specified it is assumed to be global
    gpu::AddressSpace dstAddrSpace = gpu::AddressSpace::Global;
    if (dstBufferType.getMemorySpace()) {
      dstAddrSpace = dstBufferType.getMemorySpace()
                         .cast<gpu::AddressSpaceAttr>()
                         .getValue();
    }

    int64_t iterLen;
    if (srcAddrSpace == gpu::AddressSpace::Private) {
      iterLen = srcBufferType.getNumElements();
    } else {
      // op verifiers gurantees this
      assert(dstAddrSpace == gpu::AddressSpace::Private);
      iterLen = dstBufferType.getNumElements();
    }

    // We are vectorizing in the iter dimension, not block ID or thread ID
    auto elementType = sourceView.getType().getElementType();

    // if src addr space is registers and with no transforms, there is no
    // constrain on vectorization coming from src
    const int64_t maxVectorLenBits = 128;
    int64_t srcVectorLen =
        std::min(maxVectorLenBits, srcBufferType.getNumElements());
    if (!(srcAddrSpace == gpu::AddressSpace::Private &&
          srcTransforms.empty())) {
      srcVectorLen = getMaxVectorizationForDatatype(
          srcTransforms, /*dim=*/2, iterLen, srcBufferShape, elementType);
    }

    // if dest addr space is registers and with no transforms, there is no
    // constrain on vectorization coming from dest
    int64_t destVectorLen =
        std::min(maxVectorLenBits, dstBufferType.getNumElements());
    if (!(dstAddrSpace == gpu::AddressSpace::Private &&
          dstTransforms.empty())) {
      destVectorLen = getMaxVectorizationForDatatype(
          dstTransforms, /*dim=*/2, iterLen, dstBufferShape, elementType);
    }

    int64_t vectorLen = std::min(srcVectorLen, destVectorLen);
    LLVM_DEBUG(llvm::dbgs()
               << "Max vectorization for read_into = " << vectorLen << "\n");

    Type loadType = vectorTypeOrSelf(elementType, vectorLen);
    bool forceUnroll = op.getForceUnroll();
    bool useIndexDiffs = op.getUseIndexDiffs();

    // In the future, this might get merged into the vectorizer.
    srcTransforms = collapseContiguousMerges(srcTransforms, srcBufferShape);
    dstTransforms = collapseContiguousMerges(dstTransforms, dstBufferShape);

    std::tie(srcTransforms, dstTransforms) =
        equivalizeSrcDestViews(op, srcTransforms, dstTransforms, b);

    // Constant / consistent arguments
    Value zero = b.createOrFold<arith::ConstantIndexOp>(loc, 0);
    Value bid = b.createOrFold<rock::WorkgroupIdOp>(loc, b.getIndexType());
    Value tid = b.createOrFold<rock::WorkitemIdOp>(loc, b.getIndexType());

    SmallVector<Value, 3> startCoords{bid, tid, zero};

    auto loadLoop = b.create<TransformingForOp>(
        loc, ArrayRef<ValueRange>{startCoords, startCoords},
        ArrayRef<Attribute>{srcTransforms, dstTransforms},
        ArrayRef<int64_t>{1, 1, iterLen}, ArrayRef<int64_t>{1, 1, vectorLen},
        forceUnroll, useIndexDiffs);
    {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(loadLoop.getBody());

      Value loaded;
      if (srcAddrSpace != gpu::AddressSpace::Global) {
        auto slice_n = loadLoop.getLowerCoords(/*domain=*/0).size() -
                       srcBufferType.getRank();
        loaded = b.create<InBoundsLoadOp>(
            loc, loadType, srcBuffer,
            loadLoop.getLowerCoords(/*domain=*/0).slice(slice_n));
      } else {
        loaded = b.create<GlobalLoadOp>(loc, loadType, srcBuffer,
                                        loadLoop.getValidity(/*domain=*/0),
                                        loadLoop.getLowerCoords(/*domain=*/0));
      }
      auto slice_n = loadLoop.getLowerCoords(/*domain=*/1).size() -
                     dstBufferType.getRank();
      b.create<InBoundsStoreOp>(
          loc, loaded, dstBuffer,
          loadLoop.getLowerCoords(/*domain=*/1).slice(slice_n));
    }
    b.eraseOp(op);
    return success();
  }
};

struct ThreadwiseWriteAllRewritePattern
    : public OpConversionPattern<ThreadwiseWriteAllOp> {
  using OpConversionPattern<ThreadwiseWriteAllOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(ThreadwiseWriteAllOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &b) const final;
};
} // end anonymous namespace

LogicalResult ThreadwiseWriteAllRewritePattern::matchAndRewrite(
    ThreadwiseWriteAllOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &b) const {
  Location loc = op.getLoc();
  TypedValue<MemRefType> source = adaptor.getSource();
  TypedValue<MemRefType> destView = adaptor.getDest();

  auto elementType = destView.getType().getElementType();

  auto [buffer, transforms] = untransform(b, destView, op.getExtraViews());

  int64_t numValues = source.getType().getNumElements();

  ArrayRef<int64_t> bufferShape =
      buffer.getType().cast<ShapedType>().getShape();

  // We are vectorizing in the iter dimension, not block ID or thread ID
  int64_t vectorLen = getMaxVectorizationForDatatype(
      transforms, /*dim=*/2, numValues, bufferShape, elementType);
  LLVM_DEBUG(llvm::dbgs() << "Max vectorization for write_all = " << vectorLen
                          << "\n");

  bool forceUnroll = op.getForceUnroll();
  bool useIndexDiffs = op.getUseIndexDiffs();

  transforms = collapseContiguousMerges(transforms, bufferShape);

  // Constant / consistent arguments
  Value zero = b.createOrFold<arith::ConstantIndexOp>(loc, 0);
  Value bid = b.createOrFold<rock::WorkgroupIdOp>(loc, b.getIndexType());
  Value tid = b.createOrFold<rock::WorkitemIdOp>(loc, b.getIndexType());

  SmallVector<Value, 3> writeStartCoords = {bid, tid, zero};

  auto outLoop = b.create<TransformingForOp>(
      loc, ArrayRef<ValueRange>{writeStartCoords, writeStartCoords},
      ArrayRef<Attribute>{b.getArrayAttr({}), transforms},
      ArrayRef<int64_t>{1, 1, numValues}, ArrayRef<int64_t>{1, 1, vectorLen},
      forceUnroll, useIndexDiffs);
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(outLoop.getBody());
    b.create<GlobalStoreOp>(loc, source, buffer, b.getIndexAttr(vectorLen),
                            op.getFeaturesAttr(), op.getStoreMethodAttr(),
                            outLoop.getLowerCoords(/*domain=*/0)[2],
                            outLoop.getValidity(/*domain=*/1),
                            outLoop.getLowerCoords(/*domain=*/1));
  }
  b.eraseOp(op);
  return success();
}

void RockLowerBlockwiseGemmToThreadwisePass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  {
    ConversionTarget writeAllTarget(*ctx);
    writeAllTarget.addIllegalOp<ThreadwiseMemCpyOp, ThreadwiseWriteAllOp>();
    writeAllTarget.addLegalDialect<arith::ArithDialect, rock::RockDialect,
                                   memref::MemRefDialect>();
    writeAllTarget.addLegalOp<gpu::PrintfOp>();
    RewritePatternSet writeAllPatterns(ctx);
    writeAllPatterns
        .add<ThreadwiseMemCpyRewritePattern, ThreadwiseWriteAllRewritePattern>(
            ctx);
    if (failed(applyPartialConversion(getOperation(), writeAllTarget,
                                      std::move(writeAllPatterns))))
      signalPassFailure();
  }

  ConversionTarget target(*ctx);
  target.addIllegalOp<FillOp, BlockwiseGemmOp, BlockwiseGemmAccelOp,
                      GlobalLoadOp, GlobalStoreOp>();
  target.addLegalDialect<arith::ArithDialect, rock::RockDialect, AffineDialect,
                         vector::VectorDialect, memref::MemRefDialect>();
  target.addLegalOp<gpu::PrintfOp>();

  RewritePatternSet patterns(ctx);
  patterns.add<FillRewritePattern, BlockwiseGemmRewritePattern,
               BlockwiseGemmAccelRewritePattern, GlobalLoadRewritePattern,
               GlobalStoreRewritePattern>(ctx);
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}
} // end anonymous namespace
