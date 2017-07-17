/*!
 *  Copyright (c) 2015 by Contributors
 * \file ndarray.h
 * \brief NDArray interface that handles array arithematics.
 */
#ifndef MXNET_NDARRAY_H_
#define MXNET_NDARRAY_H_

#include <dmlc/base.h>
#include <dmlc/logging.h>
#include <dmlc/io.h>
#include <dmlc/type_traits.h>
#include <dmlc/registry.h>
#include <nnvm/node.h>
#include <vector>
#include <map>
#include <string>
#include <memory>
#include "./base.h"
#include "./storage.h"
#include "./engine.h"
#if MKL_EXPERIMENTAL == 1
#include <mkl_memory.h>
#endif
// check c++11
#if DMLC_USE_CXX11 == 0
#error "cxx11 was required for ndarray module"
#endif

namespace mxnet {

namespace ndarray {
template<typename from_xpu, typename to_xpu>
void Copy(const TBlob &from, TBlob *to, Context from_ctx, Context to_ctx, RunContext ctx);
};

namespace autograd {
class AGNode;

using AGNodePtr = std::shared_ptr<AGNode>;

class AGNodeEntry {
 public:
  AGNodePtr ag_node;
  uint32_t index;
  uint32_t version;

  void clear() {
    ag_node.reset();
    index = version = 0;
  }

  nnvm::NodeEntry nn_entry() const;
  bool is_none() const;
};

class AutogradRuntime;
}  // namespace autograd

// enum for storage types
namespace csr {
enum CSRAuxType {kIndPtr, kIdx};
}

namespace rowsparse {
enum RowSparseAuxType {kIdx};
}

enum NDArrayStorageType {
  kUndefinedStorage = -1,  // undefined storage
  kDefaultStorage,         // dense
  kRowSparseStorage,       // row sparse
  kCSRStorage,             // csr
};

/*!
 * \return the number of aux data used for given storage type
 */
size_t num_aux_data(NDArrayStorageType stype);

/*!
 * \brief ndarray interface
 */
class NDArray {
 private:
  /*!
   * \brief A wrapper of VarHandle object. The data chunks
   * in an NDArray object will share the same var variable
   * using a shared_ptr of VarHolder. The destructor of
   * the VarHolder will call engine DeleteVariable.
   */
  struct VarHolder {
    explicit VarHolder(const Context& ctx, Engine::VarHandle var) : ctx_(ctx), var_(var) {}
    ~VarHolder() {
      Engine::Get()->DeleteVariable([](RunContext s) {}, ctx_, var_);
    }
    Context ctx_;
    Engine::VarHandle var_;
  };

  /*! \brief the real data chunk that backs NDArray */
  struct Chunk {
    /*! \brief storage handle from storage engine */
    Storage::Handle shandle_;

    /*!
     * \brief if this is true, this means the data do not come
     * from Storage, and do not need to be freed
     */
    bool static_data_;

    /*! \brief whether data allocation is delayed */
    bool delay_alloc_;

    /*! \brief context of data */
    Context ctx_;

    TShape storage_shape_;

    /*!
     * \brief variable from engine. The data chunks of a sparse ndarray
     * share the same var. That's why we need a shared_ptr of VarHolder
     * for holding the unique var for all data chunks in the sparse
     * ndarray. The destructor of the struct VarHolder will call
     * Engine's DeleteVariable() function to delete the var after
     * the ref counts of the var becomes zero.
     */
    std::shared_ptr<VarHolder> var_holder_;

    /*! \brief default cosntructor */
    Chunk() : static_data_(true), delay_alloc_(false) {
      var_holder_ = std::make_shared<VarHolder>(shandle_.ctx, Engine::Get()->NewVariable());
    }

    /*! \brief construct from static data */
    Chunk(const TBlob &data, int dev_id)
        : static_data_(true), delay_alloc_(false) {
      if (data.dev_mask() == cpu::kDevMask) {
        shandle_.ctx = Context::CPU(dev_id);
      } else {
        CHECK_EQ(data.dev_mask(), gpu::kDevMask);
        shandle_.ctx = Context::GPU(dev_id);
      }
      shandle_.dptr = data.dptr_;
      shandle_.size = data.shape_.Size() * mshadow::mshadow_sizeof(data.type_flag_);
      var_holder_ = std::make_shared<VarHolder>(shandle_.ctx, Engine::Get()->NewVariable());
    }

    /*! \brief construct a new chunk */
    Chunk(const TShape& storage_shape, const Context& ctx, bool delay_alloc, int dtype,
          const std::shared_ptr<VarHolder>& var_holder)
      : static_data_(false), delay_alloc_(true), ctx_(ctx),
        storage_shape_(storage_shape), var_holder_(var_holder) {
      shandle_.size = storage_shape.Size() * mshadow::mshadow_sizeof(dtype);
      shandle_.ctx = ctx_;
      if (!delay_alloc) this->CheckAndAlloc();
    }

    void CheckAndAlloc(const TShape& storage_shape, int dtype) {
      if (delay_alloc_) {
        const auto dbytes = storage_shape.Size() * mshadow::mshadow_sizeof(dtype);
        if (shandle_.size < dbytes) {
          if (shandle_.size > 0) Storage::Get()->Free(shandle_);
          shandle_ = Storage::Get()->Alloc(dbytes, ctx_);
        }
        delay_alloc_ = false;
      }
    }

    /*! \brief check if delay alloc is on, do alloc if not yet done */
    void CheckAndAlloc() {
      if (delay_alloc_) {
        shandle_ = Storage::Get()->Alloc(shandle_.size, shandle_.ctx);
        delay_alloc_ = false;
      }
    }

    /*! \brief destructor */
    ~Chunk() {
      if (static_data_ || delay_alloc_) return;
      Engine::Get()->WaitForVar(var_holder_->var_);
      Storage::Get()->Free(shandle_);
    }
  };  // struct Chunk

 public:
  /*! \brief default constructor */
  NDArray() {
#if MKL_EXPERIMENTAL == 1
    Mkl_mem_ = MKLMemHolder::create();
#endif
  }

  /*!
   * \brief constructs a new dynamic NDArray
   * \param shape the shape of array
   * \param ctx context of NDArray
   * \param delay_alloc whether delay the allocation
   * \param dtype data type of this ndarray
   */
  NDArray(const TShape &shape, Context ctx, bool delay_alloc = false,
          int dtype = mshadow::default_type_flag)
    : storage_type_(kDefaultStorage),
      ptr_(std::make_shared<Chunk>(shape.Size(), ctx, delay_alloc, dtype,
          std::make_shared<VarHolder>(ctx, Engine::Get()->NewVariable()))),
      shape_(shape), storage_shape_(shape), dtype_(dtype),
      entry_({nullptr, 0, 0}) {
#if MKL_EXPERIMENTAL == 1
    Mkl_mem_ = std::make_shared<MKLMemHolder>();
#endif
  }

  /*!
   * \brief constructor for NDArray with storage type
   */
  NDArray(const NDArrayStorageType stype, const TShape &shape, const Context& ctx,
          bool delay_alloc = true, int dtype = mshadow::default_type_flag,
          const std::vector<int>& aux_dtypes = {}, const std::vector<TShape>& aux_shapes = {},
          const TShape& storage_shape = TShape(mshadow::Shape1(0)))
    : storage_type_(stype), shape_(shape), aux_shapes_(aux_shapes),
      dtype_(dtype), aux_dtypes_(aux_dtypes), entry_({nullptr, 0, 0}) {
    // Assign default aux types if not given
    if (aux_dtypes_.empty()) {
      if (storage_type_ == kRowSparseStorage) {
        aux_dtypes_ = {mshadow::kInt64};
      } else if (storage_type_ == kCSRStorage) {
        aux_dtypes_ = {mshadow::kInt64, mshadow::kInt64};
      } else {
        LOG(FATAL) << "Unknown storage type " << stype;
      }
    }
    // Assign default shapes if not given
    // unknown shapes are intialized as {0} such that Size() would return 0
    if (aux_shapes_.empty()) {
      if (storage_type_ == kRowSparseStorage) {
        aux_shapes_ = {TShape(mshadow::Shape1(0))};
      } else if (storage_type_ == kCSRStorage) {
        // aux shapes for indptr and indices
        aux_shapes_ = {TShape(mshadow::Shape1(0)), TShape(mshadow::Shape1(0))};
      } else {
        LOG(FATAL) << "Unknown storage type " << stype;
      }
    }
    if (storage_shape.Size() == 0) {
      if (storage_type == kRowSparseStorage) {
        storage_shape = shape;
        storage_shape[0] = aux_shapes[rowsparse::kIdx][0];
      } else if (storage_type_ == kCSRStorage) {
        storage_shape = aux_shapes[csr::kIdx];
      } else {
        LOG(FATAL) << "Unknown storage type " << stype;
      }
    }
    // init ptr_
    ptr_ = std::make_shared<Chunk>(shape_.Size(), ctx, delay_alloc, dtype_,
                                   std::make_shared<VarHolder>(
                                     ctx, Engine::Get()->NewVariable()));
    // init aux_ptrs_
    aux_ptrs_.reserve(aux_shapes_.size());
    for (size_t i = 0; i < aux_shapes_.size(); ++i) {
      aux_ptrs_.emplace_back(std::make_shared<Chunk>(aux_shapes_[i].Size(), ctx, delay_alloc,
                                                     aux_dtypes_[i], ptr_->var_holder_));
    }
#if MKL_EXPERIMENTAL == 1
    Mkl_mem_ = std::make_shared<MKLMemHolder>();
#endif
  }

  /*!
   * \brief constructing a static NDArray that shares data with TBlob
   *  Use with caution: allocate ONLY ONE NDArray for each TBlob,
   *  make sure the memory region is available through out the life of NDArray
   * \param data the memory content of static data
   * \param dev_id the device id this tensor sits at
   */
  NDArray(const TBlob &data, int dev_id)
      : ptr_(std::make_shared<Chunk>(data, dev_id)), shape_(data.shape_),
        dtype_(data.type_flag_), entry_({nullptr, 0, 0}) {
#if MKL_EXPERIMENTAL == 1
    Mkl_mem_ = std::make_shared<MKLMemHolder>();
#endif
  }

  /*!
   * \brief For function call from data_ndarray and aux_ndarray
   */
  NDArray(const std::shared_ptr<Chunk>& ptr, const TShape& shape, const int dtype)
    : storage_type_(kDefaultStorage), ptr_(ptr), shape_(shape),
      storage_shape_(shape), dtype_(dtype), entry_({nullptr, 0, 0}) {
#if MKL_EXPERIMENTAL == 1
    Mkl_mem_ = std::make_shared<MKLMemHolder>();
#endif
  }

  /*!
   * \return the shape of current NDArray.
   */
  const TShape& shape() const {
    return shape_;
  }

  /*!
   *  For default storage, it is the same as shape().
   *  For row-sparse/csr storage, it is the shape of the tensor
   *  which stores the non-zero values.
   */
  const TShape& storage_shape() const {
    return ptr_->storage_shape_;
  }

  /*!
   * \brief For sparse operations, the storage shape is an estimated value
   * in the beginning for allocating enough capacity for the final result.
   * After the operation is done, the exact size of the shape is known
   * and need to be reset using this function. For example, adding
   * two CSRs with nnz1 and nnz2 as their numbers of non-zero values, respectively,
   * would allocate the array of size nnz1+nnz2 first and get the final
   * nnz that is smaller than nnz1+nnz2. Therefore, the storage shape's size
   * needs to be shrunk from nnz1+nnz2 to nnz.
   */
  void set_storage_shape(const TShape& sshape) {
    CHECK(storage_type_ != kDefaultStorage);
    storage_shape_ = sshape;
    if (kRowSparseStorage == storage_type_) {
      set_aux_shape(rowsparse::kIdx, mshadow::Shape1(storage_shape_[0]));
    } else if (kCSRStorage == storage_type_) {
      set_aux_shape(csr::kIndPtr, mshadow::Shape1(shape_[0]+1));
      set_aux_shape(csr::kIdx, mshadow::Shape1(storage_shape_[0]));
    } else {
      LOG(FATAL) << "Unsupported storage type " << storage_type_
                 << " in set_storage_shape()";
    }
  }

  /*!
   * \return the shape of aux data at ith index. If it doesn't exist, return an empty one.
   */
  inline const TShape& aux_shape(size_t i) const {
    CHECK(storage_type_ != kDefaultStorage);
    CHECK_LT(i, aux_shapes_.size());
    return aux_shapes_[i];
  }

  /*!
   * \brief For a sparse operation on a csr matrix for example,
   * the size of the column index array
   * is an estimated value in the beginning for allocating enough capacity
   * for the final result. After the operation is done, the exact size of
   * the shape is known and need to be reset using this function.
   */
  void set_aux_shape(size_t i, const TShape& shape) {
    aux_shapes_[i] = shape;
  }

  /*!
   * \return the data TBlob
   */
  const TBlob& data() const {
    if (storage_type() == kDefaultStorage) CheckAndAlloc();
    SetTBlob();
    return tblob_;
  }

  /*!
   * \return the aux TBlob
   */
  TBlob aux_data(size_t i) const {
    TBlob res;
    MSHADOW_TYPE_SWITCH(aux_dtypes_[i], DType, {
      auto dptr = static_cast<DType*>(aux_ptrs_[i]->shandle_.dptr);
      CHECK(storage_type_ == kRowSparseStorage || storage_type_ == kCSRStorage)
            << "Unexpected storage type: " << storage_type_;
      res = TBlob(dptr, aux_shapes_[i], aux_ptrs_[i]->shandle_.ctx.dev_mask(), aux_dtypes_[i]);
    });
#if MKL_EXPERIMENTAL == 1
    res.Mkl_mem_ = Mkl_mem_;
#endif
    return res;
  }

  /*!
   * \return the context of NDArray, this function is only valid when the NDArray is not empty
   */
  Context ctx() const {
    return ptr_->shandle_.ctx;
  }

  /*!
   * \return the data type of NDArray, this function is only valid when the NDArray is not empty
   */
  int dtype() const {
    return dtype_;
  }

  int aux_dtype(size_t i) const {
    CHECK(!is_none());
    return aux_dtypes_[i];
  }

  NDArrayStorageType storage_type() const {
    if (is_none()) return kUndefinedStorage;
    return storage_type_;
  }

  /*! \return whether this ndarray is not initialized */
  inline bool is_none() const {
    return ptr_ == nullptr;
  }

  /*! \return updated grad state in entry_ */
  bool fresh_out_grad() const;
  /*! \return updated grad state in entry_ */
  void set_fresh_out_grad(bool state) const;
  // returns true if a sparse ndarray's aux_data and storage are initialized
  bool storage_initialized() const {
    if (is_none()) return false;
    CHECK_NE(storage_type_, kDefaultStorage);
    if (storage_type_ == kRowSparseStorage || storage_type_ == kCSRStorage) {
      return storage_shape_.Size() != 0U;
    } else {
      LOG(FATAL) << "Unknown storage type";
    }
    return true;
  }

  /*!
   * \brief Block until all the pending write operations with respect
   *    to current NDArray are finished, and read can be performed.
   */
  void WaitToRead() const {
    if (is_none()) return;
    Engine::Get()->WaitForVar(var());
  }

  /*!
   * \brief Block until all the pending read/write operations with respect
   *    to current NDArray are finished, and write can be performed.
   */
  void WaitToWrite() const {
    if (is_none()) return;
    /*!
     * Push an empty mutable function to flush all preceding reads to the
     * variable.
     */
    Engine::Get()->PushSync([](RunContext) {}, Context{}, {}, {var()});
    Engine::Get()->WaitForVar(var());
  }

  /*! \return the associated variable of the ndarray.*/
  Engine::VarHandle var() const {
    return ptr_->var_holder_->var_;
  }

  /*!
   * \brief save the content into binary stream
   * \param strm the output stream
   */
  void Save(dmlc::Stream *strm) const;

  /*!
   * \brief load ndarrays before supporting sparse ndarrays
   * \param strm the output stream
   * \param magic the magic number used for version control
   */
  bool LegacyLoad(dmlc::Stream *strm, const uint32_t magic);

  /*!
   * \brief load the content from binary stream
   * \param strm the output stream
   * \return whether the load is successful
   */
  bool Load(dmlc::Stream *strm);

  /*!
   * \brief set all the elements in ndarray to be scalar
   * \param scalar the scalar to set
   * \return reference of self
   */
  NDArray &operator=(real_t scalar);

  /*!
   * \brief elementwise add to current space
   *  this mutate the current NDArray
   * \param src the data to add
   * \return reference of self
   */
  NDArray &operator+=(const NDArray &src);

  /*!
   * \brief elementwise add to current space
   *  this mutate the current NDArray
   * \param src the data to add
   * \return reference of self
   */
  NDArray &operator+=(const real_t &src);

  /*!
   * \brief elementwise subtract from current ndarray
   * this mutate the current NDArray
   * \param src the data to subtract
   * \return reference of self
   */
  NDArray &operator-=(const NDArray &src);

  /*!
   * \brief elementwise subtract from current ndarray
   * this mutate the current NDArray
   * \param src the data to subtract
   * \return reference of self
   */
  NDArray &operator-=(const real_t &src);

  /*!
   * \brief elementwise multiplication to current ndarray
   *  this mutate the current NDArray
   * \param src the data to subtract
   * \return reference of self
   */
  NDArray &operator*=(const NDArray &src);

  /*!
   * \brief elementwise multiplication to current ndarray
   *  this mutate the current NDArray
   * \param src the data to subtract
   * \return reference of self
   */
  NDArray &operator*=(const real_t &src);

  /*!
   * \brief elementwise division from current ndarray
   *  this mutate the current NDArray
   * \param src the data to subtract
   * \return reference of self
   */
  NDArray &operator/=(const NDArray &src);

  /*!
   * \brief elementwise division from current ndarray
   *  this mutate the current NDArray
   * \param src the data to subtract
   * \return reference of self
   */
  NDArray &operator/=(const real_t &src);

  /*!
   * \brief return transpose of current NDArray
   * \return a new transposed NDArray
   */
  NDArray T() const;

  /*!
   * \brief return a new copy this NDArray
   * \param ctx the new context of this NDArray
   * \return the new copy
   */
  NDArray Copy(Context ctx) const;

  /*!
   * \brief Do a synchronize copy from a continugous CPU memory region.
   *
   *  This function will call WaitToWrite before the copy is performed.
   *  This is useful to copy data from existing memory region that are
   *  not wrapped by NDArray(thus dependency not being tracked).
   *
   * \param data the data source to copy from.
   * \param size the size of the source array, in sizeof(DType) not raw btyes.
   */
  void SyncCopyFromCPU(const void *data, size_t size) const;

  /*!
   * \brief Do a synchronize copy to a continugous CPU memory region.
   *
   *  This function will call WaitToRead before the copy is performed.
   *  This is useful to copy data from existing memory region that are
   *  not wrapped by NDArray(thus dependency not being tracked).
   *
   * \param data the data source to copyinto.
   * \param size the memory size we want to copy into, in sizeof(DType) not raw btyes.
   */
  void SyncCopyToCPU(void *data, size_t size) const;

  /*!
   * \brief Slice a NDArray
   * \param begin begin index in first dim (inclusive)
   * \param end end index in first dim (exclusive)
   * \return sliced NDArray
   */
  NDArray Slice(index_t begin, index_t end) const;

  /*!
   * \brief Index a NDArray
   * \param idx the index
   * \return idx-th sub array NDArray
   */
  NDArray At(index_t idx) const;

  /*!
   * \brief Wrap the tblob of aux data into an NDArray which shares
   * the same variable with the current one.
   */
  const NDArray aux_ndarray(size_t i) const {
    CHECK_NE(storage_type_, kDefaultStorage);
    CHECK(i < aux_shapes_.size());
    CHECK(i < aux_dtypes_.size());
    return NDArray(aux_ptrs_[i], aux_shapes_[i], aux_dtypes_[i]);
  }

  /*!
   * \brief Wrap the tblob of data into an NDArray which shares
   * the same variable with the current one.
   */
  const NDArray data_ndarray() const {
    CHECK_NE(storage_type_, kDefaultStorage);
    return NDArray(ptr_, shape_, dtype_);
  }

  /*!
   * \brief Create a NDArray that shares memory with current one
   *  The new array must have smaller memory size than the current array.
   * \param shape new shape
   * \param dtype The data type.
   * \return NDArray in new shape and type.
   */
  NDArray AsArray(const TShape &shape, int dtype) const {
    CHECK_EQ(storage_type_, kDefaultStorage) << "Not implemented yet";
    CHECK_GE(shape_.Size() * mshadow::mshadow_sizeof(dtype_),
             shape.Size() * mshadow::mshadow_sizeof(dtype))
        << "NDArray.AsArray: target memory size is bigger";
#if MKL_EXPERIMENTAL == 1
    if (Mkl_mem_ != nullptr) {
      // convert prv to cpu
      Mkl_mem_->check_and_prv_to_cpu(ptr_->shandle.dptr);
    }
#endif
    NDArray ret = *this;
    ret.shape_ = shape;
    ret.dtype_ = dtype;
    return ret;
  }

  /*!
   * \brief Get an reshaped NDArray
   * \param shape new shape
   * \return NDArray in new shape
   */
  NDArray Reshape(const TShape &shape) const;

  /*!
   * \brief Return a copy of this NDArray without autograd history
   */
  NDArray Detach() const {
    NDArray ret(*this);
    ret.entry_ = autograd::AGNodeEntry{nullptr, 0, 0};
    return ret;
  }

  /*!
   * \brief Allocate the space if it is delayed allocated.
   * This is an internal function used by system that
   * users normally should not use.
   */
  void CheckAndAlloc() const {
    CHECK_EQ(storage_type_, kDefaultStorage);
    CHECK(ptr_ != nullptr);
    ptr_->CheckAndAlloc();
    for (auto& ptr : aux_ptrs_) {
      CHECK(ptr != nullptr);
      ptr->CheckAndAlloc();
    }
  }

  /*!
   * \brief Alloc memory for non-default storage
   * aux_shape is only known at run time.
   * The order of aux_shapes is the same as the current sparse tensor
   * of the storage_type_. This will effectively change the current
   * storage_shape_ and aux_shapes_.
   */
  void CheckAndAlloc(const std::vector<TShape>& aux_shapes) {
    CHECK_NE(storage_type_, kDefaultStorage);
    CHECK_EQ(aux_shapes.size(), num_aux_data(storage_type_));
    if (kRowSparseStorage == storage_type_) {
      storage_shape_ = shape_;
      storage_shape_[0] = aux_shapes_[rowsparse::kIdx][0];
    } else if (kCSRStorage == storage_type_) {
      storage_shape_ = aux_shapes[csr::kIdx];
    } else {
      LOG(FATAL) << "Unknown storage type = " << storage_type_
                 << " for setting storage_shape";
    }
    aux_shapes_ = aux_shapes;
    CHECK(ptr_ != nullptr);
    ptr_->CheckAndAlloc(storage_shape_.Size(), dtype_);
    for (size_t i = 0; i < aux_ptrs_.size(); ++i) {
      CHECK(aux_ptrs_[i] != nullptr);
      aux_ptrs_[i]->CheckAndAlloc(aux_shapes_[i].Size(), aux_dtypes_[i]);
    }
  }

  /*!
   * \brief Alloc memory for sparse tensor's values.
   * This must be called after aux data has been allocated.
   * For example, one will need to allocate aux data first for
   * a csr's indptr to get the total number of nnz, and then
   * allocate aux data idx and value data.
   */
  void CheckAndAllocData(const TShape& storage_shape) {
    CHECK_NE(storage_type(), kDefaultStorage);
    CHECK(ptr_ != nullptr);
    storage_shape_ = storage_shape;
    ptr_->CheckAndAlloc(storage_shape.Size(), dtype_);
  }

  /*!
   * \brief Alloc memory for aux_data(i) with aux_shape.
   * This is called before CheckAndAllocData and will
   * effectively change aux_shape_[i].
   */
  void CheckAndAllocAuxData(size_t i, const TShape& aux_shape) {
    CHECK_NE(storage_type(), kDefaultStorage);
    CHECK_LT(i, num_aux_data(storage_type_));
    CHECK_LT(i, aux_ptrs_.size());
    CHECK(aux_ptrs_[i] != nullptr);
    aux_shapes_[i] = aux_shape;
    aux_ptrs_[i]->CheckAndAlloc(aux_shape.Size(), aux_dtypes_[i]);
  }

  /*!
   * \brief Save list of ndarray into the Stream.x
   * \param fo The stream of output.
   * \param data the NDArrays to be saved.
   * \param names the name of the NDArray, optional, can be zero length.
   */
  static void Save(dmlc::Stream* fo,
                   const std::vector<NDArray>& data,
                   const std::vector<std::string>& names);

  /*!
   * \brief Load list of ndarray into from the stream.
   * \param fi The stream of the input file.
   * \param data the NDArrays to be loaded
   * \param keys the name of the NDArray, if saved in the file.
   */
  static void Load(dmlc::Stream* fi,
                   std::vector<NDArray>* data,
                   std::vector<std::string>* keys);

 private:
  friend class autograd::AutogradRuntime;

  void SetTBlob() const {
    TShape shape = shape_;
    char *dptr = static_cast<char*>(ptr_->shandle_.dptr);
    auto stype = storage_type_;
    if (stype == kDefaultStorage) {
      dptr += byte_offset_;
    } else if (stype == kCSRStorage || stype == kRowSparseStorage) {
      shape = storage_shape_;
    } else {
      LOG(FATAL) << "unknown storage type " << stype;
    }
    tblob_.dptr_ = dptr;
    tblob_.shape_ = shape;
    tblob_.type_flag_ = dtype_;
    tblob_.SetDLTensor(ptr_->shandle_.ctx.dev_mask(), ptr_->shandle_.ctx.dev_id);
#if MKL_EXPERIMENTAL == 1
    tblob_.Mkl_mem_ = Mkl_mem_;
#endif
  }

#if MKL_EXPERIMENTAL == 1
  std::shared_ptr<MKLMemHolder> Mkl_mem_;
#endif
  /*! \brief storage type */
  NDArrayStorageType storage_type_;

  /*! \brief internal data of NDArray */
  std::shared_ptr<Chunk> ptr_{nullptr};
  /*! \brief internal data array of aux data list */
  std::vector<std::shared_ptr<Chunk>> aux_ptrs_;

  /*! \brief shape of current NDArray */
  TShape shape_;
  /*! \brief storage shape of sparse tensors, equal to shape_ in dense format */
  TShape storage_shape_;
  /*! \brief aux shapes for sparse tensor's aux data */
  std::vector<TShape> aux_shapes_;

  /*! \brief type of data */
  int dtype_ = -1;
  /*! \brief aux dtypes of aux data */
  std::vector<int> aux_dtypes_;


  /*! \brief byte offset in chunk for data tblob, not aux data tblob */
  size_t byte_offset_ = 0;
  /*! \brief node entry for autograd */
  autograd::AGNodeEntry entry_;
  /*!
   * \brief internal TBlob
   * \note When user access tblob_ by some const methods like
   *     NDArray::data(), the dptr in tblob_ still need to be updated
   *     in case that allocation happens. So we make it mutable for
   *     this situation.
   */
  mutable TBlob tblob_;
};  // class NDArray

/*!
 * \brief issue an copy operation from one NDArray to another
 *  the two ndarray can sit on different devices
 *  this operation will be scheduled by the engine
 *
 * \param from the ndarray we want to copy data from
 * \param to the target ndarray
 * \param priority Priority of the action.
 * \param alloc_output whether to allocate memory for the output ndarray
 * \note The function name explicitly marks the order of from and to
 *     due to different possible convention carried by copy function.
 */
void CopyFromTo(const NDArray &from, NDArray *to, int priority = 0);

/*!
 * \brief Perform elementwise sum over each data from source, store result into out.
 * \param source the ndarray we want to sum
 * \param out the target ndarray
 * \param priority Priority of the action.
 */
void ElementwiseSum(const std::vector<NDArray> &source, NDArray *out, int priority = 0);

/*!
 * \brief elementwise add
 * \param lhs left operand
 * \param rhs right operand
 * \return a new result ndarray
 */
NDArray operator+(const NDArray &lhs, const NDArray &rhs);
/*!
 * \brief elementwise add
 * \param lhs left operand
 * \param rhs right operand
 * \return a new result ndarray
 */
NDArray operator+(const NDArray &lhs, const real_t &rhs);
/*!
 * \brief elementwise subtraction
 * \param lhs left operand
 * \param rhs right operand
 * \return a new result ndarray
 */
NDArray operator-(const NDArray &lhs, const NDArray &rhs);
/*!
 * \brief elementwise subtraction
 * \param lhs left operand
 * \param rhs right operand
 * \return a new result ndarray
 */
NDArray operator-(const NDArray &lhs, const real_t &rhs);
/*!
 * \brief elementwise multiplication
 * \param lhs left operand
 * \param rhs right operand
 * \return a new result ndarray
 */
NDArray operator*(const NDArray &lhs, const NDArray &rhs); \
/*!
 * \brief elementwise multiplication
 * \param lhs left operand
 * \param rhs right operand
 * \return a new result ndarray
 */
NDArray operator*(const NDArray &lhs, const real_t &rhs);
/*!
 * \brief elementwise division
 * \param lhs left operand
 * \param rhs right operand
 * \return a new result ndarray
 */
NDArray operator/(const NDArray &lhs, const NDArray &rhs);
/*!
 * \brief elementwise division
 * \param lhs left operand
 * \param rhs right operand
 * \return a new result ndarray
 */
NDArray operator/(const NDArray &lhs, const real_t &rhs);

/*!
 * \brief Seed the random number generator.
 * \param seed the seed to set to global random number generators.
 */
void RandomSeed(uint32_t seed);
/*!
 * \brief Sample uniform distribution for each elements of out.
 * \param begin lower bound of distribution.
 * \param end upper bound of distribution.
 * \param out output NDArray.
 */
void SampleUniform(real_t begin, real_t end, NDArray *out);
/*!
 * \brief Sample gaussian distribution for each elements of out.
 * \param mu mean of gaussian distribution.
 * \param sigma standard deviation of gaussian distribution.
 * \param out output NDArray.
 */
void SampleGaussian(real_t mu, real_t sigma, NDArray *out);
/*!
 * \brief Sample gamma distribution for each elements of out.
 * \param alpha parameter (shape) of the gamma distribution
 * \param beta parameter (scale) of the gamma distribution
 * \param out output NDArray.
 */
void SampleGamma(real_t alpha, real_t beta, NDArray *out);
/*!
 * \brief Sample exponential distribution for each elements of out.
 * \param lambda parameter (rate) of the exponential distribution
 * \param out output NDArray.
 */
void SampleExponential(real_t lambda, NDArray *out);
/*!
 * \brief Sample Poisson distribution for each elements of out.
 * \param lambda parameter (rate) of the Poisson distribution
 * \param out output NDArray.
 */
void SamplePoisson(real_t lambda, NDArray *out);
/*!
 * \brief Sample negative binomial distribution for each elements of out.
 * \param k failure limit
 * \param p success probability
 * \param out output NDArray.
 */
void SampleNegBinomial(int32_t k, real_t p, NDArray *out);
/*!
 * \brief Sample generalized negative binomial distribution for each elements of out.
 * \param mu parameter (mean) of the distribution
 * \param alpha parameter (over dispersion) of the distribution
 * \param out output NDArray.
 */
void SampleGenNegBinomial(real_t mu, real_t alpha, NDArray *out);


//--------------------------------------------------------------
// The following part are API Registration of NDArray functions.
//--------------------------------------------------------------

/*! \brief definition of NDArray function */
typedef std::function<void (NDArray **used_vars,
                            real_t *scalars,
                            NDArray **mutate_vars,
                            int num_params,
                            char **param_keys,
                            char **param_vals)> NDArrayAPIFunction;
/*! \brief mask information on how functions can be exposed */
enum NDArrayFunctionTypeMask {
  /*! \brief all the use_vars should go before scalar */
  kNDArrayArgBeforeScalar = 1,
  /*! \brief all the scalar should go before use_vars */
  kScalarArgBeforeNDArray = 1 << 1,
  /*!
   * \brief whether this function allows the handles in the target to
   *  be empty NDArray that are not yet initialized, and will initialize
   *  them when the function is invoked.
   *
   *  most function should support this, except copy between different
   *  devices, which requires the NDArray to be pre-initialized with context
   */
  kAcceptEmptyMutateTarget = 1 << 2
};
/*! \brief Registry entry for NDArrayFunction */
struct NDArrayFunctionReg
    : public dmlc::FunctionRegEntryBase<NDArrayFunctionReg,
                                        NDArrayAPIFunction> {
  /*! \brief number of variable used by this function */
  unsigned num_use_vars;
  /*! \brief number of variable mutated by this function */
  unsigned num_mutate_vars;
  /*! \brief number of scalars used by this function */
  unsigned num_scalars;
  /*! \brief information on how function should be called from API */
  int type_mask;
  /*!
   * \brief constructor
   */
  NDArrayFunctionReg()
      : num_use_vars(0),
        num_mutate_vars(0),
        num_scalars(0),
        type_mask(0) {}
  /*!
   * \brief set the function body to a NDArray setvalue function
   *  this will also auto set the parameters correctly
   * \param fsetvalue function body to set
   * \return ref to the registered entry, used to set properties
   */
  inline NDArrayFunctionReg &set_function(void (*fsetvalue)(const real_t &rhs,
                                                            NDArray *out)) {
    body = [fsetvalue] (NDArray **used_vars, real_t *s, NDArray **mutate_vars,
                        int num_params, char **param_keys, char **param_vals) {
      (*fsetvalue)(s[0], mutate_vars[0]);
    };
    num_mutate_vars = 1; num_scalars = 1;
    this->add_argument("src", "real_t", "Source input to the function.");
    return *this;
  }
  /*!
  * \brief set the function body to a ternary NDArray function
  *  this will also auto set the parameters correctly
  * \param fternary function body to set
  * \return ref to the registered entry, used to set properties
  */
  inline NDArrayFunctionReg &set_function(void(*fternary)(const NDArray &lhs,
                                                          const NDArray &mhs,
                                                          const NDArray &rhs,
                                                                NDArray *out)) {
    body = [fternary](NDArray **used_vars,
      real_t *s, NDArray **mutate_vars,
      int num_params, char **param_keys, char **param_vals) {
      (*fternary)(*used_vars[0], *used_vars[1], *used_vars[2], mutate_vars[0]);
    };
    num_use_vars = 3; num_mutate_vars = 1;
    type_mask = kNDArrayArgBeforeScalar | kAcceptEmptyMutateTarget;
    this->add_argument("lhs", "NDArray", "Left operand to the function.");
    this->add_argument("mhs", "NDArray", "Middle operand to the function.");
    this->add_argument("rhs", "NDArray", "Right operand to the function.");
    return *this;
  }
  /*!
   * \brief set the function body to a binary NDArray function
   *  this will also auto set the parameters correctly
   * \param fbinary function body to set
   * \return ref to the registered entry, used to set properties
   */
  inline NDArrayFunctionReg &set_function(void (*fbinary)(const NDArray &lhs,
                                                          const NDArray &rhs,
                                                          NDArray *out)) {
    body = [fbinary] (NDArray **used_vars, real_t *s, NDArray **mutate_vars,
                      int num_params, char **param_keys, char **param_vals) {
      (*fbinary)(*used_vars[0], *used_vars[1], mutate_vars[0]);
    };
    num_use_vars = 2; num_mutate_vars = 1;
    type_mask = kNDArrayArgBeforeScalar | kAcceptEmptyMutateTarget;
    this->add_argument("lhs", "NDArray", "Left operand to the function.");
    this->add_argument("rhs", "NDArray", "Right operand to the function.");
    return *this;
  }
  /*!
   * \brief set the function body to a binary NDArray function
   *  this will also auto set the parameters correctly
   * \param fscalar function body to set
   * \return ref to the registered entry, used to set properties
   */
  inline NDArrayFunctionReg &set_function(void (*fscalar)(const NDArray &lhs,
                                                          const real_t &rhs,
                                                          NDArray *out)) {
    body = [fscalar] (NDArray **used_vars, real_t *s, NDArray **mutate_vars,
                      int num_params, char **param_keys, char **param_vals) {
      (*fscalar)(*used_vars[0], s[0], mutate_vars[0]);
    };
    num_use_vars = 1; num_mutate_vars = 1; num_scalars = 1;
    type_mask = kNDArrayArgBeforeScalar | kAcceptEmptyMutateTarget;
    this->add_argument("lhs", "NDArray", "Left operand to the function.");
    this->add_argument("rhs", "real_t", "Right operand to the function.");
    return *this;
  }
  /*!
   * \brief set the function body to a unary NDArray function
   *  this will also auto set the parameters correctly
   * \param funary function body to set
   * \return ref to the registered entry, used to set properties
   */
  inline NDArrayFunctionReg &set_function(void (*funary)(const NDArray &src,
                                                         NDArray *out)) {
    body = [funary] (NDArray **used_vars, real_t *s, NDArray **mutate_vars,
                     int num_params, char **param_keys, char **param_vals) {
      (*funary)(*used_vars[0], mutate_vars[0]);
    };
    num_use_vars = 1; num_mutate_vars = 1;
    type_mask = kNDArrayArgBeforeScalar | kAcceptEmptyMutateTarget;
    this->add_argument("src", "NDArray", "Source input to the function.");
    return *this;
  }
  /*!
   * \brief set the function body to a unary NDArray function
   *  this will also auto set the parameters correctly
   * \param fgeneric function body to set
   * \return ref to the registered entry, used to set properties
   */
  inline NDArrayFunctionReg &set_function(
    void (*fgeneric)(NDArray **used_vars,
                     real_t *s,
                     NDArray **mutate_vars,
                     const std::map<std::string, std::string>& param)) {
    body = [fgeneric] (NDArray **used_vars, real_t *s, NDArray **mutate_vars,
                       int num_params, char **param_keys, char **param_vals) {
      std::map<std::string, std::string> param;
      for (int i = 0; i < num_params; ++i) {
        param[param_keys[i]] = param_vals[i];
      }
      fgeneric(used_vars, s, mutate_vars, param);
    };
    return *this;
  }
  /*!
   * \brief set the number of mutate variables
   * \param n number of mutate variablesx
   * \return ref to the registered entry, used to set properties
   */
  inline NDArrayFunctionReg &set_num_use_vars(unsigned n) {
    num_use_vars = n; return *this;
  }
  /*!
   * \brief set the number of mutate variables
   * \param n number of mutate variablesx
   * \return ref to the registered entry, used to set properties
   */
  inline NDArrayFunctionReg &set_num_mutate_vars(unsigned n) {
    num_mutate_vars = n; return *this;
  }
  /*!
   * \brief set the number of scalar arguments
   * \param n number of scalar arguments
   * \return ref to the registered entry, used to set properties
   */
  inline NDArrayFunctionReg &set_num_scalars(unsigned n) {
    num_scalars = n; return *this;
  }
  /*!
   * \brief set type mask
   * \param tmask typemask
   * \return ref to the registered entry, used to set properties
   */
  inline NDArrayFunctionReg &set_type_mask(int tmask) {
    type_mask = tmask; return *this;
  }
};  // NDArrayFunctionReg

/*!
 * \brief Macro to register NDArray function
 *
 * Example: the following code is example to register a plus
 * \code
 *
 * REGISTER_NDARRAY_FUN(Plus)
 * .set_function(Plus);
 *
 * \endcode
 */
#define MXNET_REGISTER_NDARRAY_FUN(name)                                 \
  DMLC_REGISTRY_REGISTER(::mxnet::NDArrayFunctionReg, NDArrayFunctionReg, name)

}  // namespace mxnet

namespace dmlc {
/*!\brief traits */
DMLC_DECLARE_TRAITS(has_saveload, mxnet::NDArray, true);
}  // namespace dmlc
#endif  // MXNET_NDARRAY_H_
