/*!
 *  Copyright (c) 2016 by Contributors
 * \file c_api_executor.cc
 * \brief C API of mxnet
 */
#include <mxnet/base.h>
#include <mxnet/c_api.h>
#include <mxnet/executor.h>
#include <nnvm/graph_attr_types.h>
#include "./c_api_common.h"

int MXExecutorPrint(ExecutorHandle handle, const char **out_str) {
  Executor *exec = static_cast<Executor*>(handle);
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();
  API_BEGIN();
  std::ostringstream os;
  exec->Print(os);
  ret->ret_str = os.str();
  *out_str = (ret->ret_str).c_str();
  API_END();
}

int MXExecutorFree(ExecutorHandle handle) {
  API_BEGIN();
  delete static_cast<Executor*>(handle);
  API_END();
}

int MXExecutorForward(ExecutorHandle handle, int is_train) {
  API_BEGIN();
  Executor *exec = static_cast<Executor*>(handle);
  exec->Forward(is_train != 0);
  API_END();
}

int MXExecutorBackward(ExecutorHandle handle,
                       mx_uint len,
                       NDArrayHandle *head_grads) {
  API_BEGIN();
  Executor *exec = static_cast<Executor*>(handle);
  std::vector<NDArray> ndarrays;
  NDArray **args_ptr = reinterpret_cast<NDArray**>(head_grads);
  for (mx_uint i = 0; i < len; ++i) {
    ndarrays.push_back(*args_ptr[i]);
  }
  exec->Backward(ndarrays);
  API_END();
}

int MXExecutorOutputs(ExecutorHandle handle,
                      mx_uint *out_size,
                      NDArrayHandle **out) {
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();
  API_BEGIN();
  Executor *exec = static_cast<Executor*>(handle);
  std::vector<NDArray> heads = exec->outputs();
  ret->ret_handles.resize(heads.size());
  for (size_t i = 0; i < heads.size(); ++i) {
    NDArray *ptr = new NDArray();
    *ptr = heads[i];
    ret->ret_handles[i] = ptr;
  }
  *out_size = heads.size();
  *out = dmlc::BeginPtr(ret->ret_handles);
  API_END();
}

int MXExecutorBind(SymbolHandle symbol_handle,
                   int dev_type,
                   int dev_id,
                   mx_uint len,
                   NDArrayHandle *in_args,
                   NDArrayHandle *arg_grad_store,
                   mx_uint *grad_req_type,
                   mx_uint aux_states_len,
                   NDArrayHandle *aux_states,
                   ExecutorHandle *out) {
  return MXExecutorBindX(symbol_handle,
                         dev_type, dev_id,
                         0, nullptr, nullptr, nullptr,
                         len, in_args, arg_grad_store, grad_req_type,
                         aux_states_len, aux_states, out);
}

int MXExecutorBindX(SymbolHandle symbol_handle,
                    int dev_type,
                    int dev_id,
                    mx_uint num_map_keys,
                    const char** map_keys,
                    const int* map_dev_types,
                    const int* map_dev_ids,
                    mx_uint len,
                    NDArrayHandle *in_args,
                    NDArrayHandle *arg_grad_store,
                    mx_uint *grad_req_type,
                    mx_uint aux_states_len,
                    NDArrayHandle *aux_states,
                    ExecutorHandle *out) {
  return MXExecutorBindEX(symbol_handle,
                          dev_type, dev_id,
                          num_map_keys, map_keys, map_dev_types, map_dev_ids,
                          len, in_args, arg_grad_store, grad_req_type,
                          aux_states_len, aux_states,
                          NULL, out);
}

int MXExecutorBindEX(SymbolHandle symbol_handle,
                     int dev_type,
                     int dev_id,
                     mx_uint num_map_keys,
                     const char** map_keys,
                     const int* map_dev_types,
                     const int* map_dev_ids,
                     mx_uint len,
                     NDArrayHandle *in_args,
                     NDArrayHandle *arg_grad_store,
                     mx_uint *grad_req_type,
                     mx_uint aux_states_len,
                     NDArrayHandle *aux_states,
                     ExecutorHandle shared_exec,
                     ExecutorHandle *out) {
  Executor* exec = nullptr;

  API_BEGIN();
  nnvm::Symbol *symb = static_cast<nnvm::Symbol*>(symbol_handle);
  Context ctx = Context::Create(static_cast<Context::DeviceType>(dev_type), dev_id);
  std::map<std::string, Context> ctx_map;
  for (mx_uint i = 0; i < num_map_keys; ++i) {
    ctx_map[std::string(map_keys[i])] = Context::Create(
        static_cast<Context::DeviceType>(map_dev_types[i]), map_dev_ids[i]);
  }
  NDArray **in_args_ptr = reinterpret_cast<NDArray**>(in_args);
  NDArray **arg_grad_ptr = reinterpret_cast<NDArray**>(arg_grad_store);
  NDArray **aux_states_ptr = reinterpret_cast<NDArray**>(aux_states);
  std::vector<NDArray> in_args_vec;
  std::vector<NDArray> arg_grad_vec;
  std::vector<OpReqType> grad_req_vec;
  std::vector<NDArray> aux_states_vec;
  for (mx_uint i = 0; i < len; ++i) {
    in_args_vec.push_back(*(in_args_ptr[i]));
    if (arg_grad_ptr[i] == nullptr) {
      arg_grad_vec.push_back(NDArray());
      grad_req_vec.push_back(kNullOp);
    } else {
      arg_grad_vec.push_back(*(arg_grad_ptr[i]));
      grad_req_vec.push_back(static_cast<OpReqType>(grad_req_type[i]));
    }
  }
  for (mx_uint i = 0; i < aux_states_len; ++i) {
    aux_states_vec.push_back(*(aux_states_ptr[i]));
  }
  *out = Executor::Bind(*symb, ctx, ctx_map, in_args_vec,
                        arg_grad_vec, grad_req_vec, aux_states_vec,
                        reinterpret_cast<Executor*>(shared_exec));
  API_END_HANDLE_ERROR(delete exec);
}

int MXExecutorSimpleBind(SymbolHandle symbol_handle,
                         int dev_type,  // default device type
                         int dev_id,  // default device id
                         const mx_uint num_g2c_keys,  // num of keys in group2ctx
                         const char** g2c_keys,  // arg names of group2ctx
                         const int* g2c_dev_types,  // ctx dev_types of group2ctx
                         const int* g2c_dev_ids,  // ctx dev_ids of group2ctx
                         const mx_uint in_arg_len,  // num of all in_args (no aux_states)
                         const int* in_arg_dev_types,  // all in_arg dev_types
                         const int* in_arg_dev_ids,  // all in_arg dev_ids
                         const mx_uint* grad_req_types,  // req types of all in_arg_grads
                         const mx_uint aux_state_len,  // number of aux_states
                         const int* aux_state_dev_types,  // aux_state ctx dev_types
                         const int* aux_state_dev_ids,  // aux_state ctx dev_ids
                         const mx_uint num_provided_args,  // #user provided in_args and aux_states
                         const char** provided_arg_shape_names,  // user provided arg names
                         const mx_uint* provided_arg_shape_data,  // provided arg shapes
                         const mx_uint* provided_arg_shape_idx,  // provided arg shape idx
                         const mx_uint num_provided_arg_dtypes,  // #provided arg dtypes
                         const char** provided_arg_dtype_names,  // provided arg dtype names
                         const int* provided_arg_dtypes,  // provided dtypes of args
                         const mx_uint num_param_names,
                         const char** param_name_list,
                         mx_uint* num_shared_data_arrays,  // negative number indicates no shared_data_arrays
                         const char*** shared_data_array_name_list,
                         NDArrayHandle** shared_data_array_handle_list,
                         const mx_uint num_shared_exec_in_args,
                         NDArrayHandle* shared_exec_in_arg_handles,
                         const mx_uint num_shared_exec_arg_grads,
                         NDArrayHandle* shared_exec_arg_grad_handles,
                         const mx_uint num_shared_exec_aux_states,
                         NDArrayHandle* shared_exec_aux_state_handles,
                         NDArrayHandle** in_args,
                         NDArrayHandle** arg_grads,
                         NDArrayHandle** aux_states,
                         ExecutorHandle shared_exec_handle,
                         ExecutorHandle* out) {
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();
  API_BEGIN();
  nnvm::Symbol *sym = static_cast<nnvm::Symbol*>(symbol_handle);
  // create default ctx
  Context ctx = Context::Create(static_cast<Context::DeviceType>(dev_type), dev_id);

  // create ctx map
  std::map<std::string, Context> ctx_map;
  for (mx_uint i = 0; i < num_g2c_keys; ++i) {
    ctx_map[std::string(g2c_keys[i])] = Context::Create(
        static_cast<Context::DeviceType>(g2c_dev_types[i]), g2c_dev_ids[i]);
  }

  // create ctxes for in_args and arg_grads
  // create grad_req_type_vec for in_arg_grads
  std::vector<Context> in_arg_ctx_vec;
  std::vector<Context> arg_grad_ctx_vec;
  std::vector<OpReqType> grad_req_type_vec;
  for (mx_uint i = 0; i < in_arg_len; ++i) {
    in_arg_ctx_vec.push_back(Context::Create(
          static_cast<Context::DeviceType>(in_arg_dev_types[i]), in_arg_dev_ids[i]));
    if (grad_req_types[i] == 0) {
      arg_grad_ctx_vec.push_back(Context());
      grad_req_type_vec.push_back(kNullOp);
    } else {
      arg_grad_ctx_vec.push_back(Context::Create(
            static_cast<Context::DeviceType>(in_arg_dev_types[i]), in_arg_dev_ids[i]));
      grad_req_type_vec.push_back(static_cast<OpReqType>(grad_req_types[i]));
    }
  }

  // create ctxes for aux_states
  std::vector<Context> aux_state_ctx_vec;
  for (mx_uint i = 0; i < aux_state_len; ++i) {
    aux_state_ctx_vec.push_back(Context::Create(
          static_cast<Context::DeviceType>(aux_state_dev_types[i]), aux_state_dev_ids[i]));
  }

  // create shape map for in_args and aux_states
  std::unordered_map<std::string, TShape> arg_shape_map;
  for (mx_uint i = 0; i < num_provided_args; ++i) {
    arg_shape_map[provided_arg_shape_names[i]] =
      TShape(provided_arg_shape_data+provided_arg_shape_idx[i],
             provided_arg_shape_data+provided_arg_shape_idx[i+1]);
  }

#if 0
  // create arg_shape vector for all input nodes including
  // in_args and aux_states that are not provided by the user
  nnvm::Graph g = Symbol2Graph(*sym);
  const size_t num_input_nodes = g.indexed_graph().input_nodes().size();
  nnvm::ShapeVector arg_shapes(num_input_nodes, TShape());
  mxnet::MatchArguments(g.indexed_graph(), arg_shape_map, &arg_shapes, "SimpleBind");
#endif

  // create dtype map for in_args and aux_states
  std::unordered_map<std::string, int> arg_dtype_map;
  for (mx_uint i = 0; i < num_provided_arg_dtypes; ++i) {
    arg_dtype_map[provided_arg_dtype_names[i]] = provided_arg_dtypes[i];
  }

#if 0
  // create arg_dtype vector for all input nodes including
  // in_args and aux_states that are not provided by the user
  nnvm::DTypeVector arg_dtypes(num_input_nodes, -1);
  mxnet::MatchArguments(g.indexed_graph(), arg_dtype_map, &arg_dtypes, "SimpleBind");
#endif

  // create para name set for sharing data array memory
  std::unordered_set<std::string> param_name_set;
  for (mx_uint i = 0; i < num_param_names; ++i) {
    param_name_set.insert(param_name_list[i]);
  }

  // create shared_data_array_map
  std::unordered_map<std::string, NDArray> shared_data_array_map;
  std::vector<NDArray> shared_exec_in_args;
  std::vector<NDArray> shared_exec_arg_grads;
  std::vector<NDArray> shared_exec_aux_states;
  bool use_shared_data_arrays = (nullptr != *shared_data_array_handle_list);
  if (use_shared_data_arrays) {
    // create shared_data_array_map
    NDArray*** shared_data_array_ptrs =
      reinterpret_cast<NDArray***>(shared_data_array_handle_list);
    for (mx_uint i = 0; i < *num_shared_data_arrays; ++i) {
      shared_data_array_map[*shared_data_array_name_list[i]] = *(*shared_data_array_ptrs)[i];
    }
    
    // create shared_exec_in_args
    NDArray** shared_exec_in_arg_ptrs =
      reinterpret_cast<NDArray**>(shared_exec_in_arg_handles);
    for (mx_uint i = 0; i < num_shared_exec_in_args; ++i) {
      shared_exec_in_args.push_back(*shared_exec_in_arg_ptrs[i]);
    }

    // create shared_exec_arg_grads
    NDArray** shared_exec_arg_grad_ptrs =
      reinterpret_cast<NDArray**>(shared_exec_arg_grad_handles);
    for (mx_uint i = 0; i < num_shared_exec_arg_grads; ++i) {
      if (nullptr == shared_exec_arg_grad_ptrs[i]) {
        shared_exec_arg_grads.push_back(NDArray());
      } else {
        shared_exec_arg_grads.push_back(*shared_exec_arg_grad_ptrs[i]);
      }
    }

    // create shared_exec_aux_states
    NDArray** shared_exec_aux_state_ptrs =
      reinterpret_cast<NDArray**>(shared_exec_aux_state_handles);
    for (mx_uint i = 0; i < num_shared_exec_aux_states; ++i) {
      shared_exec_aux_states.push_back(*shared_exec_aux_state_ptrs[i]);
    }
  }


  std::vector<NDArray> in_arg_vec;
  std::vector<NDArray> arg_grad_vec;
  std::vector<NDArray> aux_state_vec;

  *out = Executor::SimpleBind(*sym, ctx, ctx_map, in_arg_ctx_vec, arg_grad_ctx_vec,
                              aux_state_ctx_vec, arg_shape_map, arg_dtype_map, grad_req_type_vec,
                              param_name_set, shared_exec_in_args, shared_exec_arg_grads,
                              shared_exec_aux_states, &in_arg_vec, &arg_grad_vec, &aux_state_vec,
                              use_shared_data_arrays? &shared_data_array_map : nullptr,
                              reinterpret_cast<Executor*>(shared_exec_handle));

  // copy ndarray ptrs to ret->handles for persistence
  ret->ret_handles.clear();
  ret->ret_handles.reserve(in_arg_vec.size()+arg_grad_vec.size()+aux_state_vec.size()
                           +shared_data_array_map.size());
  size_t nd_idx = 0;
  for (const auto& nd : in_arg_vec) {
    if (nd.is_none()) {
      LOG(FATAL) << "Input argument NDArray cannot be un-allocated";
    }
    ret->ret_handles.push_back(new NDArray(nd));
  }
  if (in_arg_vec.size() > 0) {
    *in_args = &(ret->ret_handles[nd_idx]);
    nd_idx = ret->ret_handles.size();
  }

  for (const auto& nd : arg_grad_vec) {
    if (nd.is_none()) {
      ret->ret_handles.push_back(nullptr);
    } else {
      ret->ret_handles.push_back(new NDArray(nd));
    }
  }
  if (arg_grad_vec.size() > 0) {
    *arg_grads = &(ret->ret_handles[nd_idx]);
    nd_idx = ret->ret_handles.size();
  }

  for (const auto& nd : aux_state_vec) {
    if (nd.is_none()) {
      LOG(FATAL) << "Auxiliary argument NDArray cannot be un-allocated";
    }
    ret->ret_handles.push_back(new NDArray(nd));
  }
  if (aux_state_vec.size() > 0) {
    *aux_states = &(ret->ret_handles[nd_idx]);
    nd_idx = ret->ret_handles.size();
  }

  if (use_shared_data_arrays) {
    ret->ret_vec_charp.clear();
    ret->ret_vec_charp.reserve(shared_data_array_map.size());
    for (const auto kv : shared_data_array_map) {
      if (kv.second.is_none()) {
        LOG(FATAL) << "Shared data NDArray cannot be un-allocated";
      }
      ret->ret_handles.push_back(new NDArray(kv.second));
      ret->ret_vec_charp.push_back(kv.first.c_str());
    }
    *num_shared_data_arrays = shared_data_array_map.size();
    *shared_data_array_handle_list = &(ret->ret_handles[nd_idx]);
    *shared_data_array_name_list = &(ret->ret_vec_charp[0]);
  }

  API_END();
}

int MXExecutorSetMonitorCallback(ExecutorHandle handle,
                                 ExecutorMonitorCallback callback,
                                 void* callback_handle) {
  API_BEGIN();
  ExecutorMonitorCallback callback_temp = callback;
  void* callback_handle_temp = callback_handle;
  std::function<void(const char*, void*)> clbk
  = [callback_temp, callback_handle_temp](const char *name, void* handle) {
    callback_temp(name, handle, callback_handle_temp);
  };
  Executor *exec = static_cast<Executor*>(handle);
  exec->SetMonitorCallback(clbk);
  API_END();
}
