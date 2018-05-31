/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#ifndef MXNET_OPERATOR_SUBGRAPH_SUBGRAPH_OP_H_

#include <nnvm/graph.h>
#include <nnvm/pass.h>

namespace mxnet {

namespace op {

namespace sg {  // sg stands for subgraph

struct SimpleNode;
using SimpleNodePtr = std::shared_ptr<SimpleNode>;

struct SimpleNode {
  static SimpleNodePtr Create() {
    return std::make_shared<SimpleNode>();
  }
  SimpleNode() : label(-1), node(nullptr) {}
  int label;
  nnvm::Node* node;
  // key is node ptr
  // value is the index array standing for the entry indices
  // in key->inputs that use this->node as input node
  std::unordered_map<nnvm::Node*, std::vector<int>> outputs;
};

}

/*
 * This provides criteria for selecting nodes in a subgraph.
 * When a node is passed to this object, the selection criteria may be changed.
 * We can also specify what links we should use when traversing the neighbor
 * nodes.
 */
class SubgraphSelect {
 public:
  virtual ~SubgraphSelect() {
  }
  /*
   * Given a set of nodes that have been selected so far for a subgraph, determine
   * if the input node should be selected for a subgraph.
   */
  virtual bool Select(const nnvm::Node &n,
                      const std::vector<sg::SimpleNode*> *subgraph_nodes) = 0;
  virtual bool UseIncomingLink() const = 0;
  virtual bool UseOutgoingLink() const = 0;
  virtual void Reset() {
  }
};

using SubgraphSelectPtr = std::shared_ptr<SubgraphSelect>;

class SubgraphOpState {
  nnvm::Symbol subgraph_sym_;
public:
  SubgraphOpState(const nnvm::Symbol &sym) {
    this->subgraph_sym_ = sym;
  }

  virtual ~SubgraphOpState() {
  }

  const nnvm::Symbol &GetSubgraph() const {
    return subgraph_sym_;
  }

  virtual void Forward(const OpContext& ctx,
                       const std::vector<NDArray>& inputs,
                       const std::vector<OpReqType>& req,
                       const std::vector<NDArray>& outputs) = 0;
  virtual void Backward(const OpContext& ctx,
                        const std::vector<NDArray>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<NDArray>& outputs) = 0;
};

/*
 * This provides a set of properties for partitioning a graph into subgraphs
 * and reconstructing a new graph from the subgraphs.
 * Currently, it has two properties:
 * * the criteria of selecting the subgraph nodes,
 * * create an nnvm node for a given subgraph. Here users can customize how to
 *   execute the operators in the subgraph.
 */
class SubgraphProperty {
 public:
  virtual nnvm::NodePtr GetSubgraphNode(const nnvm::Symbol &s) const = 0;
  virtual SubgraphSelectPtr CreateSubgraphSelect() const = 0;
};

using SubgraphPropertyPtr = std::shared_ptr<SubgraphProperty>;

/*
 * This selects nodes for a subgraph that only contains operators
 * in a given set and it visits nodes via both input and output links.
 */
class ContainOpSelect: public SubgraphSelect {
  std::shared_ptr<const std::unordered_set<std::string>> op_names;

 public:
  ContainOpSelect(std::shared_ptr<const std::unordered_set<std::string>> op_names) {
    this->op_names = op_names;
  }

  virtual bool UseIncomingLink() const {
    return true;
  }

  virtual bool UseOutgoingLink() const {
    return true;
  }

  virtual bool Select(const nnvm::Node &n,
                      const std::vector<sg::SimpleNode*> *subgraph_nodes) {
    return !n.is_variable() && op_names->count(n.op()->name);
  }
};

/*
 * This subgraph property finds a subgraph whose nodes have only operators
 * within a set. The operators in the subgraph will be executed by _subgraph_op.
 */
class SimpleSubgraphProperty: public SubgraphProperty {
  std::shared_ptr<const std::unordered_set<std::string>> op_names;

 public:
  SimpleSubgraphProperty(const std::unordered_set<std::string> &op_names) {
    this->op_names = std::make_shared<std::unordered_set<std::string>>(op_names);
  }
  virtual nnvm::NodePtr GetSubgraphNode(const nnvm::Symbol &sym) const {
    nnvm::NodePtr n = nnvm::Node::Create();
    n->attrs.op = Op::Get("_subgraph_op");
    n->attrs.name = "_subgraph_op";
    n->attrs.dict.insert(std::pair<std::string, std::string>("exec_type", "default"));
    n->attrs.parsed = std::move(sym);
    return n;
  }
  virtual SubgraphSelectPtr CreateSubgraphSelect() const {
    return std::make_shared<ContainOpSelect>(op_names);
  }
};

}
}

#endif  // MXNET_OPERATOR_SUBGRAPH_SUBGRAPH_OP_H_ 
