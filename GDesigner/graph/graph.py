import shortuuid
from typing import Any, List, Optional, Dict, Tuple
from abc import ABC
import numpy as np
import torch
import asyncio
import copy

from GDesigner.graph.node import Node
from GDesigner.agents.agent_registry import AgentRegistry
from GDesigner.prompt.prompt_set_registry import PromptSetRegistry
from GDesigner.llm.profile_embedding import get_sentence_embedding
from GDesigner.gnn.gcn import GCN,MLP
from torch_geometric.utils import dense_to_sparse

class Graph(ABC):
    """
    A framework for managing and executing a network of nodes using a language model.

    This class enables the creation of a graph structure for processing and analyzing data. Each node
    in the graph can perform specific operations, allowing for complex data processing workflows.
    The graph supports integration with language models, making it suitable for tasks that require
    natural language processing capabilities.

    The communication of the node depends on the node.spatial_predecessors and node.spatial_successors.
    
    Attributes:
        domain (str): The domain for which this graph is used.
        llm_name (str): The name of the llm that used for processing within the nodes.
        nodes (dict): A collection of nodes, each identified by a unique UUID.

    Methods:
        build_graph(): Method to be implemented for constructing the graph structure.
        add_node(node): Adds a new node to the graph with a unique identifier.
        run(inputs, num_steps=10, single_agent=False): Executes the graph for a specified number of steps, processing provided inputs.
    """

    def __init__(self, 
                domain: str,
                llm_name: Optional[str],
                agent_names: List[str],
                decision_method: str,
                optimized_spatial:bool = False,
                initial_spatial_probability: float = 0.5,
                fixed_spatial_masks:List[List[int]] = None,
                optimized_temporal:bool = False,
                initial_temporal_probability: float = 0.5,
                fixed_temporal_masks:List[List[int]] = None,
                node_kwargs:List[Dict] = None,
                ):
        
        if fixed_spatial_masks is None:
            fixed_spatial_masks = [[1 if i!=j else 0 for j in range(len(agent_names))] for i in range(len(agent_names))]
        if fixed_temporal_masks is None:
            fixed_temporal_masks = [[1 for j in range(len(agent_names))] for i in range(len(agent_names))]
        fixed_spatial_masks = torch.tensor(fixed_spatial_masks).view(-1)
        fixed_temporal_masks = torch.tensor(fixed_temporal_masks).view(-1)
        assert len(fixed_spatial_masks)==len(agent_names)*len(agent_names),"The fixed_spatial_masks doesn't match the number of agents"
        assert len(fixed_temporal_masks)==len(agent_names)*len(agent_names),"The fixed_temporal_masks doesn't match the number of agents"
        
        self.id:str = shortuuid.ShortUUID().random(length=4)
        self.domain:str = domain
        self.llm_name:str = llm_name
        self.agent_names:List[str] = agent_names
        self.optimized_spatial = optimized_spatial
        self.optimized_temporal = optimized_temporal
        self.decision_node:Node = AgentRegistry.get(decision_method, **{"domain":self.domain,"llm_name":self.llm_name})
        self.nodes:Dict[str,Node] = {}
        self.potential_spatial_edges:List[List[str, str]] = []
        self.potential_temporal_edges:List[List[str,str]] = []
        self.node_kwargs = node_kwargs if node_kwargs is not None else [{} for _ in agent_names]
        
        self.init_nodes() # Corrected call, no 'self' needed as it's an instance method
        self.init_potential_edges() # Corrected call
        
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role_adj_matrix = self.construct_adj_matrix()
        self.features = self.construct_features()
        # Debug: check features dimensions
        print(f"Debug: Final self.features shape after construct_features: {self.features.shape}")
        if len(self.features.shape) < 2 or self.features.size(0) == 0:
            print(f"ERROR: Features construction failed! Shape: {self.features.shape}")
            print(f"This indicates a problem in node initialization or prompt set configuration")
            print(f"Agent names: {agent_names}")
            print(f"Domain: {domain}")
            print(f"Let me run the debug version to see what's happening...")
            
            # At this point nodes should already have been created, but let's see what we have
            print(f"Current nodes count: {len(self.nodes)}")
            print(f"Nodes: {list(self.nodes.keys())}")
            
            if len(self.nodes) == 0:
                print("Creating dummy features as fallback since no nodes were created.")
                num_agents = len(agent_names) 
                feature_dim = 384
                self.features = torch.randn(num_agents, feature_dim, dtype=torch.float32)
                print(f"Created dummy features with shape: {self.features.shape}")
            else:
                print("Nodes exist but features are empty. This shouldn't happen.")
                raise RuntimeError(f"Nodes created but features failed. Debug needed.")
        self.gcn = GCN(self.features.size(1)*2,16,self.features.size(1))
        self.mlp = MLP(384,16,16)

        init_spatial_logit = torch.log(torch.tensor(initial_spatial_probability / (1 - initial_spatial_probability))) if optimized_spatial else 10.0
        # self.spatial_logits = torch.nn.Parameter(torch.ones(len(self.potential_spatial_edges), requires_grad=optimized_spatial) * init_spatial_logit,
        #                                          requires_grad=optimized_spatial) # trainable edge logits
        self.spatial_masks = torch.nn.Parameter(fixed_spatial_masks,requires_grad=False)  # fixed edge masks

        init_temporal_logit = torch.log(torch.tensor(initial_temporal_probability / (1 - initial_temporal_probability))) if optimized_temporal else 10.0
        self.temporal_logits = torch.nn.Parameter(torch.ones(len(self.potential_temporal_edges), requires_grad=optimized_temporal) * init_temporal_logit,
                                                 requires_grad=optimized_temporal) # trainable edge logits
        self.temporal_masks = torch.nn.Parameter(fixed_temporal_masks,requires_grad=False)  # fixed edge masks
    
    def construct_adj_matrix(self):
        role_connect:List[Tuple[str,str]] = self.prompt_set.get_role_connection()
        num_nodes = self.num_nodes
        role_adj = torch.zeros((num_nodes,num_nodes))
        role_2_id = {}
        
        # First, collect all actual roles that exist in our nodes
        actual_roles = set()
        for node_id in self.nodes:
            role = self.nodes[node_id].role
            actual_roles.add(role)
            
        # Only initialize role mappings for roles that actually exist
        for role in actual_roles:
            role_2_id[role] = []
            
        # Map node indices to roles
        for i, node_id in enumerate(self.nodes):
            role = self.nodes[node_id].role
            role_2_id[role].append(i)
            
        for edge in role_connect:
            in_role, out_role = edge
            # Only process connections if both roles actually exist in our nodes
            if in_role in role_2_id and out_role in role_2_id:
                in_ids = role_2_id[in_role]
                out_ids = role_2_id[out_role]
                for in_id in in_ids:
                    for out_id in out_ids:
                        role_adj[in_id][out_id] = 1
        
        edge_index, edge_weight = dense_to_sparse(role_adj)
        return edge_index
    
    def construct_features(self):
        features = []
        print(f"Debug: self.nodes keys = {list(self.nodes.keys())}")
        print(f"Debug: Number of nodes = {len(self.nodes)}")
        
        for node_id in self.nodes:
            print(f"Debug: Processing node {node_id}")
            role = self.nodes[node_id].role
            print(f"Debug: Node {node_id} has role '{role}'")
            
            try:
                profile = self.prompt_set.get_description(role)
                print(f"Debug: Got profile for role '{role}': {profile[:100]}...")
                feature = get_sentence_embedding(profile)
                print(f"Debug: Feature shape for {role}: {np.array(feature).shape}")
                features.append(feature)
            except Exception as e:
                print(f"Error processing node {node_id} with role '{role}': {e}")
                # Create a default feature vector if there's an error
                default_feature = np.zeros(384, dtype=np.float32)  # Assuming 384-dim embeddings
                features.append(default_feature)
        
        if not features:
            print("Warning: No features generated, nodes list is empty!")
            return torch.tensor([])
            
        features_array = np.array(features, dtype=np.float32)
        print(f"Debug: Final features array shape: {features_array.shape}")
        features = torch.tensor(features_array, dtype=torch.float32)
        return features
    
    def construct_new_features(self, query):
        query_embedding = torch.tensor(get_sentence_embedding(query), dtype=torch.float32)
        
        # Ensure query_embedding has correct dimensions to match self.features
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.unsqueeze(0)  # Make it 2D: [1, embedding_dim]
        
        # Repeat to match the number of nodes in self.features
        actual_num_nodes = self.features.size(0)
        query_embedding = query_embedding.repeat((actual_num_nodes, 1))
        
        # Debug: check dimensions before concatenation
        print(f"Debug: self.features.shape = {self.features.shape}, query_embedding.shape = {query_embedding.shape}")
        
        new_features = torch.cat((self.features, query_embedding), dim=1)
        return new_features
        
    @property
    def spatial_adj_matrix(self):
        matrix = np.zeros((len(self.nodes), len(self.nodes)))
        for i, node1_id in enumerate(self.nodes):
            for j, node2_id in enumerate(self.nodes):
                if self.nodes[node2_id] in self.nodes[node1_id].spatial_successors: 
                    matrix[i, j] = 1
        return matrix

    @property
    def temporal_adj_matrix(self):
        matrix = np.zeros((len(self.nodes), len(self.nodes)))
        for i, node1_id in enumerate(self.nodes):
            for j, node2_id in enumerate(self.nodes):
                if self.nodes[node2_id] in self.nodes[node1_id].temporal_successors: 
                    matrix[i, j] = 1
        return matrix

    @property
    def num_edges(self):
        num_edges = 0
        for node in self.nodes.values():
            num_edges += len(node.spatial_successors)
        return num_edges
    
    @property
    def num_nodes(self):
        return len(self.nodes)

    def find_node(self, id: str):
        if id in self.nodes.keys():
            return self.nodes[id]
        raise Exception(f"Node not found: {id} among "
                        f"{[node.id for node in self.nodes.values()]}")
        
    def add_node(self, node: Node):
        node_id = node.id if node.id is not None else shortuuid.ShortUUID().random(length=4)
        while node_id in self.nodes:
            node_id = shortuuid.ShortUUID().random(length=4)
        node.id = node_id
        self.nodes[node_id] = node
        return node
    
    def init_nodes(self):
        """
        Creates and adds new nodes to the graph.
        """
        for agent_name,kwargs in zip(self.agent_names,self.node_kwargs):
            if agent_name in AgentRegistry.keys():
                kwargs["domain"] = self.domain
                kwargs["llm_name"] = self.llm_name
                agent_instance = AgentRegistry.get(agent_name, **kwargs)
                self.add_node(agent_instance)
    
    def init_potential_edges(self):
        """
        Creates and potential edges to the graph.
        """
        for node1_id in self.nodes.keys():
            for node2_id in self.nodes.keys():
                self.potential_spatial_edges.append([node1_id,node2_id])
                self.potential_temporal_edges.append([node1_id,node2_id])

    def clear_spatial_connection(self):
        """
        Clear all the spatial connection of the nodes in the graph.
        """
        for node_id in self.nodes.keys():
            self.nodes[node_id].spatial_predecessors = []
            self.nodes[node_id].spatial_successors = []
        self.decision_node.spatial_predecessors = []
        self.decision_node.spatial_successors = []
    
    def clear_temporal_connection(self):
        """
        Clear all the temporal connection of the nodes in the graph.
        """
        for node_id in self.nodes.keys():
            self.nodes[node_id].temporal_predecessors = []
            self.nodes[node_id].temporal_successors = []

    def connect_decision_node(self):
        for node_id in self.nodes.keys():
            self.nodes[node_id].add_successor(self.decision_node)

    def construct_spatial_connection(self, temperature: float = 1.0, threshold: float = None,): # temperature must >= 1.0
        self.clear_spatial_connection()
        log_probs = [torch.tensor(0.0, requires_grad=self.optimized_spatial)]
        
        for potential_connection, edge_logit, edge_mask in zip(self.potential_spatial_edges, self.spatial_logits, self.spatial_masks):
            out_node:Node = self.find_node(potential_connection[0])
            in_node:Node = self.find_node(potential_connection[1])
            if edge_mask == 0.0:
                continue
            elif edge_mask == 1.0 and self.optimized_spatial==False:
                if not self.check_cycle(in_node, {out_node}):
                    out_node.add_successor(in_node,'spatial')
                continue
            if not self.check_cycle(in_node, {out_node}):
                edge_prob = torch.sigmoid(edge_logit / temperature)
                if threshold:
                    edge_prob = torch.tensor(1 if edge_prob > threshold else 0)
                if torch.rand(1) < edge_prob:
                    out_node.add_successor(in_node,'spatial')
                    log_probs.append(torch.log(edge_prob))
                else:
                    log_probs.append(torch.log(1 - edge_prob))
                    
        return torch.sum(torch.stack(log_probs))
    
    def construct_temporal_connection(self, round:int = 0, temperature: float = 1.0, threshold: float = None,):  # temperature must >= 1.0
        self.clear_temporal_connection()
        log_probs = [torch.tensor(0.0, requires_grad=self.optimized_temporal)]
        if round == 0:
            return torch.sum(torch.stack(log_probs))  
        for potential_connection, edge_logit, edge_mask in zip(self.potential_temporal_edges, self.temporal_logits, self.temporal_masks):
            out_node:Node = self.find_node(potential_connection[0])
            in_node:Node = self.find_node(potential_connection[1])
            if edge_mask == 0.0:
                continue
            elif edge_mask == 1.0 and self.optimized_temporal==False:
                if not self.check_cycle(in_node, {out_node}):
                    out_node.add_successor(in_node,'temporal')
                continue
            
            edge_prob = torch.sigmoid(edge_logit / temperature)
            if threshold:
                edge_prob = torch.tensor(1 if edge_prob > threshold else 0)
            if torch.rand(1) < edge_prob:
                out_node.add_successor(in_node,'temporal')
                log_probs.append(torch.log(edge_prob))
            else:
                log_probs.append(torch.log(1 - edge_prob))
                    
        return torch.sum(torch.stack(log_probs))


    def run(self, inputs: Any, 
                  num_rounds:int = 3, 
                  max_tries: int = 3, 
                  max_time: int = 600,) -> List[Any]:
        # inputs:{'task':"xxx"}
        log_probs = 0
        for round in range(num_rounds):
            log_probs += self.construct_spatial_connection()
            log_probs += self.construct_temporal_connection(round)
            
            in_degree = {node_id: len(node.spatial_predecessors) for node_id, node in self.nodes.items()}
            zero_in_degree_queue = [node_id for node_id, deg in in_degree.items() if deg == 0]

            while zero_in_degree_queue:
                current_node_id = zero_in_degree_queue.pop(0)
                tries = 0
                while tries < max_tries:
                    try:
                        self.nodes[current_node_id].execute(inputs) # output is saved in the node.outputs
                        break
                    except Exception as e:
                        print(f"Error during execution of node {current_node_id}: {e}")
                    tries += 1
                for successor in self.nodes[current_node_id].spatial_successors:
                    if successor.id not in self.nodes.keys():
                        continue
                    in_degree[successor.id] -= 1
                    if in_degree[successor.id] == 0:
                        zero_in_degree_queue.append(successor.id)
            
            self.update_memory()
            
        self.connect_decision_node()
        self.decision_node.execute(inputs)
        final_answers = self.decision_node.outputs
        if len(final_answers) == 0:
            final_answers.append("No answer of the decision node")
            
        return final_answers, log_probs

    async def arun(self, input: Dict[str,str], 
                  num_rounds:int = 3, 
                  max_tries: int = 3, 
                  max_time: int = 600,) -> List[Any]:
        # inputs:{'task':"xxx"}
        log_probs = 0
        new_features = self.construct_new_features(input['task'])
        logits = self.gcn(new_features,self.role_adj_matrix)
        logits = self.mlp(logits)
        self.spatial_logits = logits @ logits.t()
        self.spatial_logits = min_max_norm(torch.flatten(self.spatial_logits))

        for round in range(num_rounds):
            log_probs += self.construct_spatial_connection()
            log_probs += self.construct_temporal_connection(round)
            
            in_degree = {node_id: len(node.spatial_predecessors) for node_id, node in self.nodes.items()}
            zero_in_degree_queue = [node_id for node_id, deg in in_degree.items() if deg == 0]

            while zero_in_degree_queue:
                current_node_id = zero_in_degree_queue.pop(0)
                tries = 0
                while tries < max_tries:
                    try:
                        await asyncio.wait_for(self.nodes[current_node_id].async_execute(input),timeout=max_time) # output is saved in the node.outputs
                        break
                    except Exception as e:
                        print(f"Error during execution of node {current_node_id}: {e}")
                    tries += 1
                for successor in self.nodes[current_node_id].spatial_successors:
                    if successor.id not in self.nodes.keys():
                        continue
                    in_degree[successor.id] -= 1
                    if in_degree[successor.id] == 0:
                        zero_in_degree_queue.append(successor.id)
            
            self.update_memory()
            
        self.connect_decision_node()
        await self.decision_node.async_execute(input)
        final_answers = self.decision_node.outputs
        if len(final_answers) == 0:
            final_answers.append("No answer of the decision node")
        return final_answers, log_probs
    
    def update_memory(self):
        for id,node in self.nodes.items():
            node.update_memory()
    
    def check_cycle(self, new_node, target_nodes):
        if new_node in target_nodes:
            return True
        for successor in new_node.spatial_successors:
            if self.check_cycle(successor, target_nodes):
                return True
        return False

    def update_masks(self, pruning_rate: float) -> torch.Tensor:
        if self.optimized_spatial:
            num_edges = (self.spatial_masks > 0).sum()
            num_masks = (self.spatial_masks == 0).sum()
            prune_num_edges = torch.round(num_edges*pruning_rate) if torch.round(num_edges*pruning_rate)>0 else 1
            _edge_logits = self.spatial_logits.clone()
            min_edge_logit = _edge_logits.min()
            _edge_logits[self.spatial_masks == 0] = min_edge_logit - 1.0
            sorted_edges_idx = torch.argsort(_edge_logits)
            prune_idx = sorted_edges_idx[:int(prune_num_edges + num_masks)]
            self.spatial_masks[prune_idx] = 0
        
        if self.optimized_temporal:
            num_edges = (self.temporal_masks > 0).sum()
            num_masks = (self.temporal_masks == 0).sum()
            prune_num_edges = torch.round(num_edges*pruning_rate) if torch.round(num_edges*pruning_rate)>0 else 1
            _edge_logits = self.temporal_logits.clone()
            min_edge_logit = _edge_logits.min()
            _edge_logits[self.temporal_masks == 0] = min_edge_logit - 1.0
            sorted_edges_idx = torch.argsort(_edge_logits)
            prune_idx = sorted_edges_idx[:int(prune_num_edges + num_masks)]
            self.temporal_masks[prune_idx] = 0
        return self.spatial_masks, self.temporal_masks

    def init_nodes(self):
        """Initialize nodes from agent_names"""
        print(f"Debug: Initializing nodes with agent_names: {self.agent_names}")
        
        # Check what agents are available in the registry
        available_agents = list(AgentRegistry.registry.keys()) if hasattr(AgentRegistry.registry, 'keys') else []
        print(f"Debug: Available agents in registry: {available_agents}")
        
        for i, agent_name in enumerate(self.agent_names):
            node_kwargs = self.node_kwargs[i] if i < len(self.node_kwargs) else {}
            try:
                # First try the agent_name directly
                node = AgentRegistry.get(agent_name, domain=self.domain, llm_name=self.llm_name, **node_kwargs)
                node_id = f"{agent_name}_{i}"
                self.nodes[node_id] = node
                print(f"Debug: Created node {node_id} with role {node.role}")
            except Exception as e:
                print(f"Debug: Direct agent creation failed for {agent_name}: {e}")
                # Try some common fallback agent types
                fallback_agents = ["MathSolver", "Critic", "AnalyzeAgent", "FinalDirect"]
                node_created = False
                
                for fallback_agent in fallback_agents:
                    try:
                        if fallback_agent in available_agents:
                            print(f"Debug: Trying fallback agent: {fallback_agent}")
                            node = AgentRegistry.get(fallback_agent, domain=self.domain, llm_name=self.llm_name, **node_kwargs)
                            # Override the role to match the intended agent_name, converting underscores to spaces
                            role_name = agent_name.replace('_', ' ')
                            node.role = role_name
                            node_id = f"{agent_name}_{i}"
                            self.nodes[node_id] = node
                            print(f"Debug: Created fallback node {node_id} with role {node.role}")
                            node_created = True
                            break
                    except Exception as fallback_e:
                        print(f"Debug: Fallback {fallback_agent} also failed: {fallback_e}")
                        continue
                
                if not node_created:
                    print(f"Warning: Could not create any node for {agent_name}")
                    
        print(f"Debug: Total nodes created: {len(self.nodes)}")
    
    def init_potential_edges(self):
        """Initialize potential edges between nodes"""
        node_ids = list(self.nodes.keys())
        for i, node1_id in enumerate(node_ids):
            for j, node2_id in enumerate(node_ids):
                if i != j:
                    self.potential_spatial_edges.append([node1_id, node2_id])
                    self.potential_temporal_edges.append([node1_id, node2_id])
        print(f"Debug: Created {len(self.potential_spatial_edges)} potential spatial edges")
        print(f"Debug: Created {len(self.potential_temporal_edges)} potential temporal edges")


def min_max_norm(tensor:torch.Tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    normalized_0_to_1 = (tensor - min_val) / (max_val - min_val)
    normalized_minus1_to_1 = normalized_0_to_1 * 2 - 1
    return normalized_minus1_to_1
    