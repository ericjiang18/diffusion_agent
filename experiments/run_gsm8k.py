import sys
import os
import argparse
import yaml
import json
import time
import asyncio
from pathlib import Path
import torch
import copy
from typing import List,Union,Literal
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.stdout.reconfigure(encoding='utf-8')

from GDesigner.utils.const import GDesigner_ROOT
from GDesigner.graph.graph import Graph
from GDesigner.tools.reader.readers import JSONLReader
import GDesigner.agents
from GDesigner.utils.globals import Time
from GDesigner.utils.globals import Cost, PromptTokens, CompletionTokens
from datasets.gsm8k_dataset import gsm_data_process,gsm_get_predict

from GDesigner.gdt.gtd_framework import GTDFramework
from GDesigner.gdt.proxy_reward_model import ProxyRewardModel
from GDesigner.llm.profile_embedding import get_sentence_embedding
from GDesigner.prompt.prompt_set_registry import PromptSetRegistry
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.utils import dense_to_sparse

def load_result(result_file):
    if not result_file.exists():
        with open(result_file, 'w',encoding='utf-8') as file:
            json.dump([], file)

    with open(result_file, 'r',encoding='utf-8') as file:
        data = json.load(file)
    return data

def dataloader(data_list, batch_size, i_batch):
    return data_list[i_batch*batch_size:i_batch*batch_size + batch_size]

def load_config(config_path):
    with open(config_path, 'r',encoding='utf-8') as file:
        return yaml.safe_load(file)
    
def parse_args():
    parser = argparse.ArgumentParser(description="GDesigner Experiments on gsm8k")
    parser.add_argument("--dataset_json", type=str, default="datasets/gsm8k/gsm8k.jsonl")
    parser.add_argument("--result_file", type=str, default=None)
    parser.add_argument("--llm_name", type=str, default="gpt-4o")
    parser.add_argument('--mode', type=str, default='FullConnected',
                        choices=['DirectAnswer', 'FullConnected', 'Random', 'Chain','Debate','Layered','Star', 'GTD'],
                        help="Mode of operation. Default is 'FullConnected'.")
    parser.add_argument('--lr', type=float, default=0.1,help="learning rate")
    parser.add_argument('--batch_size', type=int, default=4,help="batch size")
    parser.add_argument('--num_rounds',type=int,default=1,help="Number of optimization/inference rounds for one query")
    parser.add_argument('--pruning_rate', type=float, default=0.25,help="The Rate of Pruning. Default 0.05.")
    parser.add_argument('--num_iterations', type=int, default=10,help="The num of training iterations.")
    parser.add_argument('--domain', type=str, default="gsm8k",help="Domain (the same as dataset name), default 'gsm8k'")
    parser.add_argument('--agent_names', nargs='+', type=str, default=['MathSolver'],
                        help='Specify agent names as a list of strings')
    parser.add_argument('--agent_nums', nargs='+', type=int, default=[4],
                        help='Specify the number of agents for each name in agent_names')
    parser.add_argument('--decision_method', type=str, default='FinalRefer',
                        help='The decison method of the GDesigner')
    parser.add_argument('--optimized_spatial',action='store_true')
    parser.add_argument('--optimized_temporal',action='store_true')

    # GTD Specific Arguments
    gtd_group = parser.add_argument_group('GTD Mode Options')
    gtd_group.add_argument('--gtd_node_feat_dim', type=int, default=384, help='Node feature dimension for GTD')
    gtd_group.add_argument('--gtd_cond_dim', type=int, default=128, help='Condition vector dimension for GTD')
    gtd_group.add_argument('--gtd_task_cond_input_dim', type=int, default=384, help='Raw input dimension of the task condition embedding')
    gtd_group.add_argument('--gtd_time_emb_dim', type=int, default=128, help='Time embedding dimension for GTD')
    gtd_group.add_argument('--gtd_layers', type=int, default=2, help='Number of layers in the Graph Transformer')
    gtd_group.add_argument('--gtd_heads', type=int, default=2, help='Number of attention heads in the Graph Transformer')
    gtd_group.add_argument('--gtd_diffusion_steps', type=int, default=50, help='Number of timesteps for the diffusion model')
    gtd_group.add_argument('--gtd_candidates', type=int, default=5, help='Number of candidates for Zeroth-Order optimization')
    gtd_group.add_argument('--gtd-generate-data', action='store_true', help='[Phase 1] Generate initial dataset for GTD training.')
    gtd_group.add_argument('--gtd-train-models', action='store_true', help='[Phase 2] Train Proxy and Diffusion models from a dataset.')
    gtd_group.add_argument('--gtd-datagen-limit', type=int, default=50, help='Number of records to use for dataset generation.')
    gtd_group.add_argument('--gtd-dataset-path', type=str, default='gtd_initial_dataset.jsonl', help='Path to the GTD training dataset.')
    gtd_group.add_argument('--gtd-proxy-model-path', type=str, default='proxy_model.pth', help='Path to save/load the proxy model.')
    gtd_group.add_argument('--gtd-diffusion-model-path', type=str, default='diffusion_model.pth', help='Path to save/load the diffusion model.')
    gtd_group.add_argument('--gtd-epochs', type=int, default=10, help='Number of training epochs for Phase 2.')

    # Add missing GNN/MLP hyperparameter arguments
    parser.add_argument('--gnn_hidden_dim', type=int, default=32)
    parser.add_argument('--gnn_layers', type=int, default=2)
    parser.add_argument('--mlp_hidden_dim', type=int, default=64)
    
    args = parser.parse_args()
    result_path = GDesigner_ROOT / "result"
    os.makedirs(result_path, exist_ok=True)
    if len(args.agent_names) != len(args.agent_nums):
        parser.error("The number of agent names must match the number of agent counts.")

    return args

async def generate_initial_dataset(args, dataset):
    """PHASE 1: Generate initial dataset by running baseline topologies."""
    print("--- Starting Phase 1: Initial Dataset Generation ---")
    agent_names_list = [name for name, num in zip(args.agent_names, args.agent_nums) for _ in range(num)]
    num_nodes = len(agent_names_list)
    prompt_set = PromptSetRegistry.get(args.domain)
    agent_profiles = [prompt_set.get_description(name) for name in agent_names_list]
    node_features_base = [get_sentence_embedding(p) for p in agent_profiles]

    def generate_static_topologies(n):
        topologies = {
            'fully_connected': [[1 if i != j else 0 for j in range(n)] for i in range(n)],
            'chain': [[1 if j == i + 1 else 0 for j in range(n)] for i in range(n)],
            'star': [[1 if i == 0 and j > 0 else 0 for j in range(n)] for i in range(n)],
        }
        # Add a few random graphs for diversity
        for i in range(3):
            topologies[f'random_{i}'] = [[random.randint(0, 1) if i != j else 0 for j in range(n)] for i in range(n)]
        return topologies

    static_topologies = generate_static_topologies(num_nodes)
    generated_data = []
    proxy_data_list = []

    for i, record in enumerate(dataset):
        if i >= args.gtd_datagen_limit: # Limit dataset size for initial run
            print(f"Reached data generation limit ({args.gtd_datagen_limit} records).")
            break
        
        task_query = record["task"]
        true_answer = record["answer"]
        task_condition_embedding = get_sentence_embedding(task_query)
        print(f"\nProcessing record {i}: {task_query[:60]}...")

        for name, topology_matrix in static_topologies.items():
            print(f"  Testing topology: {name}")
            gdesigner_graph = Graph(
                domain=args.domain, llm_name=args.llm_name, agent_names=agent_names_list,
                decision_method=args.decision_method, fixed_spatial_masks=topology_matrix
            )
            raw_answer, _ = await gdesigner_graph.arun({"task": task_query}, args.num_rounds)
            
            predict_answer = gsm_get_predict(raw_answer[0])
            utility = 1.0 if predict_answer and float(predict_answer) == float(true_answer) else 0.0
            cost = sum(sum(row) for row in topology_matrix)
            
            adj_matrix = torch.tensor(topology_matrix, dtype=torch.float)
            edge_index, _ = dense_to_sparse(adj_matrix)
            proxy_data_list.append(Data(
                x=torch.tensor(node_features_base, dtype=torch.float),
                edge_index=edge_index,
                condition=torch.tensor(task_condition_embedding, dtype=torch.float).unsqueeze(0),
                true_rewards=torch.tensor([utility, cost], dtype=torch.float).unsqueeze(0)
            ))

            data_point = {
                'graph': topology_matrix,
                'condition': task_condition_embedding.tolist(),
                'node_features': [feat.tolist() for feat in node_features_base],
                'performance': {'utility': utility, 'cost': cost}
            }
            generated_data.append(data_point)

    with open(args.gtd_dataset_path, 'w') as f:
        for item in generated_data:
            f.write(json.dumps(item) + '\n')
    print(f"\n--- Phase 1 Finished: Saved {len(generated_data)} data points to {args.gtd_dataset_path} ---")

async def train_gtd_models(args):
    """PHASE 2: Train the Proxy and Diffusion models from the generated dataset."""
    print(f"--- Starting Phase 2: Training GTD Models from {args.gtd_dataset_path} ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load the dataset
    proxy_data_list = []
    diffusion_A0_list, diffusion_nodes_list, diffusion_cond_list = [], [], []
    with open(args.gtd_dataset_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            # For Proxy Model
            adj_matrix = torch.tensor(item['graph'], dtype=torch.float)
            edge_index, _ = dense_to_sparse(adj_matrix)
            rewards = item['performance']
            proxy_data_item = Data(
                x=torch.tensor(item['node_features'], dtype=torch.float),
                edge_index=edge_index,
                condition=torch.tensor(item['condition'], dtype=torch.float).unsqueeze(0),
                true_rewards=torch.tensor([rewards['utility'], rewards['cost']], dtype=torch.float).unsqueeze(0)
            )
            proxy_data_list.append(proxy_data_item)
            
            # For Diffusion Model (only use high-quality graphs)
            if rewards['utility'] > 0.5:
                diffusion_A0_list.append(item['graph'])
                diffusion_nodes_list.append(item['node_features'])
                diffusion_cond_list.append(item['condition'])

    print(f"Loaded {len(proxy_data_list)} samples for Proxy Model training.")
    print(f"Loaded {len(diffusion_A0_list)} high-quality samples for Diffusion Model training.")

    if not diffusion_A0_list:
        print("No high-quality graphs found to train the diffusion model. Aborting.")
        return

    # 2. Train Proxy Model
    proxy_model = ProxyRewardModel(
        task_cond_input_dim=args.gtd_task_cond_input_dim,
        node_feature_dim=args.gtd_node_feat_dim,
        condition_dim=args.gtd_cond_dim,
        gnn_hidden_dim=args.gnn_hidden_dim,
        gnn_layers=args.gnn_layers,
        mlp_hidden_dim=args.mlp_hidden_dim,
        num_reward_components=2
    ).to(device)
    pyg_dataloader = PyGDataLoader(proxy_data_list, batch_size=16, shuffle=True)
    optimizer = torch.optim.Adam(proxy_model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    print("\n--- Training Proxy Reward Model ---")
    proxy_model.train()
    for epoch in range(args.gtd_epochs):
        total_loss = 0
        for batch in pyg_dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred_rewards = proxy_model(batch)
            loss = criterion(pred_rewards, batch.true_rewards)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{args.gtd_epochs}, Proxy Model Loss: {total_loss / len(pyg_dataloader):.4f}")
    
    torch.save(proxy_model.state_dict(), args.gtd_proxy_model_path)
    print(f"--- Saved trained Proxy Model to {args.gtd_proxy_model_path} ---")

    # 3. Train Diffusion Model
    diffusion_model = GTDFramework(
        task_cond_input_dim=args.gtd_task_cond_input_dim, node_feature_dim=args.gtd_node_feat_dim,
        condition_dim=args.gtd_cond_dim, time_embed_dim=args.gtd_time_emb_dim, gt_num_layers=args.gtd_layers,
        gt_num_heads=args.gtd_heads, diffusion_num_timesteps=args.gtd_diffusion_steps, device=device
    )
    
    diffusion_dataset = TensorDataset(
        torch.tensor(diffusion_A0_list, dtype=torch.float),
        torch.tensor(diffusion_nodes_list, dtype=torch.float),
        torch.tensor(diffusion_cond_list, dtype=torch.float)
    )
    diffusion_dataloader = DataLoader(diffusion_dataset, batch_size=16, shuffle=True)
    
    print("\n--- Training Diffusion Model ---")
    diffusion_model.train_diffusion_model(dataloader=diffusion_dataloader, epochs=args.gtd_epochs, learning_rate=1e-4)
    torch.save(diffusion_model.diffusion_model.state_dict(), args.gtd_diffusion_model_path)
    print(f"--- Saved trained Diffusion Model to {args.gtd_diffusion_model_path} ---")


async def run_gtd_experiment(args, dataset):
    """PHASE 3: Main logic for running experiments with a pre-trained GTD Framework."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Phase 3: GTD Inference on device: {device} ---")

    # 1. Initialize GTD Framework and load trained models
    agent_names_list = [name for name, num in zip(args.agent_names, args.agent_nums) for _ in range(num)]
    num_nodes = len(agent_names_list)

    # Load Proxy Model
    proxy_model = ProxyRewardModel(
        task_cond_input_dim=args.gtd_task_cond_input_dim, node_feature_dim=args.gtd_node_feat_dim,
        condition_dim=args.gtd_cond_dim, gnn_hidden_dim=args.gnn_hidden_dim, gnn_layers=args.gnn_layers, mlp_hidden_dim=args.mlp_hidden_dim, num_reward_components=2
    )
    proxy_model.load_state_dict(torch.load(args.gtd_proxy_model_path))
    proxy_model.to(device)
    proxy_model.eval()
    
    # Add reward component names to the loaded proxy model instance
    proxy_model.reward_component_names = ['utility', 'cost']
    macp_weights = {'utility': 1.0, 'cost': -0.1}

    # Initialize GTDFramework and load Diffusion model
    gtd_framework = GTDFramework(
        task_cond_input_dim=args.gtd_task_cond_input_dim, node_feature_dim=args.gtd_node_feat_dim,
        condition_dim=args.gtd_cond_dim, time_embed_dim=args.gtd_time_emb_dim, gt_num_layers=args.gtd_layers,
        gt_num_heads=args.gtd_heads, diffusion_num_timesteps=args.gtd_diffusion_steps,
        proxy_reward_model=proxy_model, macp_weights=macp_weights,
        num_candidates_per_step=args.gtd_candidates, device=device
    )
    gtd_framework.diffusion_model.load_state_dict(torch.load(args.gtd_diffusion_model_path))
    gtd_framework.diffusion_model.to(device)
    gtd_framework.diffusion_model.eval()

    print("--- Loaded Pre-trained Proxy and Diffusion models ---")

    # 2. Setup for experiment run
    prompt_set = PromptSetRegistry.get(args.domain)
    agent_profiles = [prompt_set.get_description(name) for name in agent_names_list]
    node_features_base = torch.tensor([get_sentence_embedding(p) for p in agent_profiles]).float().to(device)

    # Result file setup
    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time
    result_dir = Path(f"{GDesigner_ROOT}/result/gtd_{args.domain}")
    result_dir.mkdir(parents=True, exist_ok=True)
    result_file = result_dir / f"{args.llm_name}_{current_time}.json"

    # 3. Batch Processing Loop
    num_batches = int(len(dataset) / args.batch_size)
    total_solved, total_executed = (0, 0)

    for i_batch in range(num_batches):
        print(f"GTD Batch {i_batch}", 80*'-')
        start_ts = time.time()
        current_batch = dataloader(dataset, args.batch_size, i_batch)
        if not current_batch:
            break
        
        for record in current_batch:
            task_query = record["task"]
            true_answer = record["answer"]

            # Generate task-specific condition
            task_condition_embedding = torch.tensor(get_sentence_embedding(task_query)).float().unsqueeze(0).to(device)

            # Generate topology for this specific task
            generated_A0_probs = gtd_framework.generate_graphs(
                num_graphs=1,
                num_nodes=num_nodes,
                node_features=node_features_base.unsqueeze(0),
                task_condition=task_condition_embedding,
                use_guidance=True
            )
            # Binarize the generated graph
            generated_adj_matrix = (generated_A0_probs.squeeze(0) > 0.5).int()

            # Create a GDesigner Graph with the generated topology
            gdesigner_graph = Graph(
                domain=args.domain,
                llm_name=args.llm_name,
                agent_names=agent_names_list,
                decision_method=args.decision_method,
                fixed_spatial_masks=generated_adj_matrix.tolist()
            )

            # Run the multi-agent simulation
            input_dict = {"task": task_query}
            raw_answer, _ = await gdesigner_graph.arun(input_dict, args.num_rounds)
            
            # Process and save results
            predict_answer = gsm_get_predict(raw_answer[0])
            is_solved = float(predict_answer) == float(true_answer) if predict_answer else False
            total_solved += is_solved
            total_executed += 1
            accuracy = total_solved / total_executed if total_executed > 0 else 0
            
            print(f"Query: {task_query[:50]}... | Solved: {is_solved} | Accuracy: {accuracy:.3f}")

            data = load_result(result_file)
            updated_item = {
                "Question": task_query, "Answer": true_answer, "Response": raw_answer,
                "Attempt answer": predict_answer, "Solved": is_solved,
                "Generated_Topology": generated_adj_matrix.tolist(),
                "Total solved": total_solved, "Total executed": total_executed, "Accuracy": accuracy
            }
            data.append(updated_item)
            with open(result_file, 'w', encoding='utf-8') as file:
                json.dump(data, file, indent=4)
        
        print(f"Batch time {time.time() - start_ts:.3f}")
        print(f"Cost {Cost.instance().value}")

async def main():
    args = parse_args()
    
    dataset = JSONLReader.parse_file(args.dataset_json)
    dataset = gsm_data_process(dataset)
    
    if args.gtd_generate_data:
        await generate_initial_dataset(args, dataset)
    elif args.gtd_train_models:
        await train_gtd_models(args)
    elif args.mode == 'GTD':
        await run_gtd_experiment(args, dataset)
    else:
        # Fallback to original logic for other modes
        result_file = None
        current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        Time.instance().value = current_time
        result_dir = Path(f"{GDesigner_ROOT}/result/gsm8k")
        result_dir.mkdir(parents=True, exist_ok=True)
        result_file = result_dir / f"{args.domain}_{args.llm_name}_{current_time}.json"
        
        agent_names = [name for name,num in zip(args.agent_names,args.agent_nums) for _ in range(num)]
        decision_method = args.decision_method
        kwargs = get_kwargs(args.mode,len(agent_names))
        graph = Graph(domain="gsm8k",
                    llm_name=args.llm_name,
                    agent_names=agent_names,
                    decision_method=decision_method,
                    optimized_spatial=args.optimized_spatial,
                    optimized_temporal=args.optimized_temporal,
                    **kwargs)
        
        optimizer = None
        if args.mode != 'GTD' and graph.gcn is not None:
            graph.gcn.train()
            optimizer = torch.optim.Adam(graph.gcn.parameters(), lr=args.lr)   
        
        num_batches = int(len(dataset)/args.batch_size)
        total_solved, total_executed = (0, 0)
        
        for i_batch in range(num_batches):
            print(f"Batch {i_batch}",80*'-')
            start_ts = time.time()
            answer_log_probs = []
            answers = []
            
            current_batch = dataloader(dataset,args.batch_size,i_batch)
            if current_batch is None:
                print("No more data available.")
                break
            
            for i_record, record in enumerate(current_batch):
                realized_graph = copy.deepcopy(graph)
                if args.mode != 'GTD':
                    realized_graph.gcn = graph.gcn
                    realized_graph.mlp = graph.mlp
                task = record["task"]
                step = record["step"]
                answer = record["answer"]
                answers.append(answer)
                input_dict = {"task": task}
                answer_log_probs.append(asyncio.create_task(realized_graph.arun(input_dict,args.num_rounds)))
            raw_results = await asyncio.gather(*answer_log_probs)
            raw_answers, log_probs = zip(*raw_results)
            loss_list: List[torch.Tensor] = []
            utilities: List[float] = []
            data = load_result(result_file)
            
            for task_record, answer, log_prob, true_answer in zip(current_batch, raw_answers, log_probs, answers):
                predict_answer = gsm_get_predict(answer[0])
                is_solved = float(predict_answer)==float(true_answer) if predict_answer else False
                total_solved = total_solved + is_solved
                total_executed = total_executed + 1
                accuracy = total_solved/ total_executed
                utility = is_solved
                utilities.append(utility)
                if log_prob is not None:
                    single_loss = -log_prob * utility
                    loss_list.append(single_loss)
                updated_item = {
                    "Question": task_record["task"],
                    "Answer": true_answer,
                    "Step": task_record["step"],
                    "Response": answer,
                    "Attempt answer": predict_answer,
                    "Solved": is_solved,
                    "Total solved": total_solved,
                    "Total executed": total_executed,
                    "Accuracy": accuracy
                }
                data.append(updated_item)
            with open(result_file, 'w',encoding='utf-8') as file:
                json.dump(data, file, indent=4)
            
            if loss_list and (args.optimized_spatial or args.optimized_temporal):
                total_loss = torch.mean(torch.stack(loss_list))
                if optimizer:
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                
                print(f"Batch time {time.time() - start_ts:.3f}")
                print(f"Accuracy: {accuracy}")
                print("utilities:", utilities)
                print("loss:", total_loss.item())
            else:
                print(f"Batch time {time.time() - start_ts:.3f}")
                print(f"Accuracy: {accuracy}")
                print("utilities:", utilities)

            if i_batch+1 == args.num_iterations:
                args.optimized_spatial = False
                args.optimized_temporal = False
                total_solved = 0
                total_executed = 0
                if graph.gcn:
                    graph.gcn.eval()
                print("Start Eval")
            
            print(f"Cost {Cost.instance().value}")
            print(f"PromptTokens {PromptTokens.instance().value}")
            print(f"CompletionTokens {CompletionTokens.instance().value}")


def get_kwargs(mode:Union[Literal['DirectAnswer'],Literal['FullConnected'],Literal['Random'],Literal['Chain'],Literal['Debate'],Literal['Layered'],Literal['Star'],Literal['GTD']]
               ,N:int):
    initial_spatial_probability: float = 0.5
    fixed_spatial_masks:List[List[int]] = None
    initial_temporal_probability: float = 0.5
    fixed_temporal_masks:List[List[int]] = None
    node_kwargs = None
    
    def generate_layered_graph(n,layer_num=2):
        adj_matrix = [[0 for _ in range(n)] for _ in range(n)]
        base_size = n // layer_num
        remainder = n % layer_num
        layers = []
        for i in range(layer_num):
            size = base_size + (1 if i < remainder else 0)
            layers.extend([i] * size)
        random.shuffle(layers)
        for i in range(n):
            current_layer = layers[i]
            for j in range(n):
                if layers[j] == current_layer + 1:
                    adj_matrix[i][j] = 1
        return adj_matrix
    
    def generate_star_graph(n):
        matrix = [[0] * n for _ in range(n)]
        for i in range(0, n):
            for j in range(i+1,n):
                matrix[i][j] = 1
        return matrix
    
    if mode=='DirectAnswer':
        fixed_spatial_masks = [[0]]
        fixed_temporal_masks = [[0]]
        node_kwargs = [{'role':'Programming Expert'}]
    elif mode=='FullConnected':
        fixed_spatial_masks = [[1 if i!=j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode=='Random':
        fixed_spatial_masks = [[random.randint(0, 1)  if i!=j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[random.randint(0, 1) for _ in range(N)] for _ in range(N)]
    elif mode=='Chain':
        fixed_spatial_masks = [[1 if i==j+1 else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 if i==0 and j==N-1 else 0 for i in range(N)] for j in range(N)]
    elif mode == 'Debate':
        fixed_spatial_masks = [[0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == 'Layered':
        fixed_spatial_masks = generate_layered_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == 'Star':
        fixed_spatial_masks = generate_star_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == 'GTD':
        pass # GTD mode does not require predefined masks.
    
    return {"initial_spatial_probability": initial_spatial_probability,
            "fixed_spatial_masks": fixed_spatial_masks,
            "initial_temporal_probability": initial_temporal_probability,
            "fixed_temporal_masks": fixed_temporal_masks,
            "node_kwargs":node_kwargs}    

if __name__ == '__main__':
    asyncio.run(main())