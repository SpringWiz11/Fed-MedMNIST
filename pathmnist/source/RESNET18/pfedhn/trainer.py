# import argparse
# import json
# import logging
# import random
# from collections import defaultdict, OrderedDict
# from pathlib import Path

# import numpy as np
# import torch
# from tqdm import trange

# from models import CNNHyper, CNNTarget
# from Basenodes import BaseNodes
# from utils import get_device, set_logger, set_seed, str2bool
# from medmnist import INFO, Evaluator

# def eval_model(nodes, num_nodes, hnet, net, criteria, device, split):
#     curr_results = evaluat(nodes, num_nodes, hnet, net, criteria, device, split=split)
#     total_correct = sum([val['correct'] for val in curr_results.values()])
#     total_samples = sum([val['total'] for val in curr_results.values()])
#     avg_loss = np.mean([val['loss'] for val in curr_results.values()])
#     avg_acc = total_correct / total_samples

#     all_acc = [val['correct'] / val['total'] for val in curr_results.values()]

#     return curr_results, avg_loss, avg_acc, all_acc

# torch.no_grad()
# def evaluat(nodes: BaseNodes, num_nodes, hnet, net, criteria, device, split='test'):
#     hnet.eval()
#     results = defaultdict(lambda: defaultdict(list))

#     for node_id in range(num_nodes):  # iterating over nodes

#         running_loss, running_correct, running_samples = 0., 0., 0.
#         if split == 'test':
#             curr_data = nodes.test_loaders[node_id]
#         elif split == 'val':
#             curr_data = nodes.val_loaders[node_id]
#         else:
#             curr_data = nodes.train_loaders[node_id]
#         # y_true = torch.tensor([])
#         # y_score = torch.tensor([])

#         for batch_count, batch in enumerate(curr_data):
#             img, label = tuple(t.to(device) for t in batch)
#             weights = hnet(torch.tensor([node_id], dtype=torch.long).to(device))
#             load_weight_sd(net, weights)
#             pred = net(img)
#             label = label.squeeze()
#             # print("The shape of the pred = ",pred.shape)
#             # print("The shape of the label = ",label.shape)
#             # print("Size of the label = ", label.size())
#             if label.size() == torch.Size([]):
#                 # label = label.unsqueeze(0)
#                 continue
#             else:
#                 running_loss += criteria(pred, label).item()
#                 running_correct += pred.argmax(1).eq(label).sum().item()
#                 running_samples += len(label)
#         results[node_id]['loss'] = running_loss / (batch_count + 1)
#         results[node_id]['correct'] = running_correct
#         results[node_id]['total'] = running_samples

#     return results

# def load_weight_sd(net, weights):
#     for layer_name, layer_params in net.named_parameters():
#         if layer_name in weights:
#             layer_params.data.copy_(weights[layer_name][0])
#         else:
#             continue


# def train( 
#         data_name : str,
#         classes_per_node : int,
#         num_nodes : int,
#         steps : int,
#         inner_steps: int,
#         optim : str,
#         lr : float,
#         inner_lr : float,
#         embed_lr : float,
#         wd: float,
#         inner_wd: float,
#         embed_dim: int,
#         hyper_hid: int,
#         n_hidden:int,
#         n_kernels: int,
#         bs: int,
#         device,
#         eval_every: int,
#         seed:int
# ) -> None:
#     nodes = BaseNodes(data_name, num_nodes, batch_size=bs, classes_per_node=classes_per_node)

#     embed_dim = embed_dim

#     if embed_dim == -1:
#         logging.info("auto embedding size")
#         embed_dim = int(1+num_nodes/4)

#     hnet = CNNHyper(n_nodes=num_nodes, embedding_dim=embed_dim )
#     net = CNNTarget(n_kernels = n_kernels)
#     net = CNNTarget()

#     hnet = hnet.to(device)
#     net = net.to(device)
    
#     for n, p in net.named_parameters():
#         print(n)
       

#     embed_lr = embed_lr if embed_lr is not None else lr

#     optimizers = {
#         'sgd': torch.optim.SGD(
#             [
#                 {'params': [p for n, p in hnet.named_parameters() if 'embed' not in n]},
#                 {'params': [p for n, p in hnet.named_parameters() if 'embed' in n], 'lr': embed_lr}
#             ],
#             lr=lr,
#             momentum=0.9,
#             weight_decay=wd
#         ),
#         'adam' : torch.optim.Adam(
#             params = hnet.parameters(),
#             lr = lr
#         )
#     }

#     optimizer = optimizers[optim]
#     criteria = torch.nn.CrossEntropyLoss()

#     last_eval = -1
#     best_step = -1
#     best_acc = -1
#     test_best_based_on_step, test_best_min_based_on_step = -1, -1
#     test_best_max_based_on_step, test_best_std_based_on_step = -1, -1
#     step_iter = trange(steps)

#     results = defaultdict(list)

#     for step in step_iter:
#         hnet.train()

#         # select client at random``
#         node_id = random.choice(range(num_nodes))
#         # print(node_id)

#         # produce & load local network weights
#         weights = hnet(torch.tensor([node_id], dtype=torch.long).to(device))
#         # load_weight_sd(net, weights)
#         net.load_state_dict(weights)
   

#         inner_optim = torch.optim.SGD(
#             net.parameters(), lr = inner_lr, momentum = .9, weight_decay = inner_wd
#         )

#         inner_state = OrderedDict({k: tensor.data for k, tensor in weights.items()})

#         with torch.no_grad():
#             net.eval()
#             batch = next(iter(nodes.test_loaders[node_id]))
#             img, label = tuple(t.to(device) for t in batch)
#             pred = net(img)
#             label = label.squeeze()

#             prvs_loss = criteria(pred, label)
#             prvs_acc = pred.argmax(1).eq(label).sum().item() / len(label)
#             net.train()


        
#         for i in range(inner_steps):
#             net.train()
#             inner_optim.zero_grad()
#             optimizer.zero_grad()

#             batch = next(iter(nodes.train_loaders[node_id]))
#             img, label = tuple(t.to(device) for t in batch)

#             pred = net(img)
#             label = label.squeeze()

#             loss = criteria(pred, label)
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(net.parameters(), 50)
#             inner_optim.step()

#         optimizer.zero_grad()
        # filtered_state_dict = {key:value for key, value in net.state_dict().items() if key in weights}
#         # print("Printing the inner_state[k]")

#         # for k in inner_state.keys():
#         #     print(k, " = ", inner_state[k].shape)

        
#         # print("Printing the filtered_state_dict[k]")

#         # for k in inner_state.keys():
#         #     print(k, " = ", filtered_state_dict[k].shape)





#         # for k in weights.keys():
#         #     print("The size of the tensor: ", filtered_state_dict[k] )
#         #     break
#         delta_theta = OrderedDict(
#             {
#                 k: inner_state[k] - filtered_state_dict[k] for k in weights.keys()
#             }
#         )
        

#         hnet_grad = torch.autograd.grad(
#             list(weights.values()), hnet.parameters(), grad_outputs=list(delta_theta.values())
#         )

#         for p, g in zip(hnet.parameters(), hnet_grad):
#             p.grad = g

#         torch.nn.utils.clip_grad_norm_(hnet.parameters(), 50)
#         optimizer.step()

#         step_iter.set_description(
#             f"Step: {step+1}, Node ID: {node_id}, Loss: {prvs_loss:.4f},  Acc: {prvs_acc:.4f}"
#         )

#         if step % eval_every == 0:
#             last_eval = step
#             step_results, avg_loss, avg_acc, all_acc = eval_model(nodes, num_nodes, hnet, net, criteria, device, split="test")

#             logging.info(f"\nStep: {step+1}, AVG Loss: {avg_loss:.4f},  AVG Acc: {avg_acc:.4f}")

#             results['test_avg_loss'].append(avg_loss)
#             results['test_avg_acc'].append(avg_acc)

#             _, val_avg_loss, val_avg_acc, _ = eval_model(nodes, num_nodes, hnet, net, criteria, device, split="val")
#             if best_acc < val_avg_acc:
#                 best_acc = val_avg_acc
#                 best_step = step
#                 test_best_based_on_step = avg_acc
#                 test_best_min_based_on_step = np.min(all_acc)
#                 test_best_max_based_on_step = np.max(all_acc)
#                 test_best_std_based_on_step = np.std(all_acc)

#             results['val_avg_loss'].append(val_avg_loss)
#             results['val_avg_acc'].append(val_avg_acc)
#             results['best_step'].append(best_step)
#             results['best_val_acc'].append(best_acc)
#             results['best_test_acc_based_on_val_beststep'].append(test_best_based_on_step)
#             results['test_best_min_based_on_step'].append(test_best_min_based_on_step)
#             results['test_best_max_based_on_step'].append(test_best_max_based_on_step)
#             results['test_best_std_based_on_step'].append(test_best_std_based_on_step)

        
        
        
        

       
        



# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(
#         description="Federated Hypernetworks meets MEDMNIST"
#     )

#     parser.add_argument(
#         "--data_name", type=str, default="pathmnist", choices=["pathmnist"], help = "As of now we are only supporting pathMNIST"
#     )
#     parser.add_argument(
#         "--num_nodes", type=int, default=20, help="Number of simulated nodes"
#     )

#     parser.add_argument(
#         "--num_steps", type=int, default = 1000
#     )
#     parser.add_argument(
#         "--optim", type = str, default = "sgd", choices = ["sgd", "adam"], help="learning rate"
#     )
#     parser.add_argument(
#         "--batch-size", type=int, default=64, help="batch size"
#     )
#     parser.add_argument(
#         "--inner-steps", type = int, default = 10, help="number of inner steps"
#     )

#     parser.add_argument(
#         "--n-hidden", type=int, default=3, help="number of hidden layers"
#     )

#     parser.add_argument(
#         "--inner_lr", type=float, default=0.001, help="Learning rate for the inner optimiser"
#     )   

#     parser.add_argument(
#         "--lr", type=float, default=1e-2, help="Learning rate for the outer optimiser"
#     )

#     parser.add_argument(
#         "--wd", type=float, default=1e-3, help="weight decay"
#     )
#     parser.add_argument(
#         "--inner-wd", type=float, default=5e-4, help="inner weight decay"
#     )
#     parser.add_argument(
#         "--embed-dim", type=int, default=-1, help="embedding dim"
#     )
#     parser.add_argument(
#         "--embed-lr", type=float, default=None, help="embedding learning rate"
#     )
#     parser.add_argument(
#         "--hyper-hid", type=int, default=100, help="hypernet hidden dim"
#     )
#     parser.add_argument(
#         "--spec-norm", type=str2bool, default=False, help="hypernet hidden dim"
#     )
#     parser.add_argument(
#         "--nkernels", type=int, default=64, help="number of kernels for cnn model"
#     )
#     parser.add_argument(
#         "--gpu", type=int, default=0, help="gpu device ID"
#     )
#     parser.add_argument(
#         "--eval-every", type=int, default=5, help="eval every X selected epochs"
#     )
#     parser.add_argument(
#         "--save-path", type=str, default="pfedhn_hetro_res", help="dir path for output file"
#     )
#     parser.add_argument(
#         "--seed", type=int, default=42, help="seed value"
#     )

#     args = parser.parse_args()
#     assert args.gpu <= torch.cuda.device_count(), f"--gpu flag should be in range [0,{torch.cuda.device_count() - 1}]"

#     set_logger()
#     set_seed(args.seed)

#     device = get_device(gpus = args.gpu)

#     args.data_name == "pathmnist"

#     args.classes_per_node = 2


#     train(
#         data_name = args.data_name,
#         classes_per_node = args.classes_per_node,
#         num_nodes = args.num_nodes,
#         steps = args.num_steps,
#         inner_steps = args.inner_steps,
#         optim=args.optim,
#         lr=args.lr,
#         inner_lr=args.inner_lr,
#         embed_lr=args.embed_lr,
#         wd=args.wd,
#         inner_wd=args.inner_wd,
#         embed_dim=args.embed_dim,
#         hyper_hid=args.hyper_hid,
#         n_hidden=args.n_hidden,
#         n_kernels=args.nkernels,
#         bs=args.batch_size,
#         device=device,
#         eval_every=args.eval_every,
#         seed = args.seed
#     )
        


import argparse
import json
import logging
import random
from collections import defaultdict, OrderedDict
from pathlib import Path

import numpy as np
import torch
from tqdm import trange

from models import CNNHyper, CNNTarget
from Basenodes import BaseNodes
from utils import get_device, set_logger, set_seed, str2bool
from medmnist import INFO, Evaluator



def eval_model(nodes, num_nodes, hnet, net, criteria, device, split):
    curr_results = evaluat(nodes, num_nodes, hnet, net, criteria, device, split=split)
    total_correct = sum([val['correct'] for val in curr_results.values()])
    total_samples = sum([val['total'] for val in curr_results.values()])
    avg_loss = np.mean([val['loss'] for val in curr_results.values()])
    avg_acc = total_correct / total_samples

    all_acc = [val['correct'] / val['total'] for val in curr_results.values()]

    return curr_results, avg_loss, avg_acc, all_acc

torch.no_grad()
def evaluat(nodes: BaseNodes, num_nodes, hnet, net, criteria, device, split='test'):
    hnet.eval()
    results = defaultdict(lambda: defaultdict(list))

    for node_id in range(num_nodes):  # iterating over nodes

        running_loss, running_correct, running_samples = 0., 0., 0.
        if split == 'test':
            curr_data = nodes.test_loaders[node_id]
        elif split == 'val':
            curr_data = nodes.val_loaders[node_id]
        else:
            curr_data = nodes.train_loaders[node_id]
        # y_true = torch.tensor([])
        # y_score = torch.tensor([])

        for batch_count, batch in enumerate(curr_data):
            img, label = tuple(t.to(device) for t in batch)
            weights = hnet(torch.tensor([node_id], dtype=torch.long).to(device))
            load_weight_sd(net, weights)
            pred = net(img)
            label = label.squeeze()
            # print("The shape of the pred = ",pred.shape)
            # print("The shape of the label = ",label.shape)
            # print("Size of the label = ", label.size())
            if label.size() == torch.Size([]):
                # label = label.unsqueeze(0)
                continue
            else:
                running_loss += criteria(pred, label).item()
                running_correct += pred.argmax(1).eq(label).sum().item()
                running_samples += len(label)
        results[node_id]['loss'] = running_loss / (batch_count + 1)
        results[node_id]['correct'] = running_correct
        results[node_id]['total'] = running_samples

    return results

def load_weight_sd(net, weights):
    for layer_name, layer_params in net.named_parameters():
        if layer_name in weights:
            layer_params.data.copy_(weights[layer_name][0])
        else:
            continue




def train( 
        data_name : str,
        classes_per_node : int,
        num_nodes : int,
        steps : int,
        inner_steps: int,
        optim : str,
        lr : float,
        inner_lr : float,
        embed_lr : float,
        wd: float,
        inner_wd: float,
        embed_dim: int,
        hyper_hid: int,
        n_hidden:int,
        n_kernels: int,
        bs: int,
        device,
        eval_every: int,
        seed:int
) -> None:
    nodes = BaseNodes(data_name, num_nodes, batch_size=bs, classes_per_node=classes_per_node)

    embed_dim = embed_dim

    if embed_dim == -1:
        logging.info("auto embedding size")
        embed_dim = int(1+num_nodes/4)
    hnet = CNNHyper(n_nodes=num_nodes, embedding_dim=embed_dim, hidden_dim=hyper_hid, n_hidden=n_hidden, n_kernels=n_kernels)
    net = CNNTarget(n_kernels = n_kernels)

    hnet = hnet.to(device)
    net = net.to(device)
    

    embed_lr = embed_lr if embed_lr is not None else lr

    optimizers = {
        'sgd': torch.optim.SGD(
            [
                {'params': [p for n, p in hnet.named_parameters() if 'embed' not in n]},
                {'params': [p for n, p in hnet.named_parameters() if 'embed' in n], 'lr': embed_lr}
            ],
            lr=lr,
            momentum=0.9,
            weight_decay=wd
        ),
        'adam' : torch.optim.Adam(
            params = hnet.parameters(),
            lr = lr
        )
    }

    optimizer = optimizers[optim]
    criteria = torch.nn.CrossEntropyLoss()

    last_eval = -1
    best_step = -1
    best_acc = -1
    test_best_based_on_step, test_best_min_based_on_step = -1, -1
    test_best_max_based_on_step, test_best_std_based_on_step = -1, -1
    step_iter = trange(steps)

    results = defaultdict(list)

    for step in step_iter:
        hnet.train()

        # select client at random``
        node_id = random.choice(range(num_nodes))
        # print(node_id)

        # produce & load local network weights
        weights = hnet(torch.tensor([node_id], dtype=torch.long).to(device))
        load_weight_sd(net, weights)

         # init inner optimizer
        inner_optim = torch.optim.SGD(
            net.parameters(), lr=inner_lr, momentum=.9, weight_decay=inner_wd
         )

         # storing theta_i for later calculating delta theta
        inner_state = OrderedDict({k: tensor.data for k, tensor in weights.items()})

        with torch.no_grad():
            net.eval()
            batch = next(iter(nodes.test_loaders[node_id]))
            img, label = tuple(t.to(device) for t in batch)
            label = label.squeeze()
            # print(img.shape)
            # print(label.shape)
            pred = net(img)
            prvs_loss = criteria(pred, label)
            prv_acc = pred.argmax(1).eq(label).sum().item() / len(label)
            net.train()

        for i in range(inner_steps):
            net.train()
            inner_optim.zero_grad()
            optimizer.zero_grad()

            batch = next(iter(nodes.train_loaders[node_id]))
            img, label = tuple(t.to(device) for t in batch)
            label = label.squeeze()
            pred = net(img)
            loss = criteria(pred, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
            inner_optim.step()
        
        optimizer.zero_grad()
        filtered_state_dict = {key:value for key, value in net.state_dict().items() if key in weights}

        # final_state = net.state_dict()

        # print(weights.keys())

        # print("""
        #       ############# Printing Inner State ##############
        #     """)
        # print(inner_state.keys())

        # print("""
        #       ############# Printing final State ##############
        #     """)
        # print(final_state.keys())

        # calculate delta theta
        delta_theta = OrderedDict({k: inner_state[k] - filtered_state_dict[k] for k in weights.keys()})

        # calculate phi gradient

        hnet_grads = torch.autograd.grad(
            list(weights.values()),hnet.parameters(),grad_outputs=list(delta_theta.values()),
        )
        # update hnet weights

        for p, g in zip(hnet.parameters(), hnet_grads):
            p.grad = g
        
        torch.nn.utils.clip_grad_norm_(hnet.parameters(), 1)
        optimizer.step()

        step_iter.set_description(
            f"Step: {step + 1}, Node ID: {node_id}, Loss:{prvs_loss:.4f}, Acc: {prv_acc:.4f}"
        )

        if step % eval_every == 0:
            last_eval = step
            step_results, avg_loss, avg_acc, all_acc = eval_model(nodes, num_nodes, hnet, net, criteria, device, split="test")
            logging.info(f"\nStep: {step+1}, AVG Loss: {avg_loss:.4f},  AVG Acc: {avg_acc:.4f}")
            # evaluate

            results['test_avg_loss'].append(avg_loss)
            results['test_avg_acc'].append(avg_acc)

            _, val_avg_loss, val_avg_acc, _ = eval_model(nodes, num_nodes, hnet, net, criteria, device, split="val")
            if best_acc < val_avg_acc:
                best_acc = val_avg_acc
                best_step = step
                test_best_based_on_step = avg_acc
                test_best_min_based_on_step = np.min(all_acc)
                test_best_max_based_on_step = np.max(all_acc)
                test_best_std_based_on_step = np.std(all_acc)

            results['val_avg_loss'].append(val_avg_loss)
            results['val_avg_acc'].append(val_avg_acc)
            results['best_step'].append(best_step)
            results['best_val_acc'].append(best_acc)
            results['best_test_acc_based_on_val_beststep'].append(test_best_based_on_step)
            results['test_best_min_based_on_step'].append(test_best_min_based_on_step)
            results['test_best_max_based_on_step'].append(test_best_max_based_on_step)
            results['test_best_std_based_on_step'].append(test_best_std_based_on_step)
        
    if step != last_eval:
        _, val_avg_loss, val_avg_acc, _ = eval_model(nodes, num_nodes, hnet, net, criteria, device, split="val")
        step_results, avg_loss, avg_acc, all_acc = eval_model(nodes, num_nodes, hnet, net, criteria, device, split="test")
        logging.info(f"\nStep: {step + 1}, AVG Loss: {avg_loss:.4f},  AVG Acc: {avg_acc:.4f}")

        results['test_avg_loss'].append(avg_loss)
        results['test_avg_acc'].append(avg_acc)

        if best_acc < val_avg_acc:
            best_acc = val_avg_acc
            best_step = step
            test_best_based_on_step = avg_acc
            test_best_min_based_on_step = np.min(all_acc)
            test_best_max_based_on_step = np.max(all_acc)
            test_best_std_based_on_step = np.std(all_acc)

        results['val_avg_loss'].append(val_avg_loss)
        results['val_avg_acc'].append(val_avg_acc)
        results['best_step'].append(best_step)
        results['best_val_acc'].append(best_acc)
        results['best_test_acc_based_on_val_beststep'].append(test_best_based_on_step)
        results['test_best_min_based_on_step'].append(test_best_min_based_on_step)
        results['test_best_max_based_on_step'].append(test_best_max_based_on_step)
        results['test_best_std_based_on_step'].append(test_best_std_based_on_step)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Federated Hypernetworks meets MEDMNIST"
    )

    parser.add_argument(
        "--data_name", type=str, default="pathmnist", choices=["pathmnist"], help = "As of now we are only supporting pathMNIST"
    )
    parser.add_argument(
        "--num_nodes", type=int, default=50, help="Number of simulated nodes"
    )

    parser.add_argument(
        "--num_steps", type=int, default = 1000
    )
    parser.add_argument(
        "--optim", type = str, default = "sgd", choices = ["sgd", "adam"], help="learning rate"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="batch size"
    )
    parser.add_argument(
        "--inner-steps", type = int, default = 50, help="number of inner steps"
    )

    parser.add_argument(
        "--n-hidden", type=int, default=3, help="number of hidden layers"
    )

    parser.add_argument(
        "--inner_lr", type=float, default=1e-4, help="Learning rate for the inner optimiser"
    )   

    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate for the outer optimiser"
    )

    parser.add_argument(
        "--wd", type=float, default=1e-4, help="weight decay"
    )
    parser.add_argument(
        "--inner-wd", type=float, default=1e-6, help="inner weight decay"
    )
    parser.add_argument(
        "--embed-dim", type=int, default=-1, help="embedding dim"
    )
    parser.add_argument(
        "--embed-lr", type=float, default=None, help="embedding learning rate"
    )
    parser.add_argument(
        "--hyper-hid", type=int, default=100, help="hypernet hidden dim"
    )
    parser.add_argument(
        "--spec-norm", type=str2bool, default=False, help="hypernet hidden dim"
    )
    parser.add_argument(
        "--nkernels", type=int, default=64, help="number of kernels for cnn model"
    )
    parser.add_argument(
        "--gpu", type=int, default=0, help="gpu device ID"
    )
    parser.add_argument(
        "--eval-every", type=int, default=30, help="eval every X selected epochs"
    )
    parser.add_argument(
        "--save-path", type=str, default="pfedhn_hetro_res", help="dir path for output file"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="seed value"
    )

    args = parser.parse_args()
    assert args.gpu <= torch.cuda.device_count(), f"--gpu flag should be in range [0,{torch.cuda.device_count() - 1}]"

    set_logger()
    set_seed(args.seed)

    device = get_device(gpus = args.gpu)

    args.data_name == "pathmnist"

    args.classes_per_node = 2

    train(
        data_name = args.data_name,
        classes_per_node = args.classes_per_node,
        num_nodes = args.num_nodes,
        steps = args.num_steps,
        inner_steps = args.inner_steps,
        optim=args.optim,
        lr=args.lr,
        inner_lr=args.inner_lr,
        embed_lr=args.embed_lr,
        wd=args.wd,
        inner_wd=args.inner_wd,
        embed_dim=args.embed_dim,
        hyper_hid=args.hyper_hid,
        n_hidden=args.n_hidden,
        n_kernels=args.nkernels,
        bs=args.batch_size,
        device=device,
        eval_every=args.eval_every,
        seed = args.seed
    )
        
