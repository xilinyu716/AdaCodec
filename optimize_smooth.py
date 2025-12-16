# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


#TODO Implement MXFP4 Quantization
#TODO Implement Optimizer for Blockwise-Diagnal Orthogonal Matrix

import datetime
import os
from logging import Logger

import datasets
import torch
import torch.distributed as dist
from torch import nn
from transformers import LlamaTokenizerFast, Trainer, default_data_collator
import transformers
from train_utils.fsdp_trainer import FSDPTrainer
from train_utils.main import prepare_model
from train_utils.modeling_llama_quant import LlamaForCausalLM as LlamaForCausalLMQuant
from train_utils.optimizer import SGDG, SGDGMuon
from utils.data_utils import CustomJsonDataset
from utils.quant_utils import ActQuantWrapper
from utils.hadamard_utils import random_hadamard_matrix
from utils.process_args import process_args_ptq
from utils.utils import get_local_rank, get_logger, pt_fsdp_state_dict
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
import torch.fx as fx
from torch.fx import symbolic_trace
from torch.optim import AdamW
log: Logger = get_logger("spinquant")



called = []

def make_hook(name):
    def hook(module, inp, out):
        called.append(name)
    return hook



class RotateModule(nn.Module):
    def __init__(self, R_init:torch.Tensor):
        super(RotateModule, self).__init__()
        self.weight = nn.Parameter(R_init.clone().to(torch.float32).to(torch.device("cuda")), requires_grad=True)

    def forward(self, x, transpose=False):
        
        raise NotImplementedError
    
        if transpose:
            return x @ self.weight
        else:
            return self.weight @ x



def block_hadamard_tensor(hidden_size, block_size=128, device="cuda"):

    assert hidden_size % block_size == 0
    n_blocks = hidden_size // block_size
    blocks = [random_hadamard_matrix(block_size, device) for _ in range(n_blocks)]
    return torch.stack(blocks, dim=0)  # shape: (n_blocks, block_size, block_size)


def train() -> None:
    
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))
    model_args, training_args, ptq_args = process_args_ptq()
    local_rank = get_local_rank()

    log.info("the rank is {}".format(local_rank))
    torch.distributed.barrier()
    
    print("Model Config")
    print(model_args.input_model)
    config = transformers.AutoConfig.from_pretrained(
        model_args.input_model, token=model_args.access_token
    )

    print("Processed Config:")
    print(config)
    # Llama v3.2 specific: Spinquant is not compatiable with tie_word_embeddings, clone lm_head from embed_tokens
    process_word_embeddings = False
    if config.tie_word_embeddings:
        config.tie_word_embeddings = False
        process_word_embeddings = True
    dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    model:nn.Module = LlamaForCausalLMQuant.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        config=config,
        torch_dtype=dtype,
        token=model_args.access_token,
    )
    
    
    # print("Origin Model")
    # print(model)
    
    # for name, mod in model.named_modules():
    #     if isinstance(mod, nn.Linear):
    #         print(name)
    #         print(mod)
    #         print(type(mod))
    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()


    # print("Starting Prepare Model !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    
    
    # debug: check the change in loss
    model = prepare_model(ptq_args, model)

        
    
    
    path = "/localssd/hyzhang/spinquant/rotation/R.bin"
    R_dict = torch.load(path)
    R1_hat = R_dict["R1"]
    model.R1 = RotateModule(R1_hat)
    
    
    # print(f"R1 shape: {model.R1.weight.shape}!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # print(f"R1 value: {model.R1.weight[0]}")
    # exit(0)
    
    
    # debug: remove R2 then check the change of loss==================================
    for i in range(model.config.num_hidden_layers):
        # Each head dim = 128 for Llama model
        R2 = R_dict[f"model.layers.{i}.self_attn.R2"]
        R2_hat = R_dict[f"model.layers.{i}.self_attn.R2_hat"]
        model.model.layers[i].self_attn.R2 = RotateModule(R2)
        model.model.layers[i].self_attn.R2_hat = RotateModule(R2_hat)
    # end debug =============================================================
    
    # debug: remove RM to check the change of loss ============================================
    for i in range(model.config.num_hidden_layers):
        RM = R_dict[f"model.layers.{i}.mlp.RM"]
        RM_hat = R_dict[f"model.layers.{i}.mlp.RM_hat"]
        
        # debug: senity check for Llama-2-7b-hf ==========================
        assert model.model.layers[i].mlp.up_proj.weight.shape[0] == 11008, "Config for Llama-2-7b-hf"
        # end debug ================================================
        model.model.layers[i].mlp.RM = RotateModule(RM)
        model.model.layers[i].mlp.RM_hat = RotateModule(RM_hat)
    # end debug ===============================================================
    
    for i in range(model.config.num_hidden_layers):
        mod_list = ["self_attn.R1_q", "self_attn.R1_k", "self_attn.R1_v",
                    "self_attn.R1_o", "mlp.R1_up", "mlp.R1_gate", "mlp.R1_down"]

        
        for mod in mod_list:
            exec(f"R1_hat = R_dict[\"model.layers.{i}.{mod}\"]")
            exec(f"model.model.layers[{i}].{mod} = RotateModule(R1_hat)")
        
            
            
    # for i in range(model.config.num_hidden_layers):
    #     print(f"{i}: \
    #         shape Q: {model.model.layers[i].self_attn.R1_q.weight.shape} \n \
    #         shape K: {model.model.layers[i].self_attn.R1_k.weight.shape} \n \
    #         shape V: {model.model.layers[i].self_attn.R1_v.weight.shape} \n \
    #         shape O: {model.model.layers[i].self_attn.R1_o.weight.shape} \n \
    #         shape Up: {model.model.layers[i].mlp.R1_up.weight.shape} \n \
    #         shape Gate: {model.model.layers[i].mlp.R1_gate.weight.shape} \n \
    #         shape Down: {model.model.layers[i].mlp.R1_down.weight.shape}")
            
    # exit(0)
    
    
    if local_rank == 0:
        log.info("Model init completed for training {}".format(model))
        log.info("Start to load tokenizer...")
    tokenizer = LlamaTokenizerFast.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
        token=model_args.access_token,
    )
    log.info("Complete tokenizer loading...")
    model.config.use_cache = False
    calibration_datasets = datasets.load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1"
    )
    
    log.info("Calibration Dataset Loaded...")
    train_data = CustomJsonDataset(
        calibration_datasets["train"],
        tokenizer,
        block_size=min(training_args.model_max_length, 2048),
    )
    log.info("Train data formed")
    
    
    for param in model.parameters():
        param.requires_grad = False
    
    MyTrainer = Trainer
    # TODO: Simplify the implementation with exec
    for i in range(model.config.num_hidden_layers):
        smooth = torch.zeros(model.config.hidden_size) 
        model.model.layers[i].self_attn.q_proj.smooth = RotateModule(smooth)
        model.model.layers[i].self_attn.q_proj.smooth_hat = RotateModule(smooth)
        model.model.layers[i].self_attn.k_proj.smooth = RotateModule(smooth)
        model.model.layers[i].self_attn.k_proj.smooth_hat = RotateModule(smooth)
        model.model.layers[i].self_attn.v_proj.smooth = RotateModule(smooth)
        model.model.layers[i].self_attn.v_proj.smooth_hat = RotateModule(smooth)
        model.model.layers[i].self_attn.o_proj.smooth = RotateModule(smooth)
        model.model.layers[i].self_attn.o_proj.smooth_hat = RotateModule(smooth)
        model.model.layers[i].mlp.up_proj.smooth = RotateModule(smooth)
        model.model.layers[i].mlp.up_proj.smooth_hat = RotateModule(smooth)
        model.model.layers[i].mlp.gate_proj.smooth = RotateModule(smooth)
        model.model.layers[i].mlp.gate_proj.smooth_hat = RotateModule(smooth)
        
        smooth_down = torch.zeros(model.model.layers[i].mlp.down_proj.weight.shape[-1])
        model.model.layers[i].mlp.down_proj.smooth = RotateModule(smooth_down)
        model.model.layers[i].mlp.down_proj.smooth_hat = RotateModule(smooth_down)
            

    
    smooth_params = [
        model.model.layers[i].self_attn.q_proj.smooth.weight
        for i in range(model.config.num_hidden_layers)
    ] + [
        model.model.layers[i].self_attn.q_proj.smooth_hat.weight
        for i in range(model.config.num_hidden_layers)
    ] + [
        model.model.layers[i].self_attn.k_proj.smooth.weight
        for i in range(model.config.num_hidden_layers)
    ] + [
        model.model.layers[i].self_attn.k_proj.smooth_hat.weight
        for i in range(model.config.num_hidden_layers)
    ] + [
        model.model.layers[i].self_attn.v_proj.smooth.weight
        for i in range(model.config.num_hidden_layers)
    ] + [
        model.model.layers[i].self_attn.v_proj.smooth_hat.weight
        for i in range(model.config.num_hidden_layers)
    ] + [
        model.model.layers[i].self_attn.o_proj.smooth.weight
        for i in range(model.config.num_hidden_layers)
    ] + [
        model.model.layers[i].self_attn.o_proj.smooth_hat.weight
        for i in range(model.config.num_hidden_layers)
    ] + [
        model.model.layers[i].mlp.up_proj.smooth.weight
        for i in range(model.config.num_hidden_layers)
    ] + [
        model.model.layers[i].mlp.up_proj.smooth_hat.weight
        for i in range(model.config.num_hidden_layers)
    ] + [
        model.model.layers[i].mlp.down_proj.smooth.weight
        for i in range(model.config.num_hidden_layers)
    ] + [
        model.model.layers[i].mlp.down_proj.smooth_hat.weight
        for i in range(model.config.num_hidden_layers)
    ] + [
        model.model.layers[i].mlp.gate_proj.smooth.weight
        for i in range(model.config.num_hidden_layers)
    ] + [
        model.model.layers[i].mlp.gate_proj.smooth_hat.weight
        for i in range(model.config.num_hidden_layers)
    ]
    
    
    train_data = CustomJsonDataset(
        calibration_datasets["train"],
        tokenizer,
        block_size=min(training_args.model_max_length, 2048),
    )
    
    
    for p in smooth_params:
        assert p.requires_grad == True
    
    optimizer = AdamW(
    smooth_params,
    lr=0.05,
    weight_decay=0.0  
)
    trainer = MyTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=None,
        data_collator=default_data_collator,
        optimizers=(optimizer, None),
    )
    model.enable_input_require_grads()
    
    with torch.autograd.set_detect_anomaly(True):
        trainer.train()
    if training_args.fsdp != "" and training_args.fsdp != []:
        cpu_state = pt_fsdp_state_dict(trainer.model)
    else:
        cpu_state = trainer.model.state_dict()



    S_dict = {
        key.replace(".weight", ""): value
        for key, value in cpu_state.items()
        if "smooth" in key
    }
    
    
    print(f"S_dict: {S_dict}")
    if local_rank == 0:
        os.makedirs(model_args.output_rotation_path, exist_ok=True)
        path = os.path.join(model_args.output_rotation_path, "S.bin")
        torch.save(
            S_dict,
            path,
        )
    dist.barrier()
    

if __name__ == "__main__":
    train()