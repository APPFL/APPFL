import time
import torch
from torch.optim import *
from .fl_base import BaseClient
from transformers import AutoTokenizer
from appfl.misc.memory import MemoryTrace
from appfl.misc.llm import load_model, get_peft_state_dict, save_peft_model
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

class GlobusComputeClientLLMFineTuneOptim(BaseClient):
    """GlobusComputeClientLLMFineTuneOptim is client optimizer for finetuning parameter large language models."""
    def __init__(self, global_state, model_name, dataloader, test_dataloader, lora_config, cfg, **kwargs):
        self.cfg = cfg
        self.device = cfg.device
        self.global_state = global_state
        self.model_name = model_name
        self.dataloader = dataloader
        self.test_dataloader = test_dataloader
        self.lora_config = LoraConfig(**lora_config, task_type=TaskType.CAUSAL_LM, inference_mode=False)
        self.gradient_accumulation_steps = cfg.custom_configs.training_configs.gradient_accumulation_steps
        self.output_name = f"{cfg.output_dirname}/lora_globus_compute.pt"
        self.__dict__.update(kwargs)

    def update(self, cli_logger):
        cli_logger.start_timer("Load Model")
        llm_model = load_model(self.model_name, quantization=True)
        llm_model = prepare_model_for_kbit_training(llm_model)
        llm_model = get_peft_model(llm_model, self.lora_config)
        if self.global_state is not None:
            llm_model.load_state_dict(self.global_state, strict=False)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        cli_logger.stop_timer("Load Model")
        optimizer = eval(self.optim)(
            llm_model.parameters(),
            **self.optim_args
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            **self.scheduler_args
        )
        cli_logger.start_timer("Train")
        results = self._train(
            llm_model, 
            self.dataloader,
            self.test_dataloader,
            optimizer,
            scheduler
        )
        cli_logger.stop_timer("Train")
        cli_logger.add_info("Training", results)
        return get_peft_state_dict(llm_model), cli_logger

    def _train(
        self,
        model, 
        train_dataloader,
        eval_dataloader, 
        optimizer, 
        lr_scheduler, 
    ):
        """
        Trains the Causal LLM model on the given dataloader
        
        Args:
            model: The model to be trained
            train_dataloader: The dataloader containing the training data
            eval_dataloader: The dataloader containing the eval data if doing evaluation
            optimizer: The optimizer used for training
            lr_scheduler: The learning rate scheduler
            
        Returns: results dictionary containing average training and validation perplexity and loss
        """
        train_perp = []
        train_loss = []
        val_perp = []
        val_loss =[]
        epoch_times = []
        checkpoint_times = []
        results = {}
        best_val_loss = float("inf")
        for epoch in range(self.num_local_epochs):
            epoch_start_time = time.perf_counter()
            with MemoryTrace() as memtrace:  # track the memory usage
                model.train()
                total_loss = 0.0
                for step, batch in enumerate(train_dataloader):
                    for key in batch.keys():
                        batch[key] = batch[key].to(self.device)              
                    loss = model(**batch).loss
                    loss = loss / self.gradient_accumulation_steps
                    total_loss += loss.detach().float()
                    loss.backward()
                    if (step + 1) % self.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                        optimizer.step()
                        optimizer.zero_grad()
          
            epoch_end_time = time.perf_counter()-epoch_start_time
            epoch_times.append(epoch_end_time)
            train_epoch_loss = total_loss / len(train_dataloader)
            train_perplexity = torch.exp(train_epoch_loss)
            train_perp.append(train_perplexity)
            train_loss.append(train_epoch_loss)

            print(f"\nMax CUDA memory allocated was {memtrace.peak} GB")
            print(f"Max CUDA memory reserved was {memtrace.max_reserved} GB")
            print(f"Peak active CUDA memory was {memtrace.peak_active_gb} GB")
            print(f"Cuda Malloc retires : {memtrace.cuda_malloc_retires}")
            print(f"CPU Total Peak Memory consumed during the train (max): {memtrace.cpu_peaked + memtrace.cpu_begin} GB")
            print(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epcoh time {epoch_end_time}s")
            
            # Update the learning rate as needed
            lr_scheduler.step()
            
            if eval_dataloader != None:
                eval_ppl, eval_epoch_loss = self._evaluate(model, eval_dataloader, self.device)
                checkpoint_start_time = time.perf_counter()
                if eval_epoch_loss < best_val_loss:
                    save_peft_model(model, self.output_name)
                    print(f"PEFT modules are saved at {self.output_name}")
                checkpoint_end_time = time.perf_counter() - checkpoint_start_time
                checkpoint_times.append(checkpoint_end_time)
                if eval_epoch_loss < best_val_loss:
                    best_val_loss = eval_epoch_loss
                    print(f"best eval loss on epoch {epoch+1} is {best_val_loss}")
                val_loss.append(best_val_loss)
                val_perp.append(eval_ppl)
            
        avg_train_prep = sum(train_perp)/len(train_perp)
        avg_train_loss = sum(train_loss)/len(train_loss)
        avg_epoch_time = sum(epoch_times)/ len(epoch_times)
        results['avg_train_prep'] = avg_train_prep
        results['avg_train_loss'] = avg_train_loss
        results["avg_epoch_time"] = avg_epoch_time
        if eval_dataloader != None:
            avg_eval_prep = sum(val_perp)/len(val_perp) 
            avg_eval_loss = sum(val_loss)/len(val_loss) 
            avg_checkpoint_time = sum(checkpoint_times)/ len(checkpoint_times) if len(checkpoint_times) > 0 else 0
            results['avg_eval_prep'] = avg_eval_prep
            results['avg_eval_loss'] = avg_eval_loss
            results["avg_checkpoint_time"] = avg_checkpoint_time
        for key, value in results.items():
            if torch.is_tensor(value):
                results[key] = value.item()
        return results

    def _evaluate(self, model, eval_dataloader, device):
        """
        Evaluates the model on the given dataloader
        
        Args:
            model: The model to evaluate
            eval_dataloader: The dataloader containing the evaluation data
            device: The device for running evaluation
        
        Returns: eval_ppl, eval_epoch_loss
        """
        model.eval()
        eval_loss = 0.0  # Initialize evaluation loss
        with MemoryTrace():
            for batch in eval_dataloader:
                for key in batch.keys():
                    batch[key] = batch[key].to(device)
                # Ensure no gradients are computed for this scope to save memory
                with torch.no_grad():
                    # Forward pass and compute loss
                    outputs = model(**batch)
                    loss = outputs.loss
                    eval_loss += loss.detach().float()
        
        # Compute average loss and perplexity
        eval_epoch_loss = eval_loss / len(eval_dataloader)
        eval_ppl = torch.exp(eval_epoch_loss)
            
        return eval_ppl, eval_epoch_loss
