import torch
import evaluate
import time
from peft import PeftModel
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup, AutoModelForCausalLM
from utils import generate_response
import numpy as np
from tensor_selector import selection_DP, downscale_t_dy_and_t_dw
from tensor_flops import compute_tensor_flops, compute_forward_flops
from utils import flops_counter, compute_squad_metric


class Trainer:
    def __init__(
        self, 
        train_loader, 
        val_loader,
        test_loader, 
        model,
        train_type,
        tokenizer,
        max_output_length,
        model_path,
    ) -> None:
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model = model
        self.train_type = train_type
        self.tokenizer = tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger = SummaryWriter(flush_secs=10)
        self.rouge_metric = evaluate.load('rouge')
        self.interval = 200
        self.max_output_length = max_output_length
        self.model_path = model_path

    def train(
        self,
        learning_rate,
        num_epochs,
        log_dir='logs/'
    ):  
        # torch.backends.cuda.matmul.allow_tf32 = True
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=(len(self.train_loader) * num_epochs),
        )
        self.model = self.model.to(self.device)
        writer = SummaryWriter(log_dir=log_dir)
        
        total_time = 0
        # self._runtime_evaluate(self.val_loader)
        for epoch in range(num_epochs):
            t_start = time.time()
            
            self.model.train()
            total_loss = 0
            for step, batch in enumerate(tqdm(self.train_loader)):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                # print(batch)
                # print(batch["input_ids"].shape)
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss
                total_loss += loss.detach().float()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            t_end = time.time()
            epoch_time = t_end - t_start
            print(f"Epoch Time: {epoch_time} (s)")
            total_time += epoch_time
            print(f"Total Time: {total_time} (s)")
            
            self.model.eval()
            eval_loss = 0
            for step, batch in enumerate(tqdm(self.val_loader)):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                loss = outputs.loss
                eval_loss += loss.detach().float()
            
            eval_epoch_loss = eval_loss / len(self.val_loader)
            eval_ppl = torch.exp(eval_epoch_loss)
            train_epoch_loss = total_loss / len(self.train_loader)
            train_ppl = torch.exp(train_epoch_loss)
            
            writer.add_scalar('Loss/train', train_epoch_loss.item(), epoch)
            writer.add_scalar('PPL/train', train_ppl.item(), epoch)
            writer.add_scalar('Loss/valid', eval_epoch_loss.item(), epoch)
            writer.add_scalar('PPL/valid', eval_ppl.item(), epoch)
            
            print(f"epoch={epoch} train_ppl={train_ppl.item()} train_loss={train_epoch_loss.item()} eval_ppl={eval_ppl.item()} eval_loss={eval_epoch_loss.item()}")
            
            self._runtime_evaluate(self.val_loader)

            self._save_model()
            
        print(f"Total Time: {total_time} (s)")
        self._runtime_evaluate(self.test_loader)
    
    def _runtime_evaluate(self, dataset):
        self.model.eval()
        # for summarization
        m_rouge1 = 0
        m_rouge2 = 0
        m_rougeL = 0
        m_rougeLsum = 0
        
        total_count = 0
        with torch.no_grad():
            for step, batch in enumerate(tqdm(dataset)):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                batch_size = batch['input_ids'].shape[0]
                    
                all_results = generate_response(
                    self.model, 
                    self.train_type,
                    self.tokenizer, 
                    batch['lp_sources'], batch['labels'], batch['input_ids_lens'],
                    max_length=self.max_output_length
                )
                
                summarization_results = self.rouge_metric.compute(predictions=all_results["outputs_text"], references=all_results["labels_text"])
                
                m_rouge1 += (summarization_results['rouge1'] * batch_size)
                m_rouge2 += (summarization_results['rouge2'] * batch_size)
                m_rougeL += (summarization_results['rougeL'] * batch_size)
                m_rougeLsum += (summarization_results['rougeLsum'] * batch_size)
                
                total_count += batch_size
        
        m_rouge1 /= total_count
        m_rouge2/= total_count
        m_rougeL /= total_count
        m_rougeLsum /= total_count
        print(f"On validation/test set, rouge1={100*m_rouge1}, rouge2={100*m_rouge2}, rougeL={100*m_rougeL}, rougeLsum={100*m_rougeLsum}")

    def evaluate(self):
        # self._load_model()
        self.model = self.model.to(self.device)
        
        self.model.eval()
        # for summarization
        m_rouge1 = 0
        m_rouge2 = 0
        m_rougeL = 0
        m_rougeLsum = 0
        
        total_count = 0
        with torch.no_grad():
            for step, batch in enumerate(tqdm(self.test_loader)):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                batch_size = batch['input_ids'].shape[0]
                    
                all_results = generate_response(
                    self.model, 
                    self.train_type,
                    self.tokenizer, 
                    batch['lp_sources'], batch['labels'], batch['input_ids_lens'],
                    max_length=self.max_output_length
                )
                
                summarization_results = self.rouge_metric.compute(predictions=all_results["outputs_text"], references=all_results["labels_text"])
                
                m_rouge1 += (summarization_results['rouge1'] * batch_size)
                m_rouge2 += (summarization_results['rouge2'] * batch_size)
                m_rougeL += (summarization_results['rougeL'] * batch_size)
                m_rougeLsum += (summarization_results['rougeLsum'] * batch_size)
                
                total_count += batch_size
        
        m_rouge1 /= total_count
        m_rouge2/= total_count
        m_rougeL /= total_count
        m_rougeLsum /= total_count
        print(f"On test set, rouge1={100*m_rouge1}, rouge2={100*m_rouge2}, rougeL={100*m_rougeL}, rougeLsum={100*m_rougeLsum}")
    
    def _save_model(self):
        self.model.save_pretrained(self.model_path)
    
    def _load_model(self):
        if self.train_type in ["lora", "adalora", "prefix_tuning"]:
            self.model = PeftModel.from_pretrained(self.model, self.model_path)
            # if self.train_type == "lora" or self.train_type == "adalora":
            #     self.model.merge_and_unload()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
