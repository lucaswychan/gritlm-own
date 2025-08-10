import logging
import json
import multiprocessing
import os
from pathlib import Path
import random

import datasets
import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, Trainer, set_seed

from .arguments import CustomTrainingArguments, DataArguments, ModelArguments
from .data import CustomCollator, CustomDataset, CustomRandomSampler
from .model import GritLMTrainModel

BASE_BOS: str = ""
TURN_SEP: str = "\n"

USER_BOS: str = ""
USER_EOS: str = "" # "</s>" for Zephyr format

EMBED_BOS: str = ""
# Am embed eos is useless as there is no generative loss on it so it won't be learned
# & it does not add anything new; It only makes sense for lasttoken pooling
EMBED_EOS: str = ""

ASSISTANT_BOS: str = ""
ASSISTANT_EOS: str = ""

logger = logging.getLogger(__name__)

def args_to_dtype(args):
    if args.bf16: return torch.bfloat16
    if args.fp16: return torch.float16
    return torch.float32

#@lucaswychan add checking of no negs since it will affect the true label in cross entropy loss
def filter_too_long_instructions_and_no_negs(dataset, tokenizer, query_max_len, passage_max_len):
    def tokens_after_instruction(instr: str, text: str) -> int:
        """How many tokens remain once the instruction part is masked out?"""
        prompt_full  = BASE_BOS + USER_BOS + instr + USER_EOS + EMBED_BOS + text + EMBED_EOS
        prompt_instr = BASE_BOS + USER_BOS + instr + USER_EOS + EMBED_BOS
        tok_full  = tokenizer(prompt_full,  add_special_tokens=False,
                            truncation=True, max_length=query_max_len)["input_ids"]
        tok_instr = tokenizer(prompt_instr, add_special_tokens=False)["input_ids"]
        return len(tok_full) - len(tok_instr)

    def filter_fn(example):
        # Filter out super long examples to avoid tokenize taking forever
        if not example["neg"]:
            return False
        if (len(example["query"][0]) > query_max_len * 10) or not(example["query"][1].strip()):
            return False
        if len(tokenizer.tokenize(BASE_BOS + USER_BOS + example["query"][0].strip("\t\n :") + USER_EOS + EMBED_BOS)) >= query_max_len:
            return False
        if tokens_after_instruction(example["query"][0], example["query"][1]) == 0:
            return False
        return True
    num_proc = max(multiprocessing.cpu_count()-2, 1) if len(dataset) > 5000 else 1
    return dataset.filter(filter_fn, num_proc=num_proc, load_from_cache_file=True)

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to bypass."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.bf16,
    )
    
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
        # Additional stability for LoRA + gradient checkpointing
        # if hasattr(training_args, 'gradient_checkpointing_kwargs'):
        #     training_args.gradient_checkpointing_kwargs.update({
        #         "use_reentrant": False,
        #         "preserve_rng_state": True
        #     })
        # else:
        #     training_args.gradient_checkpointing_kwargs = {
        #         "use_reentrant": False,
        #         "preserve_rng_state": True
        #     }

    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    # Set seed
    set_seed(training_args.seed)

    # If embedding/unified, handle grad accumulation manually inside forward of GradCacheTrainer.
    gc_chunk_size = None
    if ((training_args.gradient_accumulation_steps > 1) and \
        (training_args.negatives_cross_device) and \
        (training_args.mode in ["embedding", "unified"])) or \
        (training_args.no_gen_gas and training_args.no_emb_gas):
        gc_chunk_size = training_args.per_device_train_batch_size
        training_args.per_device_train_batch_size = \
            training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
        training_args.gradient_accumulation_steps = 1

        logger.info("Using GradCache with chunk size %d", gc_chunk_size)
    elif (training_args.no_gen_gas or training_args.no_emb_gas):
        raise ValueError("Cannot use no_gen_gas or no_emb_gas without GradCache")

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        padding_side="right", # Has to be right so masking of instruction tokens works correctly
    )
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=1,
    )
    logger.info('Config: %s', config)
    
    if not(tokenizer.pad_token) and tokenizer.bos_token:
        tokenizer.pad_token = tokenizer.bos_token
        logger.info('Set pad token to bos token: %s', tokenizer.pad_token)   

    data_files = [os.path.join(data_args.train_data, x) for x in os.listdir(data_args.train_data)] if \
        os.path.isdir(data_args.train_data) else [data_args.train_data]
    train_ds, ds_embedding_lens = [], []
    
    num_samples = None
    if data_args.num_samples:
        with open(data_args.num_samples, "r") as f:
            num_samples = json.load(f)
    
    ds_name_to_samples = {}

    if data_args.generative_max_len is None:
        data_args.generative_max_len = data_args.passage_max_len

    for file in data_files:
        if not file.endswith(".jsonl"):
            continue
        logger.info("Loading dataset %s", file)
        tmp_ds = datasets.load_dataset('json', data_files=file, split='train')
        tmp_ds_len = len(tmp_ds)
        # For testing, can add an origin column:
        # origin_col = [file] * len(tmp_ds)
        # tmp_ds = tmp_ds.add_column("origin", origin_col)
        if tmp_ds_len > data_args.max_example_num_per_dataset:
            tmp_ds = tmp_ds.select(
                random.sample(list(range(tmp_ds_len)), data_args.max_example_num_per_dataset)
            )
        # Check if has instructions separated such that they will be masked out later
        # If so filter out samples where the instructions are too long else they will all be 0s
        if training_args.mode in ["embedding", "unified"] and "query" in tmp_ds.features:
            if isinstance(tmp_ds[0]['query'], (tuple, list)):
                logger.info(f"Filtering out embedding samples with too long instructions for {file}")
                #@lucaswychan add checking of no negs since it will affect the true label in cross entropy loss
                tmp_ds = filter_too_long_instructions_and_no_negs(
                    tmp_ds,
                    tokenizer,
                    data_args.query_max_len,
                    data_args.passage_max_len,
                )
                if num_samples:
                    assert file.split("/")[-1] in num_samples, f'Missing num_samples for {file.split("/")[-1]}'
                    tmp_ds_len = len(tmp_ds)
                    samples = num_samples[file.split("/")[-1]]
                    if tmp_ds_len > samples:                    
                        tmp_ds = tmp_ds.select(random.sample(list(range(tmp_ds_len)), samples))
            ds_name_to_samples[file.split("/")[-1]] = len(tmp_ds)
            train_ds.append(tmp_ds)
            continue
        logger.info("Skipping dataset %s as its type could not be identified", file)
    if training_args.mode == "embedding":
        ds_embedding_lens = [len(t) for t in train_ds]
        ds = datasets.concatenate_datasets(train_ds)
        logger.info("Embedding mode: %d samples", len(ds))
    else:
        raise NotImplementedError(training_args.mode)

    os.makedirs(training_args.output_dir, exist_ok=True)
    with open(os.path.join(training_args.output_dir, "dataset_num_samples.json"), "w") as f:
        json.dump(ds_name_to_samples, f)

    quantization_config, load_in_4bit = None, False
    if training_args.qlora:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = GritLMTrainModel(
        model_name_or_path=model_args.model_name_or_path,
        normalized=model_args.normalized,
        pooling_method=model_args.pooling_method,
        negatives_cross_device=training_args.negatives_cross_device,
        debiased=training_args.debiased,
        tau_plus=training_args.tau_plus,
        temperature=training_args.temperature,
        mode=training_args.mode,
        projection=model_args.projection,
        attn=model_args.attn,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=args_to_dtype(training_args),
        loss_gen_type=training_args.loss_gen_type,
        loss_gen_factor=training_args.loss_gen_factor,
        use_cache=False,
        # Critical to make Mixtral work
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
        load_in_4bit=load_in_4bit,
    )
    # Add special token for embed
    if model_args.pooling_method == "lasttoken":
        embed_eos = "</e>"
        if embed_eos in tokenizer.vocab:
            logger.info("Embed eos token already in vocab: %s", embed_eos)
        else:
            logger.info("Adding embed eos token to vocab: %s", embed_eos)
            tokenizer.add_tokens([embed_eos], special_tokens=True)
            model.model.resize_token_embeddings(len(tokenizer))
        config.num_vocab += 1
    else:
        embed_eos = EMBED_EOS

    if os.getenv("BIDIRECTIONAL_ATTN", False):
        if hasattr(model.model, "model"):
            model.model.model.padding_idx = tokenizer.pad_token_id
        else:
            model.model.padding_idx = tokenizer.pad_token_id

    if (training_args.lora) or (training_args.qlora):
        if training_args.qlora:
            from peft import prepare_model_for_kbit_training
            model.model = prepare_model_for_kbit_training(
                model.model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

        from peft import get_peft_model, LoraConfig, TaskType
        # https://github.com/texttron/tevatron/blob/2e5d00ee21d5a7db0bd2ea1463c9150a572106d4/examples/repllama/repllama.py#L81
        # https://github.com/allenai/open-instruct/blob/9ebcb582cfc243a6dab75b4302fa432784db26c2/open_instruct/finetune.py#L478
        peft_config = LoraConfig(
            inference_mode=False, 
            r=16, 
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules="all-linear",
            task_type=TaskType.FEATURE_EXTRACTION
        )
        model.model.enable_input_require_grads()
        model.model = get_peft_model(model.model, peft_config)
        model.model.print_trainable_parameters()
        
    train_dataset = CustomDataset(
        ds,
        args=data_args,
        tokenizer=tokenizer,
        mode=training_args.mode,
        full_bs=training_args.per_device_train_batch_size,
        generative_bs=training_args.per_device_generative_bs,
        max_seq_len=max(data_args.query_max_len, data_args.passage_max_len, data_args.generative_max_len),
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "data_collator": CustomCollator(
            tokenizer,
            query_max_len=data_args.query_max_len,
            passage_max_len=data_args.passage_max_len,
            generative_max_len=data_args.generative_max_len,
            base_bos=BASE_BOS,
            turn_sep=TURN_SEP,
            user_bos=USER_BOS,
            user_eos=USER_EOS,
            embed_bos=EMBED_BOS,
            embed_eos=embed_eos,
            assistant_bos=ASSISTANT_BOS,
            assistant_eos=ASSISTANT_EOS,
            prefixlm=data_args.prefixlm
        ),
        "tokenizer": tokenizer,
    }

    if gc_chunk_size is not None:
        from .gradcache_trainer import GradCacheTrainer
        trainer = GradCacheTrainer(**trainer_kwargs)
        trainer.gc_chunk_size = gc_chunk_size
        trainer.emb_loss_fn = model.emb_loss_fn
        trainer.mode = training_args.mode
        trainer.no_gen_gas = training_args.no_gen_gas
        trainer.no_emb_gas = training_args.no_emb_gas
        trainer.split_emb = training_args.split_emb
        trainer.split_emb_full = training_args.split_emb_full
        trainer.emb_p_only = training_args.emb_p_only
        trainer.emb_q_only = training_args.emb_q_only
    else:
        trainer = Trainer(**trainer_kwargs)

    if len(ds_embedding_lens) > 1:
        assert training_args.dataloader_drop_last, "Multiple datasets are only supported with dropping the last incomplete batch, set `--dataloader_drop_last`"
        logger.info("Embedding dataset lengths: %s", ds_embedding_lens)
        # Multiple embedding datasets & we want to make sure each batch mostly comes from one dataset
        # Set custom sampler, see https://github.com/huggingface/transformers/blob/ccb92be23def445f2afdea94c31286f84b89eb5b/src/transformers/trainer.py#L785
        total_bs = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
        total_bs = total_bs * dist.get_world_size() if dist.is_initialized() else total_bs
        trainer._get_train_sampler = lambda _: CustomRandomSampler(
            total_batch_size=total_bs, ds_lens=ds_embedding_lens,
            _num_samples=sum(ds_embedding_lens), data_source=train_dataset,
        )

    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    # Training
    logger.info("Starting training")
    resume_from_checkpoint = None
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # The below does not save if state dict type is `SHARDED_STATE_DICT`
    trainer.save_model()

    # To be safe do another FS save
    if (trainer.is_fsdp_enabled) and (trainer.accelerator.state.fsdp_plugin.state_dict_type != "FULL_STATE_DICT"):
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
        fsd_path = os.path.join(training_args.output_dir, "full_state_dict")
        os.makedirs(fsd_path, exist_ok=True)
        trainer.save_model(fsd_path)

    # Save tokenizer & config for easy usage afterwards
    if trainer.is_world_process_zero(): 
        tokenizer.save_pretrained(training_args.output_dir)
        config.to_json_file(training_args.output_dir + "/config.json")
        
        # # Save the train configuration for reproducibility
        # all_configs = {}
        # all_configs["training_args"] = training_args.__dict__
        # all_configs["model_args"] = model_args.__dict__
        # all_configs["data_args"] = data_args.__dict__
        # with open(training_args.output_dir + "/train_all_configs.json", "w") as f:
        #     json.dump(all_configs, f)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(e)
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        raise e
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
