#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

from cProfile import label
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from datasets import load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    default_data_collator,
    set_seed
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process

from uie.extraction import constants
from uie.extraction.record_schema import RecordSchema
from uie.extraction.extraction_metrics import get_extract_metrics
from uie.extraction.noiser.spot_asoc_noiser import SpotAsocNoiser
from uie.extraction.dataset_processer import PrefixGenerator
from uie.seq2seq.constrained_seq2seq import ConstraintSeq2SeqTrainingArguments, ConstraintSeq2SeqTrainer
from uie.seq2seq.data_collator import (
    DataCollatorForMetaSeq2Seq,
    DynamicSSIGenerator,
)
from uie.seq2seq.features import RecordFeature
from uie.seq2seq.t5_bert_tokenizer import T5BertTokenizer
from uie.seq2seq.trainer_arguments import ModelArguments, DataTrainingArguments


logger = logging.getLogger(__name__)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    #获取参数
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, ConstraintSeq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)
    logger.info("Options:")
    logger.info(model_args)
    logger.info(data_args)
    logger.info(training_args)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files in the summarization task, this script will use the first column for the full texts and the
    # second column for the summaries (unless you specify column names for this with the `text_column` and
    # `record_column` arguments).
    # For translation, only JSON files are supported, with one field named "translation" containing two keys for the
    # source and target languages (unless you adapt what follows).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    
    # download the dataset. 加载数据集（下载）
    # dataset_name = None
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name)
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
    logger.info(data_files)
    datasets = load_dataset("uie_json.py", data_files=data_files)

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.
    logger.info(datasets)
    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    logger.info("Load Config: %s" % model_args.config_name if model_args.config_name else model_args.model_name_or_path)
    # 加载模型配置
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    config.max_length = data_args.max_target_length
    tokenizer_name = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
    if 'char' in tokenizer_name:
        tokenizer = T5BertTokenizer.from_pretrained(tokenizer_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    # 要移除的token列表 去除bos eos pad
    to_remove_token_list = list()
    if tokenizer.bos_token:
        to_remove_token_list += [tokenizer.bos_token]
    if tokenizer.eos_token:
        to_remove_token_list += [tokenizer.eos_token]
    if tokenizer.pad_token:
        to_remove_token_list += [tokenizer.pad_token]

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        mirror='tuna',
    )

    if training_args.do_train:
        # 添加额外的特殊标识符
        to_add_special_token = list()
        for special_token in [constants.type_start, constants.type_end, constants.text_start, constants.span_start, constants.spot_prompt, constants.asoc_prompt]:
            if special_token not in tokenizer.get_vocab():
                to_add_special_token += [special_token]
        # 添加特殊标识符（不分词）
        tokenizer.add_special_tokens(
            {"additional_special_tokens": tokenizer.special_tokens_map_extended['additional_special_tokens'] + to_add_special_token}
        )

        model.resize_token_embeddings(len(tokenizer))

    logger.info(tokenizer)

    # Set decoder_start_token_id
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
    if data_args.record_schema and os.path.exists(data_args.record_schema):
        record_schema = RecordSchema.read_from_file(data_args.record_schema)
    else:
        record_schema = None

    if data_args.source_prefix is not None:
        if data_args.source_prefix == 'schema':
            prefix = PrefixGenerator.get_schema_prefix(schema=record_schema)
        elif data_args.source_prefix.startswith('meta'):
            prefix = ""
        else:
            prefix = data_args.source_prefix
    else:
        prefix = ""
        
    logger.info(f"Prefix: {prefix}")
    logger.info(f"Prefix Length: {len(tokenizer.tokenize(prefix))}")

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = datasets["train"].column_names
    elif training_args.do_eval:
        column_names = datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # To serialize preprocess_function below, each of those four variables needs to be defined (even if we won't use
    # them all).

    text_column = data_args.text_column # "text"
    record_column = data_args.record_column # "record"
    logger.info('Using src: %s and tgt: %s' % (text_column, record_column)) 

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.error(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )
    
    # 数据集处理函数
    def preprocess_function(examples):
        inputs = examples[text_column]
        targets = examples[record_column]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(_label if _label != tokenizer.pad_token_id else -100) for _label in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        model_inputs['sample_prompt'] = [False] * len(model_inputs['input_ids'])

        if data_args.source_prefix is not None and data_args.source_prefix.startswith('meta'):
            model_inputs['spots'] = examples['spot']
            model_inputs['asocs'] = examples['asoc']
            model_inputs['spot_asoc'] = examples['spot_asoc']
            # sample_prompt=True for Finetune and Pretrain
            model_inputs['sample_prompt'] = [True] * len(model_inputs['input_ids'])
        return model_inputs

    def preprocess_function_eval(examples):
        model_inputs = preprocess_function(examples)
        # sample_prompt=False for evaluation
        model_inputs['sample_prompt'] = [False] * len(model_inputs['input_ids'])
        return model_inputs

    def postprocess_text(x_str):
        # Clean `bos` `eos` `pad` for cleaned text
        for to_remove_token in to_remove_token_list:
            x_str = x_str.replace(to_remove_token, '')

        return x_str.strip()

    # 开始处理数据
    logger.info("Start Data Preprocessing ...")

    if training_args.do_train:
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples)) # 筛出部分训练样本
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            features=RecordFeature,
        )

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        eval_dataset = datasets["validation"]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))
        eval_dataset = eval_dataset.map(
            preprocess_function_eval,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            features=RecordFeature,
        )

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        test_dataset = datasets["test"]
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_test_samples))
        test_dataset = test_dataset.map(
            preprocess_function_eval,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            features=RecordFeature,
        )

    logger.info("End Data Preprocessing ...")

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif data_args.source_prefix.startswith('meta'):
        # 添加负样例spot asoc的比例
        if data_args.spot_noise > 0 or data_args.asoc_noise > 0:
            if data_args.decoding_format == 'spotasoc':
                # 将DynamicSSIGenerator生成的负label(span asco) 以概率(spot_noise=0.2, asoc_noise=0.2) len(spot_asoc) 插入输出
                spot_asoc_nosier = SpotAsocNoiser(
                    spot_noise_ratio=data_args.spot_noise,
                    asoc_noise_ratio=data_args.asoc_noise,
                    null_span=constants.null_span,
                )
            else:
                raise NotImplementedError(
                    f"decoding_format {data_args.decoding_format} is not implemented."
                )
        else:
            spot_asoc_nosier = None
        
        data_collator = DataCollatorForMetaSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
            max_length=data_args.max_source_length,
            max_prefix_length=data_args.max_prefix_length,
            max_target_length=data_args.max_target_length,
            negative_sampler=DynamicSSIGenerator( # Sample negative spot and asoc to construct SSI（prefix）
                tokenizer=tokenizer,
                schema=record_schema, 
                positive_rate=data_args.meta_positive_rate,
                negative=data_args.meta_negative,
                ordered_prompt=data_args.ordered_prompt, 
            ),
            spot_asoc_nosier=spot_asoc_nosier, # 根据比例为label添加负span、asso
            decoding_format=data_args.decoding_format,
        )
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=False, clean_up_tokenization_spaces=False)

        decoded_preds = [postprocess_text(x) for x in decoded_preds]
        decoded_labels = [postprocess_text(x) for x in decoded_labels]
        # print("decoded_preds",decoded_preds)
        # print("decoded_labels",decoded_labels)
        # 评估 得到metrics
        result = get_extract_metrics(
            pred_lns=decoded_preds,
            tgt_lns=decoded_labels,
            label_constraint=record_schema, # 所有spot asoc
            decoding_format=data_args.decoding_format,
        )

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds] # 每个句子预测的长度
        result["gen_len"] = np.mean(prediction_lens) # 平均预测长度
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    # Initialize our Trainer
    trainer = ConstraintSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        decoding_type_schema=record_schema,
        decoding_format=data_args.decoding_format, # 解码形式
        source_prefix=prefix,
        task=data_args.task,
    )

    # Training
    if training_args.do_train:
        # 从检查点加载模型参数
        if model_args.from_checkpoint:
            if last_checkpoint is not None:
                checkpoint = last_checkpoint
            elif os.path.isdir(model_args.model_name_or_path):
                checkpoint = model_args.model_name_or_path
        checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        # 保存训练结果
        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        results = trainer.evaluate(max_length=data_args.val_max_target_length, num_beams=data_args.num_beams)
        results = {k: round(v, 4) for k, v in results.items()}

        eval_results = trainer.predict(
            eval_dataset,
            metric_key_prefix="eval",
            max_length=data_args.val_max_target_length,
            num_beams=data_args.num_beams,
        )
        # 保存输出
        output_eval_file = os.path.join(training_args.output_dir, "eval_results_seq2seq.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in sorted(results.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            if training_args.predict_with_generate:
                eval_preds = tokenizer.batch_decode(
                    eval_results.predictions, skip_special_tokens=False, clean_up_tokenization_spaces=False
                )
                eval_preds = [postprocess_text(pred) for pred in eval_preds]
                output_test_preds_file = os.path.join(training_args.output_dir, "eval_preds_seq2seq.txt")
                with open(output_test_preds_file, "w") as writer:
                    writer.write("\n".join(eval_preds))

    if training_args.do_predict:
        logger.info("*** Test ***")

        test_results = trainer.predict(
            test_dataset,
            metric_key_prefix="test",
            max_length=data_args.val_max_target_length,
            num_beams=data_args.num_beams,
        )
        test_metrics = test_results.metrics
        test_metrics["test_loss"] = round(test_metrics["test_loss"], 4)

        output_test_result_file = os.path.join(training_args.output_dir, "test_results_seq2seq.txt")
        if trainer.is_world_process_zero():
            with open(output_test_result_file, "w") as writer:
                logger.info("***** Test results *****")
                for key, value in sorted(test_metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            if training_args.predict_with_generate:
                test_preds = tokenizer.batch_decode(
                    test_results.predictions, skip_special_tokens=False, clean_up_tokenization_spaces=False
                )
                test_preds = [postprocess_text(pred) for pred in test_preds]
                output_test_preds_file = os.path.join(training_args.output_dir, "test_preds_seq2seq.txt")
                with open(output_test_preds_file, "w") as writer:
                    writer.write("\n".join(test_preds))

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
