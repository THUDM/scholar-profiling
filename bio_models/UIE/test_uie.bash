#!/usr/bin/env bash
# -*- coding:utf-8 -*-
export batch_size="16"
export model_name=uie-base-en
export data_name=absa/14lap
export task_name="meta"
export decoding_format='spotasoc'

source scripts/function_code.bash

echo "Map Config" ${map_config}
output_dir=${model_folder}_run3

python3 scripts/eval_extraction.py -p ${output_dir} -g ${data_folder} -w -m ${eval_match_mode:-"normal"}