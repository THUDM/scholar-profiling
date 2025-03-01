#!/usr/bin/env python
# -*- coding:utf-8 -*-

def convert_spot_asoc(spot_asoc_instance, structure_maker):
    """将一个 Spot-Asoc 实例转换成目标字符串

    Args:
        spot_asoc_instance ([type]): [description]
        structure_maker ([type]): [description]

    Returns:
        [type]: [description]
    """
    spot_instance_str_rep_list = list()
    for spot in spot_asoc_instance:
        spot_str_rep = [
            spot['label'],
            structure_maker.target_span_start,
            spot['span'],
        ]
        for asoc_label, asoc_span in spot.get('asoc', list()):
            asoc_str_rep = [
                structure_maker.span_start,
                asoc_label,
                structure_maker.target_span_start,
                asoc_span,
                structure_maker.span_end,
            ]
            spot_str_rep += [' '.join(asoc_str_rep)]
        spot_instance_str_rep_list += [' '.join([
            structure_maker.record_start,
            ' '.join(spot_str_rep),
            structure_maker.record_end,
        ])]
    target_text = ' '.join([
        structure_maker.sent_start,
        ' '.join(spot_instance_str_rep_list),
        structure_maker.sent_end,
    ])
    return target_text


def convert_spot_asoc_name(spot_asoc_instance, structure_maker):
    """将一个 Spot-Asoc-Name 实例转换成目标字符串

    Args:
        spot_asoc_instance ([type]): [description]
        structure_maker ([type]): [description]

    Returns:
        [type]: [description]
    """
    spot_instance_str_rep_list = list()
    for spot in spot_asoc_instance:
        spot_str_rep = [
            spot['span'],
            structure_maker.target_span_start,
            spot['label'],
        ]
        for asoc_label, asoc_span in spot.get('asoc', list()):
            asoc_str_rep = [
                structure_maker.span_start,
                asoc_span,
                structure_maker.target_span_start,
                asoc_label,
                structure_maker.span_end,
            ]
            spot_str_rep += [' '.join(asoc_str_rep)]
        spot_instance_str_rep_list += [' '.join([
            structure_maker.record_start,
            ' '.join(spot_str_rep),
            structure_maker.record_end,
        ])]
    target_text = ' '.join([
        structure_maker.sent_start,
        ' '.join(spot_instance_str_rep_list),
        structure_maker.sent_end,
    ])
    return target_text


convert_to_record_function = {
    'spotasoc': convert_spot_asoc,
    'spotasocname': convert_spot_asoc_name,
}
