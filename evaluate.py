import os
from utils import convert_to_json
from metric.evaluator import get_evaluator

# 设置代理
os.environ["HTTP_PROXY"] = "http://zhaozhengyang:XvgY8wEqHGhYU7Y04HgaProNb8kS6rxD6GMltFETKkhLPS2NQdyVZIpCC9Cw@10.1.20.50:23128/"
os.environ["HTTPS_PROXY"] = "http://zhaozhengyang:XvgY8wEqHGhYU7Y04HgaProNb8kS6rxD6GMltFETKkhLPS2NQdyVZIpCC9Cw@10.1.20.50:23128/"

def evaluate(output_list):
    # 评估 summarization 任务下的 fluency 指标
    sum_task = 'summarization'
    sum_data = convert_to_json(output_list=output_list, src_list=[''], ref_list=[''])
    sum_evaluator = get_evaluator(sum_task)
    sum_scores = sum_evaluator.evaluate(sum_data, dims=['fluency'], print_result=False)
    fluency_score = sum_scores[0].get('fluency', None)

    # 评估 dialogue 任务下的 naturalness 和 understandability 指标
    dialogue_task = 'dialogue'
    dialogue_data = convert_to_json(output_list=output_list, src_list=[''], context_list=[''])
    dialogue_evaluator = get_evaluator(dialogue_task)
    dialogue_scores = dialogue_evaluator.evaluate(dialogue_data, dims=['naturalness', 'understandability'], print_result=False)
    naturalness_score = dialogue_scores[0].get('naturalness', None)
    understandability_score = dialogue_scores[0].get('understandability', None)

    return {
        'fluency': fluency_score,
        'naturalness': naturalness_score,
        'understandability': understandability_score
    }