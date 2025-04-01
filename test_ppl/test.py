from spotlight.misc import get_model_and_tokenizer, get_tokenizer
from spotlight.misc import get_env_conf
from spotlight.misc import Evaluator, QuestEvaluator, MagicpigEvaluator
import argparse, os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_conf", type=str, default=None)
    parser.add_argument("--use_env_conf_tasks", action="store_true", default=False)
    parser.add_argument('--rmt', action='store_true', default=False)

    # Quest related arguments (https://arxiv.org/pdf/2406.10774)
    parser.add_argument('--token_budget', type=int, default=1024, help='only used for quest')
    parser.add_argument('--chunk_size', type=int, default=16, help='only used for quest')

    args = parser.parse_args()

    env_conf = get_env_conf(args.env_conf)
    test_conf = get_env_conf("test_ppl/test.json")

    if "quest" in env_conf['model']['model_method']:
        tokenizer, model = get_model_and_tokenizer(**env_conf["model"])
        model.eval()
        from quest.evaluation.quest_attention import enable_quest_attention_eval
        enable_quest_attention_eval(model.model, args)
        evaluator_class = QuestEvaluator
    elif "magicpig" in env_conf['model']['model_method']:
        from spotlight.magicpig import get_magicpig, MagicpigConfig
        tokenizer = get_tokenizer(env_conf['model']['model_name'])
        magicpig_config = MagicpigConfig(
            model_name_or_path=env_conf['model']['model_name'],
            max_seq_length=8192)
        model = get_magicpig(magicpig_config)
        evaluator_class = MagicpigEvaluator
    else:
        tokenizer, model = get_model_and_tokenizer(**env_conf["model"])
        model.eval()
        evaluator_class = Evaluator

    if args.use_env_conf_tasks:
        evaluator = evaluator_class(model, tokenizer, **env_conf["train"])
    else:
        evaluator = evaluator_class(model, tokenizer, eval=None, tasks=test_conf)

    evaluator.evaluate()
