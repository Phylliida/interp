import torch
import random
import types
types.NoneType = type(None)
from functools import partial
from collections import defaultdict
import torch.nn.functional as F
import gc
import argparse
from acdc.acdc_utils import reset_network
from acdc.TLACDCExperiment import TLACDCExperiment
from acdc.docstring.utils import AllDataThings
from acdc.acdc_graphics import (
    show,
)

from acdc.acdc_utils import (
    kl_divergence,
)

try:
    import google.colab

    IN_COLAB = True
except Exception as e:
    IN_COLAB = False
    
from IPython import get_ipython
ipython = get_ipython()

from IPython.display import display

def prompts_to_tokens(prompts, model, device='cuda'):
    toks = []
    BOS_TOKEN = model.tokenizer.convert_tokens_to_ids(['bos_token'])[0]
    for prompt in prompts:
        toks.append([BOS_TOKEN] + model.tokenizer.encode(prompt))
    return [torch.tensor(t).to(device=device) for t in toks]


def answers_to_tokens(prompts, answers, model, device='cuda', debug=True):
    answers_tokens = []
    for i, answerset in enumerate(answers):
        answerset_tokens = []
        for answer in answerset:
            toks = model.tokenizer.encode(answer)
            if len(toks) > 1:
                if debug: print("warning, answer", answer, "from prompt", prompts[i], "is more than one token")
                if debug: print("using first token", model.to_string(torch.tensor([toks[0]])))
            answerset_tokens.append(toks[0])
        answers_tokens.append(torch.tensor(answerset_tokens))
    return answers_tokens

def filter_to_same_size(prompts, answers, model, debug=True):
    tokens = prompts_to_tokens(prompts, model=model, device='cpu')
    #answer_tokens_nospace = prompts_to_tokens([x.strip() for x in answers], model=model)
    #answer_tokens_space = prompts_to_tokens([" " + x.strip() for x in answers], model=model)
    sizes = defaultdict(lambda: 0)
    for toks in tokens:
        sizes[len(toks)] += 1
    if debug: print("prompt sizes", sizes)
    most_common_size, count_of_most_common = max(list(sizes.items()), key=lambda x: x[1])
    result_prompts, result_answers = [], []
    num_removed = 0
    for i, toks in enumerate(tokens):
        if len(toks) != most_common_size:
            if debug: print("ignoring prompt", model.to_str_tokens(toks), "with answers", answers[i])
            if debug: print("because it has size", len(toks), "but most common size is", most_common_size)
            num_removed += 1
        else:
            result_prompts.append(prompts[i])
            result_answers.append(answers[i])
    if num_removed > 0:
        if debug: print("removed", num_removed, "prompts that weren't the most common size")
    return result_prompts, result_answers

# if answer_token_trim is False, only keeps stuff with one token as answer
# it answer_token_trim is True, trims answers to just the first token
def fix_answers_with_one_token(prompts, answers, model, answer_token_trim=False, debug=True):
    result_prompts = []
    result_answers = []
    for prompt, anws in zip(prompts, answers):
        approved = True
        fixed_anws = []
        for answer in anws:
            answer_tokens = model.tokenizer.encode(answer)
            if len(answer_tokens) != 1:
                if answer_token_trim:
                    if debug: print(f"answer '{answer}' from prompt '{prompt}' has more than one token, trimming to first token")
                    if debug: print("tokens are", model.to_str_tokens(torch.tensor(answer_tokens)))
                    fixed_anws.append(model.to_string(torch.tensor([answer_tokens[0]])))
                else:
                    if debug: print(f"answer '{answer}' from prompt '{prompt}' has more than one token, ignoring this prompt")
                    if debug: print("tokens are", model.to_str_tokens(torch.tensor(answer_tokens)))
                    
                    approved = False
                    break
            else:
                fixed_anws.append(answer)
        anws = fixed_anws
        if approved:
            
            result_prompts.append(prompt)
            result_answers.append(anws)
    return result_prompts, result_answers 

# removes all prompts with answers that aren't one token
# then looks at all sizes and picks the size with the largest number of samples, and gets only those samples
def clean_dataset(prompts, answers, model, answer_token_trim=False, debug=True):
    prompts, answers = fix_answers_with_one_token(prompts=prompts, answers=answers, answer_token_trim=answer_token_trim, model=model, debug=debug)
    prompts, answers = filter_to_same_size(prompts, answers, model=model, debug=debug)
    return prompts, answers

# prints dataset and
# checks dataset against model
def test_dataset(prompts, answers, model, top_n=3, sample_n=5, debug=True):
    model.reset_hooks()
    tokens = prompts_to_tokens(prompts, model=model, device='cpu')
    answers_tokens = answers_to_tokens(prompts=prompts, answers=answers, model=model, device='cpu', debug=debug)
    if debug: print("\n\n-------")
    if debug: print("dataset")
    if debug: print("-------")
    failed_datapoints = []
    success_datapoints = []
    for i, (prompt, anws) in enumerate(zip(prompts, answers)):
        if debug: print(model.to_string(tokens[i]))
        logits = model(tokens[i].view(1, -1), return_type="logits")
        probs = F.softmax(logits[:, -1], dim=-1)
        besti = torch.argsort(-probs[0])
        has_first_place = False
        for j, answer in enumerate(anws):
            answer_token = answers_tokens[i][j]
            place = ((besti == answer_token).nonzero(as_tuple=True)[0])[0].item()
            if place == 0:
                has_first_place = True
            if debug: print("  ", str(model.to_str_tokens(answers_tokens[i][j])), "pr", probs[0,answer_token].item(), "place", place)
            # print("  ", str(model.to_str_tokens(torch.tensor(model.tokenizer.encode(answer)))))
        if debug: print(f"    top{top_n}")
        for j in range(top_n):
            tok = besti[j]
            if debug: print("    ", j, model.to_str_tokens([tok]), probs[0,tok].item())
        if has_first_place:
            success_datapoints.append(i)
        else:
            failed_datapoints.append(i)
            
    if debug: print("\n\n-----------")
    if debug: print("failed data")
    if debug: print("-----------")
    if debug:
        for i in failed_datapoints:
            print(model.to_string(tokens[i]))
            logits = model(tokens[i].view(1, -1), return_type="logits")
            probs = F.softmax(logits[:, -1], dim=-1)
            besti = torch.argsort(-probs[0])
            for j, answer in enumerate(anws):
                answer_token = answers_tokens[i][j]
                place = ((besti == answer_token).nonzero(as_tuple=True)[0])[0].item()
                print("  ", str(model.to_str_tokens(answers_tokens[i][j])), "pr", probs[0,answer_token].item(), "place", place)
                # print("  ", str(model.to_str_tokens(torch.tensor(model.tokenizer.encode(answer)))))
            print(f"    top{top_n}")
            for j in range(top_n):
                tok = besti[j]
                print("      ", j, model.to_str_tokens([tok]), probs[0,tok].item())
            print("    argmax completion")
            def sample_model(toks):
                return torch.argmax(model(torch.tensor(toks).view(1, -1), return_type="logits")[0,-1]).item()
            toks = list(tokens[i])
            for t in range(sample_n):
                toks += [sample_model(toks)]
            print("       \"" + model.to_string(torch.tensor(toks[len(tokens[i]):])) + "\"")
            print("      ", model.to_str_tokens(torch.tensor(toks[len(tokens[i]):])))
            
    if len(failed_datapoints) == 0:
        if debug: print("(no failed data)")
    if debug: print("\n\nmodel can do", len(prompts)-len(failed_datapoints), "/", len(prompts), "correctly")
    return success_datapoints
    

def parse_and_clean_dataset(dataset_generator, model, answer_token_trim=False, top_n=3, sample_n=7, debug=True):
    prompts, answers = zip(*list(dataset_generator())) # this weird syntax unzips into two lists
    prompts, answers = list(prompts), list(answers)
    # in case they pass in strings, this prevents us from reading the strings as a list of chars
    answers_to_list = []
    for answer in answers:
        if type(answer) is list: answers_to_list.append(answer)
        else: answers_to_list.append([answer])
    answers = answers_to_list
    prompts, answers = clean_dataset(prompts=prompts, answers=answers, model=model, answer_token_trim=answer_token_trim, debug=debug)
    success_indices = test_dataset(prompts=prompts, answers=answers, model=model, top_n=top_n, sample_n=sample_n, debug=debug)
    return prompts, answers, success_indices


def get_all_answer_tokens(answers, model, device='cuda'):
    all_answer_tokens = set()
    for answerset in answers:
        all_answer_tokens = all_answer_tokens | set([model.tokenizer.encode(answer)[0] for answer in answerset])
    return torch.tensor(sorted(list(all_answer_tokens)))

def custom_metric(logits, tokens, answers_tokens, model, all_answer_tokens, return_one_element: bool=True):
    BOS_TOKEN = model.tokenizer.convert_tokens_to_ids(['bos_token'])[0]
    probs = F.softmax(logits[:, -1], dim=-1)
    target_tokens = []
    scores = []
    for i in range(len(probs)):
        toks = tokens[i]
        if toks[0] == BOS_TOKEN:
            prompt = model.to_string(toks[1:])
        else:
            prompt = model.to_string(toks)
        answers_toks = answers_tokens[i]
        score = 0
        score += torch.sum(probs[i,answers_toks])*2 # *2 because we also subtract by this value
        score -= torch.sum(probs[i,all_answer_tokens])
        scores.append(score)
    scores = torch.tensor(scores)
    if return_one_element:
        return -torch.mean(scores)
    else:
        return -scores


# you should get the prompts and answers from parse_and_clean_dataset(dataset_generator=get_data_func, model=model, debug=debug)
def get_custom_things(model, metric_name, prompts, answers, patch_method='swap', device="cuda", seed=27, debug=True):
    answers_tokens = answers_to_tokens(prompts=prompts, answers=answers, model=model, device=device)
    all_answer_tokens = get_all_answer_tokens(answers, model=model, device=device)
    tokens = torch.stack(prompts_to_tokens(prompts, model=model, device=device))
    # seed
    random.seed(seed)
    torch.manual_seed(seed)
    
    # split data into 4 equal sized chunks
    # one for valid, one for valid patch, one for test, one for test patch
    if patch_method == 'split':
        num_examples = len(prompts)//4

        #tokens = prompts_to_tokens(prompts, model=model, device=device)

        validation_data = tokens[:num_examples]
        validation_answers = answers_tokens[:num_examples]
        validation_patch_data = tokens[num_examples:num_examples*2]

        test_data = tokens[num_examples*2:num_examples*3]
        test_answers = answers_tokens[num_examples*2:num_examples*3]
        test_patch_data = tokens[num_examples*3:num_examples*4]
    # split data into 2 equal sized chunks
    # one for valid, one for test
    # patch for valid is shuffled valid list (so a random different sample from valid)
    # patch for test is shuffled test list (so a random different sample from test)
    elif patch_method == 'shuffle':
        num_examples = len(prompts)//2
        validation_data = tokens[:num_examples]
        validation_answers = answers_tokens[:num_examples]
        validation_patch_data = validation_data.clone()
        inds = list(range(num_examples))
        random.shuffle(inds)
        validation_patch_data[inds] = validation_data
        
        test_data = tokens[num_examples:num_examples*2]
        test_answers = answers_tokens[num_examples:num_examples*2]
        test_patch_data = tokens[num_examples:num_examples*2].clone()
        inds = list(range(num_examples))
        random.shuffle(inds)
        test_patch_data[inds] = tokens[num_examples:num_examples*2]
        
    print("num examples", num_examples)
    
    # modified from ACDC colab
    with torch.no_grad():
        base_logits = model(tokens)[:, -1, :]
        base_logprobs = F.log_softmax(base_logits, dim=-1)
        if patch_method == 'shuffle':
            base_validation_logprobs = base_logprobs[:num_examples]
            base_test_logprobs = base_logprobs[num_examples:]
        elif patch_method == 'split':
            base_validation_logprobs = base_logprobs[:num_examples]
            base_test_logprobs = base_logprobs[num_examples*2:num_examples*3]

    # from ACDC colab
    if metric_name == "custom":
        validation_metric = partial(custom_metric, tokens=validation_data, answers_tokens=validation_answers, model=model, all_answer_tokens=all_answer_tokens)
    elif metric_name == "kl_div":
        validation_metric = partial(
            kl_divergence,
            base_model_logprobs=base_validation_logprobs,
            mask_repeat_candidates=None,
            last_seq_element_only=True,
        )
    else:
        raise ValueError(f"Unknown metric {metric_name}")

    test_metrics = {
        "custom": partial(custom_metric, tokens=test_data, answers_tokens=test_answers, model=model, all_answer_tokens=all_answer_tokens),
        "kl_div": partial(
            kl_divergence,
            base_model_logprobs=base_test_logprobs,
            mask_repeat_candidates=None,
            last_seq_element_only=True,
        ),
    }

    return num_examples, prompts, answers, AllDataThings(
        tl_model=model,
        validation_metric=validation_metric,
        validation_data=validation_data,
        validation_labels=None,
        validation_mask=None,
        validation_patch_data=validation_patch_data,
        test_metrics=test_metrics,
        test_data=test_data,
        test_labels=None,
        test_mask=None,
        test_patch_data=test_patch_data)


def setup_acdc(things, label, threshold=0.71, device='cuda'):
    # much of this is from the colab
    validation_metric = things.validation_metric # metric we use (e.g KL divergence)
    toks_int_values = things.validation_data # clean data x_i
    toks_int_values_other = things.validation_patch_data # corrupted data x_i'
    tl_model = things.tl_model # transformerlens model
    TASK = label
    
    #reset_network(TASK, device, tl_model)
    
    tl_model.reset_hooks()
    # Save some mem
    gc.collect()
    torch.cuda.empty_cache()
    
    ZERO_ABLATION = True
    abs_value_threshold = True
    second_metric = None
    tl_model.reset_hooks()

    things.tl_model.do_warnings = False
    
    return TLACDCExperiment(
        model=tl_model,
        threshold=threshold,
        using_wandb=False,
        wandb_entity_name="",
        wandb_project_name="",
        wandb_run_name="",
        wandb_group_name="",
        wandb_notes="",
        wandb_dir="/tmp/wandb",
        wandb_mode="online",
        wandb_config=None,
        zero_ablation=ZERO_ABLATION,
        abs_value_threshold=abs_value_threshold,
        ds=toks_int_values,
        ref_ds=toks_int_values_other,
        metric=validation_metric,
        second_metric=second_metric,
        verbose=True,
        indices_mode="normal",
        names_mode="normal",
        corrupted_cache_cpu=True,
        hook_verbose=False,
        online_cache_cpu=True,
        add_sender_hooks=True,
        use_pos_embed=False,
        add_receiver_hooks=False,
        remove_redundant=False,
        show_full_index=False,
    )

def run_acdc(exp):
    
    for i in range(args.max_num_epochs):
        exp.step(testing=False)

        show(
            exp.corr,
            f"ims/img_new_{i+1}.png",
            show_full_index=False,
        )

        if IN_COLAB or ipython is not None:
            # so long as we're not running this as a script, show the image!
            display(Image(f"ims/img_new_{i+1}.png"))

        print(i, "-" * 50)
        print(exp.count_no_edges())

        if i == 0:
            exp.save_edges("edges.pkl")

        if exp.current_node is None or SINGLE_STEP:
            break

    exp.save_edges("another_final_edges.pkl")
