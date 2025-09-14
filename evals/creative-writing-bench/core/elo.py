import os
import logging
import math
import json
import random
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.file_io import load_json_file, save_json_file
import glicko2

##############################################
# Constants & Globally Ignored Prompt IDs
##############################################
LENGTH_TRUNCATION_CHARS = 4000
IGNORE_FOR_ELO = [
    "119","45","124","123","125","208","212","210",
    "197","207","209","215","200","196","216","217"
]
IGNORE_FOR_ELO = [
"5","16","20","21","26","28","30"
]

##############################################
# If your code had negative_criteria logic:
##############################################
def invert_if_negative(metric: str, val: float, neg_list: List[str]) -> float:
    """
    Example. If 'metric' is in neg_list, invert the 0-10 scale by doing 10 - val.
    Adjust to your real system's logic if needed.
    """
    if metric in neg_list:
        return 20.0 - val
    return val

##############################################
# Check if we ignore a given prompt ID
##############################################
def should_ignore_prompt(prompt_id: str) -> bool:
    """Check if prompt_id is in IGNORE_FOR_ELO."""
    if "__" in prompt_id:
        raw_id = prompt_id.split("__",1)[0]
        return raw_id in IGNORE_FOR_ELO
    return prompt_id in IGNORE_FOR_ELO

##############################################
# interpret_pairwise_result (original logic)
##############################################
def interpret_pairwise_result(result_dict):
    """
    Return (outcome_for_A, plus_for_A, plus_for_B) in {0,0.5,1}, plus_for_A, plus_for_B as int tallies.
    This is unchanged from your original code.
    """
    if not result_dict:
        return 0.5, 0, 0

    a_score = 0
    b_score = 0
    for key, val in result_dict.items():
        if key in ["improvement_suggestions", "theory_of_mind"]:
            continue
        # "A0493" => means model A is better for that dimension
        if "A0493" in val:
            plus_count = val.count('+')
            if plus_count > 0:
                a_score += plus_count
            if key in ["avoids_poetic_overload", "coherence", "avoids_verbosity"]:
                # punish these
                b_score -= plus_count
        elif "A0488" in val:
            plus_count = val.count('+')
            if plus_count > 0:
                b_score += plus_count
            if key in ["avoids_poetic_overload", "coherence", "avoids_verbosity"]:
                # punish these
                a_score -= plus_count

    if a_score > b_score:
        return 1.0, a_score, b_score
    elif b_score > a_score:
        return 0.0, a_score, b_score
    else:
        return 0.5, a_score, b_score

##############################################
# Margin-based fraction mapping
##############################################
def custom_blend(x: float, linear_gradient=5, sigmoid_power=0.75, transition_start=0.0, transition_end=0.11) -> float:
    """
    Transforms a value in [0,1] by blending a linear slope with a sigmoid curve
    around [transition_start..transition_end].
    """
    x = max(0.0, min(1.0, x))

    # Linear portion
    linear = linear_gradient * x
    # Sigmoid portion
    k = 3
    sig = (1.0 - math.exp(-k * (x**sigmoid_power))) / (1.0 - math.exp(-k))

    # Blend factor (smoothstep)
    if x <= transition_start:
        blend = 0.0
    elif x >= transition_end:
        blend = 1.0
    else:
        t = (x - transition_start)/(transition_end - transition_start)
        blend = t*t*(3-2*t)

    return (1.0 - blend)*linear + blend*sig


def custom_blend(x: float, linear_gradient=5, sigmoid_power=0.75, transition_start=0.0, transition_end=0.11) -> float:
    return x

def deduplicate_comparisons(comps, model_name):
    """
    Takes a list of comparison dicts (like those in pairwise_comparisons)
    and returns a new list with duplicates removed.
    """
    # Build a set of JSON-serialized signatures, or your own custom signature
    seen_signatures = set()
    unique_comps = []
    for c in comps:
        if c['pair']['test_model'] != model_name:
            # filter out comparisons that aren't for this test model
            continue
        # Turn the dict into something hashable
        sig = json.dumps(c, sort_keys=True)
        if sig not in seen_signatures:
            seen_signatures.add(sig)
            unique_comps.append(c)
    return unique_comps

def global_deduplicate_comparisons(comps):
    """
    Deduplicate across *all* models in a single pass. 
    Returns a new list with duplicates removed.
    """
    seen = set()
    unique = []
    for c in comps:
        # Build a signature that captures item_id, test/neighbor, fraction, etc.
        # Also consider plus_for_test / plus_for_other if you want the signature
        # to reflect the raw tallies. For Glicko, fraction_for_test is the key.
        
        # If you always rely on fraction_for_test, you can do:
        sig = (
            c.get("item_id", ""),
            c.get("pair", {}).get("test_model", ""),
            c.get("pair", {}).get("neighbor_model", ""),
            round(c.get("fraction_for_test", 0.5), 3),
        )
        if sig not in seen:
            seen.add(sig)
            unique.append(c)
    return unique

def compute_fraction_for_test(outcome_for_test: float, plus_for_test: int, plus_for_other: int):
    """
    1) plus_diff = abs(plus_for_test - plus_for_other)
    2) normalized = plus_diff / 45
    3) diff_blended = custom_blend(normalized)
    4) margin = diff_blended/2 + 0.5  => in [0.5..1]
    5) if outcome_for_test=1 => fraction_for_test=margin
       if outcome_for_test=0 => fraction_for_test=1 - margin
       if outcome_for_test=0.5 => fraction_for_test=0.5
    """
    diff = abs(plus_for_test - plus_for_other)
    diff_norm = diff/45.0
    diff_blend = custom_blend(diff_norm,5,0.75,0.0,0.11)
    margin = diff_blend/2 + 0.5  # [0.5..1.0]

    if outcome_for_test == 0.5:
        final_fraction = 0.5
    elif outcome_for_test == 1.0:
        final_fraction = margin
    else: # outcome_for_test==0.0
        final_fraction = 1.0 - margin

    return final_fraction, diff, diff_norm, diff_blend

##############################################
# do_pairwise_judge (restored to original approach)
##############################################
def do_pairwise_judge(
    textA: str,
    textB: str,
    prompt_id: str,
    pairwise_prompt_template: str,
    writing_prompts: Dict[str, Any],
    judge_model: str,
    api_clients: Dict[str, Any],
    item_order_idx=None
):
    # If prompt_id is something like "77_3_1", extract the actual prompt ID part ("77")
    if "_" in prompt_id:
        raw_prompt_id = prompt_id.split("_", 1)[0]
    else:
        raw_prompt_id = prompt_id
    writing_prompt = writing_prompts[raw_prompt_id]["writing_prompt"]
    
    final_prompt = pairwise_prompt_template.replace("{writing_prompt}", writing_prompt)
    final_prompt = final_prompt.replace("{model_a_analysis}", textA)
    final_prompt = final_prompt.replace("{model_b_analysis}", textB)    
    response = ""
    try:        
        response = api_clients["judge"].generate(
            judge_model, final_prompt, temperature=0.0, max_tokens=16000, include_seed=True, min_p=None
        )        
        start = response.find("{")
        end = response.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {"_item_order_idx": item_order_idx} if item_order_idx is not None else {}
        json_str = response[start:end+1]
        result = json.loads(json_str)
        if item_order_idx is not None:
            result["_item_order_idx"] = item_order_idx
        return result
    except Exception as e:
        logging.warning(response)
        logging.warning(f"Pairwise judge error: {str(e)}")
        return {"_item_order_idx": item_order_idx} if item_order_idx is not None else {}

# Normalises Elo scores in the same way as the EQ-Bench creative writing leaderboard
def normalize_elo_scores(raw_scores, anchor_models=None):
    """
    Normalize ELO scores by anchoring specific models to predefined values.
    
    Args:
        raw_scores (dict): Dictionary of model names to raw ELO scores
        anchor_models (dict, optional): Dictionary mapping model names to their anchor values.
            Default: {'deepseek/deepseek-r1': 1500, 'llama-3.2-1b-instruct': 200}
            
    Returns:
        dict: Dictionary of model names to normalized ELO scores
    """
    if anchor_models is None:
        anchor_models = {
            'deepseek/deepseek-r1': 1500,
            'meta-llama/llama-3.2-1b-instruct': 200
        }
    
    # First check if we have at least two anchor models in our raw scores
    valid_anchors = {k: v for k, v in anchor_models.items() if k in raw_scores}
    
    if len(valid_anchors) < 2:
        logging.warning(f"Not enough anchor models found in scores. "
                       f"Found {len(valid_anchors)} of {len(anchor_models)}. "
                       f"Returning raw scores.")
        return {k: v for k, v in raw_scores.items()}
    
    # Get first two valid anchors to calculate normalization
    anchor_items = list(valid_anchors.items())
    model_a, target_a = anchor_items[0]
    model_b, target_b = anchor_items[1]
    
    # Calculate the scale and shift for the linear transformation
    raw_a = raw_scores[model_a]
    raw_b = raw_scores[model_b]
    
    # Avoid division by zero
    if raw_a == raw_b:
        scale = 1.0
    else:
        scale = (target_a - target_b) / (raw_a - raw_b)
    
    shift = target_a - (scale * raw_a)
    
    # Apply the transformation to all scores
    normalized_scores = {model: (score * scale + shift) for model, score in raw_scores.items()}
    
    return normalized_scores

##############################################
# Concurrency-based partial matching
##############################################
def _judge_items_in_parallel(
    test_model_name: str,
    neighbor_model_name: str,
    test_model_items: Dict[str, str],
    neighbor_model_items: Dict[str, str],
    test_model_scores: Dict[str, float],
    neighbor_model_scores: Dict[str, float],
    concurrency: int,
    pairwise_prompt_template: str,
    writing_prompts: Dict[str, Any],
    judge_model: str,
    api_clients: Dict[str, Any],
    max_items=15
):
    """
    Overlapping item_ids: we do forward & reverse comparisons for up to 'max_items' overlap.
    Returns:
      (avg_score_for_test, comparisons_list, sum_pluses_test, sum_pluses_other)
    """
    # Overwrite concurrency for parallel calls
    concurrency = 500
    max_items = 200

    overlap_ids = sorted(set(test_model_items.keys()) & set(neighbor_model_items.keys()))
    if not overlap_ids:
        return 0.5, [], 0, 0

    chosen_ids = overlap_ids[:max_items]
    tasks = {}
    comparisons = []
    n_items = len(chosen_ids)
    #print('!!', len(chosen_ids))
    #print(chosen_ids)

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        for pid in chosen_ids:
            pieceA = test_model_items[pid][:LENGTH_TRUNCATION_CHARS]
            pieceB = neighbor_model_items[pid][:LENGTH_TRUNCATION_CHARS]

            # Forward: (test vs neighbor)
            fwd_future = executor.submit(
                do_pairwise_judge, 
                pieceA, 
                pieceB, 
                pid,
                pairwise_prompt_template,
                writing_prompts,
                judge_model,
                api_clients
            )
            tasks[fwd_future] = (pid, "forward", len(pieceA), len(pieceB))

            # Reverse: (neighbor vs test)
            rev_future = executor.submit(
                do_pairwise_judge, 
                pieceB, 
                pieceA, 
                pid,
                pairwise_prompt_template,
                writing_prompts,
                judge_model,
                api_clients
            )
            tasks[rev_future] = (pid, "reversed", len(pieceB), len(pieceA))

    total_score_for_test = 0.0
    total_comps = 0
    sum_pluses_test = 0
    sum_pluses_neighbor = 0

    for fut, (pid, direction, len_a, len_b) in tasks.items():
        try:
            result = fut.result()
            if result:
                #print(direction)
                if direction == "forward":                
                    outcome_for_test_model, test_model_plus_count, neighbor_plus_count = interpret_pairwise_result(result)
                    fracTest, diff, diff_norm, diff_blend = compute_fraction_for_test(outcome_for_test_model,test_model_plus_count,neighbor_plus_count)
                    comparisons.append({
                        "item_id": pid,
                        "pair": {
                            "test_model": test_model_name,
                            "neighbor_model": neighbor_model_name
                        },
                        "order": "A0493:test / A0488:other",
                        "judge_response": result,
                        "outcome_for_test_model": outcome_for_test_model,
                        "plus_for_test": test_model_plus_count,
                        "plus_for_other": neighbor_plus_count,
                        "item_length": {
                            "test_model": len_a,
                            "neighbor_model": len_b
                        },
                        "creative_writing_rubric_scores": {
                            test_model_name: test_model_scores.get(pid, 0.0),
                            neighbor_model_name: neighbor_model_scores.get(pid, 0.0)
                        },
                        "plus_diff": diff,
                        "plus_diff_normalized": diff_norm,
                        "plus_diff_blended": diff_blend,
                        "fraction_for_test": fracTest
                    })
                    total_score_for_test += outcome_for_test_model
                    sum_pluses_test += test_model_plus_count
                    sum_pluses_neighbor += neighbor_plus_count

                else:
                    # "reversed": A0493=neighbor / A0488:test
                    outcome_for_neighbor_model, neighbor_plus_count, test_model_plus_count = interpret_pairwise_result(result)
                    if outcome_for_neighbor_model == 1.0:
                        outcome_for_test_model = 0.0
                    elif outcome_for_neighbor_model == 0.0:
                        outcome_for_test_model = 1.0
                    else:
                        outcome_for_test_model = 0.5
                    fracTest, diff, diff_norm, diff_blend = compute_fraction_for_test(outcome_for_test_model,test_model_plus_count,neighbor_plus_count)
                    

                    comparisons.append({
                        "item_id": pid,
                        "pair": {
                            "neighbor_model": neighbor_model_name,
                            "test_model": test_model_name
                        },
                        "order": "A0493:other / A0488:test",
                        "judge_response": result,
                        "outcome_for_test_model": outcome_for_test_model,
                        "plus_for_test": test_model_plus_count,
                        "plus_for_other": neighbor_plus_count,
                        "item_length": {
                            "test_model": len_b,
                            "neighbor_model": len_a
                        },
                        "creative_writing_rubric_scores": {
                            test_model_name: test_model_scores.get(pid, 0.0),
                            neighbor_model_name: neighbor_model_scores.get(pid, 0.0)
                        },
                        "plus_diff": diff,
                        "plus_diff_normalized": diff_norm,
                        "plus_diff_blended": diff_blend,
                        "fraction_for_test": fracTest
                    })
                    total_score_for_test += outcome_for_test_model
                    sum_pluses_test += test_model_plus_count
                    sum_pluses_neighbor += neighbor_plus_count

                total_comps += 1
            else:
                comparisons.append({
                    "item_id": pid,
                    "pair": {
                        "test_model": test_model_name,
                        "neighbor_model": neighbor_model_name
                    },
                    "order": "A0493:test / A0488:other",
                    "judge_response": result,                        
                    "item_length": {
                        "test_model": len_a,
                        "neighbor_model": len_b
                    },
                    "creative_writing_rubric_scores": {
                        test_model_name: test_model_scores.get(pid, 0.0),
                        neighbor_model_name: neighbor_model_scores.get(pid, 0.0)
                    },
                    "error": "Empty judge response"                        
                })

            

        except Exception as e:
            logging.warning(f"Judging error with item {pid}, direction {direction}: {str(e)}")

    if total_comps == 0:
        return 0.5, [], 0, 0

    avg_score_for_test = total_score_for_test / total_comps
    return avg_score_for_test, comparisons, sum_pluses_test, sum_pluses_neighbor

def solve_with_glicko(
    all_models: List[str],
    pairwise_comparisons: List[Dict],
    initial_ratings: Dict[str,float],
    debug=False
) -> Dict[str,float]:
    """
    Drop-in replacement for solve_bradley_terry_fractional_distance_weighted that uses Glicko2.

    Glicko usage:
      1) Each model is a glicko2.Player(rating=..., rd=..., vol=...).
         - If 'initial_ratings' has an entry, we use that as the starting rating
           (converted to Glicko scale if you like, or 1500-based directly).
      2) We gather match results from 'pairwise_comparisons', each having
         'fraction_for_test' in [0..1]. We bin that fraction => repeated wins/losses.
      3) We do ONE rating period, i.e. each model accumulates all matches at once,
         then calls 'update_player(rating_list, rd_list, outcome_list)' once.
      4) Return final Glicko rating for each model.

    All parameters except (all_models, pairwise_comparisons, initial_ratings) are ignored.
    This keeps the same signature as your original function so you can swap it in easily.
    """

    # de-dupe
    pairwise_comparisons = global_deduplicate_comparisons(pairwise_comparisons)

    ################################################################
    # PRE-PROCESS STEP: Match up bidirectional pairs and
    # average their fraction_for_test values for downstream use.
    # Pairs are assumed to occur in adjacent positions in the list.
    # If we don't find a proper match, we discard the orphan.
    ################################################################
    paired_comparisons = []
    i = 0
    while i < len(pairwise_comparisons) - 1:
        c1 = pairwise_comparisons[i]
        c2 = pairwise_comparisons[i+1]
        # Check same item_id and reversed roles
        if (
            c1.get("item_id") == c2.get("item_id")
            and c1["pair"]["test_model"] == c2["pair"]["test_model"]
            and c1["pair"]["neighbor_model"] == c2["pair"]["neighbor_model"]
            and "fraction_for_test" in c1
            and "fraction_for_test" in c2
        ):
            avg_frac = 0.5 * (c1["fraction_for_test"] + c2["fraction_for_test"])
            paired_comparisons.append({
                "item_id": c1["item_id"],
                "pair": {
                    "test_model": c1["pair"]["test_model"],
                    "neighbor_model": c1["pair"]["neighbor_model"]
                },
                "fraction_for_test": avg_frac
            })
            i += 2
        else:
            # Discard the orphan c1 and move on
            i += 1

    # Now we work with these paired comparisons only
    pairwise_comparisons = paired_comparisons

    #############################
    # 1) Setup Glicko players
    #############################
    SHIFT_AMOUNT = 0  # example shift if you want to move 1200 -> 1500

    glicko_players = {}
    for m in all_models:
        start_rating = initial_ratings.get(m, 1200.0)
        start_rating += SHIFT_AMOUNT
        p = glicko2.Player(rating=start_rating, rd=350, vol=0.06)
        glicko_players[m] = p

    #############################
    # 2) Binning logic function
    #############################
    def bin_fraction(frac, bin_size=10):
        frac = max(0.0, min(1.0, frac))
        w_test = int(round(frac * bin_size))
        w_other = bin_size - w_test
        return w_test, w_other
    

    def bin_fraction(frac, bin_size=4):
        """
        Bins 'frac' into a discrete (wins_for_test, wins_for_other) out of
        2*bin_size total "matches," ensuring:
        - Exactly 0.5 => an equal split (bin_size each).
        - frac < 0.5 => strictly fewer wins_for_test than bin_size.
        - frac > 0.5 => strictly more wins_for_test than bin_size.
        """
        # Clamp fraction to [0,1]
        frac = max(0.0, min(1.0, frac))

        # Special-case exactly 0.5 for an equal split
        if abs(frac - 0.5) < 1e-9:
            return bin_size, bin_size

        # Total matches is 2 * bin_size
        total = 2 * bin_size

        if frac < 0.5:
            # Map frac in [0, 0.5) -> [0,1)
            # test_wins should be < bin_size
            frac_scaled = frac / 0.5  # scale from 0..0.5 up to 0..1
            test_wins = math.floor(frac_scaled * bin_size)  # 0..(bin_size-1)
            other_wins = total - test_wins
            return test_wins, other_wins
        else:
            # frac > 0.5
            # Map frac in (0.5, 1.0] -> (0,1]
            # test_wins should be > bin_size
            frac_scaled = (frac - 0.5) / 0.5  # scale from 0.5..1 to 0..1
            # We want strictly more than bin_size if frac>0.5,
            # so use ceil(...) to ensure at least 1 extra above bin_size
            extra = math.ceil(frac_scaled * bin_size)
            test_wins = bin_size + extra  # (bin_size+1)..2*bin_size
            other_wins = total - test_wins
            return test_wins, other_wins


    BIN_SIZE = 4

    #############################
    # 3) Accumulate match data
    #############################
    match_data = {}
    for m in all_models:
        match_data[m] = ([], [], [])

    for c in pairwise_comparisons:
        frac = c.get("fraction_for_test", 0.5)
        test_m = c["pair"]["test_model"]
        neigh_m = c["pair"]["neighbor_model"]
        w_test, w_other = bin_fraction(frac, BIN_SIZE)
        #print(frac, w_test, w_other)

        # test_m beats neigh_m for w_test times
        for _ in range(w_test):
            (rt_t, rd_t, out_t) = match_data[test_m]
            rt_t.append(glicko_players[neigh_m].getRating())
            rd_t.append(glicko_players[neigh_m].getRd())
            out_t.append(1.0)

            (rt_n, rd_n, out_n) = match_data[neigh_m]
            rt_n.append(glicko_players[test_m].getRating())
            rd_n.append(glicko_players[test_m].getRd())
            out_n.append(0.0)

        # neigh_m beats test_m for w_other times
        for _ in range(w_other):
            (rt_n, rd_n, out_n) = match_data[neigh_m]
            rt_n.append(glicko_players[test_m].getRating())
            rd_n.append(glicko_players[test_m].getRd())
            out_n.append(1.0)

            (rt_t, rd_t, out_t) = match_data[test_m]
            rt_t.append(glicko_players[neigh_m].getRating())
            rd_t.append(glicko_players[neigh_m].getRd())
            out_t.append(0.0)

    #############################
    # 4) Perform the rating updates (one rating period)
    #############################
    for m in all_models:
        (rating_list, rd_list, outcome_list) = match_data[m]
        if len(rating_list) == 0:
            glicko_players[m].did_not_compete()
        else:
            glicko_players[m].update_player(rating_list, rd_list, outcome_list)

    #############################
    # 5) Build final map
    #############################
    final_map = {}
    for m in all_models:
        final_map[m] = glicko_players[m].getRating() - SHIFT_AMOUNT

    if debug:
        print("[Glicko] Final Ratings (after single rating period):")
        for m in sorted(all_models, key=lambda x: final_map[x]):
            print(f"  {m:20s} => {final_map[m]:.2f}")

    return final_map

def find_optimal_position(
    test_model,
    other_models,
    existing_analyses,
    partial_match_test_vs,
    solve_elo,
    ladder_sample_size=7,
    neighbor_depth=2
):
    """
    Find the optimal position for test_model by:
    1. Using existing Elo ratings for initial position estimation
    2. Testing neighbors within specified depth
    3. Running Elo solver after each round
    4. Repeating until position stabilizes
    
    Returns:
        tuple: (final_position, all_comparisons_done)
    """
    bracket_comparisons = []
    tested_indices = set()
    
    # Extract ELO ratings for models where both ELO and rubric score exist
    model_info = []
    for m_id, info in existing_analyses.items():
        c_score = info.get("creative_writing_rubric_score_agg", None)
        c_elo = info.get("elo", None)
        if c_elo is not None and c_score is not None:
            model_info.append((m_id, float(c_elo)))
    
    # Sort by ELO rating
    model_info.sort(key=lambda x: x[1])
    
    # Map model names to their positions in the sorted list
    model_positions = {m: i for i, (m, _) in enumerate(model_info)}
    
    # Find test_model's initial position 
    if test_model in model_positions:
        test_position_in_full_list = model_positions[test_model]
    else:
        # If test_model not found with valid ELO, start in the middle
        logging.debug(f"[NEIGHBOR] Test model {test_model} not found with valid ELO, starting in middle")
        test_position_in_full_list = len(model_info) // 2
    
    # Convert to position in other_models (not including test_model)
    initial_pos_in_others = 0
    for m, _ in model_info:
        if m != test_model and m in other_models and model_positions.get(m, 0) < test_position_in_full_list:
            initial_pos_in_others += 1
    
    # Ensure initial_pos is within bounds
    initial_pos_in_others = min(max(0, initial_pos_in_others), len(other_models))
    
    logging.debug(f"[NEIGHBOR] Initial position from ELO ratings: {initial_pos_in_others} of {len(other_models)}")
    
    # If we have no other models, return position 0
    if not other_models:
        logging.debug("[NEIGHBOR] No other models to compare with")
        return 0, bracket_comparisons
    
    # Starting position for iterations
    current_pos = initial_pos_in_others
    prev_pos = -1  # Initialize to a different value to ensure first iteration
    iteration = 0
    max_iterations = 5  # Safeguard against infinite loops
    
    # Define a wrapper for partial_match_test_vs
    def run_matchups_with_idx(idx):
        if idx in tested_indices:
            logging.debug(f"[NEIGHBOR] Skipping already tested index {idx}")
            return []
        
        tested_indices.add(idx)
        neighbor = other_models[idx]
        avg_test_sc, comps, neigh_name = partial_match_test_vs(idx, ladder_sample_size)
        logging.debug(f"[NEIGHBOR] Matchup with {neigh_name} at idx={idx}, score={avg_test_sc:.3f}")
        return comps
    
    while current_pos != prev_pos and iteration < max_iterations:
        iteration += 1
        prev_pos = current_pos
        
        logging.debug(f"[NEIGHBOR] Iteration {iteration}, current position: {current_pos}")
        
        # Determine which neighbors to test
        new_comparisons = []
        for offset in range(-neighbor_depth, neighbor_depth + 1):
            idx = current_pos + offset
            if 0 <= idx < len(other_models) and idx not in tested_indices:
                comps = run_matchups_with_idx(idx)
                new_comparisons.extend(comps)
        
        if not new_comparisons:
            logging.debug(f"[NEIGHBOR] All neighbors within depth {neighbor_depth} already tested. Stopping.")
            break
        
        # Add new comparisons to overall results
        bracket_comparisons.extend(new_comparisons)
        
        # Run elo solver with all comparisons so far to update position
        current_pos, _ = solve_elo(bracket_comparisons)
        logging.debug(f"[NEIGHBOR] After Elo solve, new position: {current_pos}")
    
    logging.debug(f"[NEIGHBOR] Final position after {iteration} iterations: {current_pos}")
    
    return current_pos, bracket_comparisons

def interpolate_elo_from_rubric_scores(model_name, best_val, existing_analyses):
    """
    Interpolate an ELO rating for a model based on its rubric score relative to other models.
    Only used when there are at least two other models with valid scores.
    
    Args:
        model_name: The model to calculate ELO for
        best_val: The model's creative_writing_rubric_score_agg
        existing_analyses: Dictionary of all model analyses
        
    Returns:
        float: Interpolated ELO rating
    """
    # Count models other than the test model
    other_models = [m for m in existing_analyses.keys() if m != model_name]
    
    # If 0 or 1 other models, use default 1200
    if len(other_models) <= 1:
        return 1200.0
    
    # Build a list of (model, rubric_score, elo) tuples for models with valid data
    model_data = []
    for m in existing_analyses:
        info = existing_analyses[m]
        rubric_score = info.get("creative_writing_rubric_score_agg")
        elo_rating = info.get("elo")
        
        if rubric_score is not None and elo_rating is not None:
            model_data.append((m, float(rubric_score), float(elo_rating)))
    
    # Sort by rubric score
    model_data.sort(key=lambda x: x[1])
    
    # If we don't have at least 2 models with valid data, use default
    if len(model_data) < 2:
        return 1200.0
    
    # Find where our model's score would fit in the sorted list
    insert_index = 0
    for i, (_, score, _) in enumerate(model_data):
        if best_val <= score:
            break
        insert_index = i + 1
    
    # Handle edge cases
    if insert_index == 0:
        # Our model has the lowest score, use the lowest ELO
        return model_data[0][2]
    elif insert_index >= len(model_data):
        # Our model has the highest score, use the highest ELO
        return model_data[-1][2]
    else:
        # Interpolate between the two neighboring models
        lower_model, lower_score, lower_elo = model_data[insert_index-1]
        upper_model, upper_score, upper_elo = model_data[insert_index]
        
        # Avoid division by zero
        if upper_score == lower_score:
            return (lower_elo + upper_elo) / 2
        
        # Calculate proportional position between the two scores
        ratio = (best_val - lower_score) / (upper_score - lower_score)
        
        # Interpolate ELO
        interpolated_elo = lower_elo + ratio * (upper_elo - lower_elo)
        
        return interpolated_elo
    

##############################################
# The main function: run_elo_analysis_creative
##############################################
def run_elo_analysis_creative(
    run_key: str,
    elo_results_file: str,
    test_model: str,
    judge_model: str,
    api_clients: Dict[str,Any],
    writing_prompts: Dict[str,Any],
    concurrency: int=4,
    pairwise_prompt_file: str="data/pairwise_prompt.txt",
    negative_criteria: List[str]=[],
    creative_bench_runs_file: str=None,
    max_items_per_model: int=500,
    ladder_sample_size: int=7
):
    """
    Drop-in replacement that:
      1) Loads/aggregates iteration-level data for the run.
      2) If there's only 1 model total, no pairwise done.
      3) Otherwise, do:
         (a) binary-search bracket with "sparse sampling" (like quick ladder).
         (b) comprehensive sampling with immediate neighbors (like old code).
         (c) if new bracket emerges, do step b again on newly adjacent neighbors, skipping duplicates.
         (d) run a global Bradley–Terry solve using margin-based fraction.
    """

    # 1) load or init the elo results
    if not os.path.exists(elo_results_file):
        existing_analyses = {}
        logging.info(f"Creating new ELO store file: {elo_results_file}")
    else:
        existing_analyses = load_json_file(elo_results_file)

    # 2) load creative bench data
    if not os.path.exists(creative_bench_runs_file):
        logging.warning(f"No creative bench file at {creative_bench_runs_file}")
        return
    run_data_all = load_json_file(creative_bench_runs_file)
    if run_key not in run_data_all:
        logging.warning(f"run_key={run_key} not found in {creative_bench_runs_file}; cannot do ELO.")
        return
    run_data = run_data_all[run_key]
    tasks_dict = run_data.get("creative_tasks",{})
    if not tasks_dict:
        logging.info(f"No creative_tasks found for run={run_key}. skipping.")
        return

    # 2b) read pairwise_prompt_template
    if not os.path.exists(pairwise_prompt_file):
        logging.info(f"No pairwise prompt file at {pairwise_prompt_file}, skipping ELO.")
        return
    with open(pairwise_prompt_file,'r',encoding='utf-8') as f:
        pairwise_prompt_template = f.read()

    # 3) gather iteration-level data
    temp_data = {}
    for it_str, prompt_map in tasks_dict.items():
        for pid, tinfo in prompt_map.items():
            if should_ignore_prompt(pid):
                continue
            status = tinfo.get("status")
            if status not in ["completed","judged"]:
                continue
            model_name = tinfo.get("test_model","unknown_model")
            if model_name not in temp_data:
                temp_data[model_name] = {}
            if it_str not in temp_data[model_name]:
                temp_data[model_name][it_str] = {
                    "items": {},
                    "scores_accum": 0.0,
                    "scores_count": 0,
                    "item_scores": {}
                }

            # accumulate text & judge scores
            block_sum = 0.0
            block_count=0
            combined_text= ""
            res_mod = tinfo.get("results_by_modifier",{})
            for mod_seed, block in res_mod.items():
                txt = block.get("model_response","").strip()
                if txt:
                    combined_text+= txt+"\n"
                j_scores = block.get("judge_scores",{})
                for met,val in j_scores.items():
                    if isinstance(val,(int,float)):
                        val2 = invert_if_negative(met,val,negative_criteria)
                        block_sum+= val2
                        block_count+=1
            if combined_text:
                temp_data[model_name][it_str]["items"][pid] = combined_text
            temp_data[model_name][it_str]["scores_accum"]+= block_sum
            temp_data[model_name][it_str]["scores_count"]+= block_count
            avg_item_score = round(block_sum/block_count,2) if block_count>0 else 0.0
            temp_data[model_name][it_str]["item_scores"][pid] = avg_item_score

    # 3b) store into existing_analyses
    for m_name, iteration_map in temp_data.items():
        best_iter = None
        best_val = float("-inf")
        # Calculate the true aggregate score across all iterations
        total_score = 0
        total_count = 0
        
        for it_id, it_data in iteration_map.items():
            acc = it_data["scores_accum"]
            ccount = it_data["scores_count"]
            it_avg = acc/ccount if ccount>0 else 0.0
            it_data["creative_writing_rubric_score_iter"] = round(it_avg,2)
            total_score += acc
            total_count += ccount
            if it_avg>best_val:
                best_val = it_avg
                best_iter = it_id
        
        # Calculate the true aggregate score
        agg_score = round(total_score/total_count if total_count>0 else 0.0, 2)
        
        if m_name not in existing_analyses:
            interpolated_elo = interpolate_elo_from_rubric_scores(m_name, agg_score, existing_analyses)
            print('! interpolated elo', interpolated_elo)
            existing_analyses[m_name] = {
                "creative_writing_rubric_score_agg": agg_score,
                "elo": round(interpolated_elo, 2),
                "iterations": {},
                "best_iteration": str(best_iter)
            }
        else:
            existing_analyses[m_name]["creative_writing_rubric_score_agg"] = agg_score
            
            # Only set ELO if it doesn't exist
            if "elo" not in existing_analyses[m_name]:
                interpolated_elo = interpolate_elo_from_rubric_scores(m_name, agg_score, existing_analyses)
                print('! interpolated elo', interpolated_elo)
                existing_analyses[m_name]["elo"] = round(interpolated_elo, 2)
            
            existing_analyses[m_name]["best_iteration"] = str(best_iter)
            if "iterations" not in existing_analyses[m_name]:
                existing_analyses[m_name]["iterations"] = {}

        # store iteration-level
        for it_id, it_data in iteration_map.items():
            existing_analyses[m_name]["iterations"][it_id] = {
                "creative_writing_rubric_score_iter": it_data["creative_writing_rubric_score_iter"],
                "items": it_data["items"],
                "item_scores": it_data["item_scores"]
            }

    for model_name, info in existing_analyses.items():
        if "elo_analysis" in info and "pairwise_comparisons" in info["elo_analysis"]:
            old_list = info["elo_analysis"]["pairwise_comparisons"]
            deduped = deduplicate_comparisons(old_list, model_name)
            existing_analyses[model_name]["elo_analysis"]["pairwise_comparisons"] = deduped

    save_json_file(existing_analyses, elo_results_file)

    # if only 1 model total => no pairwise
    if len(existing_analyses)<2:
        logging.info("Only one model in store => no pairwise done.")
        return

    # ensure test_model is in store
    if test_model not in existing_analyses:
        existing_analyses[test_model] = {
            "elo": 1200.0,
            "creative_writing_rubric_score_agg": 50.0,
            "iterations": {},
            "best_iteration": "1"
        }

    # utility to get top iteration IDs
    def get_top_iterations(m: str, limit: int):
        it_map = existing_analyses[m].get("iterations",{})
        arr=[]
        for i_id, i_info in it_map.items():
            sc = i_info.get("creative_writing_rubric_score_iter",0.0)
            arr.append((i_id,sc))
        arr.sort(key=lambda x:x[1],reverse=True)
        return [x[0] for x in arr[:limit]]

    def gather_items(m: str, it_id: str):
        it_info = existing_analyses[m]["iterations"][it_id]
        return it_info.get("items",{}), it_info.get("item_scores",{})

    # Build a sorted list of the OTHER models by their current 'elo'
    # We'll do a standard binary search bracket with partial sampling
    other_models = sorted(
        [mm for mm in existing_analyses.keys() if mm!= test_model],
        key=lambda x: existing_analyses[x]["elo"]
    )

    # quick function to do partial match (like "quick ladder") using top iteration & random sampling from overlap
    def partial_match_test_vs(idx: int, sample_size: int):
        """
        We'll gather top iteration from test_model & neighbor,
        then sample from their overlap up to 'sample_size'.
        Then concurrency judge them. Return (avg_test_score, comparisons, neighbor_name).
        """
        neighbor_model = other_models[idx]
        test_iter_id = get_top_iterations(test_model, 1)[0]
        neighbor_iter_id = get_top_iterations(neighbor_model, 1)[0]

        test_items, test_scores = gather_items(test_model, test_iter_id)
        neighbor_items, neighbor_scores = gather_items(neighbor_model, neighbor_iter_id)

        # Find overlapping item IDs and sort them
        overlap_ids = sorted(set(test_items.keys()) & set(neighbor_items.keys()))
        
        # Sample evenly if we have more items than the sample size
        if len(overlap_ids) > sample_size:
            step = max(1, len(overlap_ids) // sample_size)
            overlap_ids = [overlap_ids[i] for i in range(0, len(overlap_ids), step)][:sample_size]

        # Build dictionaries with just the sampled items
        sampled_test_items = {item_id: test_items[item_id] for item_id in overlap_ids}
        sampled_test_scores = {item_id: test_scores.get(item_id, 0.0) for item_id in overlap_ids}
        sampled_neighbor_items = {item_id: neighbor_items[item_id] for item_id in overlap_ids}
        sampled_neighbor_scores = {item_id: neighbor_scores.get(item_id, 0.0) for item_id in overlap_ids}

        # Run the parallel judge
        avg_test_score, comparisons, test_pluses, neighbor_pluses = _judge_items_in_parallel(
            test_model, neighbor_model,
            sampled_test_items, sampled_neighbor_items,
            sampled_test_scores, sampled_neighbor_scores,
            concurrency,
            pairwise_prompt_template,
            writing_prompts,
            judge_model,
            api_clients,
            max_items_per_model
        )
        
        return avg_test_score, comparisons, neighbor_model


    # We'll track a global set of (modelA,modelB,item_id,iterationA,iterationB) to avoid reprocessing
    matchup_results = {}

    def do_comprehensive_with_neighbor(neighbor_idx: int):
        neighbor_model = other_models[neighbor_idx]
        all_comparisons = []
        logging.debug(f"[COMPREHENSIVE] => test_model={test_model}, neighbor={neighbor_model}")
        
        # STEP 1: Collect all items across all iterations first
        test_model_items_by_id = {}  # item_id -> list of (iteration_id, text, score)
        neighbor_model_items_by_id = {}  # item_id -> list of (iteration_id, text, score)
        
        # Collect test model items from all iterations
        test_iteration_ids = get_top_iterations(test_model, 999)  # Get all iterations
        for test_iter_id in test_iteration_ids:
            items_dict, scores_dict = gather_items(test_model, test_iter_id)
            for item_id, item_text in items_dict.items():
                item_score = scores_dict.get(item_id, 0.0)
                if item_id not in test_model_items_by_id:
                    test_model_items_by_id[item_id] = []
                test_model_items_by_id[item_id].append((test_iter_id, item_text, item_score))
        
        # Collect neighbor model items from all iterations
        neighbor_iteration_ids = get_top_iterations(neighbor_model, 999)  # Get all iterations
        for neighbor_iter_id in neighbor_iteration_ids:
            items_dict, scores_dict = gather_items(neighbor_model, neighbor_iter_id)
            for item_id, item_text in items_dict.items():
                item_score = scores_dict.get(item_id, 0.0)
                if item_id not in neighbor_model_items_by_id:
                    neighbor_model_items_by_id[item_id] = []
                neighbor_model_items_by_id[item_id].append((neighbor_iter_id, item_text, item_score))
        
        # Find common item IDs
        common_item_ids = sorted(set(test_model_items_by_id.keys()) & set(neighbor_model_items_by_id.keys()))
        
        # Sort each item's iterations by score (descending)
        for item_id in common_item_ids:
            test_model_items_by_id[item_id].sort(key=lambda x: x[2], reverse=True)
            neighbor_model_items_by_id[item_id].sort(key=lambda x: x[2], reverse=True)
        
        # STEP 2: Build matchups by tier levels (best versions first)
        reused_comparisons = []  # Will hold comparisons we've already done
        new_matchups = []  # List of new matchups to run: (item_id, test_item_data, neighbor_item_data)
        
        max_tier = 0
        for item_id in common_item_ids:
            max_tier = max(max_tier, 
                        len(test_model_items_by_id[item_id]), 
                        len(neighbor_model_items_by_id[item_id]))
        
        # Generate matchups by tier
        for tier in range(max_tier):
            for item_id in common_item_ids:
                # Skip if we don't have items at this tier for both models
                if (tier >= len(test_model_items_by_id[item_id]) or
                    tier >= len(neighbor_model_items_by_id[item_id])):
                    continue
                
                test_item_data = test_model_items_by_id[item_id][tier]
                neighbor_item_data = neighbor_model_items_by_id[item_id][tier]
                
                test_iter_id = test_item_data[0]  # First element is iteration ID
                neighbor_iter_id = neighbor_item_data[0]
                
                # Check if we've already processed this matchup
                signature = (test_model, neighbor_model, item_id, test_iter_id, neighbor_iter_id)
                
                if signature in matchup_results:
                    # We've seen this matchup before - reuse the results
                    reused_comparisons.extend(matchup_results[signature])
                else:
                    # This is a new matchup - add it to our processing list
                    new_matchups.append((item_id, test_item_data, neighbor_item_data))
        
        # Limit to max_items_per_model if needed
        if len(new_matchups) > max_items_per_model:
            logging.debug(f"[COMPREHENSIVE] Limiting from {len(new_matchups)} to {max_items_per_model} matchups")
            new_matchups = new_matchups[:max_items_per_model]
        
        logging.debug(f"[COMPREHENSIVE] Will process {len(new_matchups)} new matchups")
        logging.debug(f"[COMPREHENSIVE] Reusing {len(reused_comparisons)} cached comparisons")
        
        # STEP 3: Run parallel comparisons for new matchups
        if new_matchups:
            # Create the dictionaries needed for _judge_items_in_parallel
            test_model_texts = {}
            test_model_scores = {}
            neighbor_model_texts = {}
            neighbor_model_scores = {}
            
            # Track which original matchup each unique key corresponds to
            key_to_matchup_map = {}
            
            for matchup_idx, (item_id, test_item_data, neighbor_item_data) in enumerate(new_matchups):
                test_iter_id, test_text, test_score = test_item_data
                neighbor_iter_id, neighbor_text, neighbor_score = neighbor_item_data
                
                # Create a unique key combining item_id and iteration IDs
                unique_key = f"{item_id}_{test_iter_id}_{neighbor_iter_id}"
                
                # Store the mapping from unique key back to the original matchup info
                key_to_matchup_map[unique_key] = (item_id, test_iter_id, neighbor_iter_id)
                
                # Use the unique key as the dictionary key
                test_model_texts[unique_key] = test_text
                test_model_scores[unique_key] = test_score
                neighbor_model_texts[unique_key] = neighbor_text
                neighbor_model_scores[unique_key] = neighbor_score
            
            # Run all comparisons in parallel
            _, new_comparisons, _, _ = _judge_items_in_parallel(
                test_model, 
                neighbor_model,
                test_model_texts, 
                neighbor_model_texts,
                test_model_scores, 
                neighbor_model_scores,
                concurrency,
                pairwise_prompt_template,
                writing_prompts,
                judge_model,
                api_clients
            )

            # Cache the results and update matchup_results
            for i, (item_id, test_item_data, neighbor_item_data) in enumerate(new_matchups):
                test_iter_id = test_item_data[0]
                neighbor_iter_id = neighbor_item_data[0]
                
                # Find the corresponding comparison results
                relevant_comps = []
                for comp in new_comparisons:
                    if comp["item_id"] == item_id:
                        relevant_comps.append(comp)
                
                # Cache them
                signature = (test_model, neighbor_model, item_id, test_iter_id, neighbor_iter_id)
                matchup_results[signature] = relevant_comps
            
            # Add new comparisons to results
            all_comparisons.extend(new_comparisons)
        
        # Add cached/reused comparisons to results
        all_comparisons.extend(reused_comparisons)
        
        return all_comparisons

    # We'll do a function that re-sorts the models after a solve, to see if test_model moved
    def solve_elo(comps):
        # build model list
        all_mods = list(existing_analyses.keys())
        init_ratings = {m: existing_analyses[m]["elo"] for m in all_mods}

        final_elo = solve_with_glicko(
            all_mods,
            comps,
            init_ratings
        )
        # store
        for mm in all_mods:
            existing_analyses[mm]["elo"] = round(final_elo[mm],2)
        existing_analyses[test_model]["elo_analysis"] = {
            "final_elo_ratings": {m: round(r,2) for m,r in final_elo.items()},
            "pairwise_comparisons": comps
        }
        # see new sorted
        sorted_others = sorted([m for m in all_mods if m!= test_model],
            key=lambda x: existing_analyses[x]["elo"])
        new_pos = 0
        for i,m in enumerate(sorted_others):
            if existing_analyses[m]["elo"]< existing_analyses[test_model]["elo"]:
                new_pos= i+1
        
        inner_list = [f"{nm}={existing_analyses[nm]['elo']}" for nm in sorted_others]
        logging.debug(f"[ELO SOLVE] test_model elo => {existing_analyses[test_model]['elo']}, bracket pos => {new_pos}, sorted_others => {inner_list}")
        return new_pos, sorted_others

    # do comprehensive with immediate neighbors around bracket
    def do_local_comprehensive(bracket_idx):
        new_comps = []
        rng = range(max(0, bracket_idx-1), min(len(other_models), bracket_idx+1))
        logging.debug(f"[COMPREHENSIVE] bracket_idx={bracket_idx}, neighbor range={list(rng)}")
        for idx in rng:
            logging.debug(f"[COMPREHENSIVE] Doing neighbor idx={idx} => {other_models[idx]}")
            new_comps.extend(do_comprehensive_with_neighbor(idx))
        logging.debug(f"[COMPREHENSIVE] total new comps from local range => {len(new_comps)}")
        return new_comps

    
    
    if len(other_models) == 1:
        logging.debug("[ELO] Only 1 neighbor found. Skipping binary search, doing direct comprehensive.")
        
        # Skip partial match and go straight to comprehensive with the only neighbor
        comps = do_comprehensive_with_neighbor(0)
        
        # Combine with old comps
        old_comps = []
        for mname, info in existing_analyses.items():
            if "elo_analysis" in info and "pairwise_comparisons" in info["elo_analysis"]:
                old_comps.extend(info["elo_analysis"]["pairwise_comparisons"])
        all_combined = old_comps + comps

        
        # Do one Elo solve
        bracket_idx_after, sorted_after = solve_elo(all_combined)

        for model_name, info in existing_analyses.items():
            if "elo_analysis" in info and "pairwise_comparisons" in info["elo_analysis"]:
                old_list = info["elo_analysis"]["pairwise_comparisons"]
                deduped = deduplicate_comparisons(old_list, model_name)
                existing_analyses[model_name]["elo_analysis"]["pairwise_comparisons"] = deduped
        
        # Save and return
        save_json_file(existing_analyses, elo_results_file)
        logging.info(f"[ELO] Completed run with single neighbor, final rating for {test_model}={existing_analyses[test_model]['elo']}.")
        return

    bracket_pos, bracket_comparisons = find_optimal_position(
        test_model,
        other_models,
        existing_analyses,
        partial_match_test_vs,
        solve_elo,
        ladder_sample_size,
        neighbor_depth=2
    )

   
    

    # First pass of comprehensive in the bracket zone
    comprehensive_comparisons = do_local_comprehensive(bracket_pos)

    # Combine bracket + comp
    new_comps = bracket_comparisons + comprehensive_comparisons

    # Now do a global elo solve => see if the test_model's position changed
    old_comps = []
    for mname,info in existing_analyses.items():
        if "elo_analysis" in info and "pairwise_comparisons" in info["elo_analysis"]:
            old_comps.extend(info["elo_analysis"]["pairwise_comparisons"])

    all_combined = old_comps + new_comps

    print('>>', len(bracket_comparisons), len(comprehensive_comparisons), len(old_comps))

    # solve once
    bracket_idx_after, sorted_after = solve_elo(all_combined)

    # if bracket_idx_after changed from bracket_pos, we do another pass with new neighbors
    # and skip duplicates
    # We'll do a simple loop limit to avoid infinite re-check
    loop_guard = 0
    while bracket_idx_after != bracket_pos and loop_guard < 3:
        print('! resolving because bracket changed', bracket_idx_after, bracket_pos)
        bracket_pos = bracket_idx_after
        # re-map other_models to sorted_after
        other_models = sorted_after
        # do comprehensive in new bracket zone
        comps2 = do_local_comprehensive(bracket_pos)
        new_comps.extend(comps2)
        if comps2:
            all_combined.extend(comps2)        

        bracket_idx_after, sorted_after = solve_elo(all_combined)
        loop_guard += 1


    existing_analyses[test_model]["elo_analysis"]["pairwise_comparisons"] = new_comps

    for model_name, info in existing_analyses.items():
        if "elo_analysis" in info and "pairwise_comparisons" in info["elo_analysis"]:
            old_list = info["elo_analysis"]["pairwise_comparisons"]
            deduped = deduplicate_comparisons(old_list, model_name)
            existing_analyses[model_name]["elo_analysis"]["pairwise_comparisons"] = deduped
            

    # Normalize ELO scores
    raw_elo_scores = {model: info.get("elo", 1200) for model, info in existing_analyses.items()}
    normalized_scores = normalize_elo_scores(raw_elo_scores)
    
    # Store normalized scores
    for model, norm_score in normalized_scores.items():
        if model in existing_analyses:
            existing_analyses[model]["elo_norm"] = round(norm_score, 2)
    
    # Print normalized scores for the test model
    if test_model in normalized_scores:
        normalized_test_elo = normalized_scores[test_model]
        logging.info(f"Test model normalized ELO rating => {normalized_test_elo:.2f}")

    save_json_file(existing_analyses, elo_results_file)
    logging.info(f"Completed run_elo_analysis_creative for model={test_model}, run_key={run_key}.")
    logging.info(f"Test model final rating => {existing_analyses[test_model]['elo']}")