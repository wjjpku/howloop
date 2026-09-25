from pathlib import Path
import json
root=Path(__file__).resolve().parent
p=root/'code/reasoning_loop/paper2027_graph_g4_protocol.py'
s=p.read_text(); start=s.index('    universe = _all_permutations(node_count)'); end=s.index('    payload: dict[str, object]',start)
s=s[:start]+'''    # Rejection sampling avoids materializing 10! permutations. Each accepted
    # unseen permutation is uniform conditional on exclusions.
    excluded = set() if excluded_codes is None else set(excluded_codes.cpu().tolist())
    if permutations > math_factorial(node_count) - len(excluded):
        raise ValueError("not enough distinct graphs")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    selected = []
    seen = set(excluded)
    while len(selected) < permutations:
        proposals = torch.rand(max(64, 2 * (permutations-len(selected))), node_count, generator=generator).argsort(-1)
        for row, code in zip(proposals, permutation_codes(proposals).tolist()):
            if code not in seen:
                selected.append(row)
                seen.add(code)
                if len(selected) == permutations:
                    break
    successors = torch.stack(selected)
    codes = permutation_codes(successors)
'''+s[end:];p.write_text(s)
p=root/'code/reasoning_loop/paper2027_graph_g4_backbone.py';s=p.read_text()
s=s.replace('import math','import math\nimport os')
s=s.replace('paper2027.graph.g4.disjoint_backbone.v1','paper2027.graph.n10.disjoint_backbone.v1')
s=s.replace('node_count=8, max_depth=8','node_count=10, max_depth=8').replace('n_layers=2, max_loops=8','n_layers=2, max_loops=args.loops').replace('only for N8 D8','only for N10 D8')
s=s.replace('selection_call8','selection_endpoint').replace('call8_accuracy','endpoint_accuracy')
s=s.replace('set_seed(args.seed)\n    model', 'set_seed(args.seed + 1009 * args.loops if args.depth_mode == "uniform" else args.seed)\n    model')
s=s.replace('"loss": "final-only successor CE at call 8"','"loss": f"final-only endpoint CE at call {cfg.max_loops}", "depth_mode": args.depth_mode')
s=s.replace('"seed": args.seed,','"seed": args.seed, "pid": os.getpid(), "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),')
s=s.replace('    started = time.time()','    started = time.time()\n    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)')
s=s.replace('        optimizer.zero_grad(set_to_none=True)', '''        target = targets[:, -1]
        if args.depth_mode == "uniform":
            depth = torch.randint(1, cfg.max_depth + 1, (args.batch_size,), device=device)
            tokens[:, -2] = cfg.depth_token_base + depth - 1
            target = targets.gather(1, (depth - 1)[:, None]).squeeze(1)
        optimizer.zero_grad(set_to_none=True)''')
s=s.replace('loss = F.cross_entropy(logits, targets[:, -1])','loss = F.cross_entropy(logits, target)')
s=s.replace('"status": "complete", "protocol_id"', '"status": "complete", "elapsed_sec": time.time()-started, "peak_memory_gib": torch.cuda.max_memory_allocated(device)/2**30 if device.type == "cuda" else 0, "protocol_id"')
s=s.replace('parser.add_argument("--node-count", type=int, default=8)','parser.add_argument("--node-count", type=int, default=10)\n    parser.add_argument("--loops", type=int, choices=[6,8], default=8)\n    parser.add_argument("--depth-mode", choices=["uniform","fixed"], default="uniform")')
s=s.replace('"--eval-batch-size", type=int, default=512','"--eval-batch-size", type=int, default=500')
s=s.replace('    train(parse_args())','    torch.set_num_threads(4)\n    train(parse_args())')
p.write_text(s)
registry={
 'protocol':'n10_migration_20260923', 'node_count':10,'task_max_depth':8,
 'status':'implementation_and_smoke',
 'backbones':{'local_control':{'loops':6,'seeds':[6,3,4,5,7],'depth_mode':'uniform'},'trajectory':{'loops':8,'seeds':[0,1,6,8],'depth_mode':'uniform'},'continuation':{'loops':8,'seeds':list(range(100,112)),'depth_mode':'fixed'}},
 'steps':20000,'batch_size':512,'checkpoints':'best locked D8 selection accuracy, earliest tie; final also retained',
 'protocol_change':'All selection/test graphs excluded from backbone and J training; original local backbones did not enforce this. Record as redesigned N10 cohort, not node-count-only causal comparison.',
 'section3':'Two D8L8 and two D8L6 panels selected descriptively from complete registered cohorts; all seed results retained. One/two-step-looking readouts are not verified algorithmic steps.',
 'experiments':['native trajectories calls0-16','D8L6 same-state f9/f10 target control, 2 pureCE fits each','cross-graph pattern/content semantic counterfactual','same-input routing transfer and component controls','corrupt-restore','D8L8 on-policy J calls1-16, 12 backbones x2fits','continuation calls1-128 and executor-off/shuffle/component controls'],
 'unchanged_tasks':['Ouro','parity','KG'],
 'no_claim_until_measured':True}
(root/'MANIFEST.json').write_text(json.dumps(registry,indent=2)+'\n')
