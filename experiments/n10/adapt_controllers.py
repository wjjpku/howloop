from pathlib import Path
R=Path(__file__).resolve().parent/'code'
p=R/'reasoning_loop/paper2027_d8l6_s1.py';s=p.read_text().replace('"node_count": 8','"node_count": 10').replace('paper2027.d8l6.ae_pure_ce.seedlocked.v2','paper2027.n10.d8l6.pure_ce.disjoint.v1')
s=s.replace('from reasoning_loop.paper2027_graph_g3_controller import', 'from reasoning_loop.paper2027_graph_g4_protocol import load_unique_lock, merged_forbidden_codes, sample_training_permutations\nfrom reasoning_loop.paper2027_graph_g3_controller import')
a=s.index('@torch.no_grad()\ndef validate');b=s.index('\n\ndef state_digest',a)
s=s[:a]+'''@torch.no_grad()
def validate(model, cfg, controller, *, hop, device, lock_path, batch_size):
    lock = load_unique_lock(lock_path)
    assert lock['role'] == 'selection' and lock['node_count'] == cfg.node_count
    assert batch_size % cfg.node_count == 0
    correct = total = 0
    group = batch_size // cfg.node_count
    for first in range(0, len(lock['successors']), group):
        gs = lock['successors'][first:first+group]
        graphs = gs.repeat_interleave(cfg.node_count,0).to(device)
        starts = torch.arange(cfg.node_count,device=device).repeat(len(gs))
        tokens, targets, _, _ = fixed_depth_batch(cfg,len(starts),device,path_positions=cfg.max_depth+2,successors=graphs,start=starts)
        output = model.apply_loop(controller(terminal_h6(model,tokens,cfg)),loop_index=cfg.max_loops)
        pred = logits_from_raw_state(model,output).argmax(-1)
        correct += int(pred.eq(target_for_hop(targets,cfg,hop)).sum());total += len(starts)
    return correct/total
'''+s[b:]
s=s.replace('    model.requires_grad_(False)','    model.requires_grad_(False)\n    assert args.validation_lock in args.train_exclude_locks\n    forbidden = merged_forbidden_codes(args.train_exclude_locks, node_count=cfg.node_count)')
s=s.replace('tokens, targets, _, _ = fixed_depth_batch(cfg, args.batch_size, device, path_positions=cfg.max_depth + 2)','successors = sample_training_permutations(batch_size=args.batch_size, node_count=cfg.node_count, forbidden_codes=forbidden, device=device)\n        tokens, targets, _, _ = fixed_depth_batch(cfg, args.batch_size, device, path_positions=cfg.max_depth + 2, successors=successors)')
s=s.replace('seed=args.validation_seed, batches=args.validation_batches, batch_size=args.validation_batch_size','lock_path=args.validation_lock, batch_size=args.validation_batch_size')
s=s.replace('k:str(v) if isinstance(v,Path) else v','k:str(v) if isinstance(v,Path) else [str(x) for x in v] if isinstance(v,list) else v')
s=s.replace('train_p=sub.add_parser("train");','train_p=sub.add_parser("train"); train_p.add_argument("--validation-lock",type=Path,required=True); train_p.add_argument("--train-exclude-locks",type=Path,nargs="+",required=True);')
s=s.replace('    arguments=parse_args();','    torch.set_num_threads(4)\n    arguments=parse_args();')
p.write_text(s)
p=R/'reasoning_loop/paper2027_graph_g3_controller.py';s=p.read_text().replace('"node_count": 8','"node_count": 10').replace('N8','N10')
s=s.replace('    controller = DiagonalLowRankGraphController(cfg.d_model, args.rank).to(device)','    set_seed(args.seed)  # Seed controller initialization, not just training batches.\n    controller = DiagonalLowRankGraphController(cfg.d_model, args.rank).to(device)',1)
p.write_text(s)
p=R/'evaluate_battery.py';s=p.read_text().replace('cfg.node_count == 8','cfg.node_count == 10').replace('[:, :8]','[:, :cfg.node_count]')
s=s.replace('repeat_interleave(8, 0)','repeat_interleave(cfg.node_count, 0)').replace('torch.arange(8, device=device)','torch.arange(cfg.node_count, device=device)').replace('% 8','% cfg.node_count').replace('// 8','// cfg.node_count').replace('* 8 +','* cfg.node_count +').replace('reshape(8, 29, 4, 64)','reshape(cfg.node_count, cfg.seq_len, cfg.n_heads, cfg.d_model // cfg.n_heads)').replace('reshape(8, 29, 256)','reshape(cfg.node_count, cfg.seq_len, cfg.d_model)')
# Attention scaling sqrt(64)=8 and task depth 8 are intentionally unchanged.
p.write_text(s)
