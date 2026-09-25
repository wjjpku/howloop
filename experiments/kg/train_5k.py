"""4 controllers × curriculum m=4→32 step 4, using existing infrastructure.

Uses run_alternating (F first, J between) and truncated_unroll (moving answer_index).
"""
import sys, json, time, hashlib, argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

CODE = Path('/data/wujiaju/kg-fj-affine-resffn-m64-code-5f49178')
sys.path.insert(0, str(CODE))
from experiments.kg_fj_length.data import KGLengthConfig, PermutationWorld, sample_batch
from experiments.kg_fj_length.model import LoopedCompositionTransformer, ModelConfig
from experiments.kg_fj_length.controller import run_alternating, ControllerConfig, AffineJ, CausalAttentionJ

BACKBONE = Path('/data/wujiaju/kg-fj-nope-20260815-v1/nope_only/backbone/best.pt')
BACKBONE_SHA = '4d6bb56b501ddeb0b1056945fd41f6a5f7ce1b79651732eaef077b007b2641c5'
WORLD_SEED = 20260814
D = 256
STAGES = list(range(4, 33, 4))  # [4, 8, 12, 16, 20, 24, 28, 32]
MAX_STEPS = 5000
EARLY_STOP = 0.99
EVAL_EVERY = 250
BATCH = 128
LR = 1e-4
EVAL_N = 128
torch.set_num_threads(4)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''):
            h.update(b)
    return h.hexdigest()


class LoRA48(nn.Module):
    def __init__(self, d=D, r=48):
        super().__init__()
        self.A = nn.Parameter(torch.randn(d, r) * 0.02)
        self.B = nn.Parameter(torch.zeros(r, d))
        self.bias = nn.Parameter(torch.zeros(d))
    def forward(self, h):
        return h + (h @ self.A) @ self.B + self.bias


class MLPNoNorm(nn.Module):
    def __init__(self, d=D, hidden=256):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(d, hidden) * 0.02)
        self.b1 = nn.Parameter(torch.zeros(hidden))
        self.W2 = nn.Parameter(torch.zeros(hidden, d))
        self.b2 = nn.Parameter(torch.zeros(d))
    def forward(self, h):
        return h + F.gelu(h @ self.W1 + self.b1) @ self.W2 + self.b2


def make_controller(name):
    if name == 'lora_r48':
        return LoRA48()
    elif name == 'dense':
        return AffineJ(ControllerConfig(d_model=D, architecture='affine'), seed=301)
    elif name == 'mlp_256':
        return MLPNoNorm()
    elif name == 'attention':
        return CausalAttentionJ(ControllerConfig(d_model=D, hidden_width=256,
                                                  architecture='attention'), seed=301)


def load_backbone(dev):
    """Load with original config strict=True, then extend max_length in-place."""
    kg_orig = KGLengthConfig(entity_count=128, relation_count=16, max_length=6)
    mc = ModelConfig(d_model=D, n_heads=8, d_mlp=1024, physical_blocks=2,
                     position_encoding='none', input_injection=False)
    m = LoopedCompositionTransformer(kg_orig, mc)
    ckpt = torch.load(BACKBONE, map_location='cpu', weights_only=False)
    m.load_state_dict(ckpt['model_state'], strict=True)
    m = m.to(dev).eval().requires_grad_(False)
    # In-place extend (same as existing extend_executor_max_length)
    m.kg_config = KGLengthConfig(entity_count=128, relation_count=16, max_length=32)
    return m, m.kg_config


@torch.no_grad()
def evaluate(model, ctrl, world, kg, length, dev, n=EVAL_N):
    gen = torch.Generator().manual_seed(990000 + length * 7)
    batch = sample_batch(kg, world, n, length, gen, frozenset()).to(dev)
    logits = run_alternating(model, ctrl, batch.tokens, calls=length)
    return int(logits.argmax(-1).eq(batch.target).sum()) / n


def truncated_unroll_loss(model, ctrl, batch, world_perms, calls):
    """Exact replica of existing _controller_training_loss with truncated_unroll."""
    trace = run_alternating(model, ctrl, batch.tokens, calls=calls, return_states=True)
    current = batch.start
    losses = []
    for step, state in enumerate(trace.post_f_states, start=1):
        current = world_perms[batch.relations[:, step - 1], current]
        if step >= 2:  # Step 1 has no J before it, no controller gradient
            losses.append(F.cross_entropy(
                model.readout(state, answer_index=step + 1), current))
    return torch.stack(losses).mean()


def train_stage(model, ctrl, world, kg, stage, dev):
    opt = torch.optim.AdamW(ctrl.parameters(), lr=LR, betas=(.9, .95), weight_decay=0)
    gen = torch.Generator().manual_seed(917000 + stage * 13)
    replay_gen = torch.Generator().manual_seed(918000 + stage * 13)
    world_perms = world.permutations.to(dev)
    steps_used = 0

    for step in range(1, MAX_STEPS + 1):
        # Random length replay: 50% current stage, 50% random shorter
        if stage > 4 and torch.rand((), generator=replay_gen).item() < 0.5:
            active = int(torch.randint(4, stage + 1, (), generator=replay_gen).item())
        else:
            active = stage
        batch = sample_batch(kg, world, BATCH, active, gen, frozenset()).to(dev)
        loss = truncated_unroll_loss(model, ctrl, batch, world_perms, active)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ctrl.parameters(), 1.)
        opt.step()
        steps_used = step
        if step % EVAL_EVERY == 0:
            acc = evaluate(model, ctrl, world, kg, stage, dev, 64)
            if acc >= EARLY_STOP:
                return steps_used, acc
    acc = evaluate(model, ctrl, world, kg, stage, dev, 64)
    return steps_used, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--controller', required=True,
                    choices=['lora_r48', 'dense', 'mlp_256', 'attention'])
    ap.add_argument('--out-root', type=Path,
                    default=Path('/data/wujiaju/kg_curriculum_20260920'))
    ap.add_argument('--seed', type=int, default=None,
                    help='Seed controller initialization for a new confirmatory run')
    ap.add_argument('--save-checkpoints', action='store_true',
                    help='Save a reusable controller checkpoint after every stage')
    args = ap.parse_args()

    dev = torch.device('cuda')
    torch.cuda.set_per_process_memory_fraction(0.12, 0)
    assert sha(BACKBONE) == BACKBONE_SHA

    model, kg = load_backbone(dev)
    world = PermutationWorld.create(kg, WORLD_SEED)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    ctrl = make_controller(args.controller).to(dev)
    n_params = sum(p.numel() for p in ctrl.parameters())
    print(f'Controller: {args.controller} ({n_params} params)', flush=True)

    out_dir = args.out_root / args.controller
    out_dir.mkdir(parents=True, exist_ok=True)

    heatmap = []
    results = {'controller': args.controller, 'params': n_params,
               'backbone_sha256': sha(BACKBONE), 'world_seed': WORLD_SEED,
               'stages': STAGES, 'lr': LR, 'batch': BATCH,
               'max_steps': MAX_STEPS, 'early_stop': EARLY_STOP,
               'execution': 'F (J F)^{m-1} via run_alternating',
               'loss': 'truncated_unroll (moving answer_index, from step 2)',
               'controller_seed': args.seed,
               'heatmap': heatmap}

    for stage_idx, stage in enumerate(STAGES):
        print(f'\n  stage {stage_idx+1}/{len(STAGES)}: m={stage}', flush=True)
        t0 = time.time()
        steps, acc = train_stage(model, ctrl, world, kg, stage, dev)
        elapsed = time.time() - t0
        row = [round(evaluate(model, ctrl, world, kg, tl, dev), 4) for tl in STAGES]
        heatmap.append({'train_length': stage, 'accuracy': row,
                        'steps_used': steps, 'early_stop_acc': round(acc, 4),
                        'seconds': round(elapsed, 1)})
        print(f'  acc: {row} ({steps} steps, {elapsed:.0f}s)', flush=True)
        if args.save_checkpoints:
            torch.save({'controller': ctrl.state_dict(), 'controller_type': args.controller,
                        'stage': stage, 'controller_seed': args.seed,
                        'backbone_sha256': BACKBONE_SHA, 'world_seed': WORLD_SEED,
                        'training_step_in_stage': steps},
                       out_dir / f'controller_m{stage}.pt')
        (out_dir / 'results.json').write_text(json.dumps(results, indent=1))

    print(f'\n{args.controller} DONE', flush=True)


if __name__ == '__main__':
    main()
