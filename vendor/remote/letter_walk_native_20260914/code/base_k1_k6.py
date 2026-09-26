"""Matched base checkpoint pilot: two original graphs, k=1..6, unchanged prompts."""
import sys
import probe_ouro_v2 as probe

original_examples = probe.examples
probe.examples = lambda: [e for e in original_examples() if e['language']=='en' and e['k']<=6]
sys.argv = [sys.argv[0], '--model', '/data/paperexperiment/models/Ouro-2.6B',
            '--output', '/data/paperexperiment/letter_walk_native_20260914/ouro26_base_k1_k6_v1',
            '--limit', '12', '--max-new-tokens', '1024']
if __name__ == '__main__':
    probe.main()
