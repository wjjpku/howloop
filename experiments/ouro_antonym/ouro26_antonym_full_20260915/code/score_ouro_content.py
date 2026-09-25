"""Supplement exact-format score with conservative first-sentence answer parsing."""
import argparse,json,re
from pathlib import Path
from task import OPPOSITE

def parse(row):
    text=row['generated'].strip()
    if row['kind']=='lexical':
        match=re.match(r'^(yes|no)\b',text,re.I)
        return match.group(1).lower() if match else None
    first=re.split(r'[.!?\n]',text,maxsplit=1)[0]
    words=re.findall(r'\b(?:'+'|'.join(OPPOSITE)+r')\b',first.lower())
    return ' '.join(words) if len(words)==2 else None

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
    rows=[json.loads(line) for line in (a.root/'results.jsonl').read_text().splitlines()]
    scored=[dict(r,parsed_answer=parse(r),content_correct=parse(r)==r['answer']) for r in rows]
    def metrics(rs):
        return dict(n=len(rs),content_correct=sum(r['content_correct'] for r in rs),format_exact=sum(r['exact'] for r in rs),
            unparsed=sum(r['parsed_answer'] is None for r in rs),length_limit_reached=sum(r['length_limit_reached'] for r in rs))
    result=dict(parser='leading yes/no for lexical; exactly two vocabulary words in first sentence for cancellation; no target-dependent extraction',
        lexical=metrics([r for r in scored if r['kind']=='lexical']),cancellation=metrics([r for r in scored if r['kind']=='cancellation']),
        per_k={str(k):metrics([r for r in scored if r.get('k')==k]) for k in range(1,9)})
    (a.root/'content_scored.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in scored));(a.root/'content_summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
