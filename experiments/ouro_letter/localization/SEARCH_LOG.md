# Localization search log

All decisions use the original8 discovery pairs, not the new64 holdout.

1. Layer scan complete,54conditions×8. All16 late layers rescue8/8 and damage0/8. Every single layer rescues0/8. Layer34 damage1/8; other layers mostly little damage. Four-layer group44–47 rescues5/8 and damages to3/8; group32–35 rescues0/8 yet damages to0/8. Excluding32–35 or36–39 preserves8/8 rescue and0/8damage. Therefore explore combinations rather than treating damage-only sites as sufficient.
2. Test all six pairs of four-layer groups, plus16 leave-one-layer-out conditions, each in both directions.

3. All six group pairs: only layers40–47 retain8/8rescue and0/8damage. Last8 layers selected for head refinement. Within full late16, largest rescue probability losses on omitting layers47,41,44,43; test combinations of those four layers (all sizes2–4), plus leave-one-four-head-group-out over all last8 layers and four head-index groups. This is adaptive discovery, not a claim that those four layers are preselected confirmation sites.

4. Triple layers41,43,47 retains8/8 rescue and0/8damage (48heads); alternative41,44,47 also passes. Select41,43,47 by larger rescue mean probability. Rank four-head groups by sum of rescue probability loss and damage probability increase when omitted from last8; test nested top2,3,4,5,6,7,8,10,12 groups and single-head omissions within triple. Ranking saved in ranked_heads.json.

5. Nested top6 four-head groups (24heads) pass8/8rescue,0/8damage. Single-head omissions within layers41,43,47 ranked by rescue probability loss plus damage probability rise; top entries L47.H4,L47.H13,L41.H10,L47.H0,L41.H12,L47.H3,L43.H15,L43.H11. Test nested1,2,3,4,5,6,8,10,12,16,20,24,32,40,48 heads and top12 heads individually.

6. Nested10heads first passes both8/8rescue and0/8damage;8heads rescues6/8. All12 individual candidates rescue0/8. Ten selected positions: [(47, 4), (47, 13), (41, 10), (47, 0), (41, 12), (47, 3), (43, 15), (43, 11), (43, 13), (43, 6)]. Evaluate leave-one-out among these10 and all proper nonempty call subsets, then freeze final confirmation.

7. Tenheads call4 alone8/8rescue,0/8damage; calls2–3 alone0/8rescue,8/8damage accuracy. Two ninehead all-call variants pass; not claiming minimality. Freeze exact10head call4 intervention for untouched64 confirmation, and sizes8/16 as prespecified sensitivity arms.

8. New64 confirmation complete: raw0,J62;full256×3calls rescue62/damage0;primary10×call4 rescue50/damage5;prespecified16×call4 rescue61/damage0;8×call4 rescue44/damage12. Recommend16 with primary10 failure explicitly retained. No holdout-driven site changes. All24conditions×64rows and384generation checks audited.
