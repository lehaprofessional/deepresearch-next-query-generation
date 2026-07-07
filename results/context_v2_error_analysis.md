# Error analysis for context prompt v2

## Summary

This report compares three variants:

- `llm_context_v1`
- `llm_context_retrieval_fewshot_v2`
- `llm_context_prompt_v2`

The goal is to understand where `context_prompt_v2` improves over the previous context baseline and where it still fails.

## Average Token F1 by research question

| Research question | Examples | context_v1 | context_retrieval_fewshot_v2 | context_prompt_v2 |
|---|---:|---:|---:|---:|
| What are the distinct mechanisms of refractory wear in the snorkel tubes of an RH (Ruhrstahl-Hereaus) vacuum degasser, a... | 2 | 0.3357 | 0.2211 | 0.2912 |
| What specific operational parameters and lance configurations are most effective for minimizing skull formation and maxi... | 18 | 0.2412 | 0.2634 | 0.2303 |
| How to protect the slag line from the electric arc's radiation in a ladle furnace? | 12 | 0.3175 | 0.2787 | 0.3483 |

## Worst examples for context_prompt_v2

| query_id | F1 | Target query | Generated query |
|---:|---:|---|---|
| 7 | 0.0741 | Post combustion efficiency heat transfer efficiency BOF percent CO2 CO optimal lance distance | MgO-saturated slag basicity 3.0-3.5 FeO content skull adhesion prevention BOF high scrap |
| 14 | 0.0769 | Lance height increase higher post combustion ratio FeO foamy slag PCR converter | CFD simulation post-combustion lance swirl number 0.5 secondary oxygen 3% BOF scrap melting |
| 12 | 0.0909 | "ladle metallurgy" arc impingement refractory damage electrode misalignment | ladle furnace foamy slag carbon injection rate kg/min CO gas generation foam height |
| 4 | 0.1000 | BOF bottom stirring argon flow rate combined blowing converter | lance tip erosion copper-silver alloy skull formation BOF high scrap ratio |
| 9 | 0.1111 | Post combustion heat transfer efficiency increases scrap melting CO2 to CO ratio bath surface distance trials | BOF lance height dynamic adjustment strategy scrap ratio 25% blow start 1.2 m end 2.0 m skull reduction |
| 12 | 0.1538 | Bottom stirring improves scrap melting carbon oxygen equilibrium reduces FeO BOF | lance nozzle configuration multi-hole design angle spacing BOF high scrap ratio skull reduction post-combustion efficien... |
| 13 | 0.1600 | Double flow post combustion lance tip life doubled scrap consumption increase trials | BOF lance nozzle configuration multi-hole design angle spacing high scrap ratio post-combustion efficiency |
| 10 | 0.1667 | BOF lance height range meters above bath converter typical operating | lance swirl angle 15 degrees skull reduction post combustion BOF high scrap ratio trials |

## Best improvements over context_v1

| query_id | context_v1 F1 | context_prompt_v2 F1 | Delta | Target query | v2 generated query |
|---:|---:|---:|---:|---|---|
| 18 | 0.0833 | 0.3333 | 0.2500 | LD3 post-combustion lance secondary oxygen 3% total oxygen flow rate trials | secondary oxygen lance swirl number 0.5 skull reduction 260-ton BOF plant trials |
| 1 | 0.3529 | 0.5556 | 0.2026 | foamy slag practice EAF arc shielding refractory protection | foamy slag practice electric arc furnace radiation shielding ladle furnace |
| 11 | 0.1176 | 0.2857 | 0.1681 | "foamy slag" ladle furnace carbon injection long arc protection | ladle furnace arc radiation refractory degradation resin bonded MgO-C brick temperature profile |
| 1 | 0.4000 | 0.5556 | 0.1556 | BOF skull formation lance height prevention | lance height secondary oxygen flow rate skull formation BOF high scrap ratio |
| 15 | 0.1000 | 0.2400 | 0.1400 | Optimization swirl flow post-combustion oxygen lance parameters CFD 260-ton converter | Swirl number 0.5 secondary oxygen 3% BOF lance tip temperature skull reduction plant trials |
| 8 | 0.0741 | 0.2069 | 0.1328 | BOF specific oxygen blowing rate 2.5 to 4 Nm3 per ton per minute decarburization | CFD simulation BOF dual-flow lance secondary oxygen 3% swirl number 0.5 skull reduction |
| 4 | 0.2500 | 0.3529 | 0.1029 | MgO saturation slag ladle refractory dolomite addition | water-cooled panels ladle furnace slag line arc radiation protection refractory |
| 10 | 0.1053 | 0.2000 | 0.0947 | "resin bonded" "pitch bonded" magnesia carbon brick comparison ladle | ladle furnace arc radiation shielding refractory coating MgO-C brick thermal barrier |

## Biggest regressions compared with context_v1

| query_id | context_v1 F1 | context_prompt_v2 F1 | Delta | Target query | v2 generated query |
|---:|---:|---:|---:|---|---|
| 5 | 0.4762 | 0.2857 | -0.1905 | Post combustion ratio scrap melting heat balance BOF increase scrap | lance skull adhesion prevention refractory coating materials BOF high scrap ratio |
| 14 | 0.2500 | 0.0769 | -0.1731 | Lance height increase higher post combustion ratio FeO foamy slag PCR converter | CFD simulation post-combustion lance swirl number 0.5 secondary oxygen 3% BOF scrap melting |
| 17 | 0.4348 | 0.2963 | -0.1385 | Post-combustion oxygen secondary 3% total oxygen flow LD3 trials | Swirl number 0.5 secondary oxygen 3% lance height 1.8 m scrap rate 27% plant trials BOF |
| 12 | 0.2222 | 0.0909 | -0.1313 | "ladle metallurgy" arc impingement refractory damage electrode misalignment | ladle furnace foamy slag carbon injection rate kg/min CO gas generation foam height |
| 2 | 0.4444 | 0.3158 | -0.1287 | Basic oxygen furnace high scrap ratio post combustion efficiency | post-combustion lance dual-flow oxygen ratio scrap rate BOF PCR improvement |
| 4 | 0.2222 | 0.1000 | -0.1222 | BOF bottom stirring argon flow rate combined blowing converter | lance tip erosion copper-silver alloy skull formation BOF high scrap ratio |
| 9 | 0.1935 | 0.1111 | -0.0824 | Post combustion heat transfer efficiency increases scrap melting CO2 to CO ratio bath surface distance trials | BOF lance height dynamic adjustment strategy scrap ratio 25% blow start 1.2 m end 2.0 m skull reduction |
| 7 | 0.1481 | 0.0741 | -0.0741 | Post combustion efficiency heat transfer efficiency BOF percent CO2 CO optimal lance distance | MgO-saturated slag basicity 3.0-3.5 FeO content skull adhesion prevention BOF high scrap |
