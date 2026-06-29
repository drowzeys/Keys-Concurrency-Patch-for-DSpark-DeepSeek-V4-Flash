#!/usr/bin/env python3
"""MATH (hendrycks_math) eval on B — controlled: thinking off, \\boxed{} extraction,
math_verify equivalence scoring. Run with the eval-venv python (pyarrow+math_verify)."""
import json, re, sys, time, urllib.request, threading, glob, random
import pyarrow.parquet as pq
from math_verify import parse, verify

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://10.100.10.2:8888"
PER  = int(sys.argv[2]) if len(sys.argv) > 2 else 30   # per subject
CONC = int(sys.argv[3]) if len(sys.argv) > 3 else 8
MODEL= "deepseek-v4-flash-dspark"
ROOT = "/home/keyspark/.cache/huggingface/hub/datasets--EleutherAI--hendrycks_math/snapshots/*/"

def last_boxed(s):
    i = s.rfind("\\boxed")
    if i < 0: return None
    j = s.find("{", i)
    if j < 0: return None
    depth=0
    for k in range(j, len(s)):
        if s[k]=="{": depth+=1
        elif s[k]=="}":
            depth-=1
            if depth==0: return s[j+1:k]
    return None

probs=[]
for subj_dir in sorted(glob.glob(ROOT+"*/")):
    fs=glob.glob(subj_dir+"test-*.parquet")
    if not fs: continue
    t=pq.read_table(fs[0]).to_pydict()
    rows=list(zip(t["problem"], t["solution"]))
    for q,sol in rows[:PER]:
        g=last_boxed(sol)
        if g: probs.append((q,g))
print(f"loaded {len(probs)} MATH problems ({PER}/subject)")

res={}
def ask(idx):
    q,g=probs[idx]
    body={"model":MODEL,"messages":[{"role":"user","content":q+"\nSolve step by step. Put your final answer in \\boxed{}."}],
          "temperature":0.0,"max_tokens":1024,"stream":False,"chat_template_kwargs":{"thinking":False}}
    req=urllib.request.Request(BASE+"/v1/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    for a in range(3):
        try:
            with urllib.request.urlopen(req,timeout=300) as r:
                txt=json.loads(r.read().decode())["choices"][0]["message"]["content"] or ""
            res[idx]={"pred":last_boxed(txt),"gold":g}; return   # score later in main thread
        except Exception as e:
            if a==2: res[idx]={"pred":None,"gold":g,"err":str(e)[:40]}
            else: time.sleep(2)

from queue import Queue
work=Queue()
for i in range(len(probs)): work.put(i)
def worker():
    while not work.empty():
        try: i=work.get_nowait()
        except: return
        ask(i); work.task_done()
t0=time.perf_counter()
ths=[threading.Thread(target=worker) for _ in range(CONC)]
for t in ths: t.start()
for t in ths: t.join()
dt=time.perf_counter()-t0
# score in MAIN thread (math_verify uses signal-based timeout -> main-thread only)
for i in res:
    p=res[i]["pred"]; g=res[i]["gold"]; ok=False
    if p:
        try: ok=bool(verify(parse("$"+g+"$"), parse("$"+p+"$")))
        except: ok=False
    res[i]["ok"]=ok
correct=sum(1 for i in res if res[i]["ok"])
print(f"MATH accuracy={correct/len(probs):.4f} ({correct}/{len(probs)}) in {dt:.0f}s, conc={CONC}")
json.dump({str(i):res[i] for i in sorted(res)}, open("/home/keyspark/dspark-60/math_result.json","w"))
print("MATH_EVAL_DONE")
