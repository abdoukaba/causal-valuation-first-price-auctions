import json, cert_calib as cc
D=[[str(0.06+0.03*m) for m in (0,.05,.1)],[str(0.095+0.03*m) for m in (0,.05,.1)]]
out={}
for rm in (None,'0.2','0.1','0.05','0.02'):
    for lab,(T,lam,eps,eta) in (('long',(1,'2.25','0.05','0.45')),('short',('0.2','0.1','0.5','0.5'))):
        out[f'{lab},{rm}']=cc.best([0,.05,.1],['0.2','0.4','0.4'],D,['-0.01','0.01'],'0.2',T,lam,eps,eta,rho_min=rm,nt=4000)
        print(lab,rm,out[f'{lab},{rm}'],flush=True)
json.dump(out,open('kl_ref_table.json','w'),indent=1)
