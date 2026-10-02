import cv2, numpy as np, sys
P='/home/user/VFX/assets/plate_original.mp4'
K=sys.argv[1]
def frames(p, n=None):
    c=cv2.VideoCapture(p); out=[]
    while True:
        ok,f=c.read()
        if not ok or (n and len(out)>=n): break
        out.append(f)
    return out
pf=frames(P); kf=frames(K,40)
H,W=pf[0].shape[:2]
kf=[cv2.resize(f,(W,H),interpolation=cv2.INTER_CUBIC) for f in kf]
orb=cv2.ORB_create(4000)
bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
def g(f): return cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
for i in range(len(pf)):
    errs=[]
    for d in (-1,0,1):
        j=i+d
        if j<0 or j>=len(kf): continue
        e=np.mean(np.abs(g(pf[i])[:500].astype(int)-g(kf[j])[:500].astype(int)))
        errs.append((round(e,2),d))
    a,b=g(pf[i]),g(kf[i])
    k1,d1=orb.detectAndCompute(a,None); k2,d2=orb.detectAndCompute(b,None)
    m=bf.match(d1,d2)
    s=np.float32([k2[x.trainIdx].pt for x in m]); t=np.float32([k1[x.queryIdx].pt for x in m])
    M,inl=cv2.estimateAffinePartial2D(s,t,ransacReprojThreshold=2)
    sc=np.hypot(M[0,0],M[1,0])
    print(i,errs,'scale %.4f tx %.2f ty %.2f'%(sc,M[0,2],M[1,2]),int(inl.sum()))
