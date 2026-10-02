"""Pega solo al chico generado por Kling sobre el plate original.
Fuera de la mascara, los pixeles son los del plate sin tocar."""
import cv2, numpy as np, subprocess, sys
P='/home/user/VFX/assets/plate_original.mp4'
K=sys.argv[1]; OUT=sys.argv[2]; MATTE=sys.argv[3] if len(sys.argv)>3 else None
RX0,RY0,RX1,RY1=1189,305,1440,805          # rectangulo pegado en el plate
ROI=(1080,240,1600,1330)                     # zona donde puede estar el chico
cp,ck=cv2.VideoCapture(P),cv2.VideoCapture(K)
W,H=int(cp.get(3)),int(cp.get(4)); N=int(cp.get(7))
enc=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r','24','-i','-',
    '-i',P,'-map','0:v','-map','1:a?','-c:v','libx264','-crf','16','-preset','slow','-pix_fmt','yuv420p',
    '-c:a','copy','-shortest',OUT],stdin=subprocess.PIPE)
menc=None
if MATTE:
    menc=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','gray','-s',f'{W}x{H}','-r','24','-i','-',
        '-c:v','libx264','-crf','16','-pix_fmt','yuv420p',MATTE],stdin=subprocess.PIPE)
rng=np.random.default_rng(7)
x0,y0,x1,y1=ROI
rect=np.zeros((H,W),np.uint8); rect[RY0:RY1,RX0:RX1]=255
ker=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
def hp_std(img,m):
    g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.float32)
    return (g-cv2.GaussianBlur(g,(0,0),1.5))[m].std()
for i in range(N):
    okp,p=cp.read(); okk,k=ck.read()
    if not (okp and okk): break
    k=cv2.resize(k,(W,H),interpolation=cv2.INTER_LANCZOS4)
    pb,kb=cv2.GaussianBlur(p,(5,5),0),cv2.GaussianBlur(k,(5,5),0)
    d=cv2.absdiff(pb,kb).max(2)
    # mascara: lo que cambio dentro de la ROI
    m=np.zeros((H,W),np.uint8)
    m[y0:y1,x0:x1]=(d[y0:y1,x0:x1]>16).astype(np.uint8)*255
    m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,ker,iterations=2)
    # quedarse solo con lo conectado al chico/rectangulo
    n,lab,st,_=cv2.connectedComponentsWithStats(cv2.bitwise_or(m,rect))
    keep=np.isin(lab,[l for l in range(1,n) if (lab[RY0:RY1,RX0:RX1]==l).any()])
    m=(keep*255).astype(np.uint8)
    m=cv2.dilate(m,ker,iterations=1)
    # rellenar huecos
    cnts,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(m,cnts,-1,255,-1)
    a=cv2.GaussianBlur(m.astype(np.float32)/255,(0,0),6)
    a=np.maximum(a,cv2.GaussianBlur(cv2.dilate(rect,ker,iterations=2).astype(np.float32)/255,(0,0),3))
    # color: ajustar Kling al plate con el fondo alrededor (anillo fuera de la mascara)
    ring=(cv2.dilate(m,ker,iterations=6)>0)&(a<0.01)
    ring[:y0]=False; ring[y1:]=False
    kc=k.astype(np.float32)
    for c in range(3):
        ps,ks=p[...,c][ring].astype(np.float32),kc[...,c][ring]
        g=np.clip(ps.std()/max(ks.std(),1e-3),0.9,1.1)
        kc[...,c]=(kc[...,c]-ks.mean())*g+ps.mean()
    # grano: igualar ruido del plate
    sm=a>0.5
    ns=np.sqrt(max(hp_std(p,ring)**2-hp_std(np.clip(kc,0,255).astype(np.uint8),sm)**2,0))
    kc+=rng.normal(0,ns,(H,W,1)).astype(np.float32)
    out=p.astype(np.float32)*(1-a[...,None])+kc*a[...,None]
    enc.stdin.write(np.clip(out,0,255).astype(np.uint8).tobytes())
    if menc: menc.stdin.write((a*255).astype(np.uint8).tobytes())
    if i==0: print('grain sigma',round(float(ns),2))
enc.stdin.close(); enc.wait()
if menc: menc.stdin.close(); menc.wait()
print('frames',i+1)
