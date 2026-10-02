"""Pega solo al chico generado por Kling sobre el plate original.
Fuera de la mascara, los pixeles son los del plate sin tocar.
uso: composite.py kling.mov salida.mp4 [matte.mp4]"""
import cv2, numpy as np, subprocess, sys
P='/home/user/VFX/assets/plate_original.mp4'
K=sys.argv[1]; OUT=sys.argv[2]; MATTE=sys.argv[3] if len(sys.argv)>3 else None
RX0,RY0,RX1,RY1=1189,305,1440,805            # rectangulo pegado en el plate
x0,y0,x1,y1=1080,240,1600,1330                # zona donde puede estar el chico
GRAIN=0.45                                    # sigma del grano (calibrado contra el plate)
SHARP=0.1                                     # realce minimo (mas, y se ve recortado)

def read(p):
    c=cv2.VideoCapture(p); fs=[]
    while True:
        ok,f=c.read()
        if not ok: break
        fs.append(f)
    return fs
pf=read(P); H,W=pf[0].shape[:2]; N=len(pf)
kf=[cv2.resize(f,(W,H),interpolation=cv2.INTER_LANCZOS4) for f in read(K)[:N]]

rect=np.zeros((H,W),np.uint8); rect[RY0:RY1,RX0:RX1]=255
ell=lambda r:cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(r,r))

# 1) mascara por cuadro: lo que cambio, conectado al chico
masks=[]
for p,k in zip(pf,kf):
    d=cv2.absdiff(cv2.GaussianBlur(p,(5,5),0),cv2.GaussianBlur(k,(5,5),0)).max(2)
    m=np.zeros((H,W),np.uint8)
    m[y0:y1,x0:x1]=(d[y0:y1,x0:x1]>16).astype(np.uint8)*255
    m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    m=cv2.bitwise_or(m,rect)
    # conectar con un radio grande para que un brazo que cruza no corte las piernas
    n,lab,st,_=cv2.connectedComponentsWithStats(cv2.dilate(m,ell(41)))
    ids=[l for l in range(1,n) if (lab[RY0:RY1,RX0:RX1]==l).any()]
    m=cv2.bitwise_and(m,np.isin(lab,ids).astype(np.uint8)*255)
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,ell(15),iterations=2)
    cnts,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(m,cnts,-1,255,-1)
    masks.append(cv2.dilate(m,ell(15)))
# 2) continuidad: union con los cuadros vecinos, sin saltos de un cuadro a otro
alphas=[]
for i in range(N):
    m=np.maximum.reduce([masks[j] for j in range(max(0,i-1),min(N,i+2))])
    # quitar islas sueltas (destellos)
    n,lab,st,_=cv2.connectedComponentsWithStats(m)
    m=np.isin(lab,[l for l in range(1,n) if st[l,4]>3000]).astype(np.uint8)*255
    a=cv2.GaussianBlur(m.astype(np.float32)/255,(0,0),6)
    a=np.maximum(a,cv2.GaussianBlur(cv2.dilate(rect,ell(15),iterations=2).astype(np.float32)/255,(0,0),3))
    alphas.append(a)
# 3) color: ganancia/offset por canal con el fondo alrededor, suavizado en el tiempo
params=[]
for p,k,m in zip(pf,kf,masks):
    ring=(cv2.dilate(m,ell(15),iterations=6)>0)&(m==0)
    ring[:y0]=False; ring[y1:]=False
    pr=[]
    for c in range(3):
        ps,ks=p[...,c][ring].astype(np.float32),k[...,c][ring].astype(np.float32)
        pr.append((ks.mean(),np.clip(ps.std()/max(ks.std(),1e-3),0.92,1.08),ps.mean()))
    params.append(np.array(pr))
params=np.array(params)
sm=np.copy(params)
for i in range(N):
    sm[i]=params[max(0,i-3):i+4].mean(0)

rng=np.random.default_rng(7)
enc=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r','24','-i','-',
    '-i',P,'-map','0:v','-map','1:a?','-c:v','libx264','-crf','12','-preset','slow','-pix_fmt','yuv420p',
    '-c:a','copy','-shortest',OUT],stdin=subprocess.PIPE)
menc=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','gray','-s',f'{W}x{H}','-r','24','-i','-',
    '-c:v','libx264','-crf','16','-pix_fmt','yuv420p',MATTE],stdin=subprocess.PIPE) if MATTE else None
for i in range(N):
    p,k,a=pf[i],kf[i].astype(np.float32),alphas[i]
    for c in range(3):
        km,g,pm=sm[i][c]; k[...,c]=(k[...,c]-km)*g+pm
    k=k+SHARP*(k-cv2.GaussianBlur(k,(0,0),1.2))
    # grano tipo celular: mayormente en luz, grano un poco grueso
    n=cv2.GaussianBlur(rng.normal(0,1,(H,W)).astype(np.float32),(0,0),0.6)
    n=n/n.std()*GRAIN
    k=k+n[...,None]
    out=p.astype(np.float32)*(1-a[...,None])+k*a[...,None]
    enc.stdin.write(np.clip(out,0,255).astype(np.uint8).tobytes())
    if menc: menc.stdin.write((a*255).astype(np.uint8).tobytes())
enc.stdin.close(); enc.wait()
if menc: menc.stdin.close(); menc.wait()
print('frames',N,'mask areas',[int((a>0.5).sum()/1000) for a in alphas])
