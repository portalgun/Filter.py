# TODO

import scipy.fft as ft
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import math
import importlib



TAU=2*np.pi
SELF=importlib.import_module("Filter")

def minC(val):
    return np.fmin(np.min(val.real),np.min(val.imag))

def maxC(val):
    return np.fmax(np.max(val.real),np.max(val.imag))

def minmaxC(val):
    return (minC(val),maxC(val),max(np.abs(val)))

def iscomplex(val):
    return np.any(np.imag(val) != 0)


#- plot
def plotFT(x,y,colorR='blue',colorI='red',colorA='black',linestyleR='-',linestyleI='-',linestyleA=':',**kwargs):
    if iscomplex(y):
        plt.plot(x,np.real(y),color=colorR,linestyle=linestyleR,**kwargs)
        plt.plot(x,np.imag(y),color=colorI,linestyle=linestyleI,**kwargs)
        plt.plot(x, np.abs(y),color=colorA,linestyle=linestyleA,**kwargs)
    else:
        plt.plot(x,y,color=colorR,linestyle=linestyleR,**kwargs)
    #plt.plot(x,-np.abs(y),color=colorA,linestyle=linestyleA,*kwargs)

def fft(x,yS,bCtrOut=None):
    if bCtrOut is None:
        bCtrOut=x.bCtrF
    bCtrIn=x.bCtrS

    if isinstance(bCtrOut,tuple):
        bCtrOut=list(bCtrOut)
    elif not isinstance(bCtrOut,list):
        bCtrOut=[bCtrOut]

    if isinstance(bCtrIn,tuple):
        bCtrIn=list(bCtrIn)
    elif not isinstance(bCtrIn,list):
        bCtrIn=[bCtrIn]


    ctrIn=np.arange(len(bCtrIn))[bCtrIn]
    ctrOut=np.arange(len(bCtrOut))[bCtrOut]

    if any(bCtrIn):
        out=ft.fftn(ft.ifftshift(yS,axes=ctrIn))
    else:
        out=ft.fftn(yS)

    if any(bCtrOut):
        return ft.fftshift(out,axes=ctrOut)
    else:
        return out

        #return ft.ifftshift(out)

def ifft(x,yF,bCtrOut=None):
    if bCtrOut is None:
        bCtrOut=x.bCtrS
    bCtrIn=x.bCtrF

    if isinstance(bCtrOut,tuple):
        bCtrOut=list(bCtrOut)
    elif not isinstance(bCtrOut,list):
        bCtrOut=[bCtrOut]

    if isinstance(bCtrIn,tuple):
        bCtrIn=list(bCtrIn)
    elif not isinstance(bCtrIn,list):
        bCtrIn=[bCtrIn]


    ctrIn=np.arange(len(bCtrIn))[bCtrIn]
    ctrOut=np.arange(len(bCtrOut))[bCtrOut]

    if any(bCtrIn):
        out= ft.ifftn(ft.ifftshift(yF,axes=ctrIn))
    else:
        out=ft.ifftn(yF)

    if any(bCtrOut):
        return ft.fftshift(out,axes=ctrOut)
    else:
        return out

def spread(x,y):
    return np.sqrt( np.sum( np.square(x-mean_time(x,y)) * np.square(np.abs(y)) ) )

def mean_time(x,y):
# mean time -> centrality
    return np.sum( x * np.square(np.abs(y) ))/ np.sum( np.square(np.abs(y)) ) 

def flipl(xF,yF,x0=0):
    # flip parity -> make temporal real not imaginary
    yF[xF<x0]=yF.real[xF<x0] - 1j*yF.imag[xF<0]
    return yF

def nApprxF(n,f,totS):
    # only works for integers
    nmin=totS/math.gcd(totS,f)+1
    return np.int64(np.ceil(n/nmin)*nmin)

def carS(xS,f,phi=0,bQuad=True,loc=0):
    if isinstance(xS,X):
        xS=xS.s

    # provide loc in reference to envelope
    k=TAU*f *(xS-loc)+ phi
    return np.cos(k) + ( int(bQuad)* 1j * np.sin(k))


def carF(xF,f,phi=0,bQuad=True,loc=0):
    if isinstance(xF,X):
        x=xF
    else:
        x=xF2x(xF)

    return fft(x,carS(x.s,f,phi=phi,bQuad=bQuad,loc=loc))/len(x.f)

def carSF(xF,phi=0,loc=0,f=0):
    if isinstance(xF,X):
        xF=xF.f
    # provide f in *de*ference from enevelop
    #  TODO check sign
    return np.exp(-1j*loc*(xF+f)*TAU + 1j*phi)


def carFApprx(x,f,phi=0,bQuad=True,loc=0):
    # TODO mag and phase
    # an approximation
    # http://www.nicholson.com/rhn/dsp.html#4

    if isinstance(x,X):
        n=x.n
        smpS=x.smpS
        x=x.f
    else:
        n=len(x)
        smpS=np.max(x)-np.min(x) # XXX Check

    n=len(x)
    A=n

    #Hm = np.sin((n) * np.pi*(f-x)/smpS) / ((n) * np.sin(np.pi*(f-x)/smpS))
    #Hc = np.sin((n) * np.pi*(f+x)/smpS) / ((n) * np.sin(np.pi*(f+x)/smpS))


    if f==0:
        m=0
    else:
        m=TAU*loc/f

    out=Hm
    out=A*(Hm**2 + Hc**2 + 2*Hm*Hc*np.cos(2*phi))
    #out=A*(Hm**2 + Hc**2 - 2*Hm*Hc*np.cos(2*(phi-m)) )
    #out[np.isnan(out)]=A
    out=out * np.exp(1j*(phi))



    if bQuad:
        out[x<0]=0
    else:
        flipl(x,out)

    return out

def dirichlet(x):
    n=len(x)
    return np.sin((n+1/2)*x)/np.sin(x/2)

def _id(w):
    return w

def xS2x(xS):
    return X(n=len(xS),totS=(xS[-1]-xS[0])+(xS[1]-xS[0]))

def xF2x(xF):
    return X(n=len(xF),totF=(xF[-1]-xF[0])+(xF[1]-xF[0]))


#- x and y
class X():
    #def __set_name__(self, owner, name):
    #    self.name = name


    #def __set__(self, instance, value):
    #    vars(getattr(instance,self.cname))[self.cattr] = value


    def __init__ (self,ndim=None,**kwargs):
        # TODO shape XOR ndim

        if ndim is None:
            dims=[]
            for value in kwargs.values():
                if (isinstance(value,list) or isinstance(value,tuple)):
                    dims.append(len(value))
                ndim=max(dims)


        # parse types and lengths
        for key,value in kwargs.items():
            if isinstance(value,tuple):
                if len(value)==ndim:
                    kwargs[key]=list(value)
                elif len(value)==1:
                    kwargs[key]=list(value)*ndim
                else:
                    raise Exception(key + ' should have a length of 1 or ' + str(ndim) + '. has ' + str(len(value)))
            elif not isinstance(value,list):
                kwargs[key]=[value]*ndim
            elif isinstance(value,list):
                if len(value)==1:
                    kwargs[key]=value*ndim
                elif len(value)!=ndim:
                    raise Exception(key + ' should have a length of 1 or ' + str(ndim) + '. has ' + str(len(value)))

        self._dims=[]
        for i in range(ndim):
            dic={}
            for key,value in kwargs.items():
                dic[key]=value[i]
            self._dims.append(_X(**dic))

    @property
    def ndim(self):
        return np.size(self._dims)

    @property
    def shape(self):
        return self.n

    @property
    def f(self):
        return self._x_fld('f')

    @property
    def s(self):
        return self._x_fld('s')

    @property
    def F(self):
        return XVal(self._x_fld('f'))

    @property
    def S(self):
        return XVal(self._x_fld('s'))

    def _x_fld(self,fld):
        return np.meshgrid(*tuple(getattr(x,fld) for x in self._dims),indexing='xy')

    @property
    def extents(self):
        return np.concat(self.lims)

    @property
    def extentf(self):
        return np.concat(self.limf)

    @property
    def lims(self):
        return self._xlim_fld('s')

    @property
    def limf(self):
        return self._xlim_fld('f')

    @property
    def fl(self):
        return self._xl_fld('f')

    @property
    def sl(self):
        return self._xl_fld('s')

    def _xlim_fld(self,fld):
        return [getattr(x,fld)[np.array([0, -1])] for x in self._dims]

    def _xl_fld(self,fld):
        return tuple(getattr(x,fld) for x in self._dims)

    def __getattr__(self,attr):
        if self.ndim==1:
            return getattr(self._dims[0],attr)
        else:
            return tuple(getattr(x,attr) for x in self._dims)

    def __getitem__(self,ind):
        return self._dims[ind]

    def __setitem__(self,ind,value):
        self._dims[ind]=value

class XVal():
    def __init__(self,value):
        self.value=value


    def __call__(self):
        return self.value

    def __repr__(self):
        return f"_XVal({self.value})"

    def __getitem__(self,ind):
        out=[]
        for i in range(len(self.value)):
            out.append(self.value[i][ind])
        return out

    def __gt__(self,other):
        return np.bitwise_and.reduce([arr > other for arr in self.value])

    def __lt__(self,other):
        return np.bitwise_and.reduce([arr < other for arr in self.value])

    def __ge__(self,other):
        return np.bitwise_and.reduce([arr >= other for arr in self.value])

    def __le__(self,other):
        return np.bitwise_and.reduce([arr <= other for arr in self.value])

    def __eq__(self,other):
        return np.bitwise_and.reduce([arr == other for arr in self.value])

    def __ne__(self,other):
        return np.bitwise_and.reduce([arr != other for arr in self.value])


    @property
    def shape(self):
        return np.shape(self.value[0])

    @property
    def ndim(self):
        return np.ndim(self.value)


class _X():
    # s
    # f
    #

    # n
    # totS
    # totF
    # smpS
    # smpF
    # dS
    # dF
    # bCtrS
    # bCtrF
    # ctrS
    # ctrF
    def __init__(self,n=None,totS=None,totF=None,smpS=None,smpF=None,dS=None,dF=None,bCtrS=True,bCtrF=True):
        self.bCtrS=bCtrS
        self.bCtrF=bCtrF

        # n and something else
        if totS and (smpS or totF) and not n:
            if totF:
                smpS=totF

            totS=totS
            n=int(totS*smpS)
        elif totS is None and n:
            if  smpS is not None:
                totS=n / smpS
            elif dS is not None:
                totS=n * dS
            elif totF is not None:
                totS=n / totF
            elif smpF is not None:
                totS=smpF
            elif dF is not None:
                totS=1 / dF

        # defaults
        if totS:
            self.totS=totS
        else:
            self.totS=1
        if n:
            self.n=n
        else:
            self.n=1000

    @property
    def maxS(self):
        return np.max(self.s)

    @property
    def maxF(self):
        return np.max(self.f)

    @property
    def smpS(self):
        return self.totF

    @property
    def smpF(self):
        return self.totS
    #---------
    @property
    def dS(self):
        return self.totS / self.n

    @property
    def dF(self):
        # XXX
        return self.smpS/self.n

    @property
    def totF(self):
        # XXX
        return self.n / self.totS

    @property
    def ctrS(self):
        return np.ceil((self.n-1)/2) * np.int64(self.bCtrS)

    @property
    def ctrF(self):
        return np.ceil((self.n-1)/2) * np.int64(self.bCtrF)
    ###

    @property
    def s(self):
        # TODO MORE ACCURATE?
        c=self.ctrS
        return np.linspace( -c*self.dS, (self.n-c-1)*self.dS, self.n)
        #return (np.arange(0,self.n) - self.ctrS) * self.dS

    @property
    def f(self):
        # TODO MORE ACCURATE?
        c=self.ctrF
        return np.linspace( -c*self.dF, (self.n-c-1)*self.dF, self.n)
        #return (np.arange(0,self.n) - self.ctrF) * self.dF

    @property
    def stdDevS(self):
        return np.sqrt(self.totS/(TAU*self.totF))

    @property
    def stdDevF(self):
        return np.sqrt(self.totF/(TAU*self.totS))



#- parts
class Part:
    def __init__(self,x,paramsS=None,paramsF=None,amp=1,funcS=None,funcF=None,yS=None,yF=None,par=None,priority=None,typeS=None,typeF=None,ftNormType='none'):
        # TODO ftNormType
        self.amp=amp
        self._funcS=funcS
        self._funcF=funcF
        self._yS_val=yS
        self._yF_val=yF


        self._priority=priority
        self._typeS=typeS
        self._typeF=typeF

        self.paramsS=paramsS
        self.paramsF=paramsF

        self.par=par
        self.x_=x

        self.ftNormType=ftNormType

    def len(self):
        len(self.x)


    @classmethod
    def get(cls,name,x,**kwargs):
        if name is None:
            class_=cls
        else:
            class_=getattr(SELF,name)
            if not issubclass(class_,getattr(SELF,cls.__name__)):
                raise Exception( name + ' is not a valid ' + cls.__name__ + ' type')
        return class_(x,**kwargs)

    @property
    def sibs(self):
        if not self.par:
            return None
        children=self.par.children
        return tuple(x for x in children if x!=self)

    def get_sibs_attr(self,attr):
        if not self.par:
            return None
        children=self.par.children
        return tuple(getattr(x,attr) for x in children if x!=self)

    def get_par_attr(self,attr):
        if not self.par:
            return None
        return getattr(self.par,attr)

    def get_children_attr(self,attr):
        if not ( hasattr(self,'children') ):
            return None

        children=self.par.children
        return tuple(getattr(x,attr) for x in children if x!=self)


    @property
    def nrmS(self):
        if self.ftNormType=='stdS':
                return self.x.n
        elif self.ftNormType=='stdF':
                return 1/self.x.n
        elif self.ftNormType=='unitary':
            return 1/np.sqrt(self.x.n)
        elif self.ftNormType=='none':
            return 1

    @property
    def nrmF(self):
        if self.ftNormType=='stdF':
                return self.x.n
        elif self.ftNormType=='stdS':
                return 1/self.x.n
        elif self.ftNormType=='unitary':
            return 1/np.sqrt(self.x.n)
        elif self.ftNormType=='none':
            return 1

    def _parse_multi(self,name,dict,default,defaultV,bSet=True):
        k=default
        v=defaultV
        s=0

        for key, value in dict.items():
            s=+int(value is not None)
            if value:
                k=key
                v=value

        if s > 1:
            raise Exception('Only one ' + name  + ' type may be set.')

        if bSet:
            setattr(self,k,v)

        return (k,v)

    @property
    def x(self):
        if self.par and self.par.x:
            return self.par.x
        else:
            return self.x_


    #- funcs
    @property
    def pfuncS(self):
        l=[]
        if not self.paramsS:
            return l

        for p in self.paramsS:
            l.append(getattr(self,p))
        return l


    @property
    def pfuncF(self):
        l=[]
        if not self.paramsF:
            return l

        for p in self.paramsF:
            l.append(getattr(self,p))
        return l

    @property
    def funcS(self):
        if self._funcS:
            return self._funcS
        else:
            return None
            return lambda *_ : np.full_like(self.x.s,np.nan)

    @property
    def funcF(self):
        if self._funcF:
            return self._funcF
        else:
            return None
            return lambda *_ : np.full_like(self.x.f,np.nan)


    #- types
    @property
    def priority(self):
        if self._priority:
            return self._priority
        else:
            return ['val','val_ft','func','func_ft','div','div_ft']

    @property
    def types(self):
        valS=(self._yS_val is True)
        valF=(self._yF_val is True)
        funS=callable(self.funcS)
        funF=callable(self.funcF)
        divS=self.par is not None and isinstance(self.par,Filter) and  (self.par.typeS=="val" or self.par.typeS=="func")
        divF=self.par is not None and isinstance(self.par,Filter) and  (self.par.typeF=="val" or self.par.typeF=="func")

        return {
             "val" :       [valS, valF],
             "val_ft":     [valF, valS],
             "func":       [funS, funF],
             "func_ft":    [funF, funS],
             "div" :       [divS, divF],
             "div_ft" :    [divF, divS],
        }

    @property
    def typeS(self):
        if self._typeS:
            return self._typeS

        types=self.types
        for p in self.priority:
            if types[p][0]:
                return p

    @property
    def typeF(self):
        if self._typeF:
            return self._typeF

        types=self.types
        for p in self.priority:
            if types[p][1]:
                return p

    #- y
    @property
    def yS(self):
        return self.nrmS * getattr(self, 'yS_' + self.typeF )

    @property
    def yF(self):
        return self.nrmS * getattr(self, 'yF_' + self.typeF )

    #- y components
    @property
    def yIS(self):
        return self.yS.real

    @property
    def yQS(self):
        return 1j*self.yS.imag

    @property
    def yAS(self):
        return np.abs(self.yS)

    @property
    def yIF(self):
        return self.yF.real

    @property
    def yQF(self):
        return 1j*self.yF.imag

    @property
    def yAF(self):
        return np.abs(self.yF)

    #- derivatives
    @property
    def dyS(self):
        return self.nrmS * ifft(self.x,1j*self.x.f*self.yF)

    @property
    def dyF(self):
        return self.nrmF * fft(self.x,1j*self.x.s*self.yS)

    #- derivative components
    @property
    def dyIS(self):
        return self.dyS.real

    @property
    def dyQS(self):
        return 1j*self.dyS.imag

    @property
    def dyAS(self):
        return np.abs(self.dyS)

    @property
    def dyIF(self):
        return self.dyF.real

    @property
    def dyQF(self):
        return 1j*self.dyF.imag

    @property
    def dyAF(self):
        return np.abs(self.dyF)

    #- antiderivatives
    @property
    def DyS(self):
    # XXX check
        yF=self.yF
        xF=self.x.f

        # handle x.f==0
        xind=xF==0
        if any(xind):
            xF[xind]=1
            DC=yF[xind]*self.x.s

            y=self.yF/(1j*xF)
            y[xind]=0
        else:
            y=self.yF/(1j*xF)
            DC=0

        return self.nrmS * ifft(self.x,y) + DC

    @property
    def DyF(self):
        yS=self.yS
        xS=self.x.s

        # handle x.s==0
        xind=xS==0
        if any(xind):
            xS[xind]=1
            DC=yS[xind]*self.x.f

            y=self.yS/(1j*xS)
            y[xind]=0
        else:
            y=self.yS/(1j*xS)
            DC=0

        return self.nrmF * fft(self.x,y) + DC

    #- antiderivative components
    @property
    def DyIS(self):
        return self.DyS.real

    @property
    def DyQS(self):
        return 1j*self.DyS.imag

    @property
    def DyAS(self):
        return np.abs(self.DyS)

    @property
    def DyIF(self):
        return self.DyF.real

    @property
    def DyQF(self):
        return 1j*self.DyF.imag

    @property
    def DyAF(self):
        return np.abs(self.DyF)

    #- val

    @property
    def yS_val(self):
        if not self.types['val'][0]:
            return None
        return self._yS_val

    @property
    def yF_val(self):
        if not self.types['val'][1]:
            return None
        return self._yF_val

    #- val fft
    @property
    def yS_val_ft(self):
        if not self.types['val_ft'][0]:
            return None
        return self.nrmS * ifft(self.x,self._yF_val)

    @property
    def yF_val_ft(self):
        if not self.types['val_ft'][1]:
            return None
        return self.nrmF *  fft(self.x,self._yS_val)

    #- func
    @property
    def yS_func(self):
        if not self.types['func'][0]:
            return None
        return self.nrmS * self.amp*self.funcS(self.x.s,*self.pfuncS)

    @property
    def yF_func(self):
        if not self.types['func'][1]:
            return None
        return self.nrmF * self.amp*self.funcF(self.x.f,*self.pfuncF)

    #- fft
    @property
    def yS_func_ft(self):
        if not self.types['func_ft'][0]:
            return None
        return self.nrmS * ifft(self.x,self.yF_func)

    @property
    def yF_func_ft(self):
        if not self.types['func_ft'][1]:
            return None
        return self.nrmF * fft(self.x,self.yS_func)

    def _div_func(self,fld):
        sibs=self.get_sibs_attr(fld)
        if sibs is not None:
            if len(sibs)==1:
                y=sibs[0]
            else:
                y=np.multiply.reduce(sibs)
            return self.get_par_attr(fld) / y

        children=self.get_children_attr(fld)
        if children is not None:
            if len(children)==1:
                return children[0]
            else:
                return np.multiply.reduce(children)

        return None


    @property
    def yS_div(self):
        if not self.types['div'][0]:
            return None
        return self._div_func('yS')

    @property
    def yF_div(self):
        if not self.types['div'][1]:
            return None
        return self._div_func('yF')

    #- div fft
    @property
    def yS_div_ft(self):
        if not self.types['div_ft'][0] and hasattr(self,'yF_div'):
            return None
        return self.nrmS*ifft(self.x,self.yF_div)

    @property
    def yF_div_ft(self):
        if not self.types['div_ft'][1] and hasattr(self,'yF_div'):
            return None
        return self.nrmF*fft(self.x,self.yS_div)


    #- Plotting
    def plotS(self,yflds=None,**kwargs):
        self._plot('s',yflds=yflds,**kwargs)

    def plotF(self,yflds=None,**kwargs):
        self._plot('f',yflds=yflds,**kwargs)

    def plotSF(self,yflds=None,bFitY=False,title="__None__",subs=(1,2,1),**kwargs):
        cls=type(self).__name__

        # ylims
        if bFitY:
            maxS=np.max(np.abs(self.yS))
            minS=np.fmin(minC(self.yS),0)
            rS=abs(minS/maxS)

            maxF=np.max(np.abs(self.yF))
            minF=np.fmin(minC(self.yF),0)
            rF=abs(minF/maxF)
            k=1.05
            if rS >= rF:
                ylimS=(minS*k,     maxS*k)
                ylimF=(-rS*maxF*k, maxF*k)
            else:
                ylimF=(minF*k,     maxF*k)
                ylimS=(-rF*maxS*k, maxF*k)

        else:
            ylimS=None
            ylimF=None

        plt.subplot(subs[0],subs[1],subs[2])
        self._plot('s',yflds=yflds,ylim=ylimS,title=None,**kwargs)
        plt.subplot(subs[0],subs[1],subs[2]+1)
        self._plot('f',yflds=yflds,ylim=ylimF,title=None,**kwargs)

        # title
        if title=="__None__":
            plt.suptitle(cls)
        elif title:
            plt.suptitle(title)

    def plot_stack(self):
        flds=['y','dyI','dyQ','DyI','DyQ']
        n=len(flds)

        for i in range(n):
            self.plotSF(yflds=flds[i],subs=(n,2,(i+1)*2-1))


    def _plot(self,b,yflds=None,bFitX=False,ylim=None,xlim=None,title="__None__",**kwargs):
        if not yflds:
            yflds=tuple('y')
        elif isinstance(yflds,str):
            yflds=tuple([yflds])

        x=getattr(self.x,b)
        for yfld in yflds:
            y=getattr(self,yfld+b.upper())
            plotFT(x,y,**kwargs)

        cls=type(self).__name__

        # xlabel
        if b:
            xlbl='x'+b
        else:
            xlbl='x'
        plt.xlabel(xlbl)

        # xlim
        if bFitX:
            if b=='s' and hasattr(self,'stdDevS'):
                w=self.stdDevS*4
                c1=self.loc
                c2=c1
            elif b=='f' and hasattr(self,'stdDevF'):
                w=self.stdDevF*4
                c=self.f
                if self.bQuad:
                    c1=c
                    c2=c
                else:
                    if c > 0:
                        c1=-c
                        c2=c
                    else:
                        c1=c
                        c2=-c
            else:
                raise Exception('bFitX not handled for class ' + cls )
            plt.xlim(c1-w,c2+w)
        elif xlim:
            plt.xlim(xlim)

        #ylim
        if ylim:
            plt.ylim(ylim)

        # title
        if title=="__None__":
            plt.title(cls)
        elif title:
            plt.title(title)



class Env(Part):
    def __init__(self,x,
                      amp=1,
                      funcS=None,yS=None,
                      funcF=None,yF=None,
                      typeS=None, typeF=None,
                      priority=None,
                      paramsS=None,paramsF=None,
                      par=None
                ):

        super().__init__(x,paramsS=paramsS,paramsF=paramsF,
                           amp=amp,
                           funcS=funcS,funcF=funcF,
                           yS=yS,yF=yF,
                           typeS=typeS,typeF=typeF,
                           par=par,priority=priority)

    @property
    def f(self):
        if self.par and isinstance(self.par,Filter) and hasattr(self.par,'f'):
            return self.par.f
        elif self._f:
            return self._f
        else:
            return 0


    def spreadS(self):
    # uncertainty
        return spread(self.x.s,self.yS)

    def spreadF(self):
    # uncertainty
        return spread(self.x.f,self.yF)



class Car(Part):
    # x
    # y
    # f
    # phi
    # bQuad
    def __init__(self,x,f,phi=0,amp=1,bQuad=True,yS=None,yF=None,typeS=None,typeF=None,priority=None,par=None):

        paramsS=["f","phi","bQuad"]
        paramsF=["f","phi","bQuad"]
        funcS=carS
        funcF=carF


        super().__init__(x,paramsS=paramsS,paramsF=paramsF,
                           amp=amp,
                           funcS=funcS,funcF=funcF,
                           yS=yS,yF=yF,
                           typeS=typeS,typeF=typeF,
                           par=par,priority=priority)
        self.f=f
        self.phi=phi
        self.bQuad=bQuad


class Chr(Part):
    def __init__(self,x,
                 amp=1,
                 funcS=None,funcF=None,
                 paramsS={},paramsF={},
                 yS=None,yF=None,
                 typeS=None,typeF=None,
                 priority=None,
                 par=None):

        super().__init__(x,paramsS=paramsS,paramsF=paramsF,
                           amp=amp,
                           funcS=funcS,funcF=funcF,
                           yS=yS,yF=yF,
                           typeS=typeS,typeF=typeF,
                           par=par,priority=priority)


#- Descriptors
class ChildProp:
    def __init__(self, cname,cattr=None,default=None):
        self.cname=cname
        self._cattr=cattr
        self.default = default

    @property
    def cattr(self):
        if self._cattr:
            return self._cattr
        else:
            return self.name

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        if instance is not None:
            return getattr( getattr(instance,self.cname), self.cattr)
        return self

    def __set__(self, instance, value):
        vars(getattr(instance,self.cname))[self.cattr] = value



class Filter(Part):
    # x
    # y
    #
    # car
    # env
    # xfuncS
    loc   =ChildProp('env')
    widthS=ChildProp('env')
    widthF=ChildProp('env')
    phi   =ChildProp('car')


    def __init__(self,x,
                      car=None,
                      env=None,
                      chr=None,
                      carDict={},
                      envDict={},
                      chrDict={},
                      envType=None,
                      chrType=None,
                      bChr=None,

                      paramsS=None,
                      paramsF=None,
                      amp=1,
                      funcS=None, funcF=None,
                      typeS=None, typeF=None,
                      priority=None,
                      yS=None, yF=None,
                     ):

        if bChr is None:
            bChr=chrType is not None or chr is not None or bool(chrDict)

        self.bChr=bChr

        super().__init__(x,paramsS=paramsS,paramsF=paramsF,
                           amp=amp,
                           funcS=funcS,funcF=funcF,
                           yS=yS,yF=yF,
                           typeS=typeS,typeF=typeF,
                           par=None,priority=priority)

        #- Car
        if car is not None and isinstance(car,Car):
            self.car=car
            self.car.par=self
        else:
            self.car=Car(x,**carDict,par=self)

        #- Env
        if env is not None and isinstance(env,Env):
            self.env=env
            self.env.par=self
        else:
            self.env=Env.get(envType,x,**envDict,par=self)

        #- Chr
        if car is not None and isinstance(chr,Chr):
            self.chr=chr
            self.chr.par=self
        else:
            self.chr=Chr.get(chrType,x,*chrDict,par=self)

    @property
    def children(self):
        if self.bChr:
            return (self.car, self.env, self.chr)
        else:
            return (self.car, self.env)



    #- methods
    def respondS(self,stimS):
        return np.dot(self.yS.flatten(),stimS.flatten())

    def respondF(self,stimF):
        return np.dot(self.yF.flatten(),stimF.flatten())


class GaborF(Filter):
    stdDevS=ChildProp('env')
    stdDevF=ChildProp('env')
    hw     =ChildProp('env')

    def __init__(self,x,f=None,
                      phi=0,
                      loc=0,
                      stdDevS=None,stdDevF=None,hw=None,
                      a=None,obw=None,
                      bQuad=True,
                      amp=1,
                     ):


        # a and obw
        if a and obw:
            raise Exception('only "a" or "obw" can be set')
        elif obw:
            a=GaborF.obw2a(obw)

        # a and widths
        if a and f and (stdDevS or stdDevF or hw):
            raise Exception('only "a" and f, or "a" and a width pramater can be set')
        elif a and f:
            stdDevS=a/f/TAU
        elif not a and not f:
            f=1
        # NOTE see below for not a and f


        self.bQuad=bQuad

        envType='GaussE'
        envDict={
                 'loc':loc,
                 'stdDevS':stdDevS,
                 'stdDevF':stdDevF,
                 'hw':hw
                }

        carDict={
                 'amp':amp,
                 'f':f,
                 'phi':phi,
                 'bQuad':bQuad
                }

        funcS=GaborF.gS_1D
        funcF=GaborF.gF_1D

        paramsS=["f","phi","loc","stdDevS","bQuad"]
        paramsF=["f","phi","loc","stdDevF","bQuad"]


        # TODO
        typeS=None
        typeF=None
        priority=None


        super().__init__(x,
                          carDict=carDict,
                          envDict=envDict,
                          chrDict={},
                          envType=envType,

                          paramsS=paramsS,
                          paramsF=paramsF,
                          funcS=funcS, funcF=funcF,
                          typeS=typeS, typeF=typeF,
                          priority=priority,
                          yS=None, yF=None
         )


        if a and not f:
            self.f=a/self.stdDevS/TAU


    #- funcs
    @staticmethod
    def gS_1D(xS,f,phi,loc,stdDevS,bQuad=True):
        if isinstance(xS,X):
            xS=xS.s

        return  GaussE.gS(xS,loc,stdDevS) * carS(xS,f,phi,bQuad,loc)
        #return  GaussE.gS(xS,loc,stdDevS)

    @staticmethod
    def gF_1D(xF,f,phi,loc,stdDevF,bQuad=True):
        if isinstance(xF,X):
            xF=xF.f

        env=lambda xF : np.exp(-np.square(xF-f)/np.square(stdDevF)/2)
        car=lambda phi: np.exp(-1j*loc*xF*TAU + 1j*phi)
        out=env(xF)*car(phi) + int(not bQuad)*env(-xF)*car(-phi)

        return out/np.sum(np.abs(out))

    #- f
    @property
    def f(self):
        return self.car.f

    @f.setter
    def f(self,val):
        self.car.f=val

    #- a
    @property
    def a(self):
    # no setter - read only
        return self.stdDevS * self.f * TAU

    #- obw
    @property
    def obw(self):
       return GaborF.a2obw(self.a)

    @obw.setter
    def obw(self,val):
    # obw=0.8, 1.4, 2.4
    # a  =4.4, 2.6, 1.7
       self.stdDevS=GaborF.obw2a(val)

    @staticmethod
    def a2obw(a):
        c=np.sqrt(2*np.log(2));
        return np.log2((a+c)/(a-c));

    @staticmethod
    def obw2a(obw):
        return np.sqrt(np.log(4))*(np.exp2(obw)+1)/(np.exp2(obw)-1)

class GaussE(Env):
    def __init__(self,x,
                 loc=0,
                 f=0,
                 stdDevS=None,stdDevF=None,hw=None,
                 par=None
                 ):

        funcS=GaussE.gS
        funcF=GaussE.gF
        paramsS=["loc","stdDevS"]
        paramsF=["f","stdDevF"]


        super().__init__(x,
                         amp=1,
                         yS=None, yF=None,
                         funcS=funcS,funcF=funcF,
                         priority=None,
                         paramsS=paramsS,paramsF=paramsF,
                         par=par
                        )



        widths={
                'stdDevS':stdDevS,
                'stdDevF':stdDevF,
                'hw':hw,
               }

        self._width_parse(widths)
        self.loc=loc
        self._f=f


    def _width_parse(self,widths):
        self._parse_multi('widths',widths,'stdDevS',self.x.stdDevS)

    @staticmethod
    def gS(xS,loc,stdDevS):
        return np.exp(-(xS-loc)**2/(np.square(stdDevS))/2)

    @staticmethod
    def gF(xF,f,stdDevF):
        out=np.exp(-np.square(xF-f)/np.square(stdDevF)/2)
        return out/np.sum(np.abs(out))

    @staticmethod
    def GS(xS,f,stdDevS):
    # ifft of gF -- has frq
        return np.exp(-np.square(xS/stdDevS)/2 )*carS(xS,f,phi=0,bQuad=True,loc=0)

    @staticmethod
    def GF(xF,loc,stdDevF):
    # fft of gS -- has loc
        out=np.exp(-np.square(xF/stdDevF)/2 )* np.exp(-1j*loc*xF*TAU)
        # fsum
        return out/np.sum(np.abs(out))


    @property
    def widthF(self):
       return GaussE.stdDevS2F(self.stdDevS)

    #- stdDevF
    @property
    def stdDevF(self):
       return GaussE.stdDevS2F(self.stdDevS)

    @stdDevF.setter
    def stdDevF(self,val):
       self.stdDevS=GaussE.stdDevF2S(val)

    @staticmethod
    def stdDevS2F(stdDevS):
        return 1/stdDevS/TAU # uncertainty principle

    @staticmethod
    def stdDevF2S(stdDevF):
        return 1/stdDevF/TAU # uncertainty principle

    #- hw
    @property
    def hw(self):
       return GaussE.sigma2hw(self.stdDevS)

    @hw.setter
    def hw(self,val):
       self.stdDevS=GaussE.hw2sigma(val)

    @staticmethod
    def sigma2hw(sigma):
        return  np.sqrt(2*np.log(2))*sigma

    @staticmethod
    def hw2sigma(hw):
        return hw/(np.sqrt(2*np.log(2)))

